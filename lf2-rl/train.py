"""Phase 2: PPO 训练（Stable-Baselines3）

三件事都只为一个目的：一次几小时的长跑不要白费。

1. 每 2 万步存一个 checkpoint —— 中途 Ctrl+C 最多丢 8 分钟
2. 每 10 万步跑一次贪婪评估，`(胜率, 平均奖励)` 破纪录就存 `checkpoints/best_model.zip`
   —— 口径与 `train_DQN.py:745` 的 `best_model.pt` 完全一致
3. 训完再存一份 `lf2_ppo.zip`（最后一个模型，不代表最好）

为什么评估不能用 SB3 自带的 `EvalCallback`：`env.py` 把 ws 服务器写死在 8765 上，
而且是**按实例**起的，一个进程里只能有一个 `LF2Env`。`EvalCallback` 会新建 env，
第二个实例绑定端口失败后线程静默死掉，`reset()` 会干等 120 秒再抛 `TimeoutError`。
所以这里复用训练中的那个 env，并在用完把 `_last_obs` 同步回去。

`smoke_ppo_best.py` 会本文件里的 `LF2Env` / 各路径常量换成假的，用来跑无浏览器的管道测试，
所以这些都留成模块级全局，`main()` 里按名字查表取，不要在函数内重新算。
"""

import argparse
import csv
import os
import time

import numpy as np

from env import DIFF_NAMES, LF2Env, diff_arg, resolve_diff
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

# 1200_000大概8小时
# 训练总步数，可用环境变量覆盖（冒烟测试：LF2_TIMESTEPS=2048 python train.py）
TOTAL_TIMESTEPS = int(os.environ.get("LF2_TIMESTEPS", "300_000"))
# 页面服务端口：只用于打印提示，本脚本本身不监听它（训练用的是 ws 8765，写死在 env.py 里）
HTTP_PORT = os.environ.get("LF2_PORT", "8000")
# 每多少步存一个 checkpoint（可用 LF2_CKPT_FREQ 覆盖）。40fps 下 20000 步≈8 分钟：
# 中途 Ctrl+C 最多丢这么多，不加的话整场训练白跑
CKPT_FREQ = int(os.environ.get("LF2_CKPT_FREQ", "20000"))
# 每多少训练步跑一次贪婪评估。0 = 关闭（短跑 / 调试用）
BEST_EVERY = int(os.environ.get("LF2_BEST_EVERY", "100000"))
# 每次评估跑几局
BEST_EPISODES = int(os.environ.get("LF2_BEST_EPISODES", "5"))
# 落盘位置
SAVE_DIR = os.environ.get("LF2_SAVE_DIR", "checkpoints")
FINAL_PATH = os.environ.get("LF2_FINAL_PATH", "lf2_ppo")
TB_DIR = os.environ.get("LF2_TB_DIR", "./logs/")
# 最优模型；PPO.save() 会自动补 .zip
BEST_PATH = os.environ.get("LF2_BEST_PATH", os.path.join(SAVE_DIR, "best_model"))
# 单次评估里一局最多跑多少步，纯属兜底（正常由 env 的 MAX_EPISODE_STEPS 兜住）
MAX_EVAL_STEPS = 20000

EVAL_CSV_FIELDS = [
    "total_steps", "n", "wins", "loses", "win_rate", "avg_reward",
    "avg_len", "timeout_rate", "best_win_rate",
]


class BestModelCallback(BaseCallback):
    """按「贪婪评估胜率」保存 checkpoints/best_model.zip。

    口径与 train_DQN.py 的 best_model.pt 一致：`(win_rate, avg_reward)` 字典序比较，
    胜率打平才比平均奖励。`best_win_rate` 从 0.0、`best_avg_reward` 从 -inf 起，
    所以**第一次评估必定落盘** —— 和 DQN 一样，一局没赢也会有 best，报告里引用时看清胜率。

    接管 env 的安全性（SB3 2.5 实测）：
      - `on_rollout_end()` 在 `collect_rollouts()` 末尾触发，此时 rollout buffer
        已经 `compute_returns_and_advantage()` 算完；buffer 要等下一次
        `collect_rollouts()` 开头才 `reset()`，所以往里塞别的局面不会污染训练数据。
      - `rollout_buffer` / `ep_info_buffer` / `num_timesteps` 都不受
        `venv.step()` 影响，评估消耗的游戏帧不计入训练预算。
      - 唯一会脏的是 `model._last_obs` 和 `model._last_episode_starts`
        （`collect_rollouts()` 每步更新它们）。评估结束后必须重新 reset 并把它们改回来，
        否则下一次 rollout 会拿着过期观测继续走，那一局的奖励归因全错。
    """

    def __init__(self, every_steps, n_episodes, save_path, verbose=1):
        super().__init__(verbose)
        self.every_steps = every_steps
        self.n_episodes = max(1, n_episodes)
        self.save_path = save_path
        self.best_win_rate = 0.0
        self.best_avg_reward = float("-inf")
        self.last_eval_step = 0
        self.n_evals = 0
        self._csv_path = None

    # ---- 触发 ----------------------------------------------------------

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        if self.every_steps <= 0:
            return
        if self.num_timesteps - self.last_eval_step < self.every_steps:
            return
        self.last_eval_step = self.num_timesteps
        self._evaluate()

    # ---- 评估 ----------------------------------------------------------

    def _evaluate(self) -> None:
        venv = self.training_env
        t0 = time.time()
        self.n_evals += 1
        print(f"\n[评估] 第 {self.num_timesteps} 步，贪婪策略跑 {self.n_episodes} 局...")

        results, rewards, lengths = [], [], []
        failed = None
        try:
            for i in range(self.n_episodes):
                obs = venv.reset()
                ep_rew, ep_len, info = 0.0, 0, {}
                while ep_len < MAX_EVAL_STEPS:
                    actions, _ = self.model.predict(obs, deterministic=True)
                    obs, rews, dones, infos = venv.step(actions)
                    ep_rew += float(rews[0])
                    ep_len += 1
                    info = infos[0]
                    if dones[0]:
                        break
                result = info.get("result", "unknown")
                results.append(result)
                rewards.append(ep_rew)
                lengths.append(ep_len)
                print(f"  [{i + 1}/{self.n_episodes}] {result:<10} "
                      f"reward={ep_rew:>8.1f}  {ep_len:>5} 步")
        except Exception as e:                      # noqa: BLE001
            failed = e
        finally:
            # 评估把游戏跑到别的局里去了，不管跑没跑完都必须把训练现场收回来，
            # 否则下一次 collect_rollouts 会拿着过期观测继续走。
            # 这里的 reset 若也失败，说明浏览器已经不可用，让它抛出去（训练继续下去也是垃圾）。
            obs = venv.reset()
            self.model._last_obs = obs
            if hasattr(self.model, "_last_episode_starts"):
                self.model._last_episode_starts = np.ones((venv.num_envs,), dtype=bool)
            if hasattr(self.model, "_last_original_obs"):  # 只在套了 VecNormalize 时存在
                self.model._last_original_obs = obs

        if failed is not None:
            # 一次评估跑失败不该连累几小时的训练：跳过这次，best 保持上一次的
            print(f"[评估] 没跑完，已跳过（训练继续）：{type(failed).__name__}: {failed}")
            return

        n = len(results)
        wins = results.count("win")
        loses = results.count("lose")
        win_rate = wins / n
        avg_reward = sum(rewards) / n
        avg_len = sum(lengths) / n
        timeout_rate = sum(1 for r in results if r in ("max_steps", "timeout")) / n

        improved = (win_rate, avg_reward) > (self.best_win_rate, self.best_avg_reward)
        if improved:
            self.best_win_rate = win_rate
            self.best_avg_reward = avg_reward
            folder = os.path.dirname(self.save_path)
            if folder:
                os.makedirs(folder, exist_ok=True)
            self.model.save(self.save_path)
            print(f"  ★ 新最佳胜率 {win_rate:.1%}！模型已存到 {self.save_path}.zip")

        print(f"[评估] 胜率 {win_rate:.1%} ({wins}胜/{loses}负/{n - wins - loses}其他)  "
              f"平均奖励 {avg_reward:.1f}  平均 {avg_len:.0f} 步  "
              f"超时率 {timeout_rate:.0%}  用时 {time.time() - t0:.0f}s"
              + ("" if improved else f"（未破纪录，best 仍是 {self.best_win_rate:.1%}）"))

        if self._tb is not None:
            self._tb.record("eval/win_rate", win_rate)
            self._tb.record("eval/best_win_rate", self.best_win_rate)
            self._tb.record("eval/avg_reward", avg_reward)
            self._tb.record("eval/timeout_rate", timeout_rate)
            self._tb.record("eval/avg_len", avg_len)

        self._append_csv(dict(
            total_steps=self.num_timesteps, n=n, wins=wins, loses=loses,
            win_rate=round(win_rate, 4), avg_reward=round(avg_reward, 2),
            avg_len=round(avg_len, 1), timeout_rate=round(timeout_rate, 4),
            best_win_rate=round(self.best_win_rate, 4),
        ))

    # ---- 落 CSV --------------------------------------------------------

    @property
    def _tb(self):
        """SB3 的 logger。`BaseCallback.logger` 是个转发到 `model.logger` 的 property，
        模型还没 `_setup_learn()` 过时会直接抛 AttributeError（单测里直接调 _evaluate 就会踩到），
        所以这里自己取，取不到就当没有。"""
        return getattr(self.model, "_logger", None)

    def _append_csv(self, row):
        if self._csv_path is None:
            # 有 TensorBoard 时 logger.dir 就是 logs/PPO_N；没有就落到 checkpoints/
            folder = getattr(self._tb, "dir", None)
            folder = folder or os.path.dirname(self.save_path) or "."
            os.makedirs(folder, exist_ok=True)
            self._csv_path = os.path.join(folder, "ppo_eval.csv")
        new_file = not os.path.isfile(self._csv_path)
        with open(self._csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=EVAL_CSV_FIELDS)
            if new_file:
                writer.writeheader()
            writer.writerow(row)


def main(diff=None):
    print("初始化环境...")
    # 难度在本地先解析一遍：冒烟测试会把 LF2Env 换成没有 .diff 的假环境，
    # 打印不能依赖 env 实例上的属性。
    level = resolve_diff(diff)
    env = LF2Env(diff=diff)

    print("请先启动静态服务和浏览器（两个终端，端口要一致）：")
    print(f"  python serve.py --port {HTTP_PORT}")
    print(f"  python launch_browser.py --port {HTTP_PORT}")
    print("或者手动打开页面，地址为：")
    print(f"  http://localhost:{HTTP_PORT}/LF2-RL/game/game.html?debug&headless")
    print("（8000 被占用时：set LF2_PORT=8010 再跑本脚本；ws 桥固定 8765，不用管）")
    if level is not None:
        print(f"[难度] 请求 {DIFF_NAMES[level]}（diff={level}）；"
              f"env.py 会切换已打开的浏览器并校验，切不过去会直接报错。")
    else:
        print("[难度] 未指定 --diff，沿用页面 URL 里的难度（旧行为）。")
    print("等待连接...")

    model = PPO(
        "MlpPolicy",
        env,
        n_steps=2048,        # 原来 512，调大让每次更新见到更多样本
        batch_size=64,
        learning_rate=1e-4,
        n_epochs=5,
        gamma=0.99,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log=TB_DIR,
    )

    print("开始训练！浏览器最小化即可，训练在后台继续")
    # 定期把权重存进 checkpoints/，这样中途 Ctrl+C 也不会整场白跑
    checkpoint = CheckpointCallback(
        save_freq=CKPT_FREQ,
        save_path=SAVE_DIR,
        name_prefix="ppo",
    )
    # 定期贪婪评估，破纪录就把权重存成 checkpoints/best_model.zip
    best_cb = BestModelCallback(BEST_EVERY, BEST_EPISODES, BEST_PATH)
    if BEST_EVERY > 0:
        print(f"[最佳模型] 每 {BEST_EVERY} 步贪婪评估 {BEST_EPISODES} 局，"
              f"破纪录存 {BEST_PATH}.zip")
    else:
        print("[最佳模型] LF2_BEST_EVERY=0，已关闭贪婪评估")

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[checkpoint, best_cb],   # 原来 100k，至少跑 500k 才能看到学习效果
    )
    model.save(FINAL_PATH)

    print(f"训练完成！最后一个模型已保存为 {FINAL_PATH}.zip")
    if best_cb.best_avg_reward > float("-inf"):
        print(f"最优模型（贪婪胜率 {best_cb.best_win_rate:.1%}）"
              f"已保存为 {BEST_PATH}.zip —— 评估/部署用这个，不是 {FINAL_PATH}.zip")
    else:
        print("这次训练没有跑过任何一次评估，所以没有 best_model.zip")

    return model, best_cb


def _parse_args(argv=None):
    """只解析 --diff。

    刻意不在 main() 里解析：smoke_ppo_best.py 直接无参调用 train.main()，
    若 main() 自己读 sys.argv，冒烟测试就会被它自己的参数搞挂。
    """
    parser = argparse.ArgumentParser(description="PPO 训练（Stable-Baselines3）")
    parser.add_argument(
        "--diff",
        type=diff_arg,
        choices=("easy", "normal", "difficult"),
        default=None,
        help="对手难度，可写 easy / normal / difficult 或 0 / 1 / 2；"
             "不给则沿用页面 URL 里的设置（旧行为）。"
             "指定后 env.py 会切换已打开的浏览器并校验。",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    main(diff=_parse_args().diff)
