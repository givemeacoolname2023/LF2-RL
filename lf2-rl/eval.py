"""
eval.py —— 用训好的模型（PPO 或 DQN）打 N 局，统计胜率和行为特征。

为什么需要它：
    train.py 只打 ep_rew_mean，而那个指标会被「距离奖励」灌水 ——
    模型站在中距离不动也能拿 400 多分，看起来"在学"，其实一招没出。
    这个脚本直接数三件事：赢没赢、出没出招、站多远。这三个数骗不了人。

随机策略的对照基线：11 个动作里含 att 的有 5 个，所以随机出招占比 = 45.5%。
模型如果远低于这个数，说明它学的是「别打」，而不是「怎么打」。

用法（需要 serve.py + launch_browser.py 已启动，且 train.py 已停止 —— 8765 独占）：
    python eval.py                                      # 默认跑 checkpoints 里最新的模型，20 局
    python eval.py --model lf2_ppo.zip --episodes 50     # 跑指定的模型
    set LF2_MAX_STEPS=400 && python eval.py --episodes 2 # 短局冒烟，几分钟跑完

    
评估可视化
# 终端 A（serve.py）—— 全程不用管，一直在跑
# 终端 B
python launch_browser.py --port 8010 --visible --path "/LF2-RL/game/game.html?debug"
# 终端 C
python eval.py --episodes 1

不做的事：
    不修改 env.py / train.py，只读地用它们。模型是 deterministic 选动作（不打采样噪声），
    这样同一局条件固定时结果可复现。
"""

import argparse
import glob
import json
import os
import re
import socket
import sys
import time
from collections import Counter

import numpy as np
import torch

from env import (ACTION_KEYS, DIFF_NAMES, LF2Env, OBS_SCALE,
                 diff_arg, resolve_diff)

# 观测下标 / 出招判定 / Wilson CI 统一定义在 rl_metrics.py，
# 因为 train_DQN.py 的 evaluate() 也要用同一套口径，两边不能各写一份。
from rl_metrics import (        # noqa: F401 —— 这些名字仍然是 eval.py 的公开属性
    ATTACK_ACTIONS, IDX_D_HP, IDX_DX, IDX_E_HP,
    RANDOM_ATTACK_RATE, action_name, wilson,
)

WS_PORT = 8765



def ws_port_busy(port=WS_PORT):
    """8765 有没有人占着。

    不能用 connect 探测：env.py 的 _handler 会把每一个新连接都当成游戏页面
    （`self.ws = websocket`），我们的探测会把正在训练的进程挤掉一次。
    这里只是试 bind，不设 SO_REUSEADDR —— Windows 下端口处于 LISTEN 时
    bind 会失败，够用了。
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(('127.0.0.1', port))
        return False
    except OSError:
        return True
    finally:
        s.close()


def resolve_model(path):
    """把 --model 补成真实存在的路径；找不到返回 None。.zip（PPO）和 .pt（DQN）都试。"""
    for cand in (path, path + '.zip', path + '.pt'):
        if os.path.isfile(cand):
            return cand
    return None


def latest_checkpoint(folder='checkpoints'):
    """PPO：优先 checkpoints/best_model.zip，否则取步数最大的 .zip，找不到返回 None。

    和 latest_dqn_checkpoint() 的优先级保持一致 —— 训练中贪婪评估胜率最高的那个，
    才是报告里要引用的模型；最后那个 lf2_ppo.zip 只代表「跑完了」，不代表「最好」。
    """
    preferred = os.path.join(folder, 'best_model.zip')
    if os.path.isfile(preferred):
        return preferred
    best, best_step = None, -1
    for f in glob.glob(os.path.join(folder, '*.zip')):
        m = re.search(r'_(\d+)_steps', os.path.basename(f))
        step = int(m.group(1)) if m else 0
        if step > best_step:
            best, best_step = f, step
    return best


def latest_dqn_checkpoint(folder='checkpoints'):
    """DQN：按「最可能想要」的顺序挑一个 .pt，找不到返回 None。

    best_model.pt 是评估胜率最高那次的裸 state_dict，报告里引用的一般就是它；
    没有的话退到 final_model.pt / last.pt，再退到 ep{N}.pt 里 N 最大的，
    最后才按 mtime 取最新的 .pt（例如 interrupted.pt）。
    """
    for name in ('best_model.pt', 'final_model.pt', 'last.pt', 'interrupted.pt'):
        p = os.path.join(folder, name)
        if os.path.isfile(p):
            return p

    best, best_n = None, -1
    for f in glob.glob(os.path.join(folder, 'ep*.pt')):
        m = re.search(r'ep(\d+)\.pt$', os.path.basename(f))
        if m and int(m.group(1)) > best_n:
            best, best_n = f, int(m.group(1))
    if best:
        return best

    pts = glob.glob(os.path.join(folder, '*.pt'))
    return max(pts, key=os.path.getmtime) if pts else None


def detect_algo(model_path):
    """按扩展名猜算法：.pt → dqn，其余（.zip）→ ppo。"""
    return 'dqn' if model_path.lower().endswith('.pt') else 'ppo'


class QNetworkPolicy:
    """把 DQN 的 .pt 包成和 SB3 的 model.predict() 一样的接口。

    这样下面「跑 N 局统计」那段代码一行都不用改，PPO 和 DQN 的评估口径天然一致。
    """

    def __init__(self, model_path, device='cpu', stochastic=False, temperature=1.0):
        # 延迟导入：只在真的要评 DQN 时才拖进 torch 训练模块
        from train_DQN import ACT_DIM, OBS_DIM, QNetwork, load_checkpoint

        self.device = torch.device(device)
        self.stochastic = stochastic
        self.temperature = max(1e-6, float(temperature))

        ckpt = load_checkpoint(model_path, device=self.device)
        cfg = ckpt.get('config') or {}
        obs_dim = int(cfg.get('obs_dim', OBS_DIM))
        act_dim = int(cfg.get('act_dim', ACT_DIM))

        self.net = QNetwork(obs_dim, act_dim).to(self.device)
        self.net.load_state_dict(ckpt['model'])
        self.net.eval()

        self.meta = {'algo': 'dqn', 'format': ckpt.get('format'),
                     'episode': ckpt.get('episode'), 'total_steps': ckpt.get('total_steps'),
                     'best_win_rate': ckpt.get('best_win_rate')}

    def predict(self, obs, deterministic=True):
        with torch.no_grad():
            q = self.net(torch.as_tensor(np.asarray(obs), dtype=torch.float32)
                         .unsqueeze(0).to(self.device))[0]
        if deterministic or not self.stochastic:
            return int(q.argmax().item()), None
        # DQN 没有可采样的策略分布，这里用 softmax(Q/T) 近似 —— 它测的是
        # 「argmax 策略有多脆」，不能和 PPO 的「按策略分布采样」当同一个口径比。
        probs = torch.softmax(q / self.temperature, dim=-1)
        return int(torch.multinomial(probs, 1).item()), None

    def describe(self):
        """一句话说明这个模型是怎么选动作的。"""
        if self.stochastic:
            return f'DQN softmax(Q/{self.temperature:g}) 采样'
        return 'DQN argmax deterministic'


class PPOPolicy:
    """薄包装，让 PPO 也被传进来的 env 一致地处理（保持原有行为）。"""

    def __init__(self, model_path, env=None, device='cpu', stochastic=False):
        from stable_baselines3 import PPO
        self.model = PPO.load(model_path, env=env, device=device)
        self.stochastic = stochastic
        self.meta = {'algo': 'ppo'}

    def predict(self, obs, deterministic=True):
        action, _ = self.model.predict(obs, deterministic=deterministic)
        return int(action), None

    def describe(self):
        return ('PPO 采样 stochastic' if self.stochastic
                else 'PPO argmax deterministic')


def build_policy(algo, model_path, env=None, device='cpu', stochastic=False,
                 temperature=1.0):
    if algo == 'dqn':
        return QNetworkPolicy(model_path, device=device,
                              stochastic=stochastic, temperature=temperature)
    return PPOPolicy(model_path, env=env, device=device, stochastic=stochastic)


def _ensure_utf8_stdout():
    """重定向到文件 / 管道时 Windows 会用 GBK 编码 stdout，中文和 emoji 会抛
    UnicodeEncodeError。训练和评估都跑很久，输出一般都要存盘，先修好。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding='utf-8', errors='replace')
        except Exception:                       # noqa: BLE001
            pass


def fmt_eta(seconds):
    seconds = int(seconds)
    if seconds < 60:
        return f'{seconds} 秒'
    return f'{seconds // 60} 分 {seconds % 60} 秒'


def main():
    parser = argparse.ArgumentParser(
        description='用训好的模型（PPO 或 DQN）打 N 局，统计胜率与行为特征')
    parser.add_argument('--model', default=None,
                        help='模型路径，.zip / .pt 后缀可省；'
                             '默认按 --algo 取 checkpoints/ 下的模型')
    parser.add_argument('--algo', choices=('auto', 'ppo', 'dqn'), default='auto',
                        help='ppo 读 .zip，dqn 读 .pt；auto 按扩展名判断（默认）')
    parser.add_argument('--episodes', type=int, default=20, help='跑几局（默认 20）')
    parser.add_argument('--device', default='cpu', help='推理设备（默认 cpu，13 维 MLP 用不上 GPU）')
    parser.add_argument('--stochastic', action='store_true',
                        help='按策略分布采样选动作，而不是取 argmax。'
                             '默认 deterministic：每局几乎一模一样，20 局约等于同一条轨迹跑 20 遍；'
                             '加这个开关才看得出策略的真实方差（也要用它才能和训练时的表现对比）。'
                             'DQN 没有可采样的分布，用 softmax(Q/温度) 代替。')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='仅 DQN 的 --stochastic 用：softmax(Q/T) 的温度（默认 1.0）')
    parser.add_argument('--diff', type=diff_arg,
                        choices=('easy', 'normal', 'difficult'), default=None,
                        help='对手难度，可写 easy / normal / difficult 或 0 / 1 / 2；'
                             '不给则沿用页面 URL 里的设置（旧行为）。'
                             '指定后 env.py 会切换已打开的浏览器并校验，'
                             '切不过去会直接报错，绝不评估错难度的对手。')
    parser.add_argument('--json', default=None, metavar='PATH',
                        help='把汇总结果另存为 JSON（写报告时引用）')
    args = parser.parse_args()
    _ensure_utf8_stdout()

    # ---- 1. 找模型 ----
    if args.model:
        model_path = resolve_model(args.model)
        if model_path is None:
            print(f'[错误] 找不到模型：{args.model}')
            return 1
    elif args.algo == 'dqn':
        model_path = latest_dqn_checkpoint()
        if model_path is None:
            print('[错误] checkpoints/ 下没有 .pt。先用 train_DQN.py 训练。')
            return 1
    else:
        model_path = latest_checkpoint()
        if model_path is None:
            print('[错误] checkpoints/ 下没有 .zip。先用 train.py 训练，'
                  '或改用 --model lf2_ppo.zip；要评 DQN 就加 --algo dqn')
            return 1

    algo = detect_algo(model_path) if args.algo == 'auto' else args.algo
    if args.algo != 'auto' and detect_algo(model_path) != args.algo:
        print(f'[提示] --algo {args.algo} 但文件是 {model_path}，按 --algo 处理。')

    # ---- 2. 端口预检（抢不到 8765 会表现成"等 120 秒后超时"，很难排查）----
    if ws_port_busy():
        print(f'[错误] 端口 {WS_PORT} 已被占用 —— 多半是 train.py 还在跑。')
        print('       env.py 把训练端口写死在 8765，两边的对局会互相抢。')
        print('       先把训练停掉（Ctrl+C），确认 netstat -ano | findstr :8765 没有输出，再运行本脚本。')
        return 1

    print(f'[模型] {model_path}    [算法] {algo.upper()}')
    print(f'[局数] {args.episodes}    [设备] {args.device}')
    if args.diff:
        diff_level = resolve_diff(args.diff)
        diff_msg = (f'请求 {DIFF_NAMES[diff_level]}（diff={diff_level}），'
                    f'env.py 会切换并校验')
    else:
        diff_msg = '未指定 --diff，沿用页面 URL 里的设置'
    print(f'[难度] {diff_msg}')
    print(f'[提示] 需要 serve.py + 浏览器页面已打开；一局约 45~65 秒。')
    print()

    # ---- 3. 建环境和模型（只能建一个 LF2Env，8765 写死了）----
    try:
        env = LF2Env(diff=args.diff)
    except Exception as e:
        print(f'[错误] 创建环境失败：{e}')
        return 1
    try:
        model = build_policy(algo, model_path, env=env, device=args.device,
                             stochastic=args.stochastic, temperature=args.temperature)
    except Exception as e:
        print(f'[错误] 加载模型失败：{type(e).__name__}: {e}')
        env.close()
        return 1

    print(f'[选动作] {model.describe()}')
    meta = getattr(model, 'meta', {}) or {}
    if algo == 'dqn' and meta.get('episode') is not None:
        print(f'[检查点] ep={meta.get("episode")}  steps={meta.get("total_steps")}  '
              f'best_win_rate={meta.get("best_win_rate")}')
    print()

    episodes = []
    action_counter = Counter()
    t_start = time.time()

    try:
        for ep in range(1, args.episodes + 1):
            try:
                obs, _ = env.reset()
            except (TimeoutError, RuntimeError) as e:
                print(f'[错误] 游戏环境不可用：{e}')
                print('       检查 launch_browser.py 的页面是否还开着、是否被最小化冻结。')
                break

            # 用首帧血量作基准，不假设初始满血是多少
            hp0_d = float(obs[IDX_D_HP]) * OBS_SCALE[IDX_D_HP]
            hp0_e = float(obs[IDX_E_HP]) * OBS_SCALE[IDX_E_HP]

            steps = att = 0
            dist_sum = 0.0
            info = {}
            t_ep = time.time()

            while True:
                action, _ = model.predict(obs, deterministic=not args.stochastic)
                action = int(action)
                action_counter[action] += 1
                att += action in ATTACK_ACTIONS

                obs, reward, terminated, truncated, info = env.step(action)
                steps += 1
                dist_sum += abs(float(obs[IDX_DX]) * OBS_SCALE[IDX_DX])

                if terminated or truncated:
                    break

            result = info.get('result', 'unknown')
            hp_d = float(obs[IDX_D_HP]) * OBS_SCALE[IDX_D_HP]
            hp_e = float(obs[IDX_E_HP]) * OBS_SCALE[IDX_E_HP]
            rec = {
                'ep': ep,
                'result': result,
                'steps': steps,
                'att': att,
                'att_rate': att / steps if steps else 0.0,
                'mean_dist': dist_sum / steps if steps else 0.0,
                'dealt': max(0.0, hp0_e - hp_e),
                'taken': max(0.0, hp0_d - hp_d),
                'secs': time.time() - t_ep,
            }
            episodes.append(rec)

            left = args.episodes - ep
            eta = (time.time() - t_start) / ep * left
            print(f"[{ep:>3}/{args.episodes}] {result:<9} 步数={steps:<5} "
                  f"出招={att:<4}({rec['att_rate'] * 100:>4.1f}%) "
                  f"距离={rec['mean_dist']:>5.0f}px "
                  f"打掉={rec['dealt']:>4.0f} 挨打={rec['taken']:>4.0f} "
                  f"| {rec['secs']:>4.0f}s 剩 {fmt_eta(eta)}")
    except KeyboardInterrupt:
        print('\n[中断] 已停止，下面只统计跑完的局。')

    if not episodes:
        print('\n没有完成任何一局，无法统计。')
        return 1

    # ---- 4. 汇总 ----
    n = len(episodes)
    wins = sum(e['result'] == 'win' for e in episodes)
    loses = sum(e['result'] == 'lose' for e in episodes)
    timeouts = sum(e['result'] == 'max_steps' for e in episodes)
    frame_to = sum(e['result'] == 'timeout' for e in episodes)
    other = n - wins - loses - timeouts - frame_to

    total_steps = sum(e['steps'] for e in episodes)
    total_att = sum(e['att'] for e in episodes)
    att_rate = total_att / total_steps if total_steps else 0.0
    mean_steps = total_steps / n
    mean_dist = sum(e['mean_dist'] for e in episodes) / n
    mean_dealt = sum(e['dealt'] for e in episodes) / n
    mean_taken = sum(e['taken'] for e in episodes) / n

    lo, hi = wilson(wins, n)

    print()
    print('=' * 68)
    print(f'评估结果：{model_path}，共 {n} 局，{algo.upper()}，{model.describe()}')
    print('=' * 68)
    line = f'胜 {wins} / 负 {loses} / 超时 {timeouts}'
    if frame_to:
        line += f' / 帧超时 {frame_to}'
    if other:
        line += f' / 其它 {other}'
    print(line)
    print(f'胜率        {wins / n * 100:5.1f}%   (95% Wilson CI {lo * 100:.1f}% – {hi * 100:.1f}%)')

    decisive = wins + loses
    if decisive:
        print(f'决胜局胜率  {wins / decisive * 100:5.1f}%   ({wins}/{decisive}，'
              f'不含 {timeouts} 局超时)')
    else:
        print('决胜局胜率    n/a   （一局都没打出人命，全部超时）')

    n_distinct = len({e['steps'] for e in episodes})
    if not args.stochastic and n_distinct <= max(1, n // 5):
        print(f'           注意：{n} 局里只有 {n_distinct} 种不同局长 —— deterministic 模式下')
        print('           这些局几乎是同一条轨迹，上面的置信区间偏乐观。加 --stochastic 再看一遍。')

    print()
    print(f'平均局长    {mean_steps:7.0f} 帧（约 {mean_steps / 45:.0f} 秒/局）')
    print(f'平均距离    {mean_dist:7.0f} px')
    print(f'出招占比    {att_rate * 100:7.1f}%   '
          f'<- 随机策略基线 {RANDOM_ATTACK_RATE * 100:.1f}%')
    if att_rate < RANDOM_ATTACK_RATE * 0.5:
        print('            ↑ 明显低于随机：模型学到的是「别出招」，不是「怎么打」')
    print(f'平均打掉    {mean_dealt:7.0f} hp / 平均挨打 {mean_taken:.0f} hp')

    print()
    print('动作分布（全部 %d 步）：' % total_steps)
    for a, c in action_counter.most_common():
        bar = '#' * int(c / max(action_counter.values()) * 40)
        tag = ' <- 出招' if a in ATTACK_ACTIONS else ''
        print(f'  {a:>2} {str(ACTION_KEYS[a]):<14} {c:>6}  {c / total_steps * 100:>5.1f}%  {bar}{tag}')

    print()
    elapsed = time.time() - t_start
    print(f'总耗时 {fmt_eta(elapsed)}')

    if args.json:
        summary = {
            'model': model_path, 'algo': algo,
            'action_selection': model.describe(),
            'n': n, 'wins': wins, 'loses': loses,
            'timeouts': timeouts, 'frame_timeouts': frame_to, 'other': other,
            'win_rate': wins / n, 'win_rate_ci': [lo, hi],
            'win_rate_decisive': (wins / decisive) if decisive else None,
            'mean_len': mean_steps, 'mean_dist': mean_dist,
            'attack_rate': att_rate, 'random_attack_rate': RANDOM_ATTACK_RATE,
            'mean_dealt': mean_dealt, 'mean_taken': mean_taken,
            'distinct_len': n_distinct, 'elapsed_secs': round(elapsed, 1),
            'action_counts': {action_name(a): c for a, c in action_counter.most_common()},
            'checkpoint_meta': meta,
            # 难度自证：写报告时不用再猜这次是在打谁
            'difficulty_requested': args.diff,
            'difficulty_index': env.diff,
            'difficulty_observed': env.observed_difficulty,
            'ai_observed': env.observed_ai,
        }
        try:
            with open(args.json, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=1)
            print(f'汇总已写入 {args.json}')
        except OSError as e:
            print(f'[警告] 写 {args.json} 失败：{e}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
