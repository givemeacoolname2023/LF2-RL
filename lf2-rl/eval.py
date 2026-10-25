"""
eval.py —— 用训好的 PPO 模型打 N 局，统计胜率和行为特征。

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

耗时：一局 2000~3000 帧，约 45 帧/秒，也就是 45~65 秒。20 局约 20 分钟。

不做的事：
    不修改 env.py / train.py，只读地用它们。模型是 deterministic 选动作（不打采样噪声），
    这样同一局条件固定时结果可复现。
"""

import argparse
import glob
import math
import os
import re
import socket
import sys
import time
from collections import Counter

from stable_baselines3 import PPO

from env import ACTION_KEYS, LF2Env, OBS_SCALE

# 观测里的下标（对应 env.py 的 _state_to_obs）
IDX_D_HP = 2    # davis hp  = obs[2] * OBS_SCALE[2]
IDX_E_HP = 8    # dennis hp = obs[8] * OBS_SCALE[8]
IDX_DX = 12     # dx        = obs[12] * OBS_SCALE[12]

# 含 'att' 的动作算一次出招：5=攻 7=左攻 8=右攻 9=跳攻 10=下攻
ATTACK_ACTIONS = sorted(i for i, keys in enumerate(ACTION_KEYS) if 'att' in keys)
RANDOM_ATTACK_RATE = len(ATTACK_ACTIONS) / len(ACTION_KEYS)

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
    """把 --model 补成真实存在的路径；找不到返回 None。"""
    for cand in (path, path + '.zip'):
        if os.path.isfile(cand):
            return cand
    return None


def latest_checkpoint(folder='checkpoints'):
    """checkpoints/ 里步数最大的那个，找不到返回 None。"""
    best, best_step = None, -1
    for f in glob.glob(os.path.join(folder, '*.zip')):
        m = re.search(r'_(\d+)_steps', os.path.basename(f))
        step = int(m.group(1)) if m else 0
        if step > best_step:
            best, best_step = f, step
    return best


def wilson(k, n, z=1.96):
    """二项比例的 Wilson 置信区间。小样本下比正态近似靠谱得多。"""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def fmt_eta(seconds):
    seconds = int(seconds)
    if seconds < 60:
        return f'{seconds} 秒'
    return f'{seconds // 60} 分 {seconds % 60} 秒'


def main():
    parser = argparse.ArgumentParser(
        description='用训好的 PPO 模型打 N 局，统计胜率与行为特征')
    parser.add_argument('--model', default=None,
                        help='模型路径，.zip 可省；默认取 checkpoints/ 里步数最大的')
    parser.add_argument('--episodes', type=int, default=20, help='跑几局（默认 20）')
    parser.add_argument('--device', default='cpu', help='推理设备（默认 cpu，13 维 MLP 用不上 GPU）')
    parser.add_argument('--stochastic', action='store_true',
                        help='按策略分布采样选动作，而不是取 argmax。'
                             '默认 deterministic：每局几乎一模一样，20 局约等于同一条轨迹跑 20 遍；'
                             '加这个开关才看得出策略的真实方差（也要用它才能和训练时的表现对比）。')
    args = parser.parse_args()

    # ---- 1. 找模型 ----
    if args.model:
        model_path = resolve_model(args.model)
        if model_path is None:
            print(f'[错误] 找不到模型：{args.model}')
            return 1
    else:
        model_path = latest_checkpoint()
        if model_path is None:
            print('[错误] checkpoints/ 下没有 .zip。先用 train.py 训练，'
                  '或改用 --model lf2_ppo.zip')
            return 1

    # ---- 2. 端口预检（抢不到 8765 会表现成"等 120 秒后超时"，很难排查）----
    if ws_port_busy():
        print(f'[错误] 端口 {WS_PORT} 已被占用 —— 多半是 train.py 还在跑。')
        print('       env.py 把训练端口写死在 8765，两边的对局会互相抢。')
        print('       先把训练停掉（Ctrl+C），确认 netstat -ano | findstr :8765 没有输出，再运行本脚本。')
        return 1

    print(f'[模型] {model_path}')
    print(f'[局数] {args.episodes}    [设备] {args.device}    '
          f"[选动作] {'采样 stochastic' if args.stochastic else 'argmax deterministic'}")
    print(f'[提示] 需要 serve.py + 浏览器页面已打开；一局约 45~65 秒。')
    print()

    # ---- 3. 建环境和模型（只能建一个 LF2Env，8765 写死了）----
    try:
        env = LF2Env()
    except Exception as e:
        print(f'[错误] 创建环境失败：{e}')
        return 1
    model = PPO.load(model_path, env=env, device=args.device)

    episodes = []
    action_counter = Counter()
    t_start = time.time()

    try:
        for ep in range(1, args.episodes + 1):
            try:
                obs, _ = env.reset()
            except TimeoutError as e:
                print(f'[错误] 等不到游戏状态：{e}')
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
    print(f'评估结果：{model_path}，共 {n} 局，'
          f"{'采样 stochastic' if args.stochastic else 'argmax deterministic'}")
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
    print(f'总耗时 {fmt_eta(time.time() - t_start)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
