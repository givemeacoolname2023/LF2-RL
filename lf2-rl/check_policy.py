"""check_policy.py —— 不开浏览器，直接问「策略在各个状态下到底选什么动作」。

为什么需要它：
    `train.py` 的曲线（ep_rew_mean / entropy_loss / 训练内 5 局评估）都看不出
    策略是不是在**用观测**。1.2M 步那次一路看着在涨，实际策略在 175 个状态上
    给出同一个 argmax —— 整局按死一个键，eval 出来就是「站桩空挥、不分左右」。
    这个脚本只加载 .zip、只用 `state_to_obs` 造观测，几秒钟就能看出来。

判定（任何一条不过就退出码 1）：
    ① 左右：dx<0（对手在左）时 P(左移类) 应高于 P(右移类)，dx>0 反过来；
    ② 纵深：dx=0、|dz|=60 时，朝向对手的 up/down 概率应高于 dz=0 时；
    ③ argmax 在整张网格上至少出现 2 种（1 种 = 常数动作策略）。

坐标约定（LF/livingobject.js:484 `dirv`）：up → vz<0，down → vz>0，
    所以 dz = davis_z - dennis_z > 0 时（自己在对手下方）应该按 up。

用法（在 lf2-rl 目录下，conda activate lf2rl）：
    python check_policy.py
    python check_policy.py --model lf2_ppo.zip
    python check_policy.py --model checkpoints/ppo_100000_steps.zip
"""

import argparse
import os
import sys

import numpy as np
import torch

from env import ACTION_KEYS, state_to_obs

LEFT = [i for i, k in enumerate(ACTION_KEYS) if 'left' in k]
RIGHT = [i for i, k in enumerate(ACTION_KEYS) if 'right' in k]
UP = [i for i, k in enumerate(ACTION_KEYS) if 'up' in k]
DOWN = [i for i, k in enumerate(ACTION_KEYS) if 'down' in k]

DX_GRID = (-300, -150, -60, 0, 60, 150, 300)
DZ_GRID = (-60, 0, 60)
FRAME_GRID = (0, 20, 40)


def build_state(dx, dz, frame=0, hp=1000, ehp=1000):
    """造一帧和真实对局同量级的状态（x 在场地中间，z 用 100 的基准）。"""
    return {
        'davis': {'x': 200, 'y': 0, 'z': 100, 'hp': hp, 'mp': 0,
                  'frame': frame, 'dir': 'right'},
        'dennis': {'x': 200 + dx, 'y': 0, 'z': 100 - dz, 'hp': ehp, 'mp': 0,
                   'frame': frame, 'dir': 'left'},
    }


def action_probs(model, obs):
    """返回 (n_actions,) 的策略概率。兼容 SB3 2.x 的返回值差异。"""
    t = torch.as_tensor(np.asarray(obs, dtype=np.float32)[None], dtype=torch.float32)
    dist = model.policy.get_distribution(t)
    if isinstance(dist, tuple):          # 老版本返回 (distribution, _)
        dist = dist[0]
    return dist.distribution.probs.detach().numpy()[0]


def name(i):
    return '_'.join(ACTION_KEYS[i]) or 'noop'


def load_model(path):
    from stable_baselines3 import PPO
    if not os.path.isfile(path) and os.path.isfile(path + '.zip'):
        path += '.zip'
    if not os.path.isfile(path):
        raise FileNotFoundError(f'找不到模型：{path}')
    return PPO.load(path, device='cpu')


def main():
    ap = argparse.ArgumentParser(description='离线检查策略是否真的在用观测')
    ap.add_argument('--model', default=None,
                    help='模型路径（.zip 可省）；默认走 eval.latest_checkpoint()，'
                         '即 checkpoints/best_model.zip')
    ap.add_argument('--margin', type=float, default=0.02,
                    help='判定「有反应」所需的最小概率差（默认 0.02）')
    args = ap.parse_args()

    path = args.model
    if path is None:
        from eval import latest_checkpoint
        path = latest_checkpoint() or 'checkpoints/best_model.zip'

    model = load_model(path)
    m = args.margin

    print('=' * 78)
    print(f'策略检查：{path}')
    print(f'  观测 {model.observation_space.shape}  动作 {model.action_space.n}  '
          f'（左移 {LEFT} / 右移 {RIGHT} / up {UP} / down {DOWN}）')
    print('=' * 78)

    # ---------------- ① 左右 ----------------
    print('\n[1] 左右：dx = davis.x - dennis.x（负=对手在左）')
    print(f'    {"dx":>5}  {"P(左移)":>8}  {"P(右移)":>8}  {"argmax":<14}  判定')
    lr_ok = lr_bad = 0
    argmaxes = set()
    gaps = {}
    for dx in DX_GRID:
        pl = pr = 0.0
        ams = []
        for frame in FRAME_GRID:
            p = action_probs(model, state_to_obs(build_state(dx, 0, frame)))
            pl += p[LEFT].sum()
            pr += p[RIGHT].sum()
            am = int(p.argmax())
            ams.append(am)
            argmaxes.add(am)
        pl /= len(FRAME_GRID)
        pr /= len(FRAME_GRID)
        gaps[dx] = pr - pl          # 带符号：正 = 偏右移
        if dx < 0:
            good = pl > pr + m
        elif dx > 0:
            good = pr > pl + m
        else:
            good = True
        lr_ok += good
        lr_bad += (not good)
        print(f'    {dx:>5}  {pl:>8.3f}  {pr:>8.3f}  '
              f'{name(ams[0]):<14}  {"OK" if good else "FAIL"}')
    # 诊断用：P(右移)-P(左移) 随 dx 的变化量。常数偏好策略这里接近 0
    #（固定偏右的模型两端都偏右，单看某一边会误判成「听 dx」）。
    print(f'    -> P(右)-P(左)：{gaps[-300]:+.3f} @dx=-300 → {gaps[300]:+.3f} @dx=+300'
          f'，变化量 {gaps[300] - gaps[-300]:+.3f}')

    # ---------------- ② 纵深 ----------------
    # 比较方式：同一个方向自己跟 dz=0 时比 —— up 和 down 的动作个数不一样（3 vs 4），
    # 直接比「朝对手 vs 远离」会被动作个数带偏。
    print('\n[2] 纵深：dx=0，看 |dz|=60 时该方向的概率有没有抬起来')
    print(f'    {"dz":>5}  {"P(up)":>7}  {"P(down)":>8}  {"argmax":<14}')
    p_up, p_down = {}, {}
    for dz in DZ_GRID:
        pu = pd = 0.0
        ams = []
        for frame in FRAME_GRID:
            p = action_probs(model, state_to_obs(build_state(0, dz, frame)))
            pu += p[UP].sum()
            pd += p[DOWN].sum()
            am = int(p.argmax())
            ams.append(am)
            argmaxes.add(am)
        p_up[dz] = pu / len(FRAME_GRID)
        p_down[dz] = pd / len(FRAME_GRID)
        print(f'    {dz:>5}  {p_up[dz]:>7.3f}  {p_down[dz]:>8.3f}  '
              f'{name(ams[0]):<14}')
    up_rise = p_up[60] - p_up[0]        # 自己在对手下方，该按 up
    down_rise = p_down[-60] - p_down[0]  # 自己在对手上方，该按 down
    dz_good = (up_rise > m) and (down_rise > m)
    print(f'    -> 该按 up 时 P(up) {up_rise:+.3f}，该按 down 时 P(down) {down_rise:+.3f}'
          f'（都要 > {m:+.3f}）')

    # ---------------- ③ 不是常数动作 ----------------
    multi = len(argmaxes) >= 2

    print('\n' + '-' * 78)
    print(f'  [{"OK" if lr_bad == 0 else "FAIL"}] 左右听 dx：{lr_ok}/{len(DX_GRID)} 个 dx 方向正确')
    print(f'  [{"OK" if dz_good else "FAIL"}] 纵深听 dz')
    print(f'  [{"OK" if multi else "FAIL"}] argmax 种类 {len(argmaxes)}：'
          f'{sorted(name(a) for a in argmaxes)}')
    if not multi:
        print('       只有一种 argmax = 常数动作策略：eval 时整局按死一个键，')
        print('       表现就是「站桩空挥、不分左右前后」。别拿它跑通宵。')
    print('-' * 78)

    ok = (lr_bad == 0) and dz_good and multi
    print('结论：' + ('策略在用观测，可以继续训练/评估' if ok
                     else '策略没学会看观测 —— 先改奖励，别再花一晚'))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
