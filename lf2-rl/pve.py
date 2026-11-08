"""
pve.py —— 真人对模型：模型接管左边的 Davis，你操作右边的 Dennis。

为什么模型一定在左边：
    LF/match.js 把 RL 桥接的注入目标写死成 `$.character[char_keys[0]].con`，
    也就是 0 号角色（Davis）。而 Dennis 在训练配置里是 `type: 'computer'`，拿到的是
    AIscript 控制器，往它的 buf 里塞按键没有意义。所以要让模型和人各控一个，
    只能人让到右边 —— 这样模型侧一个字都不用改，行为和评测时完全一致。

和 eval.py 的区别（很重要）：
    eval.py 管回合：每局 reset、超时截断、算胜负。
    pve.py 不管回合：不 reload、不算步数、不算 reward，就是一直读状态一直发动作，
    直到你 Ctrl+C。否则你正打着它会把页面重开。

用法：
    终端 A：python serve.py --port 8010                     # 若已在跑就跳过
    终端 B：python launch_browser.py --port 8010 --visible \\
                --path "/LF2-RL/game/game.html?pve"           # 注意不要带 &headless
    终端 C：python pve.py            # 默认 argmax，和 eval.py 一致
            python pve.py --stochastic  # 改成每帧采样，用来打破卡死

键位（你操作右边的 Dennis）：
    U 上   M 下   H 左   K 右   I 跳   , 防   J 攻
    （W/A/S/D/Q/Z/S 是左边 Davis 的，被模型占用；按了也会被模型每帧清掉）

打完一局按 F5 重开，pve.py 不用重启 —— 页面会自己重连。
"""

import argparse
import asyncio
import json
import sys
import time

import websockets

from env import ACTION_KEYS, state_to_obs
from eval import (build_policy, detect_algo, latest_checkpoint,
                  latest_dqn_checkpoint, resolve_model, ws_port_busy)

WS_PORT = 8765


async def serve(model, heartbeat, deterministic=True):
    """每收到一帧状态就当场算动作并回发。事件驱动，不轮询。"""
    # 只认最新的连接：页面刷新后旧连接可能还在关，别让它继续吃动作
    active = {'ws': None}
    state_box = {'last': time.time()}

    async def handler(ws):
        active['ws'] = ws
        print('游戏已连接！')
        try:
            async for message in ws:
                if active['ws'] is not ws:
                    break
                try:
                    state = json.loads(message)
                except Exception:
                    continue
                # 只处理真正的对局状态；其它消息（没有 davis/dennis）直接忽略
                if 'davis' not in state or 'dennis' not in state:
                    continue

                obs = state_to_obs(state)
                action = int(model.predict(obs, deterministic=deterministic)[0])
                await ws.send(json.dumps({
                    'action': action,
                    'keys': ACTION_KEYS[action],
                }))

                now = time.time()
                if heartbeat > 0 and now - state_box['last'] >= heartbeat:
                    state_box['last'] = now
                    print(f"[对局] t={state.get('time')}  "
                          f"Davis hp={state['davis']['hp']}  "
                          f"Dennis hp={state['dennis']['hp']}")
        except websockets.ConnectionClosed:
            pass
        finally:
            if active['ws'] is ws:
                active['ws'] = None

    async with websockets.serve(handler, 'localhost', WS_PORT):
        print('等待游戏连接...（用 ?pve 打开页面后会自动连上）')
        await asyncio.Future()


def main():
    parser = argparse.ArgumentParser(
        description='真人对模型：模型操控左边的 Davis，你用键盘操控右边的 Dennis')
    parser.add_argument('--model', default=None,
                        help='模型路径，.zip / .pt 后缀可省；默认按 --algo 取 checkpoints/ 下的模型')
    parser.add_argument('--algo', choices=('auto', 'ppo', 'dqn'), default='auto',
                        help='ppo 读 .zip，dqn 读 .pt；auto 按扩展名判断（默认）')
    parser.add_argument('--device', default='cpu', help='推理设备（默认 cpu）')
    parser.add_argument('--heartbeat', type=float, default=2.0,
                        help='每隔几秒打印一次双方血量（0 = 不打印，默认 2）')
    parser.add_argument('--stochastic', action='store_true',
                        help='每帧从策略分布采样而不是取 argmax。默认是 argmax（和 '
                             'eval.py 一致）。加上它是因为确定性策略一旦进了不动点 '
                             '（蹲着不动、对着墙出拳），观测不变、动作也不变，会永远卡住；'
                             '采样能打破这个循环。')
    args = parser.parse_args()

    # ---- 1. 找模型（PPO 的 .zip 和 DQN 的 .pt 都行）----
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
                  '或改用 --model lf2_ppo.zip；要比 DQN 就加 --algo dqn')
            return 1

    algo = detect_algo(model_path) if args.algo == 'auto' else args.algo

    # ---- 2. 端口预检 ----
    if ws_port_busy():
        print(f'[错误] 端口 {WS_PORT} 已被占用 —— train.py / eval.py 还开着？')
        print('       先停掉它们（Ctrl+C），确认 netstat -ano | findstr :8765 没有输出。')
        return 1

    # ---- 3. 加载模型（不需要 env，纯推理）----
    try:
        model = build_policy(algo, model_path, device=args.device,
                             stochastic=args.stochastic)
    except Exception as e:
        print(f'[错误] 加载模型失败：{type(e).__name__}: {e}')
        return 1
    print(f'[模型] {model_path}    [算法] {algo.upper()}')
    print(f'[策略] {model.describe()}')
    print('[键位] 你操作右边的 Dennis：U 上 / M 下 / H 左 / K 右 / I 跳 / , 防 / J 攻')
    print('[提示] 页面要用 ?pve 打开，并且浏览器要 --visible（画面得看得见）')
    print('[重启] 打完一局按 F5 重开，本脚本不用重启')
    print('[退出] Ctrl+C\n')

    try:
        asyncio.run(serve(model, args.heartbeat, deterministic=not args.stochastic))
    except KeyboardInterrupt:
        print('\n[停止] 已退出')
    return 0


if __name__ == '__main__':
    sys.exit(main())
