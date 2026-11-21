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
from stable_baselines3 import PPO

from action_selector import ACTION_MODES, ActionSelector, action_mode
from env import ACTION_KEYS, state_to_obs
from eval import latest_checkpoint, resolve_model

WS_PORT = 8765


async def serve(model, heartbeat, action_repeat=1, mode='greedy'):
    """每收到一帧状态就当场算动作并回发。事件驱动，不轮询。"""
    # 只认最新的连接：页面刷新后旧连接可能还在关，别让它继续吃动作
    active = {'ws': None}
    state_box = {'last': time.time()}

    async def handler(ws):
        active['ws'] = ws
        selector = ActionSelector(model, mode)
        last_action_at = None
        session = None
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
                frame = state['time']
                if state.get('session') != session or (last_action_at is not None and frame < last_action_at):
                    selector.reset()
                    last_action_at = None
                    session = state.get('session')
                frames = action_repeat if last_action_at is None else frame - last_action_at
                if frames < action_repeat:
                    continue
                last_action_at = frame

                obs = state_to_obs(state)
                action = int(selector.predict(obs, frames))
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
                        help='PPO 模型路径（.zip 可省）')
    parser.add_argument('--algo', choices=('ppo',), help=argparse.SUPPRESS)
    parser.add_argument('--device', default='cpu', help='推理设备（默认 cpu）')
    parser.add_argument('--heartbeat', type=float, default=2.0,
                        help='每隔几秒打印一次双方血量（0 = 不打印，默认 2）')
    policy_args = parser.add_mutually_exclusive_group()
    policy_args.add_argument('--action-mode', choices=ACTION_MODES, help='出招方式（默认沿用存档）')
    policy_args.add_argument('--stochastic', dest='action_mode', action='store_const', const='sample',
                             help='从策略分布采样动作')
    args = parser.parse_args()

    model_path = resolve_model(args.model) if args.model else latest_checkpoint()
    if model_path is None:
        parser.error(f'找不到 PPO .zip 模型（{args.model or "checkpoints/"}）。')

    # ---- 3. 加载模型（不需要 env，纯推理）----
    try:
        model = PPO.load(model_path, device=args.device)
    except Exception as e:
        print(f'[错误] 加载模型失败：{type(e).__name__}: {e}')
        return 1
    action_repeat = getattr(model, 'lf2_config', {}).get('action_repeat', 1)
    mode = action_mode(model, args.action_mode)
    print(f'[模型] {model_path}    [算法] PPO    [动作重复] {action_repeat} 帧    [出招] {mode}')
    print('[键位] 你操作右边的 Dennis：U 上 / M 下 / H 左 / K 右 / I 跳 / , 防 / J 攻')
    print('[提示] 页面要用 ?pve 打开，并且浏览器要 --visible（画面得看得见）')
    print('[重启] 打完一局按 F5 重开，本脚本不用重启')
    print('[退出] Ctrl+C\n')

    try:
        asyncio.run(serve(model, args.heartbeat, action_repeat, mode))
    except OSError as exc:
        print(f'[错误] WebSocket 端口 {WS_PORT} 无法使用：{exc}')
        return 1
    except KeyboardInterrupt:
        print('\n[停止] 已退出')
    return 0


if __name__ == '__main__':
    sys.exit(main())
