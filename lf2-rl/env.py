import asyncio
import websockets
import json
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import threading
import time

ACTION_KEYS = [
    [],
    ['left'],
    ['right'],
    ['jump'],
    ['down'],
    ['att'],
    ['def'],
    ['left', 'att'],
    ['right', 'att'],
    ['jump', 'att'],
    ['down', 'att'],
]

OBS_SCALE = np.array([
    400.0, 100.0, 500.0, 500.0, 200.0, 1.0,
    400.0, 100.0, 500.0, 500.0, 200.0, 1.0,
    400.0,
], dtype=np.float32)

OBS_OFFSET = np.array([
    1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0,
], dtype=np.float32)

# ---------------------------------------------------------------------------
# 回合管理参数
# ---------------------------------------------------------------------------
# 同一页面内，time < FRESH_MATCH_TIME 才算“刚开局”。
# 注意时间单位是帧（约 45 帧/秒），10 帧只有 0.22 秒，页面重连都来不及，
# 所以真正的“新局”判定靠 match.js 里的 session id，这里只是兜底。
FRESH_MATCH_TIME = 60
# reset() 最多等多久让浏览器连上（允许先启动训练脚本、再打开页面）。
CONNECT_TIMEOUT = 120.0
# 当前这一局已经跑了一会儿时，先给它一点时间自然结束（角色阵亡时页面会自动刷新）。
GRACE_TIME = 3.0
# 主动请求页面重开之后，等新局状态的时长。
FRESH_TIMEOUT = 30.0
# 每局最多走多少帧。到这个数还没人死就截断（truncated），避免“两个人都站着不动”
# 的僵局让一局无限长（match.js 里那句 MAX_STEPS 注释指的就是它）。
# 按 45 帧/秒算，3000 帧约 66 秒；可用环境变量 LF2_MAX_STEPS 覆盖。
MAX_EPISODE_STEPS = int(os.environ.get('LF2_MAX_STEPS', '3000'))


def state_to_obs(state):
    """把一帧游戏状态转成 13 维观测。

    独立成模块级函数（原来是 LF2Env 的方法，但方法体根本不用 self）。这样 pve.py
    这种「只驱动模型、不管回合」的脚本可以直接复用，不必再抄一份 OBS_SCALE / OBS_OFFSET
    —— 以后改观测定义只需要改这一处。
    """
    d = state['davis']
    e = state['dennis']
    dir_map = {'right': 1.0, 'left': -1.0}
    dx = d['x'] - e['x']
    raw = np.array([
        d['x'], d['y'], d['hp'], d['mp'],
        float(d['frame']), dir_map.get(d['dir'], 0.0),
        e['x'], e['y'], e['hp'], e['mp'],
        float(e['frame']), dir_map.get(e['dir'], 0.0),
        dx,
    ], dtype=np.float32)
    obs = raw / OBS_SCALE - OBS_OFFSET
    return np.clip(obs, -2.0, 2.0)


class LF2Env(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(
            low=-2.0, high=2.0, shape=(13,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(11)
        self.ws = None
        self.latest_state = None
        self.prev_state = None
        self._lock = threading.Lock()
        self._episode = 0
        self._step_count = 0
        # 当前这一局所在页面的 session（match.js 每次载入都会换一个）
        self._session_in_use = None
        # 已经确认“不是新开局”的 session，同一个 session 不会再被当成新局
        self._rejected_sessions = set()
        self._start_server()

    def _start_server(self):
        self.loop = asyncio.new_event_loop()
        t = threading.Thread(target=self._run_loop, daemon=True)
        t.start()

    def _run_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._serve())

    async def _serve(self):
        async with websockets.serve(self._handler, "localhost", 8765):
            await asyncio.Future()

    async def _handler(self, websocket):
        # 只认最新的一个连接：多个标签页同时开着时，旧页面发来的状态不再参与训练
        self.ws = websocket
        print("游戏已连接！")
        try:
            async for message in websocket:
                if self.ws is not websocket:
                    break
                try:
                    data = json.loads(message)
                except Exception:
                    continue
                with self._lock:
                    self.latest_state = data
        finally:
            if self.ws is websocket:
                self.ws = None
                print("游戏断开连接")

    def _send_raw(self, payload):
        """向浏览器发一条 JSON 消息，不等待结果。"""
        if self.ws is None:
            return False
        try:
            asyncio.run_coroutine_threadsafe(
                self.ws.send(json.dumps(payload)), self.loop
            )
            return True
        except Exception:
            # 页面正在刷新时连接可能已经关闭，忽略即可
            return False

    def _send_action(self, action):
        if self.ws is None:
            return
        keys = ACTION_KEYS[action]
        self._send_raw({"action": int(action), "keys": keys})

    def _request_reload(self):
        """请浏览器刷新页面、重开一局（JS 侧收到 {"cmd":"reload"} 后 location.reload）。"""
        return self._send_raw({"cmd": "reload"})

    def _state_to_obs(self, state):
        # 具体实现在模块级 state_to_obs()，这里只做委托，语义和以前完全一致
        return state_to_obs(state)

    def _compute_reward(self, prev, curr):
        if prev is None:
            return 0.0

        d_hp_delta = curr['davis']['hp'] - prev['davis']['hp']
        e_hp_delta = curr['dennis']['hp'] - prev['dennis']['hp']
        reward = 0.0

        # 打中 Dennis 最重要
        reward += (-e_hp_delta) * 2.0

        # 被打惩罚小
        reward += d_hp_delta * 0.1

        # 胜负
        if curr['dennis']['hp'] <= 0:
            reward += 200.0
        if curr['davis']['hp'] <= 0:
            reward -= 10.0

        # 距离奖励：越近越好
        dx = abs(curr['davis']['x'] - curr['dennis']['x'])
        reward += max(0.0, (300.0 - dx) / 300.0) * 0.5

        # 每帧小惩罚，鼓励尽快出手
        reward -= 0.01

        return reward

    def _wait_for_state(self, timeout=60.0):
        """等浏览器发来任意一帧状态；返回是否等到。"""
        start = time.time()
        while True:
            with self._lock:
                if self.latest_state is not None:
                    return True
            if time.time() - start > timeout:
                return False
            time.sleep(0.01)

    def _is_new_round(self, s):
        """这一帧能不能当作新一局的起点。

        页面每次重新载入（match.js 顶部）都会生成新的 rl_session_id：
          * session 变过 → 页面确实重开过，直接接受，不必等 time 变小；
          * session 没变 → 只有刚开局（time < FRESH_MATCH_TIME）才算；
          * 已经否决过的 session（当时正在中局）永远不再接受。
        """
        sess = s.get('session')
        t = s.get('time')
        if sess is None or t is None:
            # 旧页面没有 session 字段，只能看开局时间
            return t is not None and t < FRESH_MATCH_TIME
        if sess in self._rejected_sessions:
            return False
        if self._session_in_use is None:
            # 还没打过任何一局：页面可能是很久以前打开的中局，
            # 没否决过任何 session 之前，只认“刚开局”
            return t < FRESH_MATCH_TIME or bool(self._rejected_sessions)
        return sess != self._session_in_use or t < FRESH_MATCH_TIME

    def _wait_new_round(self, timeout):
        """等一帧属于“新一局”的状态；返回是否等到。"""
        start = time.time()
        while time.time() - start < timeout:
            with self._lock:
                s = self.latest_state
            if s is not None and self._is_new_round(s):
                return True
            time.sleep(0.01)
        return False

    def _ensure_fresh(self):
        """保证接下来的一局是从 t≈0 开始的干净一局。"""
        # 1) 等浏览器连上并真的在发状态
        #    注意给冷启动留时间：页面要加载十几 MB 素材，太早重载会一直打断加载
        deadline = time.time() + CONNECT_TIMEOUT
        silent = 0.0
        silent_reloads = 0
        while not self._wait_for_state(5.0):
            if time.time() > deadline:
                raise TimeoutError(
                    "等待游戏状态超时：浏览器没有连上 ws://localhost:8765，"
                    "或页面打开后一直没有开始比赛。请确认游戏页面已打开、没有卡死。"
                )
            # 只在“已经连上”时累计沉默时间：还没连上属于浏览器没开，重载也没用
            if self.ws is not None:
                silent += 5.0
            if silent >= 15.0 and silent_reloads < 3:
                # 连上了却一直不发状态：页面多半停在主菜单/加载失败
                print("[Reset] 连上了但 15 秒没有状态（页面可能停在主菜单），请求重开页面...")
                self._request_reload()
                silent = 0.0
                silent_reloads += 1
        # 2) 如果当前已经是新局，直接用
        if self._wait_new_round(GRACE_TIME):
            return
        # 3) 否则主动请页面重开（最多 4 次）
        for attempt in range(4):
            with self._lock:
                s = self.latest_state
            print(
                f"[Reset] 当前不是新局（time={s['time'] if s else 'None'}），"
                f"请求浏览器重开（第 {attempt + 1} 次）..."
            )
            # 记下这局的 session，重开之后同一个 session 的残留状态不会被误当成新局
            if s and s.get('session'):
                self._rejected_sessions.add(s['session'])
            with self._lock:
                self.latest_state = None
            self._request_reload()
            if self._wait_new_round(FRESH_TIMEOUT):
                return
            with self._lock:
                s = self.latest_state
            print(f"[Reset] {FRESH_TIMEOUT:.0f} 秒内没等到新局（此刻 time={s['time'] if s else 'None'}）")
        raise TimeoutError(
            "浏览器没有重开新局。请检查：页面是否被最小化/冻结、"
            "是否已经关闭、控制台是否有报错。"
        )

    def reset(self, seed=None, options=None):
        self._episode += 1
        print(f"[Reset] 第 {self._episode} 局开始，等待游戏连接...")
        with self._lock:
            self.latest_state = None
            self.prev_state = None
        self._step_count = 0
        # 等连接 + 保证是干净的新一局（必要时请求页面重开）
        self._ensure_fresh()
        with self._lock:
            state = self.latest_state
            obs = self._state_to_obs(state)
            self._session_in_use = state.get('session', self._session_in_use)
        print(f"[Reset] 已连接，开始第 {self._episode} 局（t={state['time']}）")
        return obs, {}

    def step(self, action):
        with self._lock:
            self.prev_state = self.latest_state
            prev_time = self.latest_state['time'] if self.latest_state else -1

        self._send_action(action)

        # 等下一帧，最多等 3 秒
        deadline = time.time() + 3.0
        frame_received = False
        while True:
            with self._lock:
                curr = self.latest_state
            if curr and curr['time'] != prev_time:
                frame_received = True
                break
            if time.time() > deadline:
                break
            time.sleep(0.001)

        with self._lock:
            curr = self.latest_state

        # 帧超时：浏览器卡住了
        if not frame_received:
            print(
                f"[结果] 🔄 帧超时 | prev_time={prev_time} | curr_time={curr['time'] if curr else 'None'}")
            with self._lock:
                s = self.latest_state
            # 连接已经断了的时候 latest_state 可能是 None，给一个占位观测
            obs = self._state_to_obs(s) if s is not None else np.zeros(13, dtype=np.float32)
            # 主动让页面重开一局，避免下一局从中局开始
            self._request_reload()
            return obs, -5.0, False, True, {'result': 'timeout'}

        obs = self._state_to_obs(curr)
        reward = self._compute_reward(self.prev_state, curr)

        # 正常结束：有人死亡
        terminated = curr['davis']['hp'] <= 0 or curr['dennis']['hp'] <= 0

        info = {}
        if terminated:
            if curr['dennis']['hp'] <= 0:
                info['result'] = 'win'
                print(
                    f"[结果] 🏆 Davis 赢！davis hp={curr['davis']['hp']} dennis hp={curr['dennis']['hp']}")
            else:
                info['result'] = 'lose'
                print(
                    f"[结果] 💀 Davis 输！davis hp={curr['davis']['hp']} dennis hp={curr['dennis']['hp']}")

        # 步数上限：没人死也截断，避免“双方都不动”的僵局无限长
        self._step_count += 1
        truncated = (not terminated) and self._step_count >= MAX_EPISODE_STEPS
        if truncated:
            info['result'] = 'max_steps'
            print(f"[结果] ⏱ 到达步数上限 {MAX_EPISODE_STEPS}，截断本局")

        return obs, reward, terminated, truncated, info

    def close(self):
        pass
