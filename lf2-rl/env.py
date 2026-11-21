import argparse
import asyncio
import websockets
import json
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import threading
import time

POLICY_CHARACTERS = ('davis', 'henry')

ACTION_KEYS = [
    [],
    ['left'],
    ['right'],
    ['jump'],
    ['down'],      # 纵深向下（朝镜头），不是蹲
    ['att'],
    ['def'],
    ['left', 'att'],
    ['right', 'att'],
    ['jump', 'att'],
    ['down', 'att'],
    ['up'],        # 纵深向上（远离镜头）。必须与 match.js 的 ACTION_MAP 一致
    # 12–15：斜向移动。没有斜向动作时一帧只能追一个轴，追 x 和追 z 互斥；
    # 引擎的移动判定读 con.state 的两个轴（character.js:216-219），手动确认可斜走。
    ['left', 'down'],   # 12: 左+纵深向下
    ['right', 'down'],  # 13: 右+纵深向下
    ['left', 'up'],     # 14: 左+纵深向上
    ['right', 'up'],    # 15: 右+纵深向上
]
ACTION_KEYS += [
    ['def', 'toward', 'att'],
    ['toward', 'def', 'down', 'att'],
    ['toward', 'def', 'up', 'att'],
    ['toward', 'def', 'up', 'jump'],
    ['approach'],
    ['toward', 'att'],
]

# 前 13 项是原观测（x/y/hp/mp/frame/dir ×2 + dx），后 3 项是纵深：
# Davis 的 z、Dennis 的 z、dz = davis_z - dennis_z。
# 纵深实际范围约 289–510（背景数据各不相同），所以用 200 缩放、2.0 平移，
# 落在约 ±0.55（绝对 z）和 ±1.1（dz），不会被 ±2 截断。
OBS_SCALE = np.array([
    400.0, 100.0, 500.0, 500.0, 200.0, 1.0,
    400.0, 100.0, 500.0, 500.0, 200.0, 1.0,
    400.0,
    200.0, 200.0, 200.0,
], dtype=np.float32)

OBS_OFFSET = np.array([
    1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0,
    2.0, 2.0, 0.0,
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

# ---------------------------------------------------------------------------
# 奖励常数（完整推导和调参建议见项目根目录 reward.md）
# ---------------------------------------------------------------------------
# 距离奖励：每「净靠近 1 像素」给 MOVE_REWARD，靠近和远离对称。
#
#   d_t = hypot(dx_t, Z_WEIGHT * dz_t)
#   r_t += MOVE_REWARD * (d_{t-1} - d_t)
#
# 三代版本的教训（都是在这份代码上实测出来的，别凭感觉回调）：
#   1. 望远镜势差 0.5*(γΦ'-Φ)（commit 83b638c）：整局总和上界 0.5，
#      一次胜利 ≈ 500hp*2.0 + 200 ≈ 1200 —— 0.04%，PPO 的 advantage 归一化之后
#      等于 0，1.2M 步训出的策略在几百个状态上给同一个 argmax（见 check_policy.py）。
#   2. 逐帧值 0.5*(1 - d/600)：量级够了（整局 ~750），但它是「只要在附近就一直给」，
#      而 Easy AI 会自己追过来 —— 站着不动也照拿，策略于是塌成常数动作
#      （300k/320k 的 checkpoint 整局按 left_down）。刷得到分，学不到走位。
#   3. 本版：只认「距离变小」。没变就没分，退一步还扣回去，所以
#      「站着不动」和「来回踱步」（只奖靠近不罚远离会出现）都刷不到分。
#      全速靠近 5px/帧 → +0.5/帧；整局净靠近 470px → ~47，相对一次胜利 4%。
#      注意这里**不截断距离**：截断会让 600px 之外完全没有梯度，逃跑阶段就废了；
#      望远镜求和本身有界（≤ MOVE_REWARD * 场地宽度），不需要 clip。
MOVE_REWARD = 0.1
# 距离里 dz 的权重。
#   z 走路速度只有 x 的一半（davis.js: walking_speed 5 vs walking_speedz 2.5）。
#   Z_WEIGHT=2 时两轴每帧等价：MOVE*v_x = MOVE*W*v_z = 0.5/帧。
#   但 1.2M 那版实测：残血追击阶段（AI 往 z 跑）策略完全不碰纵深 ——
#   |dz| 从 0 拉到 80，P(朝对手的纵向) 反而从 0.29 掉到 0.19，argmax 始终横向攻击。
#   原因：追 80px 纵深要连按 37 帧、全程只有 ~13 分，而一拳命中是 40–100，
#   信号被噪声盖住；横向是从 400px 外开始追、累计上百，才学得会。
#   取 5 → 每帧纵深的收益是横向的 2.5 倍：追击时纵向 1.17/帧 vs 横向 0.18/帧，
#   追满 80px 约 46 分，和一拳同量级。仍是双边差分，退开照样扣分，没有刷分漏洞。
Z_WEIGHT = 5.0
# 伤害：按「净血量优势」对称给分，打掉对手 1 点和自己掉 1 点等值。
#
# 旧版是 2.0 / −0.5（打中给得多、挨打扣得少），盈亏点在 `打掉 > 0.25 × 挨打` ——
# 也就是说打掉 1 点血、挨打 2 点血，一个明显吃亏的交换仍然是 +1.0 的正收益。
# 实测后果（2026-11-18，1.2M 步闭环实验）：
#   模型把「打掉 464 / 挨打 352」当成好成绩（奖励 +752），
#   但换成对称计数只有 +112，而 20 局胜率停在 40% 不动 —— 训练回报从 544 涨到 770，
#   胜率一步没涨。所以这一项必须对称，否则策略优化的根本不是「赢」。
DAMAGE_REWARD = 1.0       # 每点净血量优势
STEP_PENALTY = 0.01       # 每帧小惩罚，鼓励尽快结束战斗
# 纵深没对上还乱挥的惩罚。命中要求 |dz| <= 12（zwidth），这里留 8px 余量抵消
# 动作延迟 1–2 帧（z 速度 2.5px/帧），所以阈值用 20。按「按下攻击键时看到的状态」判。
WHIFF_DZ = 20.0
WHIFF_PENALTY = 0.05
# 胜负与超时。终局也必须对称，否则「输」的代价远小于「赢」的收益，
# 策略会倾向于稳一点打完而不是去争取胜利。
WIN_BONUS = 200.0
LOSS_PENALTY = 200.0
# 超时必须明显大于「僵持的收益」：对称化之后，双方各掉 350 血的僵持是 0 分，
# 若超时只扣 20，那「缩着不打」就比「冒险进攻但可能输（−200）」划算得多。
TIMEOUT_PENALTY = 50.0
# 难度下标 → 名字，只为打印。数值来自 manager.js 的 diff_list，
# 对手 AI 由 URL 的 ?debug&diff=0/1/2 决定（0 是 dumbass，2 是 Crusher）。
DIFF_NAMES = {0: 'Easy', 1: 'Normal', 2: 'Difficult'}
# 难度名 → URL 里的 diff 下标（与 manager.js 的 difficulty_index 一一对应）
DIFF_LEVELS = {'easy': 0, 'normal': 1, 'difficult': 2}
# 切换难度后，等新页面报出目标难度的时长。页面重载要重新加载素材，和 FRESH_TIMEOUT 同量级。
DIFF_TIMEOUT = 30.0
# 切换尝试次数。全都失败就报错 —— 绝不带着错误的难度继续训练。
DIFF_ATTEMPTS = 3


def resolve_diff(value):
    """把 --diff 的取值统一成 0/1/2。

    None 表示「不干预」：完全沿用页面 URL 里的设置（旧行为），返回 None。
    接受 'easy'/'normal'/'difficult'（大小写、首尾空格都容忍），
    也接受 0/1/2 以及命令行传进来的 '0'/'1'/'2'。
    非法值直接抛 ValueError —— 在绑定 8765 之前就失败，别等浏览器连上来才发现。
    """
    if value is None:
        return None
    if isinstance(value, str):
        key = value.strip().lower()
        if key in DIFF_LEVELS:
            return DIFF_LEVELS[key]
        if key in ('0', '1', '2'):
            return int(key)
    elif isinstance(value, int) and not isinstance(value, bool):
        if value in DIFF_NAMES:
            return value
    raise ValueError(
        f"未知难度：{value!r}（可用 easy / normal / difficult，或 0 / 1 / 2）"
    )


def diff_arg(value):
    """argparse 的 type：`--diff` 同时接受名字和 0/1/2，统一返回小写难度名。

    返回名字而不是数字，是为了让入口脚本里已有的打印、TensorBoard 的
    `diff_name`、argparse 的 choices 校验都继续按字符串走。
    非法值抛 ArgumentTypeError，argparse 会原样打印中文提示。
    """
    try:
        level = resolve_diff(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e))
    if level is None:
        raise argparse.ArgumentTypeError("不能为空")
    return DIFF_NAMES[level].lower()


def state_to_obs(state):
    """把一帧游戏状态转成 16 维观测。

    独立成模块级函数（原来是 LF2Env 的方法，但方法体根本不用 self）。这样 pve.py
    这种「只驱动模型、不管回合」的脚本可以直接复用，不必再抄一份 OBS_SCALE / OBS_OFFSET
    —— 以后改观测定义只需要改这一处。

    注意坐标约定：y 是跳跃高度（落地恒为 0），z 才是屏幕上的纵深。
    纵深特征追加在最后，原有下标（hp=2/8、dx=12）保持不变。
    """
    d = state['davis']
    e = state['dennis']
    if 'z' not in d or 'z' not in e:
        raise KeyError(
            "游戏状态里没有 z：页面还在跑旧版 match.js。"
            "请用 serve.py（禁缓存）重新打开浏览器，或强制刷新页面。"
        )
    dir_map = {'right': 1.0, 'left': -1.0}
    dx = d['x'] - e['x']
    dz = d['z'] - e['z']
    raw = np.array([
        d['x'], d['y'], d['hp'], d['mp'],
        float(d['frame']), dir_map.get(d['dir'], 0.0),
        e['x'], e['y'], e['hp'], e['mp'],
        float(e['frame']), dir_map.get(e['dir'], 0.0),
        dx,
        d['z'], e['z'], dz,
    ], dtype=np.float32)
    obs = raw / OBS_SCALE - OBS_OFFSET
    return np.clip(obs, -2.0, 2.0)


class LF2Env(gym.Env):
    def __init__(self, diff=None, seed_base=None, bg=None):
        super().__init__()
        # 本次运行要求的难度；None = 完全沿用页面 URL 的设置（旧行为）。
        # 放在 _start_server() 之前：非法的 diff 在占端口之前就抛错。
        self.diff = resolve_diff(diff)
        self.diff_name = DIFF_NAMES.get(self.diff)
        # 每局不同的随机种子：第 i 局（从 0 起）用 seed_base + i。
        # None = 不发 seed（页面用 manager.js 里写死的 824163532，训练的老行为）。
        # 不改它的话，每一局都是同一个出生点 + 同一串 AI 随机决策，20 局等于 1 局。
        if seed_base is not None:
            seed_base = int(seed_base)
            if seed_base < 0:
                raise ValueError(f"seed_base 必须 >= 0（收到 {seed_base}）")
        self.seed_base = seed_base
        self._current_seed = None
        # 固定背景场地（bg_list 下标）。None = 沿用页面原本的随机选择。
        # 各场地宽度 794~3200 差 4 倍，而观测缩放只对 ~800 宽成立 ——
        # 换场地等于换一个游戏，所以「换开局」的实验必须把它钉住。
        if bg is not None:
            bg = int(bg)
            if bg < 0:
                raise ValueError(f"bg 必须 >= 0（收到 {bg}）")
        self.bg = bg
        # 最近一局实际观察到的难度/对手/种子，供 eval.py 的 JSON 汇总引用
        self.observed_difficulty = None
        self.observed_ai = None
        self.observed_seed = None
        self.observed_bg = None
        self.observation_space = spaces.Box(
            low=-2.0, high=2.0, shape=(len(OBS_SCALE),), dtype=np.float32
        )
        self.action_space = spaces.Discrete(len(ACTION_KEYS))
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
        # 上一次打印过的对手 AI 名字，用来在难度变化时提醒一次
        self._announced_ai = None
        if self.diff is not None:
            print(f"[难度] 本次运行要求 {self.diff_name}（diff={self.diff}），"
                  f"连上页面后会自动切换并校验")
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

    def _request_reload(self, diff=None, seed=None, bg=None):
        """请浏览器刷新页面、重开一局；带 diff / seed / bg 时顺带改掉页面 URL 里的参数。

        diff 缺省取 self.diff、seed 缺省取 self._current_seed、bg 缺省取 self.bg，
        所以 _ensure_fresh / step 里现有的调用点都不用改，难度/种子/场地会自动跨局保留。
        三者都取不到时，发出去的内容和以前逐字一致（{"cmd": "reload"}）。
        """
        diff = self.diff if diff is None else diff
        seed = self._current_seed if seed is None else seed
        bg = self.bg if bg is None else bg
        payload = {"cmd": "reload"}
        if diff is not None:
            payload["diff"] = int(diff)
        if seed is not None:
            payload["seed"] = int(seed)
        if bg is not None:
            payload["bg"] = int(bg)
        return self._send_raw(payload)

    def _state_to_obs(self, state):
        # 具体实现在模块级 state_to_obs()，这里只做委托，语义和以前完全一致
        return state_to_obs(state)

    def _compute_reward(self, prev, curr, action):
        if prev is None:
            return 0.0

        # hp 差值：受伤为负；取负号得到“对对手造成的伤害”
        d_hp_delta = curr['davis']['hp'] - prev['davis']['hp']
        e_hp_delta = curr['dennis']['hp'] - prev['dennis']['hp']
        reward = 0.0

        # 净血量优势：打掉对手的血 减去 自己损失的血，等权重（见 DAMAGE_REWARD 注释）。
        # 双方各掉 350 血 → 0 分；对手掉 10、自己掉 5 → +5。
        reward += (d_hp_delta - e_hp_delta) * DAMAGE_REWARD
        #                             ↑ d_hp_delta <= 0（挨打），取「对手掉血 − 自己掉血」

        # 距离：只奖「净靠近」，靠近和远离对称（推导见 MOVE_REWARD 的注释）。
        # 二维距离 = hypot(dx, Z_WEIGHT * dz)：命中要求纵深对齐（zwidth=12），
        # 所以「对齐纵深」和「横向靠近」用同一条距离；z 加权见 Z_WEIGHT。
        # 不截断：望远镜求和本身有界（≤ MOVE_REWARD * 场地宽度），
        # 截断反而会让远处完全没有梯度。
        dx_prev = prev['davis']['x'] - prev['dennis']['x']
        dz_prev = prev['davis']['z'] - prev['dennis']['z']
        dx_curr = curr['davis']['x'] - curr['dennis']['x']
        dz_curr = curr['davis']['z'] - curr['dennis']['z']
        dist_prev = float(np.hypot(dx_prev, Z_WEIGHT * dz_prev))
        dist_curr = float(np.hypot(dx_curr, Z_WEIGHT * dz_curr))
        reward += MOVE_REWARD * (dist_prev - dist_curr)

        # 每帧小惩罚，鼓励尽快结束战斗
        reward -= STEP_PENALTY

        # 纵深没对上还乱挥：命中窗口只有 |dz| <= 12，这里放在 20px 之外才罚，
        # 给动作延迟留余量，免得把“下一帧就进窗口”的攻击也罚掉。
        # 用 dz_prev 是因为它才是模型做这次决策时看到的状态。
        if 'att' in ACTION_KEYS[action] and abs(dz_prev) > WHIFF_DZ:
            reward -= WHIFF_PENALTY

        # 胜负
        if curr['dennis']['hp'] <= 0:
            reward += WIN_BONUS
        if curr['davis']['hp'] <= 0:
            reward -= LOSS_PENALTY

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

    def _wait_difficulty(self, timeout):
        """等一帧 difficulty == self.diff 的状态；返回该帧或 None。"""
        start = time.time()
        while time.time() - start < timeout:
            with self._lock:
                s = self.latest_state
            if s is not None and s.get('difficulty') == self.diff:
                return s
            time.sleep(0.01)
        return None

    def _apply_difficulty(self):
        """把已经开着的浏览器切到 self.diff，并确认页面真的报出了这个难度。

        self.diff is None 时是零成本的 no-op。切换靠 match.js 收到
        {"cmd":"reload","diff":N} 后改写 URL 再导航，重载会换一个新的
        rl_session_id —— 这正是 _ensure_fresh 认的“新局”，所以切换完可以直接开局。
        """
        if self.diff is None:
            return
        with self._lock:
            s = self.latest_state
        if s is not None and s.get('difficulty') == self.diff:
            return                      # 已经是目标难度：不重载，零延迟

        for attempt in range(1, DIFF_ATTEMPTS + 1):
            with self._lock:
                s = self.latest_state
            if s is not None and s.get('difficulty') == self.diff:
                return                  # 上一轮重载生效得慢，这里补认一次
            seen = None if s is None else s.get('difficulty')
            print(f"[难度] 当前 diff={seen}，切换到 {self.diff_name}"
                  f"（diff={self.diff}），第 {attempt}/{DIFF_ATTEMPTS} 次...")
            # 和 _ensure_fresh 第 3 步同样的处理：旧 session 永久否决。
            # 否则首次 reset 时 _is_new_round 的 `_session_in_use is None` 分支
            # 可能因为新页面已经跑过 FRESH_MATCH_TIME 而拒收，导致白重载一次。
            if s and s.get('session'):
                self._rejected_sessions.add(s['session'])
            with self._lock:
                self.latest_state = None
            if not self._request_reload(self.diff):
                # ws 正好在重连：给连接一点时间，下一轮再发
                self._wait_for_state(5.0)
                continue
            confirmed = self._wait_difficulty(DIFF_TIMEOUT)
            if confirmed is not None:
                print(f"[难度] 已确认 {self.diff_name}"
                      f"（对手 AI：{confirmed.get('ai')}）")
                return
            with self._lock:
                s = self.latest_state
            print(f"[难度] {DIFF_TIMEOUT:.0f} 秒内没切换成功"
                  f"（此刻 difficulty={None if s is None else s.get('difficulty')}）")

        raise RuntimeError(
            f"无法把游戏切换到 {self.diff_name}（diff={self.diff}）："
            f"页面上报的 difficulty 一直不是 {self.diff}。请检查：\n"
            f"  1) 游戏页面是否带 ?debug（只有 debug 模式才会开局并发送状态，"
            f"主菜单 / ?demo 永远不会报出 difficulty）；\n"
            f"  2) 是否用 serve.py 提供页面（它发 no-store，重载能拿到最新的 "
            f"LF/match.js；旧缓存的 match.js 不认识 reload 消息里的 diff 字段）；\n"
            f"  3) 是否有另一个标签页同时在改难度。\n"
            f"  仍不行就重开 launch_browser.py 后再跑。"
        )

    def _ensure_fresh(self):
        """保证接下来的一局是从 t≈0 开始的干净一局。"""
        # 1) 等浏览器连上并真的在发状态
        #    注意给冷启动留时间：页面要加载十几 MB 素材，太早重载会一直打断加载
        deadline = time.time() + CONNECT_TIMEOUT
        silent = 0.0
        silent_reloads = 0
        while not self._wait_for_state(5.0):
            if time.time() > deadline:
                msg = ("等待游戏状态超时：浏览器没有连上 ws://localhost:8765，"
                       "或页面打开后一直没有开始比赛。请确认游戏页面已打开、没有卡死。")
                if self.diff is not None:
                    msg += (f"\n另外：本次要求 {self.diff_name}，"
                            f"游戏页面必须带 ?debug 才会开局并发送状态。")
                raise TimeoutError(msg)
            # 只在“已经连上”时累计沉默时间：还没连上属于浏览器没开，重载也没用
            if self.ws is not None:
                silent += 5.0
            if silent >= 15.0 and silent_reloads < 3:
                # 连上了却一直不发状态：页面多半停在主菜单/加载失败
                print("[Reset] 连上了但 15 秒没有状态（页面可能停在主菜单），请求重开页面...")
                self._request_reload()
                silent = 0.0
                silent_reloads += 1
        # 2) 把难度切到 --diff 要求的值（没指定时是 no-op）。
        #    必须在“认领新局”之前，否则会先接受一局错误难度的对局。
        self._apply_difficulty()
        # 3) 如果当前已经是新局、且种子/场地都对得上，直接用。
        #    对不上（例如第一局时页面还是默认种子、或换到了别的场地）必须当“没准备好”，
        #    走下面的重开流程。
        with self._lock:
            s = self.latest_state
        seed_ok = (self._current_seed is None or
                   (s is not None and s.get('seed') is not None
                    and int(s['seed']) == self._current_seed))
        bg_ok = (self.bg is None or
                 (s is not None and s.get('bg') is not None
                  and int(s['bg']) == self.bg))
        if seed_ok and bg_ok and self._wait_new_round(GRACE_TIME):
            return
        # 4) 否则主动请页面重开（最多 4 次）
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
        # 本局的随机种子：第 1 局 = seed_base + 0
        self._current_seed = (None if self.seed_base is None
                              else self.seed_base + (self._episode - 1))
        seed_note = '' if self._current_seed is None else f"，seed={self._current_seed}"
        print(f"[Reset] 第 {self._episode} 局开始{seed_note}，等待游戏连接...")
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
        self.observed_difficulty = state.get('difficulty')
        self.observed_ai = state.get('ai')
        self.observed_seed = state.get('seed')
        self.observed_bg = state.get('bg')
        # 场地校验：各场地宽度差 4 倍而观测只对 ~800 宽成立，跑错场地等于跑错游戏。
        if self.bg is not None:
            if self.observed_bg is None:
                raise RuntimeError(
                    "页面状态里没有 bg：浏览器还在跑旧版 match.js。请强制刷新页面。")
            if int(self.observed_bg) != self.bg:
                raise RuntimeError(
                    f"场地校验失败：要求 bg={self.bg}，页面报告 {self.observed_bg}。")
        # 种子校验：页面报的种子必须就是本局要的那个，否则同样的种子记录没法复现。
        # 旧版页面（match.js 里没有 seed 字段）会在这里报 None —— 强制刷新即可。
        if self._current_seed is not None:
            if self.observed_seed is None:
                raise RuntimeError(
                    "页面状态里没有 seed：浏览器还在跑旧版 match.js。"
                    "请用 serve.py（禁缓存）重新打开页面，或强制刷新。")
            if int(self.observed_seed) != self._current_seed:
                raise RuntimeError(
                    f"种子校验失败：要求 seed={self._current_seed}，"
                    f"页面报告 {self.observed_seed}。拒绝在这个种子下继续评估。")
        # 最后一道闸：_apply_difficulty 之后如果又发生重载/人工干预，这里也不会放过
        if self.diff is not None and self.observed_difficulty != self.diff:
            raise RuntimeError(
                f"难度校验失败：要求 diff={self.diff}（{self.diff_name}），"
                f"页面报告 {self.observed_difficulty}（对手 AI：{self.observed_ai}）。"
                f"拒绝在这个难度下继续训练/评估。")
        # 难度/对手换过就报一次：跑几小时之后，日志里必须能看出这条曲线是在打谁
        ai = state.get('ai')
        if ai and ai != self._announced_ai:
            self._announced_ai = ai
            print(f"[难度] {DIFF_NAMES.get(state.get('difficulty'), '?')}（对手 AI：{ai}）")
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
            obs = self._state_to_obs(s) if s is not None else np.zeros(len(OBS_SCALE), dtype=np.float32)
            # 主动让页面重开一局，避免下一局从中局开始
            self._request_reload()
            return obs, -5.0, False, True, {'result': 'timeout'}

        obs = self._state_to_obs(curr)
        reward = self._compute_reward(self.prev_state, curr, action)

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
            reward -= TIMEOUT_PENALTY
            print(f"[结果] ⏱ 到达步数上限 {MAX_EPISODE_STEPS}，截断本局")

        return obs, reward, terminated, truncated, info

    def close(self):
        pass
