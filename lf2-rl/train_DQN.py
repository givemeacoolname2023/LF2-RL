"""
train_DQN.py — Davis vs Dennis Double DQN 訓練主程式
使用 Double DQN + Replay Buffer + ε-greedy

除了训练本身，这版还补上了三件事：
    1. TensorBoard 训练曲线 —— 写进 logs/DQN_N/，tag 命名对齐 SB3 的 PPO（logs/PPO_N），
       所以 `tensorboard --logdir logs` 里两条路线的曲线可以直接叠在一张图上比；
    2. 富评估 —— evaluate() 返回的指标和 eval.py 的报告同口径（胜率 + Wilson CI、
       出招占比 vs 41.7% 随机基线、平均水平距离、平均纵深 |dz| 与对齐率、打掉/挨打血量），
       既写进 TB 也写 CSV；
    3. 断点续训 —— checkpoint 存 optimizer / total_steps / episode，`--resume` 接着跑。

用法（前提：8765 空闲，serve.py + 浏览器页面已开）：
    python train_DQN.py                              # 默认 5000 局
    python train_DQN.py --episodes 2                 # 冒烟
    python train_DQN.py --resume                     # 从 checkpoints/last.pt 续训
    python train_DQN.py --resume checkpoints/ep500.pt
    python train_DQN.py --no-tb                      # 不写 TensorBoard

环境变量（dqn_bench.py 会导入这里的常量，所以指标同时保留模块级定义）：
    LF2_EPISODES / LF2_EVAL_EVERY / LF2_EVAL_EPISODES / LF2_CKPT_EVERY
    LF2_LOG_DIR / LF2_LOG_WINDOW / LF2_TB / LF2_EPSILON_DECAY

注意：`dqn_bench.py` 从本文件导入 QNetwork / ReplayBuffer / select_action / update /
get_epsilon / OBS_DIM / ACT_DIM / LR / BUFFER_SIZE / WARMUP_STEPS / TARGET_UPDATE /
EPISODES / EVAL_EVERY / EVAL_EPISODES，并且忽略 update() 的返回值。
改这些名字或改 update() 的返回类型都会把它弄坏。
"""

import argparse
import csv
import json
import os
import random
import sys
import time
from collections import Counter, deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from env import (ACTION_KEYS, DIFF_NAMES, LF2Env, OBS_SCALE,
                 diff_arg, resolve_diff)
from rl_metrics import (
    ATTACK_ACTIONS, RANDOM_ATTACK_RATE,
    action_entropy, action_name, depth_aligned, dist_px, dz_px,
    hp_from_obs, is_attack, wilson,
)

# =====================================================
# 超參數
# =====================================================
EPISODES        = int(os.environ.get('LF2_EPISODES', '5000'))   # 總訓練場數
# 从 env.py 推导，别再手写数字：观测/动作空间改动后这里自动跟上
OBS_DIM         = len(OBS_SCALE)   # 觀察值維度
ACT_DIM         = len(ACTION_KEYS) # 動作數量

GAMMA           = 0.99    # 折扣因子
LR              = 1e-3    # 學習率
BATCH_SIZE      = 64      # 每次更新抽幾筆
BUFFER_SIZE     = 100_000 # Replay buffer 大小
WARMUP_STEPS    = 1000    # 多少步後才開始訓練
TARGET_UPDATE   = 1000    # 每幾步同步 target network

EPSILON_START   = 1.0     # 初始探索率
EPSILON_END     = 0.05    # 最低探索率
# 注意：5 万步在一轮 1090 万步的训练里只占 0.46%（TRAINING_FIXES.md §17.6）。
# 默认值保持原样不偷改；想改就用 LF2_EPSILON_DECAY 覆盖，曲线上的 explore/epsilon 会直接显示出来。
EPSILON_DECAY   = int(os.environ.get('LF2_EPSILON_DECAY', '50000'))

EVAL_EVERY      = int(os.environ.get('LF2_EVAL_EVERY', '100'))   # 每幾場評估一次
EVAL_EPISODES   = int(os.environ.get('LF2_EVAL_EPISODES', '10')) # 評估時跑幾場（原 20）
CKPT_EVERY      = int(os.environ.get('LF2_CKPT_EVERY', '100'))   # 每幾局寫一次 last.pt
FULL_CKPT_EVERY = int(os.environ.get('LF2_FULL_CKPT_EVERY', '500'))  # 每幾局寫一次 ep{N}.pt
LOG_WINDOW      = int(os.environ.get('LF2_LOG_WINDOW', '100'))   # 滚动平均窗口（局）
LOG_DIR         = os.environ.get('LF2_LOG_DIR', 'logs')
RUN_PREFIX      = 'DQN'   # 目录名 logs/DQN_1、logs/DQN_2 ...
SAVE_DIR        = 'checkpoints'
TB_ENABLED      = os.environ.get('LF2_TB', '1') not in ('0', 'false', 'False')

RESULT_LABEL = {'win': '🏆 贏', 'lose': '💀 輸', 'timeout': '⏱ 超時',
                'max_steps': '⏱ 超時'}


def _ensure_utf8_stdout():
    """让 print 里的 emoji 在「重定向到文件 / 管道」时也不炸。

    Windows 控制台重定向时 Python 用 GBK 编码 stdout，`🏆` 会直接
    UnicodeEncodeError 打断训练。三天长跑一般都会 `> log.txt`，所以必须先修好。
    reconfigure 是幂等的，重复调用没有代价。
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding='utf-8', errors='replace')
        except Exception:                       # noqa: BLE001 —— 老版本/非常规流就算了
            pass

# =====================================================
# Q 網路（小 MLP，適合這個問題規模）
# =====================================================
class QNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, act_dim)
        )

    def forward(self, x):
        return self.net(x)


# =====================================================
# Replay Buffer
# =====================================================
class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, s_next, done):
        self.buf.append((s, a, r, s_next, done))

    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        s, a, r, s_next, done = zip(*batch)
        return (
            torch.FloatTensor(np.array(s)),
            torch.LongTensor(a),
            torch.FloatTensor(r),
            torch.FloatTensor(np.array(s_next)),
            torch.FloatTensor(done)
        )

    def __len__(self):
        return len(self.buf)


# =====================================================
# ε-greedy 動作選擇
# =====================================================
def select_action(q_net, obs, epsilon, device):
    if random.random() < epsilon:
        return random.randint(0, ACT_DIM - 1)
    with torch.no_grad():
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
        q_vals = q_net(obs_t)
        return q_vals.argmax().item()


def get_epsilon(step):
    """線性衰減 epsilon"""
    ratio = min(1.0, step / EPSILON_DECAY)
    return EPSILON_START + ratio * (EPSILON_END - EPSILON_START)


# =====================================================
# Double DQN 更新
# =====================================================
def update(q_net, target_net, optimizer, buffer, device, stats=None):
    """回傳 loss（float）。

    stats 是可選的出參（dict），傳了就會累加 q_sum / target_sum / grad_sum / n，
    用來給 TensorBoard 畫 Q 值發散和梯度爆炸 —— 這兩種是 DQN 特有的壞法，
    光看 reward 曲線要很久才看得出来。不傳就完全沒有額外開銷（dqn_bench.py 就是這樣）。
    """
    if len(buffer) < BATCH_SIZE:
        return 0.0

    s, a, r, s_next, done = buffer.sample(BATCH_SIZE)
    s      = s.to(device)
    a      = a.to(device)
    r      = r.to(device)
    s_next = s_next.to(device)
    done   = done.to(device)

    # 目前 Q 值
    q_vals = q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

    # Double DQN target：用 q_net 選動作，用 target_net 估值
    with torch.no_grad():
        next_actions = q_net(s_next).argmax(1)
        next_q = target_net(s_next).gather(1, next_actions.unsqueeze(1)).squeeze(1)
        target = r + GAMMA * next_q * (1 - done)

    loss = nn.MSELoss()(q_vals, target)

    optimizer.zero_grad()
    loss.backward()
    grad_norm = nn.utils.clip_grad_norm_(q_net.parameters(), 10)  # 梯度裁剪，防爆炸
    optimizer.step()

    if stats is not None:
        stats['q_sum']      = stats.get('q_sum', 0.0) + float(q_vals.mean())
        stats['target_sum'] = stats.get('target_sum', 0.0) + float(target.mean())
        stats['grad_sum']   = stats.get('grad_sum', 0.0) + float(grad_norm)
        stats['n']          = stats.get('n', 0) + 1

    return loss.item()


# =====================================================
# 評估（關掉 ε-greedy，純用 Q 值選動作）
# =====================================================
def evaluate(q_net, env, device, n_episodes=EVAL_EPISODES):
    """跑 n_episodes 局贪婪策略，返回指标 dict。

    口径和 eval.py 完全一致（同一个 rl_metrics），所以这里的 eval/win_rate
    和离线 `python eval.py --algo dqn` 报出来的胜率是可比的。
    """
    q_net.eval()

    results = Counter()
    actions = Counter()
    steps_list = []
    total_steps = total_att = 0
    dist_sum = dz_sum = 0.0
    aligned = 0
    dealt_sum = taken_sum = total_reward = 0.0

    with torch.no_grad():
        for _ in range(n_episodes):
            obs, _ = env.reset()
            hp0_d, hp0_e = hp_from_obs(obs)
            ep_reward, steps, done, info = 0.0, 0, False, {}

            while not done:
                obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
                action = int(q_net(obs_t).argmax().item())
                actions[action] += 1
                total_att += is_attack(action)

                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                ep_reward += reward
                steps += 1
                dist_sum += dist_px(obs)
                dz_sum += dz_px(obs)
                aligned += depth_aligned(obs)

            hp_d, hp_e = hp_from_obs(obs)
            dealt_sum += max(0.0, hp0_e - hp_e)
            taken_sum += max(0.0, hp0_d - hp_d)
            results[info.get('result', 'unknown')] += 1
            steps_list.append(steps)
            total_steps += steps
            total_reward += ep_reward

    q_net.train()

    n = max(1, n_episodes)
    wins, loses = results['win'], results['lose']
    decisive = wins + loses
    ci_low, ci_high = wilson(wins, n)

    return {
        'n': n_episodes,
        'wins': wins,
        'loses': loses,
        'timeouts': results['max_steps'] + results['timeout'],
        'results': results,
        'win_rate': wins / n,
        'win_rate_decisive': (wins / decisive) if decisive else None,
        'ci_low': ci_low,
        'ci_high': ci_high,
        'avg_reward': total_reward / n,
        'avg_len': total_steps / n,
        'timeout_rate': (results['max_steps'] + results['timeout']) / n,
        'attack_rate': total_att / total_steps if total_steps else 0.0,
        'mean_dist': dist_sum / total_steps if total_steps else 0.0,
        'mean_dz': dz_sum / total_steps if total_steps else 0.0,
        'align_rate': aligned / total_steps if total_steps else 0.0,
        'dealt': dealt_sum / n,
        'taken': taken_sum / n,
        'action_counts': actions,
        'distinct_len': len(set(steps_list)),
    }


# =====================================================
# 記錄：TensorBoard 寫入器 / CSV / checkpoint
# =====================================================
class _NullWriter:
    """LF2_TB=0 或 tensorboard 坏了时候的替身：所有写操作都是空操作。

    日志永远不该弄死一场三天的训练，所以 TensorBoard 出任何问题都退化成它。
    """

    def add_scalar(self, *a, **k):
        pass

    def add_text(self, *a, **k):
        pass

    def add_hparams(self, *a, **k):
        pass

    def flush(self):
        pass

    def close(self):
        pass


def resolve_run_dir(log_dir=LOG_DIR, prefix=RUN_PREFIX, run_name=None):
    """(run_name, run_dir)。run_name 给了就用它（续训接上原目录），否则取第一个空位 DQN_N。

    和 SB3 的 logs/PPO_N 并列，`tensorboard --logdir logs` 里两条路线一起出现。
    """
    if not log_dir:
        return None, None
    os.makedirs(log_dir, exist_ok=True)
    if run_name:
        run_dir = os.path.join(log_dir, run_name)
        os.makedirs(run_dir, exist_ok=True)
        return run_name, run_dir
    i = 1
    while os.path.isdir(os.path.join(log_dir, f'{prefix}_{i}')):
        i += 1
    run_dir = os.path.join(log_dir, f'{prefix}_{i}')
    os.makedirs(run_dir, exist_ok=True)
    return f'{prefix}_{i}', run_dir


def make_writer(run_dir, enabled=True):
    if not enabled or not run_dir:
        return _NullWriter()
    try:
        from torch.utils.tensorboard import SummaryWriter
        return SummaryWriter(log_dir=run_dir, flush_secs=10)
    except Exception as e:                      # noqa: BLE001 —— 故意兜住一切
        print(f'[警告] TensorBoard 不可用（{e}），本次不写曲线，终端输出照旧。')
        return _NullWriter()


class CsvAppender:
    """逐行追加 + 立即 flush。整份文件重写在 5000 局的规模下会变成 O(n²)。"""

    def __init__(self, path, fieldnames):
        self.path = path
        self.fieldnames = fieldnames
        self._fh = None
        self._writer = None
        self._warned = False

    def write(self, row):
        try:
            if self._fh is None:
                is_new = not os.path.exists(self.path)
                self._fh = open(self.path, 'a', newline='', encoding='utf-8')
                self._writer = csv.DictWriter(self._fh, fieldnames=self.fieldnames,
                                              extrasaction='ignore')
                if is_new:
                    self._writer.writeheader()
            self._writer.writerow(row)
            self._fh.flush()
        except OSError as e:
            if not self._warned:
                print(f'[警告] 写 {self.path} 失败（{e}），本次不再写这个文件。')
                self._warned = True

    def close(self):
        if self._fh is not None:
            try:
                self._fh.close()
            finally:
                self._fh = None


EP_FIELDS = ['episode', 'total_steps', 'result', 'reward', 'steps', 'epsilon',
             'loss_mean', 'q_mean', 'target_mean', 'grad_norm', 'attack_rate',
             'action_entropy', 'fps', 'secs']

EVAL_FIELDS = ['episode', 'total_steps', 'n', 'wins', 'loses', 'win_rate',
               'ci_low', 'ci_high', 'win_rate_decisive', 'avg_reward', 'avg_len',
               'timeout_rate', 'attack_rate', 'mean_dist', 'mean_dz',
               'align_rate', 'dealt', 'taken',
               'distinct_len', 'eval_secs'] + [f'a{a:02d}' for a in range(ACT_DIM)]


def _atomic_save(obj, path):
    """先写 .tmp 再 os.replace。

    训练写到一半被 Ctrl+C / 浏览器崩掉时，直接 torch.save 会留下半截文件，
    而 best_model.pt 是三天训练里唯一值得留的东西 —— 不能让它变成垃圾。
    """
    tmp = path + '.tmp'
    torch.save(obj, tmp)
    os.replace(tmp, path)


def save_checkpoint(path, q_net, target_net, optimizer, total_steps, episode,
                    best_win_rate, best_avg_reward, config=None, run_name=None,
                    plain=False):
    """存 checkpoint。

    plain=True 时只存裸 state_dict —— best_model.pt 走这条路，因为
    TRAINING_FIXES.md §13.6 公开的加载片段是 `net.load_state_dict(torch.load(...))`。
    """
    if plain:
        _atomic_save(q_net.state_dict(), path)
        return
    _atomic_save({
        'format': 2,
        'model': q_net.state_dict(),
        'target': target_net.state_dict() if target_net is not None else None,
        'optimizer': optimizer.state_dict() if optimizer is not None else None,
        'total_steps': total_steps,
        'episode': episode,
        'best_win_rate': best_win_rate,
        'best_avg_reward': best_avg_reward,
        'config': config or {},
        'run_name': run_name,
    }, path)


def load_checkpoint(path, device='cpu'):
    """读 checkpoint，旧版的裸 state_dict 也能读。

    返回 dict，至少有 model / total_steps / episode / best_win_rate。
    ε 不用存：get_epsilon(total_steps) 是纯函数，续训时自动接着衰减。
    """
    obj = torch.load(path, map_location=device, weights_only=False)

    if isinstance(obj, dict) and isinstance(obj.get('model'), dict):
        ckpt = dict(obj)
        ckpt.setdefault('format', 2)
        ckpt.setdefault('target', None)
        ckpt.setdefault('optimizer', None)
        ckpt.setdefault('total_steps', 0)
        ckpt.setdefault('episode', 0)
        ckpt.setdefault('best_win_rate', 0.0)
        ckpt.setdefault('best_avg_reward', float('-inf'))
        ckpt.setdefault('config', {})
        ckpt.setdefault('run_name', None)
        return ckpt

    # 旧格式：裸 state_dict（best_model.pt / ep{N}.pt 的老版本）
    if isinstance(obj, dict) and obj and all(torch.is_tensor(v) for v in obj.values()):
        return {
            'format': 1, 'model': obj, 'target': None, 'optimizer': None,
            'total_steps': 0, 'episode': 0,
            'best_win_rate': 0.0, 'best_avg_reward': float('-inf'),
            'config': {}, 'run_name': None,
        }

    raise ValueError(f'{path} 不是可识别的 checkpoint（既不是训练存档也不是裸 state_dict）')


def config_dict(run_name, diff=None):
    level = resolve_diff(diff)
    return {
        'obs_dim': OBS_DIM, 'act_dim': ACT_DIM, 'gamma': GAMMA, 'lr': LR,
        'batch_size': BATCH_SIZE, 'buffer_size': BUFFER_SIZE,
        'warmup_steps': WARMUP_STEPS, 'target_update': TARGET_UPDATE,
        'epsilon_start': EPSILON_START, 'epsilon_end': EPSILON_END,
        'epsilon_decay': EPSILON_DECAY, 'episodes': EPISODES,
        'eval_every': EVAL_EVERY, 'eval_episodes': EVAL_EPISODES,
        'ckpt_every': CKPT_EVERY, 'log_window': LOG_WINDOW,
        # 本次要求的难度（None = 沿用页面 URL）。写进 TensorBoard 和 checkpoint，
        # 以后看曲线/续训时才知道这条胜率是在打谁。
        'diff': level, 'diff_name': DIFF_NAMES.get(level),
        'run_name': run_name, 'torch_version': torch.__version__,
    }


# =====================================================
# 主訓練迴圈
# =====================================================
def train(resume=None, episodes=None, eval_every=None, eval_episodes=None,
          ckpt_every=None, log_dir=None, device=None, new_run=False,
          run_name=None, diff=None):
    """训练主循环。

    参数为 None 时一律回落到模块级常量（环境变量已经处理过一遍）。
    resume 传路径则从该 checkpoint 续训；log_dir 传空字符串表示不写 TensorBoard。
    run_name 是显式指定日志目录名（冒烟测试用），平时不用传。
    """
    _ensure_utf8_stdout()

    episodes      = EPISODES if episodes is None else episodes
    eval_every    = EVAL_EVERY if eval_every is None else eval_every
    eval_episodes = EVAL_EPISODES if eval_episodes is None else eval_episodes
    ckpt_every    = CKPT_EVERY if ckpt_every is None else ckpt_every
    log_dir       = LOG_DIR if log_dir is None else log_dir

    device = torch.device(device) if device else \
        torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用裝置：{device}')

    # ---- 1. 先读 checkpoint，好决定这次开哪个日志目录 ----
    ckpt = None
    if resume:
        if not os.path.isfile(resume):
            print(f'[错误] 找不到 checkpoint：{resume}')
            print('       从头训练就别加 --resume；或确认 checkpoints/last.pt 存在。')
            return []
        ckpt = load_checkpoint(resume, device=torch.device('cpu'))
        if ckpt['config'].get('obs_dim', OBS_DIM) != OBS_DIM or \
                ckpt['config'].get('act_dim', ACT_DIM) != ACT_DIM:
            print(f"[错误] checkpoint 的网络维度和当前 env 不一致"
                  f"（存档 {ckpt['config'].get('obs_dim')}/{ckpt['config'].get('act_dim')}，"
                  f"当前 {OBS_DIM}/{ACT_DIM}）。观测或动作空间改过？")
            return []

    # 续训默认接回原来那个 logs/DQN_N，曲线才是连续的；--new-run 强制开新的
    explicit_name = run_name
    if not explicit_name and not new_run:
        explicit_name = (ckpt or {}).get('run_name')
    run_name, run_dir = resolve_run_dir(log_dir, run_name=explicit_name)
    writer = make_writer(run_dir, enabled=TB_ENABLED)
    cfg = config_dict(run_name, diff)

    print(f'[日志] {run_dir or "（TensorBoard 已关闭）"}    [局数] {episodes}    '
          f'[评估] 每 {eval_every} 局跑 {eval_episodes} 局')
    if cfg['diff'] is not None:
        print(f'[难度] 请求 {cfg["diff_name"]}（diff={cfg["diff"]}）；'
              f'env.py 会切换并校验，切不过去会直接报错。')
    else:
        print('[难度] 未指定 --diff，沿用页面 URL 里的难度（旧行为）。')
    if run_dir:
        print(f'        tensorboard --logdir {log_dir}')
    writer.add_text('meta/config_json', json.dumps(cfg, ensure_ascii=False), 0)
    writer.add_scalar('train/learning_rate', LR, 0)

    # ---- 2. 建立環境（會等瀏覽器連接）----
    env = LF2Env(diff=diff)

    # ---- 3. 建立網路 ----
    q_net      = QNetwork(OBS_DIM, ACT_DIM).to(device)
    target_net = QNetwork(OBS_DIM, ACT_DIM).to(device)
    optimizer  = optim.Adam(q_net.parameters(), lr=LR)
    buffer     = ReplayBuffer(BUFFER_SIZE)

    total_steps     = 0
    best_win_rate   = 0.0
    best_avg_reward = float('-inf')
    start_episode   = 1
    history         = []   # 每轮评估一条

    if ckpt is not None:
        # 续训时难度库不一致要提醒：不同难度的胜率不能直接对比。
        # 旧 checkpoint 没有 diff 这个键，取到 None 就不报警。
        prev_diff = (ckpt.get('config') or {}).get('diff')
        new_diff = cfg['diff']
        if prev_diff is not None and new_diff is not None and prev_diff != new_diff:
            print(f'[警告] 这个 checkpoint 是在 diff={prev_diff}'
                  f'（{DIFF_NAMES.get(prev_diff, "?")}）下训练的，本次要求 diff={new_diff}'
                  f'（{DIFF_NAMES.get(new_diff, "?")}）。两边的胜率不能直接对比。')
        q_net.load_state_dict(ckpt['model'])
        if ckpt.get('target') is not None:
            target_net.load_state_dict(ckpt['target'])
        else:
            target_net.load_state_dict(q_net.state_dict())
        if ckpt.get('optimizer') is not None:
            optimizer.load_state_dict(ckpt['optimizer'])
        else:
            print('[提示] 这个 checkpoint 没存 optimizer（旧格式），Adam 的动量重新开始累计。')
        total_steps     = ckpt['total_steps']
        start_episode   = ckpt['episode'] + 1
        best_win_rate   = ckpt['best_win_rate']
        best_avg_reward = ckpt['best_avg_reward']

        print(f'[续训] {resume}')
        print(f'       已完成 {ckpt["episode"]} 局 / {total_steps} 步，'
              f'从现在起接着跑；ε={get_epsilon(total_steps):.3f}')
        # Replay buffer 不存档（1 万条经验序列化要几十 MB 且很慢）。
        # update() 自带 len(buffer) < BATCH_SIZE 的守卫，所以前 64 步只是不更新。
        print('       经验池需要重新填充：前 64 步不更新，之后再恢复正常。')
        writer.add_text('meta/status',
                        f'resumed from {resume} at ep {ckpt["episode"]} step {total_steps}', 0)
    else:
        writer.add_text('meta/status', 'started', 0)

    os.makedirs(SAVE_DIR, exist_ok=True)

    csv_ep   = CsvAppender(os.path.join(run_dir, 'episodes.csv'), EP_FIELDS) if run_dir else None
    csv_eval = CsvAppender(os.path.join(run_dir, 'eval.csv'), EVAL_FIELDS) if run_dir else None
    history_path = os.path.join(run_dir, 'history.json') if run_dir else None

    # 最近 LOG_WINDOW 局的滚动统计
    roll = deque(maxlen=max(1, LOG_WINDOW))

    def write_history():
        if not history_path:
            return
        try:
            with open(history_path, 'w', encoding='utf-8') as f:
                json.dump(history, f, ensure_ascii=False, indent=1)
        except OSError as e:
            print(f'[警告] 写 {history_path} 失败：{e}')

    def dump(path, **kw):
        save_checkpoint(path, q_net, target_net, optimizer, total_steps,
                        current_episode, best_win_rate, best_avg_reward,
                        config=cfg, run_name=run_name, **kw)

    current_episode = start_episode - 1
    t_start = time.perf_counter()
    t_eval_total = 0.0

    print(f'\n開始訓練！共 {episodes} 场（从第 {start_episode} 场开始）')
    print('=' * 72)

    try:
        for episode in range(start_episode, episodes + 1):
            current_episode = episode
            t_ep0 = time.perf_counter()

            obs, _ = env.reset()
            hp0_d, hp0_e = hp_from_obs(obs)
            done = False
            ep_reward   = 0.0
            ep_steps    = 0
            ep_att      = 0
            ep_dist_sum = 0.0
            ep_loss_sum = 0.0
            ep_loss_cnt = 0
            ep_actions  = Counter()
            ep_stats    = {}

            while not done:
                epsilon = get_epsilon(total_steps)
                action  = select_action(q_net, obs, epsilon, device)
                ep_actions[action] += 1
                ep_att += is_attack(action)

                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                # 只有真正阵亡（terminated）才是终止状态。到达 3000 帧上限或帧超时只是
                # 被截断（truncated），下一步仍然有 Q 值；若在这里写 float(done)，
                # target 会退化成 r，长线策略（躲开再打）被系统性低估，速攻换血被高估。
                buffer.push(obs, action, reward, next_obs, float(terminated))
                obs = next_obs

                ep_reward += reward
                ep_steps  += 1
                total_steps += 1
                ep_dist_sum += dist_px(obs)

                # 開始訓練（warmup 結束後）
                if total_steps >= WARMUP_STEPS:
                    loss = update(q_net, target_net, optimizer, buffer, device, ep_stats)
                    ep_loss_sum += loss
                    ep_loss_cnt += 1

                # 同步 target network
                if total_steps % TARGET_UPDATE == 0:
                    target_net.load_state_dict(q_net.state_dict())

            # ---------- 每局結束 ----------
            secs      = time.perf_counter() - t_ep0
            avg_loss  = ep_loss_sum / ep_loss_cnt if ep_loss_cnt else 0.0
            epsilon   = get_epsilon(total_steps)
            hp_d, hp_e = hp_from_obs(obs)
            result    = info.get('result', 'unknown')
            n_upd     = ep_stats.get('n', 0)
            q_mean    = ep_stats.get('q_sum', 0.0) / n_upd if n_upd else 0.0
            tgt_mean  = ep_stats.get('target_sum', 0.0) / n_upd if n_upd else 0.0
            grad_norm = ep_stats.get('grad_sum', 0.0) / n_upd if n_upd else 0.0
            att_rate  = ep_att / ep_steps if ep_steps else 0.0
            entropy   = action_entropy(ep_actions, ep_steps)

            roll.append({'reward': ep_reward, 'len': ep_steps, 'win': result == 'win',
                         'timeout': result in ('max_steps', 'timeout'),
                         'steps': ep_steps, 'secs': secs})

            print(f'[Ep {episode:4d}] {RESULT_LABEL.get(result, "❓ 中斷")} | '
                  f'reward={ep_reward:7.2f} | steps={ep_steps:4d} | '
                  f'ε={epsilon:.3f} | loss={avg_loss:7.4f} | '
                  f'{ep_steps / secs if secs else 0:4.1f} fps')

            # ---------- 写 TensorBoard / CSV ----------
            win_rate_w  = sum(r['win'] for r in roll) / len(roll)
            timeout_w   = sum(r['timeout'] for r in roll) / len(roll)
            rew_w       = sum(r['reward'] for r in roll) / len(roll)
            len_w       = sum(r['len'] for r in roll) / len(roll)
            fps_w       = sum(r['steps'] for r in roll) / max(1e-9, sum(r['secs'] for r in roll))

            s = total_steps   # 所有标量统一用步数当横轴，才能和 PPO 的 total_timesteps 对齐
            writer.add_scalar('episode/reward', ep_reward, s)
            writer.add_scalar('episode/length', ep_steps, s)
            writer.add_scalar('explore/epsilon', epsilon, s)
            writer.add_scalar('explore/action_entropy', entropy, s)
            for a in range(ACT_DIM):
                writer.add_scalar(f'explore/action_rate/{action_name(a)}',
                                  ep_actions.get(a, 0) / ep_steps if ep_steps else 0.0, s)
            writer.add_scalar('train/td_loss', avg_loss, s)
            writer.add_scalar('train/q_mean', q_mean, s)
            writer.add_scalar('train/target_mean', tgt_mean, s)
            writer.add_scalar('train/grad_norm', grad_norm, s)
            writer.add_scalar('rollout/ep_rew_mean', rew_w, s)
            writer.add_scalar('rollout/ep_len_mean', len_w, s)
            writer.add_scalar('rollout/ep_win_rate', win_rate_w, s)
            writer.add_scalar('rollout/ep_timeout_rate', timeout_w, s)
            writer.add_scalar('time/fps', fps_w, s)

            if csv_ep is not None:
                csv_ep.write({'episode': episode, 'total_steps': s, 'result': result,
                              'reward': round(ep_reward, 4), 'steps': ep_steps,
                              'epsilon': round(epsilon, 5), 'loss_mean': round(avg_loss, 6),
                              'q_mean': round(q_mean, 6), 'target_mean': round(tgt_mean, 6),
                              'grad_norm': round(grad_norm, 6),
                              'attack_rate': round(att_rate, 5),
                              'action_entropy': round(entropy, 5),
                              'fps': round(fps_w, 3), 'secs': round(secs, 2)})

            # ---------- 定期存档（和评估节奏解耦）----------
            if episode % ckpt_every == 0:
                dump(os.path.join(SAVE_DIR, 'last.pt'))
            if episode % FULL_CKPT_EVERY == 0:
                dump(os.path.join(SAVE_DIR, f'ep{episode}.pt'))
                print(f'  Checkpoint 已儲存：{SAVE_DIR}/ep{episode}.pt')

            writer.flush()

            # ---------- 定期评估 ----------
            if episode % eval_every == 0:
                print(f'\n--- 第 {episode} 场评估（{eval_episodes} 场）---')
                t_eval0 = time.perf_counter()
                ev = evaluate(q_net, env, device, eval_episodes)
                eval_secs = time.perf_counter() - t_eval0
                t_eval_total += eval_secs

                ci = f'（95% CI {ev["ci_low"] * 100:.1f}%–{ev["ci_high"] * 100:.1f}%）'
                print(f'  勝率：{ev["win_rate"]:.1%} {ci}  |  平均 reward：{ev["avg_reward"]:.3f}')
                print(f'  局长 {ev["avg_len"]:.0f} 帧 | 出招 {ev["attack_rate"] * 100:.1f}% '
                      f'(随机基线 {RANDOM_ATTACK_RATE * 100:.1f}%) | '
                      f'纵深 {ev["mean_dz"]:.0f}px 对齐 {ev["align_rate"] * 100:.1f}% | '
                      f'打掉 {ev["dealt"]:.0f} / 挨打 {ev["taken"]:.0f} hp')

                writer.add_scalar('eval/win_rate', ev['win_rate'], s)
                writer.add_scalar('eval/win_rate_ci_low', ev['ci_low'], s)
                writer.add_scalar('eval/win_rate_ci_high', ev['ci_high'], s)
                if ev['win_rate_decisive'] is not None:
                    writer.add_scalar('eval/win_rate_decisive', ev['win_rate_decisive'], s)
                writer.add_scalar('eval/avg_reward', ev['avg_reward'], s)
                writer.add_scalar('eval/avg_len', ev['avg_len'], s)
                writer.add_scalar('eval/timeout_rate', ev['timeout_rate'], s)
                writer.add_scalar('eval/attack_rate', ev['attack_rate'], s)
                writer.add_scalar('eval/mean_dist', ev['mean_dist'], s)
                writer.add_scalar('eval/mean_dz', ev['mean_dz'], s)
                writer.add_scalar('eval/align_rate', ev['align_rate'], s)
                writer.add_scalar('eval/dealt', ev['dealt'], s)
                writer.add_scalar('eval/taken', ev['taken'], s)
                writer.add_scalar('eval/distinct_len', ev['distinct_len'], s)
                writer.add_scalar('eval/n_episodes', ev['n'], s)
                writer.add_scalar('eval/time_secs', eval_secs, s)
                writer.add_scalar('time/eval_frac',
                                  t_eval_total / max(1e-9, time.perf_counter() - t_start), s)
                for a in range(ACT_DIM):
                    writer.add_scalar(
                        f'eval/action_rate/{action_name(a)}',
                        ev['action_counts'].get(a, 0) / max(1, ev['avg_len'] * ev['n']), s)

                if csv_eval is not None:
                    row = {k: ev[k] for k in ('n', 'wins', 'loses', 'win_rate',
                                              'ci_low', 'ci_high', 'avg_reward',
                                              'avg_len', 'timeout_rate', 'attack_rate',
                                              'mean_dist', 'mean_dz', 'align_rate',
                                              'dealt', 'taken',
                                              'distinct_len')}
                    row.update({'episode': episode, 'total_steps': s,
                                'win_rate_decisive': ev['win_rate_decisive'],
                                'eval_secs': round(eval_secs, 2)})
                    for a in range(ACT_DIM):
                        row[f'a{a:02d}'] = ev['action_counts'].get(a, 0)
                    csv_eval.write(row)

                # 胜率相同时比平均 reward，避免 best 停在噪声高点
                if (ev['win_rate'], ev['avg_reward']) > (best_win_rate, best_avg_reward):
                    best_win_rate   = ev['win_rate']
                    best_avg_reward = ev['avg_reward']
                    path = os.path.join(SAVE_DIR, 'best_model.pt')
                    dump(path, plain=True)      # 裸 state_dict，保持旧片段可用
                    print(f'  ★ 新最佳勝率！模型已儲存到 {path}')

                writer.add_scalar('eval/best_win_rate', best_win_rate, s)

                history.append({
                    'episode': episode, 'total_steps': s, 'eval_secs': round(eval_secs, 2),
                    'n': ev['n'], 'wins': ev['wins'], 'loses': ev['loses'],
                    'timeouts': ev['timeouts'],
                    'win_rate': ev['win_rate'], 'ci_low': ev['ci_low'],
                    'ci_high': ev['ci_high'],
                    'win_rate_decisive': ev['win_rate_decisive'],
                    'avg_reward': ev['avg_reward'], 'avg_len': ev['avg_len'],
                    'attack_rate': ev['attack_rate'], 'mean_dist': ev['mean_dist'],
                    'mean_dz': ev['mean_dz'], 'align_rate': ev['align_rate'],
                    'dealt': ev['dealt'], 'taken': ev['taken'],
                    'distinct_len': ev['distinct_len'],
                    'action_counts': {action_name(a): c
                                      for a, c in ev['action_counts'].most_common()},
                })
                write_history()
                writer.flush()
                print(f'{"-" * 43}\n')

    except KeyboardInterrupt:
        print('\n[中断] Ctrl+C —— 正在保存现场...')
        try:
            dump(os.path.join(SAVE_DIR, 'interrupted.pt'))
            write_history()
            writer.add_text('meta/status', f'interrupted at ep {current_episode} '
                                           f'step {total_steps}', total_steps)
            print(f'  已存：{SAVE_DIR}/interrupted.pt')
            print(f'  继续训练：python train_DQN.py --resume {SAVE_DIR}/interrupted.pt')
        except Exception as e:                  # noqa: BLE001
            print(f'  [警告] 保存失败：{e}')
    except Exception:
        # 浏览器崩了 / 帧超时抛 TimeoutError 就是走这里 —— 先存盘再往上抛，别让三天白跑
        try:
            dump(os.path.join(SAVE_DIR, 'interrupted.pt'))
            write_history()
            writer.add_text('meta/status', f'crashed at ep {current_episode} '
                                           f'step {total_steps}', total_steps)
        except Exception:                       # noqa: BLE001
            pass
        raise
    else:
        print('\n訓練完成！')
        print(f'最佳勝率：{best_win_rate:.1%}')
        dump(os.path.join(SAVE_DIR, 'final_model.pt'))
        dump(os.path.join(SAVE_DIR, 'last.pt'))
        write_history()
        writer.add_text('meta/status', f'finished at ep {current_episode} '
                                       f'step {total_steps}', total_steps)
        print(f'模型已保存：{SAVE_DIR}/final_model.pt、{SAVE_DIR}/last.pt'
              + (f'、{SAVE_DIR}/best_model.pt' if best_win_rate > 0 or
                 best_avg_reward > float('-inf') else ''))
    finally:
        if csv_ep is not None:
            csv_ep.close()
        if csv_eval is not None:
            csv_eval.close()
        writer.flush()
        writer.close()
        env.close()

    return history


def main(argv=None):
    p = argparse.ArgumentParser(
        description='Double DQN 训练（TensorBoard 曲线 + 富评估 + 断点续训）')
    p.add_argument('--resume', nargs='?', const='', default=None, metavar='PATH',
                   help='从 checkpoint 续训；不写路径时用 checkpoints/last.pt')
    p.add_argument('--episodes', type=int, default=None, help=f'总局数（默认 {EPISODES}）')
    p.add_argument('--eval-every', type=int, default=None,
                   help=f'每多少局评估一次（默认 {EVAL_EVERY}）')
    p.add_argument('--eval-episodes', type=int, default=None,
                   help=f'每次评估跑几局（默认 {EVAL_EPISODES}）')
    p.add_argument('--ckpt-every', type=int, default=None,
                   help=f'每多少局写一次 last.pt（默认 {CKPT_EVERY}）')
    p.add_argument('--log-dir', default=None, help=f'TensorBoard 根目录（默认 {LOG_DIR}）')
    p.add_argument('--new-run', action='store_true',
                   help='续训时强制开新的 logs/DQN_N（默认接回原来的目录，曲线连续）')
    p.add_argument('--no-tb', action='store_true', help='不写 TensorBoard')
    p.add_argument('--diff', type=diff_arg, choices=('easy', 'normal', 'difficult'),
                   default=None,
                   help='对手难度，可写 easy / normal / difficult 或 0 / 1 / 2；'
                        '不给则沿用页面 URL 里的设置（旧行为）。'
                        '指定后 env.py 会切换已打开的浏览器并校验。')
    p.add_argument('--device', default=None, choices=('cpu', 'cuda'))
    args = p.parse_args(argv)

    resume = args.resume
    if resume == '':
        resume = os.path.join(SAVE_DIR, 'last.pt')

    log_dir = args.log_dir
    if args.no_tb:
        log_dir = ''

    train(resume=resume, episodes=args.episodes, eval_every=args.eval_every,
          eval_episodes=args.eval_episodes, ckpt_every=args.ckpt_every,
          log_dir=log_dir, device=args.device, new_run=args.new_run,
          diff=args.diff)


if __name__ == '__main__':
    main()
