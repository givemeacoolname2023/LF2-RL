"""观测下标、出招判定、Wilson 置信区间。"""

import math
from collections import Counter

from env import ACTION_KEYS, OBS_SCALE

MIN_CENTER_DAMAGE_RATIO = 0.70

# 观测里的下标（对应 env.py 的 state_to_obs）
IDX_D_HP = 2    # davis hp  = obs[2] * OBS_SCALE[2]
IDX_E_HP = 8    # dennis hp = obs[8] * OBS_SCALE[8]
IDX_DX = 12     # dx        = obs[12] * OBS_SCALE[12]
# 纵深特征追加在末尾，原有下标不受影响
IDX_D_Z = 13    # davis z   = (obs[13] + OBS_OFFSET[13]) * OBS_SCALE[13]
IDX_E_Z = 14    # dennis z
IDX_DZ = 15     # dz = davis_z - dennis_z = obs[15] * OBS_SCALE[15]

# 含 'att' 的动作算一次出招：5=攻 7=左攻 8=右攻 9=跳攻 10=下攻
ATTACK_ACTIONS = sorted(i for i, keys in enumerate(ACTION_KEYS) if 'att' in keys)
HENRY_SPECIAL_ACTIONS = {16: 'gust', 17: 'piercing_arrow', 18: 'multiple_shot', 19: 'sonata'}
# 随机策略的出招占比基线：出招动作数 / 总动作数（随 ACTION_KEYS 自动变化）
RANDOM_ATTACK_RATE = len(ATTACK_ACTIONS) / len(ACTION_KEYS)


def hp_from_obs(obs):
    """从观测里取 (davis_hp, dennis_hp)，还原成真实血量。"""
    return (float(obs[IDX_D_HP]) * OBS_SCALE[IDX_D_HP],
            float(obs[IDX_E_HP]) * OBS_SCALE[IDX_E_HP])


def dist_px(obs):
    """双方水平距离（像素，取绝对值）。"""
    return abs(float(obs[IDX_DX] * OBS_SCALE[IDX_DX]))


def dz_px(obs):
    """双方纵深距离（像素，取绝对值）。"""
    return abs(float(obs[IDX_DZ] * OBS_SCALE[IDX_DZ]))


def depth_aligned(obs, zwidth=12.0):
    """纵深是否落在可以命中的带宽内（默认 zwidth=12，见 LF/global.js）。"""
    return dz_px(obs) <= zwidth


def is_attack(action, character='davis'):
    """这个动作算不算出招。"""
    return action in ATTACK_ACTIONS or (character == 'henry' and action in HENRY_SPECIAL_ACTIONS)


def action_name(action, character='davis'):
    """动作的可读名字，也是 TensorBoard 的 tag 片段（不要带空格/斜杠）。"""
    if character == 'henry' and action in HENRY_SPECIAL_ACTIONS:
        return HENRY_SPECIAL_ACTIONS[action]
    keys = ACTION_KEYS[action]
    return '_'.join(keys) if keys else 'noop'


def action_entropy(counts, total=None):
    """动作分布的香农熵（nat）。n 个动作的上限是 ln(n)（随 ACTION_KEYS 变化）。

    用来一条曲线看探索有没有塌掉：接近 0 就是策略退化成了单动作。
    """
    if total is None:
        total = sum(counts.values()) if isinstance(counts, Counter) else sum(counts)
    if not total:
        return 0.0
    values = counts.values() if isinstance(counts, Counter) else counts
    ent = 0.0
    for c in values:
        if c:
            p = c / total
            ent -= p * math.log(p)
    return ent


def wilson(k, n, z=1.96):
    """二项比例的 Wilson 置信区间。小样本下比正态近似靠谱得多。"""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)
