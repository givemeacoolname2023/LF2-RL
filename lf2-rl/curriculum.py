"""Validate and resolve the reward and curriculum settings used by PPO."""

import json
import math

from env import resolve_diff
from parallel_env import DEFAULT_REWARD


def reward_config(overrides, base=DEFAULT_REWARD):
    if not isinstance(overrides, dict) or overrides.keys() - DEFAULT_REWARD.keys():
        raise ValueError(f'奖励配置只支持这些字段：{", ".join(DEFAULT_REWARD)}')
    result = {**base, **overrides}
    for key, value in result.items():
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f'奖励参数 {key} 必须为有限数值。')
        if key not in ('win', 'loss', 'draw', 'timeout') and value < 0:
            raise ValueError(f'奖励参数 {key} 必须非负。')
    if result['win'] <= max(result['loss'], result['draw'], result['timeout']):
        raise ValueError('奖励 win 必须大于 loss、draw 和 timeout。')
    if result['center_damage'] > 1:
        raise ValueError('center_damage 必须在 [0, 1] 内。')
    return result


def load_config(path, *, target, timesteps, threshold, disabled=False, previous=None):
    config = json.loads(path.read_text(encoding='utf-8')) if path else (previous or {})
    if not isinstance(config, dict) or config.keys() - {'reward', 'stages'}:
        raise ValueError('训练配置只支持 reward 和 stages。')
    base_reward = reward_config(config.get('reward', {}))
    if disabled:
        definitions = [{'difficulty': target}]
    elif path and 'stages' in config:
        definitions = config['stages']
    elif previous and 'stages' in config and config['stages'][-1]['difficulty'] == target:
        definitions = config['stages']
    else:
        definitions = []
        for difficulty, fraction in enumerate((0.15, 0.25)[:target]):
            budget = max(1, round(timesteps * fraction))
            definitions.append(dict(difficulty=difficulty, min_steps=budget // 3, max_steps=budget,
                                    promote_win_rate=threshold))
        definitions.append({'difficulty': target})
    if not isinstance(definitions, list) or not definitions:
        raise ValueError('stages 必须是非空列表。')
    stages = []
    fields = {'difficulty', 'min_steps', 'max_steps', 'promote_win_rate', 'reward'}
    for index, stage in enumerate(definitions):
        if not isinstance(stage, dict) or stage.keys() - fields or 'difficulty' not in stage:
            raise ValueError(f'课程第 {index + 1} 阶段配置无效。')
        difficulty = resolve_diff(stage['difficulty'])
        if difficulty is None or (stages and difficulty <= stages[-1]['difficulty']):
            raise ValueError('课程难度必须严格递增。')
        minimum = stage.get('min_steps', 0)
        maximum = stage.get('max_steps')
        rate = stage.get('promote_win_rate', threshold)
        last = index == len(definitions) - 1
        if (type(minimum) is not int or minimum < 0
                or (not last and (type(maximum) is not int or maximum < max(1, minimum)))
                or (last and maximum is not None)
                or type(rate) not in (int, float) or not 0 < rate <= 1):
            raise ValueError('中间阶段需要 0 <= min_steps <= max_steps，且 max_steps > 0。'
                             '最终阶段不设 max_steps，晋级胜率须在 (0, 1] 内。')
        stages.append(dict(difficulty=difficulty, min_steps=minimum, max_steps=maximum,
                           promote_win_rate=rate, reward=reward_config(stage.get('reward', {}), base_reward)))
    if stages[-1]['difficulty'] != target:
        raise ValueError('课程最终阶段的难度必须与 --diff 一致。')
    return {'reward': base_reward, 'stages': stages}
