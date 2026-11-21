"""Evaluate a PPO checkpoint on reproducible LF2 matches."""

import argparse
from collections import Counter
from contextlib import ExitStack
import json
from pathlib import Path
import re
import time

from stable_baselines3 import PPO
import torch

from action_selector import ACTION_MODES, ActionSelector, action_mode
from env import DIFF_NAMES, OBS_SCALE, POLICY_CHARACTERS, diff_arg, resolve_diff
from rl_metrics import (IDX_D_HP, IDX_E_HP, action_name,
                        depth_aligned, dist_px, dz_px, is_attack, wilson)
from sync_env import DEFAULT_ASSETS, DEFAULT_MAX_FRAMES, DEFAULT_SEED, SyncLF2Env, game_server


def resolve_model(path):
    path = Path(path)
    if path.suffix != '.zip':
        path = Path(f'{path}.zip')
    return str(path) if path.is_file() else None


def latest_checkpoint(folder='checkpoints'):
    folder = Path(folder)
    preferred = resolve_model(folder / 'best_model.zip')
    if preferred:
        return preferred
    candidates = []
    for path in folder.glob('ppo_*_steps.zip'):
        match = re.fullmatch(r'ppo_(\d+)_steps', path.stem)
        if match:
            candidates.append((int(match[1]), str(path)))
    return max(candidates)[1] if candidates else resolve_model('lf2_ppo.zip')


def evaluate_episode(env, selector, seed):
    obs, _ = env.reset(seed=seed)
    selector.reset()
    start_hp = obs[[IDX_D_HP, IDX_E_HP]] * OBS_SCALE[[IDX_D_HP, IDX_E_HP]]
    action_counts = Counter()
    steps = attacks = aligned = 0
    distance = depth = 0.0
    while True:
        action = int(selector.predict(obs, env.action_repeat))
        action_counts[action] += 1
        attacks += is_attack(action, env.character)
        obs, _, terminated, truncated, info = env.step(action)
        steps += 1
        distance += dist_px(obs)
        depth += dz_px(obs)
        aligned += depth_aligned(obs)
        if terminated or truncated:
            break
    end_hp = obs[[IDX_D_HP, IDX_E_HP]] * OBS_SCALE[[IDX_D_HP, IDX_E_HP]]
    record = {
        'seed': seed, 'bg': env.observed_bg, 'result': info['result'],
        'center_damage_ratio': info['center_damage_ratio'], 'center_win': info['center_win'],
        'steps': steps, 'frames': info['game_frames'], 'attacks': attacks,
        'mean_dist': distance / steps, 'mean_dz': depth / steps, 'align_rate': aligned / steps,
        'dealt': float(max(0, start_hp[1] - max(0, end_hp[1]))),
        'taken': float(max(0, start_hp[0] - max(0, end_hp[0]))),
    }
    return record, action_counts


def main():
    parser = argparse.ArgumentParser(description='评估 PPO 模型的胜率与行为特征')
    parser.add_argument('--model', help='PPO 模型路径（.zip 可省）')
    parser.add_argument('--character', choices=POLICY_CHARACTERS, help='主队角色（默认沿用存档）')
    parser.add_argument('--opponent-model', help='客队 PPO 模型路径（.zip 可省，默认对战 Dennis AI）')
    parser.add_argument('--algo', choices=('ppo',), help=argparse.SUPPRESS)
    parser.add_argument('--episodes', type=int, default=20)
    parser.add_argument('--device', default='cpu')
    policy_args = parser.add_mutually_exclusive_group()
    policy_args.add_argument('--action-mode', choices=ACTION_MODES, help='出招方式（默认沿用存档）')
    policy_args.add_argument('--stochastic', dest='action_mode', action='store_const', const='sample',
                             help='从策略分布采样动作')
    parser.add_argument('--seed-base', type=int, default=20000, help='种子起点。-1 每局使用同一个默认种子。')
    parser.add_argument('--diff', type=diff_arg)
    parser.add_argument('--bg', type=int, choices=range(9))
    parser.add_argument('--action-repeat', type=int)
    parser.add_argument('--max-frames', type=int)
    parser.add_argument('--assets', type=Path, default=DEFAULT_ASSETS)
    parser.add_argument('--browser')
    parser.add_argument('--json', type=Path, metavar='PATH')
    args = parser.parse_args()
    if args.episodes < 1 or not -1 <= args.seed_base <= 2 ** 32 - args.episodes:
        parser.error('局数须为正数，种子范围须在无符号 32 位整数内。')
    model_path = resolve_model(args.model) if args.model else latest_checkpoint()
    if model_path is None:
        parser.error(f'找不到 PPO .zip 模型（{args.model or "checkpoints/"}）。')
    opponent_path = resolve_model(args.opponent_model) if args.opponent_model else None
    if args.opponent_model and opponent_path is None:
        parser.error(f'找不到对手 PPO .zip 模型（{args.opponent_model}）。')

    torch.set_num_threads(1)
    model = PPO.load(model_path, device=args.device)
    opponent_model = PPO.load(opponent_path, device=args.device) if opponent_path else None
    mode = action_mode(model, args.action_mode)
    selector = ActionSelector(model, mode)
    config = getattr(model, 'lf2_config', {})
    character = args.character or config.get('character', 'davis')
    bg = args.bg if args.bg is not None else config.get('bg', 0)
    repeat = args.action_repeat if args.action_repeat is not None else config.get('action_repeat', 1)
    max_frames = args.max_frames if args.max_frames is not None else config.get('max_frames', DEFAULT_MAX_FRAMES)
    difficulty = resolve_diff(args.diff) if args.diff is not None else config.get('diff', 0)
    episodes = []
    action_counts = Counter()
    started = time.monotonic()
    with ExitStack() as resources:
        url = resources.enter_context(game_server(args.assets))
        env = SyncLF2Env(url, diff=difficulty, bg=bg, action_repeat=repeat,
                         max_frames=max_frames, gamma=model.gamma, browser=args.browser,
                         action_count=int(model.action_space.n), character=character,
                         opponent_model=opponent_model)
        resources.callback(env.close)
        opponent_mode = env.opponent_selector.mode if env.opponent_selector else None
        opponent_label = f'{env.opponent.title()} PPO ({opponent_mode})' if env.opponent else (
            f'Dennis {DIFF_NAMES[difficulty]}')
        print(f'[评估] {character.title()} PPO ({mode}) vs {opponent_label} | '
              f'{model_path} | bg={bg} | 每步 {repeat} 帧')
        try:
            for index in range(args.episodes):
                seed = args.seed_base + index if args.seed_base >= 0 else DEFAULT_SEED
                if 'sample' in (mode, opponent_mode) and args.seed_base >= 0:
                    torch.manual_seed(seed)
                record, counts = evaluate_episode(env, selector, seed)
                episodes.append(record)
                action_counts.update(counts)
                print(f'[{index + 1}/{args.episodes}] seed={seed} {record["result"]} '
                      f'{record["frames"]} 帧 | 打掉 {record["dealt"]:.0f} / 挨打 {record["taken"]:.0f} hp',
                      flush=True)
        except KeyboardInterrupt:
            print('[中断] 汇总已完成的对局。')
    if not episodes:
        return 1

    results = Counter(record['result'] for record in episodes)
    count = len(episodes)
    wins, losses = results['win'], results['lose']
    center_wins = sum(record['center_win'] for record in episodes)
    lower, upper = wilson(wins, count)
    mean_frames = sum(record['frames'] for record in episodes) / count
    mean_dist = sum(record['mean_dist'] for record in episodes) / count
    mean_dz = sum(record['mean_dz'] for record in episodes) / count
    align_rate = sum(record['align_rate'] for record in episodes) / count
    attack_rate = sum(record['attacks'] for record in episodes) / action_counts.total()
    print(f'胜 {wins} / 负 {losses} / 平 {results["draw"]} / 超时 {results["max_steps"]}')
    print(f'胜率 {wins / count:.1%}（95% Wilson 区间 {lower:.1%}–{upper:.1%}）')
    print(f'场中胜 {center_wins}/{count} ({center_wins / count:.1%})')
    print(f'平均 {mean_frames:.0f} 帧 | 水平距离 {mean_dist:.1f}px | 纵深 {mean_dz:.1f}px | '
          f'纵深对齐 {align_rate:.1%} | 出招 {attack_rate:.1%}')
    if args.json:
        action_space_size = int(model.action_space.n)
        summary = {
            'model': model_path, 'algo': 'ppo', 'action_selection': 'argmax' if mode == 'greedy' else mode,
            'character': character, 'opponent_character': env.opponent or 'dennis',
            'opponent_model': opponent_path, 'opponent_action_selection': opponent_mode,
            'n': count, 'wins': wins, 'loses': losses, 'draws': results['draw'],
            'timeouts': results['max_steps'], 'frame_timeouts': 0, 'other': results['draw'],
            'win_rate': wins / count, 'win_rate_ci': [lower, upper],
            'center_wins': center_wins, 'center_win_rate': center_wins / count,
            'win_rate_decisive': wins / (wins + losses) if wins + losses else None,
            'mean_len': mean_frames, 'mean_dist': mean_dist, 'mean_dz': mean_dz,
            'align_rate': align_rate, 'alignment_zwidth': 12.0,
            'attack_rate': attack_rate,
            'random_attack_rate': sum(is_attack(action, character) for action in range(action_space_size))
                                  / action_space_size,
            'mean_dealt': sum(record['dealt'] for record in episodes) / count,
            'mean_taken': sum(record['taken'] for record in episodes) / count,
            'action_counts': {action_name(action, character): count for action, count in action_counts.most_common()},
            'checkpoint_meta': {'algo': 'ppo', 'num_timesteps': model.num_timesteps},
            'difficulty_requested': args.diff, 'difficulty_index': difficulty,
            'difficulty_observed': env.observed_difficulty, 'ai_observed': env.observed_ai,
            'bg_requested': args.bg, 'bg_observed': env.observed_bg,
            'bgs': [record['bg'] for record in episodes],
            'action_repeat': repeat, 'max_frames': max_frames,
            'seed_base': None if args.seed_base < 0 else args.seed_base,
            'seeds': [record['seed'] for record in episodes],
            'elapsed_secs': round(time.monotonic() - started, 1), 'episodes': episodes,
        }
        args.json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
