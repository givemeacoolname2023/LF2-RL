"""Train PPO with parallel headless LF2 environments and win-rate evaluation."""

import argparse
from contextlib import ExitStack
import csv
import math
import os
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
import torch

from action_selector import ACTION_MODES, ActionSelector, action_mode
from curriculum import load_config
from env import ACTION_KEYS, DIFF_NAMES, POLICY_CHARACTERS, diff_arg, resolve_diff
from parallel_env import ParallelLF2Env, default_workers
from sync_env import DEFAULT_ASSETS, DEFAULT_MAX_FRAMES, DEFAULT_REPEAT, DEFAULT_SEED, SyncLF2Env, game_server


def save_model(model, path):
    temporary = path.with_name(f'.{path.stem}.{os.getpid()}.zip')
    try:
        model.save(temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def append_csv(path, columns, row):
    with path.open('a', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        if stream.tell() == 0:
            writer.writerow(columns)
        writer.writerow(row)


class WinRateCallback(BaseCallback):
    def __init__(self, eval_env, *, every, episodes, seed_base, stages, progress, best_path, evaluate_initial):
        super().__init__()
        self.eval_env = eval_env
        self.every = every
        self.episodes = episodes
        self.seed_base = seed_base
        self.stages = stages
        self.progress = progress
        self.target = stages[-1]['difficulty']
        self.best_path = best_path
        self.evaluate_initial = evaluate_initial
        self.best_wins = -1
        self.last_eval = None

    def _on_training_start(self):
        self.last_eval = self.num_timesteps
        self.log_stage('start')
        if self.evaluate_initial and self.every:
            self.evaluate()

    def _on_step(self):
        return True

    def _on_rollout_start(self):
        index = self.progress['stage']
        stage = self.stages[index]
        used = self.num_timesteps - self.progress['started_at']
        due = self.every > 0 and self.num_timesteps - self.last_eval >= self.every
        reason = None
        rate = None
        if index < len(self.stages) - 1:
            if used >= stage['max_steps']:
                reason = 'max_steps'
            elif due and used >= stage['min_steps']:
                rate = self.score(stage['difficulty']) / self.episodes
                if rate >= stage['promote_win_rate']:
                    reason = 'win_rate'
        if reason:
            self.progress.update(stage=index + 1, started_at=self.num_timesteps)
            next_stage = self.stages[index + 1]
            self.training_env.set_stage(next_stage['difficulty'], next_stage['reward'], restart=True)
            self.log_stage(reason, used, rate)
        self.logger.record('curriculum/stage', self.progress['stage'] + 1)
        self.logger.record('curriculum/difficulty', self.stages[self.progress['stage']]['difficulty'])
        if due:
            self.evaluate()

    def _on_training_end(self):
        if self.every and self.num_timesteps != self.last_eval:
            self.evaluate()
            self.logger.dump(self.num_timesteps)

    def evaluate(self):
        self.model.policy.set_training_mode(False)
        target_wins = self.score(self.target)
        if target_wins >= self.best_wins:
            self.best_wins = target_wins
            save_model(self.model, self.best_path)
        self.last_eval = self.num_timesteps

    def log_stage(self, reason, previous_steps=0, rate=None):
        index = self.progress['stage']
        difficulty = self.stages[index]['difficulty']
        print(f'[课程] {self.num_timesteps} 步 | 阶段 {index + 1}/{len(self.stages)} '
              f'{DIFF_NAMES[difficulty]} | {reason} | 新开对局生效', flush=True)
        append_csv(Path(self.logger.get_dir()) / 'ppo_curriculum.csv',
                   ['total_steps', 'stage', 'difficulty', 'reason', 'previous_stage_steps', 'win_rate'],
                   [self.num_timesteps, index + 1, difficulty, reason, previous_steps, rate])

    def score_parallel(self, difficulty):
        env = self.eval_env
        env.set_stage(difficulty, env.reward)
        env.seed(self.seed_base)
        obs = env.reset()
        pending = np.ones(self.episodes, dtype=bool)
        wins = center_wins = 0
        selector = ActionSelector(self.model, self.model.lf2_action_mode, self.episodes)
        while pending.any():
            actions = selector.predict(obs, env.action_repeat)
            obs, _, done, infos = env.step(actions)
            selector.reset(done)
            for index in np.flatnonzero(pending & done):
                wins += int(infos[index]['is_success'])
                center_wins += int(infos[index]['center_win'])
                pending[index] = False
        return wins, center_wins

    def score_browser(self, difficulty):
        wins = center_wins = 0
        selector = ActionSelector(self.model, self.model.lf2_action_mode)
        for index in range(self.episodes):
            obs, _ = self.eval_env.reset(seed=self.seed_base + index, options={'diff': difficulty})
            selector.reset()
            while True:
                action = selector.predict(obs, self.eval_env.action_repeat)
                obs, _, terminated, truncated, info = self.eval_env.step(int(action))
                if terminated or truncated:
                    wins += int(info['result'] == 'win')
                    center_wins += int(info['center_win'])
                    break
        return wins, center_wins

    def score(self, difficulty):
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(self.seed_base)
            wins, center_wins = (self.score_parallel(difficulty) if isinstance(self.eval_env, ParallelLF2Env)
                                 else self.score_browser(difficulty))
        stage = next(stage for stage in self.stages if stage['difficulty'] == difficulty)
        rate = wins / self.episodes
        center_rate = center_wins / self.episodes
        label = f'{self.eval_env.opponent.title()} PPO' if self.eval_env.opponent else DIFF_NAMES[difficulty]
        print(f'[评估] {self.num_timesteps} 步 | {label} | 胜 {wins}/{self.episodes} ({rate:.1%})'
              f' | 场中胜 {center_wins}/{self.episodes} ({center_rate:.1%})',
              flush=True)
        metric = label.lower().replace(' ', '_')
        self.logger.record(f'eval/{metric}_win_rate', rate)
        self.logger.record(f'eval/{metric}_center_win_rate', center_rate)
        append_csv(Path(self.logger.get_dir()) / 'ppo_eval.csv',
                   ['total_steps', 'difficulty', 'wins', 'episodes', 'win_rate', 'seed_base'],
                   [self.num_timesteps, difficulty, wins, self.episodes, rate, self.seed_base])
        return center_wins if stage['reward']['center_damage'] else wins


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='并行训练 PPO，按目标难度的评估胜率选模型')
    parser.add_argument('--timesteps', type=int, default=int(os.getenv('LF2_TIMESTEPS', '1200000')))
    parser.add_argument('--n-envs', type=int, default=4, help='并行环境数（默认 4）')
    parser.add_argument('--n-workers', type=int, help='Node.js 进程数（默认按可用 CPU，最多 16）')
    parser.add_argument('--n-steps', type=int, help='每个环境每轮决策步数（默认随环境数缩放）')
    parser.add_argument('--batch-size', type=int, help='PPO 的小批量大小')
    parser.add_argument('--n-epochs', type=int, help='PPO 每轮数据的更新遍数')
    parser.add_argument('--diff', type=diff_arg, help='目标难度（新训练默认 difficult）')
    parser.add_argument('--no-curriculum', action='store_true', help='直接在目标难度训练')
    parser.add_argument('--promote-win-rate', type=float, default=0.7)
    parser.add_argument('--config', type=Path, help='课程和奖励 JSON 配置')
    parser.add_argument('--action-mode', choices=ACTION_MODES, help='选模和对战出招方式（新训练默认 reengage）')
    parser.add_argument('--eval-every', type=int, default=int(os.getenv('LF2_BEST_EVERY', '32768')),
                        help='评估间隔（所有环境合计决策步），0 只按阶段预算晋级')
    parser.add_argument('--eval-episodes', type=int, default=int(os.getenv('LF2_BEST_EPISODES', '20')))
    parser.add_argument('--eval-seed-base', type=int, default=10000)
    parser.add_argument('--seed-base', type=int, default=DEFAULT_SEED)
    parser.add_argument('--bg', type=int, choices=range(9), help='场地下标（新训练默认 0）')
    parser.add_argument('--character', choices=POLICY_CHARACTERS, help='PPO 控制的角色（默认 Davis）')
    parser.add_argument('--opponent-model', type=Path, help='固定的对手 PPO 存档')
    parser.add_argument('--action-repeat', type=int, help='每次决策推进的帧数（新训练默认 4）')
    parser.add_argument('--max-frames', type=int, help='每局最多游戏帧数（新训练默认 3000）')
    parser.add_argument('--assets', type=Path, default=DEFAULT_ASSETS, help='LF2_19 素材目录')
    parser.add_argument('--browser', help='使用指定的 Chrome / Chromium 做训练中评估')
    parser.add_argument('--output-dir', type=Path, default=Path('.'))
    parser.add_argument('--resume', help='继续训练的 PPO .zip 文件；--timesteps 是追加步数')
    parser.add_argument('--ent-coef', type=float)
    parser.add_argument('--gamma', type=float, help='PPO 折扣因子（默认 1）')
    args = parser.parse_args(argv)
    if (min(args.n_envs, args.timesteps, args.eval_episodes) < 1 or args.eval_every < 0
            or not 0 < args.promote_win_rate <= 1):
        parser.error('步数、环境数和评估局数必须为正数，升难度阈值必须在 (0, 1] 内。')
    if not (0 <= args.seed_base < 2 ** 32 and 0 <= args.eval_seed_base <= 2 ** 32 - args.eval_episodes):
        parser.error('随机种子必须在无符号 32 位范围内。')
    if any(value is not None and value < 1 for value in (args.n_workers, args.n_steps, args.n_epochs)):
        parser.error('进程数、rollout 长度和更新遍数必须为正数。')
    if args.batch_size is not None and args.batch_size < 2:
        parser.error('PPO 小批量大小必须至少为 2。')
    if args.gamma is not None and not 0 < args.gamma <= 1:
        parser.error('--gamma 必须在 (0, 1] 内。')
    if args.ent_coef is not None and (not math.isfinite(args.ent_coef) or args.ent_coef < 0):
        parser.error('--ent-coef 必须是非负有限数。')
    return args


def main(args):
    torch.set_num_threads(1)
    previous = PPO.load(args.resume, device='cpu') if args.resume else None
    opponent_model = PPO.load(args.opponent_model, device='cpu') if args.opponent_model else None
    old_config = getattr(previous, 'lf2_config', {})
    config = {
        'character': args.character or old_config.get('character', 'davis'),
        'action_count': int(previous.action_space.n) if previous else len(ACTION_KEYS),
        'bg': args.bg if args.bg is not None else old_config.get('bg', 0),
        'action_repeat': (args.action_repeat if args.action_repeat is not None
                          else old_config.get('action_repeat', 1 if previous else DEFAULT_REPEAT)),
        'max_frames': (args.max_frames if args.max_frames is not None
                       else old_config.get('max_frames', DEFAULT_MAX_FRAMES)),
    }
    if not 1 <= config['action_repeat'] <= config['max_frames']:
        raise SystemExit('要求 1 <= --action-repeat <= --max-frames。')
    target = resolve_diff(args.diff) if args.diff is not None else old_config.get('diff', 2)
    old_curriculum = getattr(previous, 'lf2_curriculum', None)
    try:
        if old_curriculum is not None:
            for entry in [old_curriculum, *old_curriculum['stages']]:
                entry.get('reward', {}).pop('opponent_damage', None)
            old_curriculum = load_config(None, target=old_config.get('diff', 2), timesteps=args.timesteps,
                                         threshold=args.promote_win_rate, previous=old_curriculum)
        curriculum = load_config(args.config, target=target, timesteps=args.timesteps,
                                 threshold=args.promote_win_rate, disabled=args.no_curriculum or opponent_model is not None,
                                 previous=old_curriculum)
    except (ValueError, OSError) as exc:
        raise SystemExit(f'课程或奖励配置无效。{exc}') from exc
    stages = curriculum['stages']
    prior_steps = previous.num_timesteps if previous else 0
    progress = (dict(previous.lf2_progress) if curriculum == old_curriculum
                else {'stage': 0, 'started_at': prior_steps})
    stage = stages[progress['stage']]
    gamma = args.gamma if args.gamma is not None else (previous.gamma if previous else 1.0)
    n_steps = args.n_steps or (previous.n_steps if previous and previous.n_envs == args.n_envs
                             else max(8, min(512, 8192 // args.n_envs)))
    if n_steps < 1 or n_steps * args.n_envs < 2:
        raise SystemExit('PPO 的每轮合计决策步数必须至少为 2。')
    rollout_steps = n_steps * args.n_envs
    train_seed = args.seed_base + prior_steps
    last_seed = train_seed + math.ceil(args.timesteps / rollout_steps) * rollout_steps + args.n_envs - 1
    if last_seed >= 2 ** 32:
        raise SystemExit('训练种子可能越过 32 位整数范围，请降低 --seed-base。')
    if args.eval_every and train_seed < args.eval_seed_base + args.eval_episodes and last_seed >= args.eval_seed_base:
        raise SystemExit('训练和评估的种子范围可能重叠，请调整 --seed-base 或 --eval-seed-base。')
    default_batch = 256 if rollout_steps <= 2048 else 1024
    batch_size = args.batch_size or min(rollout_steps, default_batch)
    if batch_size > rollout_steps:
        raise SystemExit('--batch-size 不能超过 --n-steps × --n-envs。')
    workers = min(args.n_envs, args.n_workers or default_workers(args.n_envs))
    checkpoints = args.output_dir / 'checkpoints'
    checkpoints.mkdir(parents=True, exist_ok=True)
    logs = str(args.output_dir / 'logs')

    with ExitStack() as resources:
        venv = ParallelLF2Env(args.assets, n_envs=args.n_envs, n_workers=workers,
                              diff=stage['difficulty'], gamma=gamma, reward=stage['reward'],
                              opponent_model=opponent_model, **config)
        resources.callback(venv.close)
        epochs = args.n_epochs or (previous.n_epochs if previous else 5)
        if (previous and previous.n_envs == args.n_envs and previous.n_steps == n_steps
                and previous.gamma == gamma):
            model = previous
            model.set_env(venv)
            model.tensorboard_log = logs
            model.batch_size = args.batch_size or min(previous.batch_size, rollout_steps)
            model.n_epochs = epochs
        elif previous:
            model = PPO.load(args.resume, env=venv, device='cpu', tensorboard_log=logs,
                             n_steps=n_steps, batch_size=batch_size, n_epochs=epochs, gamma=gamma)
        else:
            model = PPO(
                'MlpPolicy', venv, n_steps=n_steps, batch_size=batch_size, n_epochs=epochs,
                learning_rate=3e-4, gamma=gamma, gae_lambda=0.98, ent_coef=0.01,
                seed=args.seed_base, device='cpu', verbose=1, tensorboard_log=logs,
            )
        venv.seed(train_seed)
        model.ep_info_buffer = model.ep_success_buffer = None
        model.lf2_config = {**config, 'diff': target}
        model.lf2_action_mode = action_mode(previous, args.action_mode, default='reengage')
        model.__dict__.pop('lf2_reward_model', None)
        model.lf2_curriculum = curriculum
        model.lf2_progress = progress
        if args.ent_coef is not None:
            model.ent_coef = args.ent_coef

        callbacks = []
        eval_env = None
        if args.eval_every:
            if args.browser:
                url = resources.enter_context(game_server(args.assets))
                eval_env = SyncLF2Env(url, diff=target, gamma=gamma, browser=args.browser,
                                     opponent_model=opponent_model, **config)
            else:
                eval_env = ParallelLF2Env(args.assets, n_envs=args.eval_episodes,
                                          n_workers=min(4, workers), diff=target, gamma=gamma,
                                          reward=curriculum['reward'], opponent_model=opponent_model, **config)
            resources.callback(eval_env.close)
        callbacks.append(WinRateCallback(
            eval_env, every=args.eval_every, episodes=args.eval_episodes, seed_base=args.eval_seed_base,
            stages=stages, progress=progress, best_path=checkpoints / 'best_model.zip', evaluate_initial=bool(previous),
        ))
        ckpt_freq = int(os.getenv('LF2_CKPT_FREQ', '20000'))
        if ckpt_freq > 0:
            callbacks.append(CheckpointCallback(math.ceil(ckpt_freq / args.n_envs), str(checkpoints), 'ppo'))
        label = f'{venv.opponent.title()} PPO' if venv.opponent else f'Dennis {DIFF_NAMES[target]}'
        print(f'[PPO] {config["character"].title()} vs {label} | {workers} 进程 | '
              f'{args.n_envs} 环境 × {n_steps} 步/轮 | 动作重复 {config["action_repeat"]} 帧 | '
              f'{model.lf2_action_mode}',
              flush=True)
        try:
            model.learn(total_timesteps=args.timesteps, callback=callbacks, reset_num_timesteps=not args.resume)
        except KeyboardInterrupt:
            save_model(model, checkpoints / 'interrupted.zip')
            print('[中断] 已保存 interrupted.zip')
            return
        save_model(model, args.output_dir / 'lf2_ppo.zip')
        print(f'[完成] 模型已保存到 {args.output_dir / "lf2_ppo.zip"}')


if __name__ == '__main__':
    main(parse_args())
