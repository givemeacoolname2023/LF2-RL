"""Vectorized LF2 simulation on a pool of headless Node.js engine workers."""

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import subprocess

from gymnasium import spaces
import numpy as np
from stable_baselines3.common.vec_env import VecEnv

from action_selector import ActionSelector, action_mode
from env import ACTION_KEYS, OBS_OFFSET, OBS_SCALE, POLICY_CHARACTERS
from rl_metrics import MIN_CENTER_DAMAGE_RATIO

DEFAULT_REWARD = dict(win=1.0, loss=-1.0, draw=-1.0, timeout=-1.25, health_margin=0.25,
                      health_potential=0.5, distance_potential=0.08, depth_weight=4.0,
                      win_speed=0.10, frame_cost=0.0, distance_cost=0.0, center_damage=0.0)
TEAM_HP = (3, 10)
DAMAGE_INDEX, CENTER_DAMAGE_INDEX = 15, 16
FULL_HP = 500
OUTCOMES = ('max_steps', 'lose', 'win', 'draw')


def observations(states, *, opponent=False):
    first, second = (8, 1) if opponent else (1, 8)
    delta_x = states[:, first] - states[:, second]
    first_z, second_z = states[:, first + 6], states[:, second + 6]
    raw = np.column_stack((states[:, first:first + 6], states[:, second:second + 6], delta_x,
                           first_z, second_z, first_z - second_z)).astype(np.float32)
    return np.clip(raw / OBS_SCALE - OBS_OFFSET, -2.0, 2.0)


def measures(states, depth_weight):
    health = np.maximum(states[:, TEAM_HP], 0)
    lead = np.clip((health[:, 0] - health[:, 1]) / FULL_HP, -1, 1)
    distance = np.minimum(np.hypot(states[:, 1] - states[:, 8],
                                   depth_weight * (states[:, 7] - states[:, 14])) / 800, 1)
    return lead, distance


class EngineWorker:
    def __init__(self, node, assets):
        self.process = subprocess.Popen(
            [node, str(Path(__file__).with_name('simulator.cjs')), str(assets)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, start_new_session=True,
        )

    def run(self, request):
        try:
            self.process.stdin.write(json.dumps(request, separators=(',', ':')).encode() + b'\n')
            self.process.stdin.flush()
            reply = self.process.stdout.readline()
        except OSError as exc:
            raise RuntimeError('LF2 模拟进程通信失败。') from exc
        if not reply:
            raise RuntimeError(f'LF2 模拟进程提前退出（退出码 {self.process.poll()}）。')
        result = json.loads(reply)
        if isinstance(result, dict):
            raise RuntimeError(result['error'])
        return result

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.process.stdin.close()
        self.process.stdout.close()


class ParallelLF2Env(VecEnv):
    render_mode = None

    def __init__(self, assets, *, n_envs, n_workers, bg, diff, action_repeat, max_frames, gamma, reward,
                 action_count=len(ACTION_KEYS), character='davis', opponent_model=None):
        assets = Path(assets).expanduser().resolve()
        if not (assets / 'manifest.js').is_file():
            raise FileNotFoundError(f'找不到游戏素材 {assets / "manifest.js"}。请设置 --assets。')
        node = shutil.which('node')
        if not node:
            raise RuntimeError('训练需要 Node.js，请安装 Node.js 并将 node 加入 PATH。')
        self.bg, self.diff = bg, diff
        self.character = character
        self.opponent = getattr(opponent_model, 'lf2_config', {}).get('character', 'davis') if opponent_model else None
        if character not in POLICY_CHARACTERS or (self.opponent and self.opponent not in POLICY_CHARACTERS):
            raise ValueError('未知的 PPO 角色。')
        self.opponent_selector = (ActionSelector(opponent_model, action_mode(opponent_model), n_envs)
                                  if opponent_model else None)
        self.action_repeat, self.max_frames, self.gamma = action_repeat, max_frames, gamma
        self.reward = reward
        self._weights = {key: np.empty(n_envs) for key in DEFAULT_REWARD}
        self._sequences = np.arange(n_envs, dtype=np.int64)
        self._active_diff = np.full(n_envs, diff, dtype=int)
        self._returns = np.zeros(n_envs)
        self._lengths = np.zeros(n_envs, dtype=int)
        self._state = None
        self._pending = None
        self._restart = False
        self._workers = []
        self._pool = None
        super().__init__(n_envs, spaces.Box(-2.0, 2.0, (len(OBS_SCALE),), np.float32),
                         spaces.Discrete(action_count))
        offsets = np.linspace(0, n_envs, min(n_envs, n_workers), endpoint=False, dtype=int).tolist()
        self._shards = [slice(start, end) for start, end in zip(offsets, offsets[1:] + [n_envs])]
        try:
            for _ in self._shards:
                self._workers.append(EngineWorker(node, assets))
            self._pool = ThreadPoolExecutor(max_workers=len(self._workers))
        except BaseException:
            self.close()
            raise

    def _submit(self, actions=None):
        futures = []
        opponent_actions = None
        if actions is not None and self.opponent_selector:
            opponent_actions = self.opponent_selector.predict(observations(self._state, opponent=True), self.action_repeat)
        for shard, worker in zip(self._shards, self._workers):
            request = {'seeds': (self._sequences[shard] % (2 ** 32)).tolist(),
                       'difficulty': self.diff, 'background': self.bg,
                       'character': self.character, 'opponent': self.opponent}
            if actions is not None:
                request.update(actions=actions[shard].tolist(), repeat=self.action_repeat,
                               limit=self.max_frames, restart=self._restart)
                if opponent_actions is not None:
                    request['opponent_actions'] = opponent_actions[shard].tolist()
            futures.append(self._pool.submit(worker.run, request))
        return futures

    def _mark_reset(self, indices):
        for index in indices:
            self.reset_infos[index] = {'seed': int(self._sequences[index] % (2 ** 32)),
                                       'difficulty': self.diff, 'bg': self.bg}
        self._sequences[indices] += self.num_envs
        self._active_diff[indices] = self.diff
        for key, default in DEFAULT_REWARD.items():
            self._weights[key][indices] = self.reward.get(key, default)

    def set_stage(self, diff, reward, *, restart=False):
        self.diff, self.reward = diff, reward
        self._restart = restart

    def reset(self):
        if self._pending is not None:
            for future in self._pending:
                future.result()
            self._pending = None
        self._restart = False
        if self.opponent_selector:
            self.opponent_selector.reset()
        if self._seeds[0] is not None:
            self._sequences[:] = self._seeds
        self._state = np.concatenate([future.result() for future in self._submit()], dtype=np.float64)
        self._mark_reset(np.arange(self.num_envs))
        self._returns.fill(0)
        self._lengths.fill(0)
        self._reset_seeds()
        self._reset_options()
        return observations(self._state)

    def step_async(self, actions):
        self._pending = self._submit(np.asarray(actions).reshape(self.num_envs))

    def step_wait(self):
        replies = [future.result() for future in self._pending]
        self._pending = None
        self._restart = False
        next_state = np.concatenate([rows for rows, _ in replies], dtype=np.float64)
        done_indices, final_states = [], []
        for shard, (_, finals) in zip(self._shards, replies):
            for index, state in finals:
                done_indices.append(shard.start + index)
                final_states.append(state)
        finished = np.asarray(done_indices, dtype=int)
        dones = np.zeros(self.num_envs, dtype=bool)
        dones[finished] = True
        current = next_state.copy()
        if final_states:
            current[finished] = final_states
        terminal_dones = dones & (np.any(current[:, TEAM_HP] <= 0, axis=1) | (current[:, 0] >= self.max_frames))
        terminated = np.flatnonzero(terminal_dones)
        weights = self._weights
        previous_lead, previous_dist = measures(self._state, weights['depth_weight'])
        lead, dist = measures(current, weights['depth_weight'])
        frames = current[:, 0] - self._state[:, 0]
        discount = self.gamma * ~terminal_dones
        center_mix = weights['center_damage']
        health_shaping = discount * lead - previous_lead
        center_damage = current[:, CENTER_DAMAGE_INDEX] - self._state[:, CENTER_DAMAGE_INDEX]
        hp_change = np.maximum(0, current[:, 3]) - np.maximum(0, self._state[:, 3])
        health_reward = (1 - center_mix) * health_shaping + center_mix * (center_damage + hp_change) / FULL_HP
        rewards = (weights['health_potential'] * health_reward
                   + weights['distance_potential'] * (previous_dist - discount * dist)
                   - frames * (weights['frame_cost'] + weights['distance_cost'] * dist))
        codes = (current[terminated, 3] <= 0).astype(int) + 2 * (current[terminated, 10] <= 0)
        for code, key in ((0, 'timeout'), (1, 'loss'), (3, 'draw')):
            indices = terminated[codes == code]
            rewards[indices] += weights[key][indices] + weights['health_margin'][indices] * lead[indices]
        wins = terminated[codes == 2]
        damage = current[:, DAMAGE_INDEX]
        center_ratio = current[:, CENTER_DAMAGE_INDEX] / np.where(damage > 0, damage, 1)
        win_speed = weights['win_speed'][wins] * (1 - current[wins, 0] / self.max_frames)
        win_bonus = weights['win'][wins] + weights['health_margin'][wins] * lead[wins] + win_speed
        rewards[wins] += (1 - center_mix[wins] * (1 - center_ratio[wins])) * win_bonus
        self._returns += rewards
        self._lengths += 1
        infos = [{} for _ in range(self.num_envs)]
        for index, terminal in zip(finished, observations(current[finished])):
            infos[index] = {'terminal_observation': terminal, 'TimeLimit.truncated': not bool(terminal_dones[index])}
        for index, code in zip(terminated, codes):
            infos[index].update(result=OUTCOMES[code], is_success=bool(code == 2),
                                center_damage_ratio=float(center_ratio[index]),
                                center_win=bool(code == 2 and center_ratio[index] >= MIN_CENTER_DAMAGE_RATIO),
                                game_frames=int(current[index, 0]), difficulty=int(self._active_diff[index]),
                                episode={'r': float(self._returns[index]), 'l': int(self._lengths[index])})
        self._state = next_state
        if finished.size:
            if self.opponent_selector:
                self.opponent_selector.reset(finished)
            self._mark_reset(finished)
            self._returns[finished] = 0
            self._lengths[finished] = 0
        return observations(next_state), rewards.astype(np.float32), dones, infos

    def close(self):
        for worker in self._workers:
            worker.close()
        self._workers.clear()
        if self._pool is not None:
            self._pool.shutdown()
            self._pool = None

    def get_attr(self, attr_name, indices=None):
        return [getattr(self, attr_name) for _ in self._get_indices(indices)]

    def set_attr(self, attr_name, value, indices=None):
        raise NotImplementedError('Use set_stage to change difficulty and rewards on the next reset.')

    def env_method(self, method_name, *args, indices=None, **kwargs):
        raise NotImplementedError('The simulator does not expose per-match Python methods.')

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False for _ in self._get_indices(indices)]


def default_workers(n_envs):
    available = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count() or 1
    return min(available, 16, n_envs)
