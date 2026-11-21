from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
import torch

from parallel_env import DEFAULT_REWARD, EngineWorker, ParallelLF2Env, observations


ASSETS = Path('/tmp/lf2-rl-assets')
RUN = Path(__file__).resolve().parent
ARENA_CENTER = 397
SHARD_SIZE = 64


def collect(shard):
    seeds = list(range(82000 + SHARD_SIZE * shard, 82000 + SHARD_SIZE * (shard + 1)))
    worker = EngineWorker('node', ASSETS)
    request = dict(seeds=seeds, difficulty=2, background=0, repeat=4, limit=2400)
    active = np.ones(SHARD_SIZE, dtype=bool)
    features, labels, finished_episodes = [], [], []
    depth_limit = (8, 10, 12, 15)[shard % 4]
    edge_limit = (180, 210, 225, 245)[shard // 4 % 4]
    try:
        states = np.asarray(worker.run(request))
        while active.any():
            delta_depth = states[:, 14] - states[:, 7]
            distance = abs(states[:, 8] - states[:, 1])
            aligned = abs(delta_depth) <= depth_limit
            have_mana = states[:, 4] >= 40
            spacing = np.where(delta_depth < 0, 11, 4)
            fighting = np.where(have_mana, 16, np.where(distance < 70, 21, 6))
            off_center = abs((states[:, 1] + states[:, 8]) / 2 - ARENA_CENTER) > edge_limit
            recenter = np.where(states[:, 1] < ARENA_CENTER, 2, 1)
            actions = np.where(off_center, recenter, np.where(aligned, fighting, spacing))
            features.append(observations(states))
            labels.append(actions)
            request['actions'] = actions.tolist()
            updated, finals = worker.run(request)
            states = np.asarray(updated)
            for index, final in finals:
                if not active[index]:
                    continue
                active[index] = False
                hp_ratio = final[16] / (final[15] or 1)
                win = final[3] > 0 and final[10] <= 0
                steps = len(features)
                if hp_ratio >= 0.70 and (win or final[16] >= 280):
                    finished_episodes.append((index, steps, win))
    finally:
        worker.close()
    episode_features = np.asarray(features).swapaxes(0, 1)
    episode_labels = np.asarray(labels).T
    kept = np.concatenate([episode_features[index, :steps] for index, steps, _ in finished_episodes])
    target = np.concatenate([episode_labels[index, :steps] for index, steps, _ in finished_episodes])
    print(f'shard={shard} center_wins={sum(win for _, _, win in finished_episodes)} rows={len(kept)}', flush=True)
    return kept, target


def main():
    torch.set_num_threads(1)
    torch.manual_seed(82000)
    with ThreadPoolExecutor(max_workers=12) as pool:
        demonstrations = list(pool.map(collect, range(20)))
    observations_batch, labels = (torch.as_tensor(np.concatenate(parts)) for parts in zip(*demonstrations))
    env = ParallelLF2Env(ASSETS, n_envs=512, n_workers=12, bg=0, diff=2, action_repeat=4,
                         max_frames=2400, gamma=0.997, reward=DEFAULT_REWARD)
    try:
        model = PPO('MlpPolicy', env, n_steps=64, batch_size=2048, n_epochs=4,
                    learning_rate=0.00015, gamma=0.997, gae_lambda=0.98, ent_coef=0.003,
                    policy_kwargs={'net_arch': [128, 128]}, device='cpu', seed=82000, verbose=1)
        optimizer = torch.optim.Adam(model.policy.parameters(), lr=0.001)
        for epoch in range(12):
            losses = []
            for indices in torch.randperm(len(labels)).split(1024):
                action_distribution = model.policy.get_distribution(observations_batch[indices])
                loss = -action_distribution.log_prob(labels[indices]).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach()))
            print(f'epoch={epoch + 1} demonstration_loss={np.mean(losses):.3f}', flush=True)
        model.lf2_config = {'bg': 0, 'diff': 2, 'action_repeat': 4, 'max_frames': 2400,
                            'action_count': int(env.action_space.n)}
        model.lf2_action_mode = 'greedy'
        model.save(RUN / 'warm_start.zip')
    finally:
        env.close()


if __name__ == '__main__':
    main()
