"""Select PPO actions with an optional recovery from prolonged disengagement."""

import numpy as np

from env import OBS_SCALE

ACTION_MODES = ('reengage', 'greedy', 'sample')
REENGAGE_FRAMES = 180
MAX_ENGAGEMENT_DX = 60
MAX_ENGAGEMENT_DZ = 10
DX_INDEX = 12
DZ_INDEX = 15


def action_mode(model, override=None, *, default='greedy'):
    return override or getattr(model, 'lf2_action_mode', default)


class ActionSelector:
    def __init__(self, model, mode, n_envs=1):
        self.model = model
        self.mode = mode
        self.far_frames = np.zeros(n_envs, dtype=np.int64)

    def reset(self, done=None):
        if done is None:
            self.far_frames.fill(0)
        else:
            self.far_frames[done] = 0

    def predict(self, obs, frames):
        actions, _ = self.model.predict(obs, deterministic=self.mode != 'sample')
        if self.mode != 'reengage':
            return actions

        states = np.atleast_2d(obs)
        dx = states[:, DX_INDEX] * OBS_SCALE[DX_INDEX]
        dz = states[:, DZ_INDEX] * OBS_SCALE[DZ_INDEX]
        horizontal = np.abs(dx) > MAX_ENGAGEMENT_DX
        depth = np.abs(dz) > MAX_ENGAGEMENT_DZ
        self.far_frames += frames
        self.far_frames[~(horizontal | depth)] = 0
        recover = self.far_frames >= REENGAGE_FRAMES
        if not recover.any():
            return actions

        move_x = np.where(dx[recover] > 0, 1, 2)
        move_z = np.where(dz[recover] > 0, 11, 4)
        diagonal = 12 + 2 * (dz[recover] > 0) + (dx[recover] < 0)
        recovery = np.where(horizontal[recover], np.where(depth[recover], diagonal, move_x), move_z)
        actions = np.asarray(actions).copy()
        actions.reshape(-1)[recover] = recovery
        return actions
