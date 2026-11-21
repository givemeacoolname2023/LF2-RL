"""Synchronous LF2 simulation with a private headless browser per environment."""

from contextlib import contextmanager
from functools import partial
from http.server import ThreadingHTTPServer
import math
import os
from pathlib import Path
import shutil
from threading import Thread

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from playwright.sync_api import TimeoutError as BrowserTimeout, sync_playwright

from action_selector import ActionSelector, action_mode
from env import ACTION_KEYS, OBS_SCALE, POLICY_CHARACTERS, resolve_diff, state_to_obs
from rl_metrics import MIN_CENTER_DAMAGE_RATIO
from serve import NoCacheHandler

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ASSETS = ROOT.parent / 'LF2_19'
DEFAULT_MAX_FRAMES = int(os.environ.get('LF2_MAX_STEPS', '3000'))
DEFAULT_REPEAT = 4
DEFAULT_SEED = 824163532


class GameHandler(NoCacheHandler):
    def __init__(self, *args, assets, **kwargs):
        self.assets = assets
        super().__init__(*args, directory=ROOT, **kwargs)

    def translate_path(self, path):
        relative = Path(super().translate_path(path)).relative_to(ROOT)
        mount, *parts = relative.parts or ('',)
        if mount == 'LF2_19':
            return str(self.assets.joinpath(*parts))
        if mount == 'LF2-RL':
            return str(ROOT.joinpath(*parts))
        return str(ROOT / 'game' / 'game.html')


@contextmanager
def game_server(assets=DEFAULT_ASSETS):
    assets = Path(assets).expanduser().resolve()
    if not (assets / 'manifest.js').is_file():
        raise FileNotFoundError(f'找不到游戏素材 {assets / "manifest.js"}。请设置 --assets。')
    with ThreadingHTTPServer(('127.0.0.1', 0), partial(GameHandler, assets=assets)) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f'http://127.0.0.1:{server.server_port}/LF2-RL/game/game.html?rlstep&headless'
        finally:
            server.shutdown()
            thread.join()


def health_lead(state):
    davis, dennis = state['davis'], state['dennis']
    lead = (max(0, davis['hp']) - max(0, dennis['hp'])) / 500
    return max(-1.0, min(lead, 1.0))


def reward_potential(state):
    davis, dennis = state['davis'], state['dennis']
    distance = math.hypot(davis['x'] - dennis['x'], 4 * (davis['z'] - dennis['z']))
    return 0.5 * health_lead(state) - 0.08 * min(distance / 800, 1.0)


def score_transition(previous, current, max_frames, gamma):
    davis_dead = current['davis']['hp'] <= 0
    dennis_dead = current['dennis']['hp'] <= 0
    result = None
    if davis_dead and dennis_dead:
        result = 'draw'
    elif dennis_dead:
        result = 'win'
    elif davis_dead:
        result = 'lose'
    elif current['time'] >= max_frames:
        result = 'max_steps'

    if result is not None:
        outcome = 1.0 if result == 'win' else -1.0
        return outcome + 0.25 * health_lead(current) - reward_potential(previous), result
    return gamma * reward_potential(current) - reward_potential(previous), None


class SyncLF2Env(gym.Env):
    def __init__(self, url, *, diff=0, bg=0, seed_base=DEFAULT_SEED, seed_stride=1,
                 action_repeat=DEFAULT_REPEAT, max_frames=DEFAULT_MAX_FRAMES, gamma=1.0,
                 browser=None, action_count=len(ACTION_KEYS), character='davis', opponent_model=None):
        if not 0 <= bg <= 8 or not 1 <= action_repeat <= max_frames:
            raise ValueError('bg 必须为 0–8，且 1 <= action_repeat <= max_frames。')
        self.url = url
        self.character = character
        self.opponent = getattr(opponent_model, 'lf2_config', {}).get('character', 'davis') if opponent_model else None
        if character not in POLICY_CHARACTERS or (self.opponent and self.opponent not in POLICY_CHARACTERS):
            raise ValueError('未知的 PPO 角色。')
        self.opponent_selector = ActionSelector(opponent_model, action_mode(opponent_model)) if opponent_model else None
        self.diff = resolve_diff(diff)
        self.bg = bg
        self.action_repeat = action_repeat
        self.max_frames = max_frames
        self.gamma = gamma
        self.browser_path = browser or os.environ.get('LF2_BROWSER') or next(
            (path for name in ('google-chrome', 'chromium', 'chromium-browser', 'chrome')
             if (path := shutil.which(name))), None
        )
        self.observation_space = spaces.Box(-2.0, 2.0, shape=(len(OBS_SCALE),), dtype=np.float32)
        self.action_space = spaces.Discrete(action_count)
        self.observed_seed = self.observed_bg = self.observed_difficulty = self.observed_ai = None
        self._next_seed = seed_base
        self._seed_stride = seed_stride
        self._playwright = self._browser = self._page = None
        self._state = None
        self._done = True

    def _open_page(self):
        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(
            executable_path=self.browser_path, handle_sigint=False,
        )
        self._page = self._browser.new_page()
        self._page.goto(self.url, wait_until='domcontentloaded')
        self._page.wait_for_function('typeof window.rl_reset_match === "function"')

    def set_training_difficulty(self, diff):
        self.diff = resolve_diff(diff)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._next_seed = seed
        game_seed = self._next_seed % (2 ** 32)
        self._next_seed += self._seed_stride
        diff = resolve_diff((options or {}).get('diff', self.diff))
        try:
            if self._page is None:
                self._open_page()
            self._page.evaluate('config => window.rl_reset_match(config)',
                                {'diff': diff, 'seed': game_seed, 'bg': self.bg,
                                 'character': self.character, 'opponent': self.opponent})
            state = self._page.wait_for_function(
                '() => window.rl_match?.time && window.rl_match.game_state()', polling=5
            ).json_value()
        except BrowserTimeout as exc:
            self.close()
            raise RuntimeError('等待 LF2 页面或对局超时。请检查浏览器和 LF2_19 素材。') from exc
        if (state['time'], state['difficulty'], state['seed'], state['bg']) != (
            0, diff, game_seed, self.bg
        ):
            raise RuntimeError(f'对局初始化与请求不符：{state}')
        if (state['davis']['character'].lower(), state['dennis']['character'].lower()) != (
                self.character, self.opponent or 'dennis'):
            raise RuntimeError(f'对局角色与请求不符：{state}')
        if self.opponent_selector:
            self.opponent_selector.reset()
        self.observed_seed = state['seed']
        self.observed_bg = state['bg']
        self.observed_difficulty = state['difficulty']
        self.observed_ai = state['ai']
        self._state, self._done = state, False
        return state_to_obs(state), {'seed': game_seed, 'difficulty': diff, 'bg': self.bg}

    def step(self, action):
        if self._done:
            raise RuntimeError('回合已结束，请先 reset()。')
        if not self.action_space.contains(action):
            raise ValueError(f'无效动作：{action}')
        frames = min(self.action_repeat, self.max_frames - self._state['time'])
        opponent_action = None
        if self.opponent_selector:
            opponent_obs = state_to_obs({'davis': self._state['dennis'], 'dennis': self._state['davis']})
            opponent_action = int(self.opponent_selector.predict(opponent_obs, self.action_repeat))
        state = self._page.evaluate(
            '([action, frames, opponent]) => window.rl_match.rl_step(action, frames, opponent)',
            [int(action), frames, opponent_action],
        )
        advanced = state['time'] - self._state['time']
        reward, result = score_transition(self._state, state, self.max_frames, self.gamma)
        if not 1 <= advanced <= frames or (result is None and advanced != frames):
            raise RuntimeError(f'游戏帧推进异常：请求 {frames} 帧，实际 {advanced} 帧。')
        self._state, self._done = state, result is not None
        info = {'frames': advanced, 'game_frames': state['time']}
        if self._done:
            center_ratio = state['center_damage'] / (state['damage'] or 1)
            info.update(result=result, is_success=result == 'win', center_damage_ratio=center_ratio,
                        center_win=result == 'win' and center_ratio >= MIN_CENTER_DAMAGE_RATIO)
        return state_to_obs(state), reward, self._done, False, info

    def close(self):
        try:
            if self._browser is not None:
                self._browser.close()
        finally:
            if self._playwright is not None:
                self._playwright.stop()
            self._playwright = self._browser = self._page = None
