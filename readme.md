# LF2-RL: Training a Reinforcement Learning Agent to Play Little Fighter 2

> **Based on [F.LF](https://github.com/Project-F/F.LF)** — an open-source HTML5 reimplementation of Little Fighter 2 by Project-F. The game engine, character data, and JS controller are inherited from that project. We added a WebSocket bridge and RL training code on top.

Training an RL agent (Davis) to defeat the built-in AI opponent (Dennis) in Little Fighter 2, a classic 2.5D fighting game.

The agent went through two training phases, achieving a **50–64% rolling win rate** against the hard-difficulty AI in the final phase.

---

## Results

| Phase | Method | Win Rate (Hard) |
|-------|--------|-----------------|
| Phase 2 | Custom Env + PPO | 9.6% |
| Phase 3 | Double DQN + Curriculum | **50–64%** |

---

## Repository Structure

The engine loads its sprites and data from a **sibling** directory `LF2_19` (configured in
`game/game.html` as `"package": "../LF2_19"`), so the layout has to look like this:

```
group_project/
├── LF2-RL/                  # this repository
│   ├── lf2-rl/              # all RL code
│   │   ├── env.py           # Gymnasium environment — WebSocket bridge, reward, obs
│   │   ├── bridge.py        # Low-level WebSocket server (JS ↔ Python communication)
│   │   ├── train.py         # Phase 2: PPO training loop (Stable-Baselines3)
│   │   ├── train_DQN.py     # Phase 3: Double DQN training loop
│   │   ├── test_bridge.py   # Bridge connection test utility
│   │   ├── serve.py         # Static server that sends Cache-Control: no-store
│   │   ├── launch_browser.py# Headless Chrome launcher (background-throttling disabled)
│   │   └── logs/            # Training logs (TensorBoard event files)
│   ├── LF/                  # F.LF game engine (inherited, with RL modifications)
│   │   ├── match.js         # Match logic (modified to push state via WebSocket)
│   │   └── controller.js    # Key injection controller
│   ├── game/game.html       # actual game page (training entry point, see step 4)
│   └── index.html           # upstream F.LF landing page — not used for training
└── LF2_19/                  # game assets, NEXT TO LF2-RL (not inside it)
```

All training code lives in `lf2-rl/`. Use `train.py` for Phase 2 (PPO) and `train_DQN.py` for Phase 3 (Double DQN).

---

## How It Works

The game runs in a browser (F.LF). A Python Gym environment communicates with it over WebSocket:

```
Browser (F.LF JS game)
        ↕  WebSocket (localhost:8765)
Python LF2Env (env.py)
        ↕  step() / reset()
RL Agent (Double DQN)
        ↓  action (0–10) injected via JS controller
```

Each game frame, the JS side pushes a JSON state object (HP, MP, position, facing direction for both characters) to Python. The agent picks one of 11 discrete macro-actions; the controller injects the corresponding key inputs.

---

## Quickstart

**1. Install dependencies**
```bash
pip install torch numpy gymnasium websockets stable-baselines3
```

Verified with Python 3.10 / torch 2.5.1 / stable-baselines3 2.5.0 (conda env `lf2rl`).
If `import stable_baselines3` fails with `ImportError: C extension: None not built`, the
active interpreter has a broken pandas — switch interpreters, it is not a project bug.

**2. Get the game assets**

`LF2_19` must be a sibling of `LF2-RL`, not a subdirectory:

```bash
cd /path/to/group_project        # the directory that contains LF2-RL/
git clone https://github.com/Project-F/LF2_19.git LF2_19
```

**3. Serve the game**
```bash
cd LF2-RL/lf2-rl
python serve.py --port 8000      # --port 8010 if 8000 is already taken
```

`serve.py` is a drop-in replacement for `python -m http.server` that also answers with
`Cache-Control: no-store`. Without it Chrome may keep running an old `match.js` after
you edit it (heuristic caching), which looks like "my change did nothing".

**4. Open the game**
```bash
python launch_browser.py --port 8000
```

This launches a dedicated headless Chrome with background throttling disabled, so
training keeps going when the window is minimized or you switch to another app.
To open it manually instead, use the same URL:

```
http://localhost:8000/LF2-RL/game/game.html?debug&headless
```

`?debug` starts Davis vs Dennis automatically, `?headless` skips canvas rendering.
To actually watch the fight, drop `?headless` and pass `--visible` to `launch_browser.py`.

Query parameter cheatsheet:

| URL | What you get |
|---|---|
| `game/game.html` | normal F.LF main menu — pick characters, stage and difficulty yourself |
| `game/game.html?debug` | skips the menu, jumps straight into Davis vs Dennis (the training matchup) |
| `game/game.html?demo` | built-in AI vs AI demo, `Esc` to leave |
| `game/game.html?debug&headless` | training / unattended (no rendering) |

**5. Run training**
```bash
cd lf2-rl

# Phase 2: PPO
python train.py

# Phase 3: Double DQN (recommended)
python train_DQN.py
```

The script waits for the browser to connect, then starts training automatically.
Do not run both scripts at once — the WebSocket port (`8765`) is fixed, so the second
process fails to bind.

**6. Watch the curves, evaluate, resume**

```bash
# Both algorithms write TensorBoard runs side by side: logs/PPO_*, logs/DQN_*
tensorboard --logdir logs

# Evaluate either checkpoint type (auto-detected: .zip → PPO, .pt → DQN)
python eval.py --algo ppo --model checkpoints/best_model.zip   # best greedy win rate
python eval.py --model lf2_ppo.zip                             # merely the last one
python eval.py --algo dqn --model checkpoints/best_model.pt --episodes 100 --json out.json

# Resume a DQN run from where it stopped (Ctrl+C now saves interrupted.pt first)
python train_DQN.py --resume
```

DQN also writes `logs/DQN_N/episodes.csv`, `eval.csv` and `history.json`, so the numbers
are readable without TensorBoard. The in-training `eval/*` scalars use the same metric
definitions as `eval.py` (`rl_metrics.py`), so the two can be quoted interchangeably.

Both trainers now keep a `best_model`: DQN writes `checkpoints/best_model.pt` on every
eval that beats the previous `(win rate, avg reward)`, and PPO writes
`checkpoints/best_model.zip` after a greedy eval every 100k steps
(`LF2_BEST_EVERY=0` disables it, `LF2_BEST_EPISODES=5` sets the episodes per eval).
PPO's in-training numbers land in `logs/PPO_N/ppo_eval.csv`.

`python smoke_dqn.py` runs the whole DQN pipeline against a fake env in ~30 s — no
browser needed — and asserts the TensorBoard tags, CSVs and resume behaviour.

> **Still want `python -m http.server 8000`?** It works, but it sends no cache headers,
> so after editing anything under `LF/` Chrome may serve the old copy until you hard
> reload (`Ctrl+Shift+R`). That is the only reason `serve.py` exists.

For the full Chinese troubleshooting guide (background throttling, round detection,
page reloads, manual play, watching demos), see [TRAINING_FIXES.md](TRAINING_FIXES.md).

---

## Action Space

11 discrete macro-actions:

| ID | Action | ID | Action |
|----|--------|----|--------|
| 0 | Idle | 6 | Defend |
| 1 | Left | 7 | Left + Attack |
| 2 | Right | 8 | Right + Attack |
| 3 | Jump | 9 | Jump + Attack |
| 4 | Crouch | 10 | Down + Attack (Shoryuken) |
| 5 | Attack | | |

Special moves are packed as single atomic actions — no key-sequence learning required.

---

## Key Design Decisions

- **Off-policy (Double DQN + Replay Buffer)**: each experience is reused many times, critical when browser steps are slow
- **Macro-actions**: special moves packed as single actions to avoid needing recurrent memory
- **Anti-passivity reward**: per-step penalty to prevent stalling strategies
- **Timeout penalty > loss penalty**: prevents the agent from dragging out losing matches
- **3-stage curriculum**: Dumbass AI → Crusher AI → hardest AI, advancing on 70% rolling win rate

---

## Acknowledgements

This project builds on [F.LF](https://github.com/Project-F/F.LF) by Project-F, which provides the HTML5 game engine, character sprites, and physics. We modified `match.js` to broadcast game state over WebSocket and added the entire `lf2-rl/` directory for RL training.

