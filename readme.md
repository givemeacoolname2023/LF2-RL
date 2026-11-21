# LF2-RL: Training a PPO Agent to Play Little Fighter 2

Davis learns to defeat the built-in AI opponent Dennis in the F.LF HTML5 reimplementation of Little Fighter 2.

## Quickstart

Run from `lf2-rl/` with Python and Node.js on `PATH`.

```bash
pip install -r requirements.txt
git clone https://github.com/Project-F/LF2_19.git ../../LF2_19
python train.py --n-envs 4 --diff difficult --output-dir runs/difficult-ppo
```

When the asset pack lives elsewhere, pass `--assets /path/to/LF2_19` to training and evaluation. Standalone `eval.py` uses Chrome or Chromium. Set `--browser` to select an executable, or run `python -m playwright install chromium`.

### Training

- `--n-envs` sets the number of independent matches. `--n-workers` sets the number of Node.js processes that share the simulation load. Each process shares asset data across its matches. The default uses up to 16 available CPU cores.
- `--n-steps` sets decisions per environment per rollout. The default is 16 for 512 environments and 8 for 1024, giving 8192 decisions per rollout. Each decision advances four game frames by default. Set `--action-repeat` and `--max-frames` to change the decision interval and round length.
- Training evaluates target-difficulty wins on separate environments and fixed seeds every `--eval-every` aggregate decisions, at a rollout boundary. The later model wins ties for `checkpoints/best_model.zip`. `--eval-every 0` uses only the stage budgets. Set `--browser` to run training-time evaluation in the specified browser.
- The output directory contains `lf2_ppo.zip`, periodic checkpoints, and TensorBoard, `ppo_eval.csv`, and `ppo_curriculum.csv` logs under `logs/`.

```bash
python train.py --n-envs 512 --n-workers 8 --timesteps 8000000 \
  --config curriculum.example.json --output-dir runs/ppo-512
python train.py --n-envs 1024 --n-workers 16 --timesteps 8000000 \
  --config curriculum.example.json --output-dir runs/ppo-1024
```

### Curriculum and Rewards

The default curriculum plays Easy (`dumbass`), Normal (`CHALLANGAR`), then Difficult (`CRUSHER`). Training advances after beating the current opponent in at least 70% of evaluation rounds and meeting the minimum stage budget. The Easy and Normal stages also advance once they reach 15% and 25% of the requested decision count, respectively. Each promotion starts all environments on the new stage after one final decision in the previous stage. PPO bootstraps unfinished rounds at this boundary and excludes them from completed-round metrics. Use `--no-curriculum` for a single target difficulty.

`--config` accepts the JSON schema in `lf2-rl/curriculum.example.json`. Each non-final stage has a `min_steps`, a `max_steps`, and a `promote_win_rate`. Budgets count decisions across all environments. The final stage corresponds to `--diff` and continues until training ends. At 1024 environments, 768000 decisions cover 750 decisions per environment, enough for a full 3000-frame round with action repeat 4.

Set global reward weights under `reward` and override any of them inside a stage. `win`, `loss`, `draw`, and `timeout` are terminal rewards. `health_margin` scales the final HP difference, `win_speed` rewards earlier wins, and `health_potential`, `distance_potential`, and `depth_weight` control shaping. `frame_cost` and `distance_cost` add optional per-frame penalties. `--gamma`, `--ent-coef`, `--batch-size`, and `--n-epochs` adjust PPO.

Set `center_damage` to a value from 0 to 1 to blend in a center-weighted reward. At 1, every point of opponent HP lost is weighted by `1 - d²`, where `d` is the distance from the fighters' horizontal midpoint to the arena center, divided by half the arena width. The KO bonus is scaled by the damage-weighted mean of those weights. Training logs and `eval.py` report raw wins and wins with a mean weight of at least 0.70. A stage with `center_damage` enabled uses the latter for progression and checkpoint selection. At the default of 0, selection uses raw wins. Both rates count draws and timeouts in the denominator.

```bash
python train.py --resume runs/difficult-ppo/lf2_ppo.zip --timesteps 200000 \
  --output-dir runs/difficult-ppo-continued
tensorboard --logdir runs/difficult-ppo/logs
```

PPO finishes the current rollout, so the actual decision count can round up to the next multiple of `--n-envs × --n-steps`. Resume preserves the checkpoint's discount, action repeat, arena, frame limit, and curriculum progress by default. Passing a different curriculum starts at its first stage. Older PPO checkpoints without environment settings default to one frame per decision.

New PPO policies can select Davis's energy blast, rapid punches, uppercut, leaping strike, approach, and a basic attack aimed at the opponent. The engine executes the normal key sequences, with the character's usual range, recovery frames and MP costs. Resuming a checkpoint preserves its action space.

### Evaluation

```bash
python eval.py --model runs/difficult-ppo/checkpoints/best_model.zip \
  --episodes 100 --seed-base 20000 --json result.json
```

Evaluation inherits the checkpoint's target difficulty, arena, action repeat, and frame limit. Override them with `--diff`, `--bg`, `--action-repeat`, and `--max-frames`. Win rate counts all completed rounds, including draws and timeouts. Use the same seeds, arena, opponent, frame limit, and action mode when comparing checkpoints. The default standalone evaluation seeds start at 20000 and training-time evaluation seeds start at 10000.

New training runs default to `--action-mode reengage` for curriculum assessment and checkpoint selection. During evaluation and play, a checkpoint in this mode uses the PPO argmax until the fighters spend 180 consecutive frames outside the engagement range (60 pixels horizontally or 10 in depth). A movement action then closes the gap until they return to range. `--action-mode greedy` always takes the PPO argmax, and `--action-mode sample` samples from PPO's action distribution. Training rollouts sample from PPO. Both `eval.py` and `pve.py` read the stored mode and accept an override. Older checkpoints without a mode default to greedy when opened directly for evaluation or play.

### PPO vs PPO

Use `--character henry` to train Henry through the Dennis AI curriculum. Henry's special actions use his original Gust, Piercing Arrow, Multiple Shot, and Sonata inputs. A fixed `--opponent-model` replaces the AI curriculum with matches against that checkpoint. It controls the second fighter with its stored character and action mode. Checkpoints without a character use Davis.

```bash
python train.py --character henry --n-envs 512 --action-mode reengage \
  --output-dir runs/henry-curriculum
python train.py --resume runs/henry-curriculum/checkpoints/best_model.zip \
  --opponent-model runs/difficult-ppo/checkpoints/best_model.zip \
  --n-envs 512 --output-dir runs/henry-vs-davis
python eval.py --model runs/henry-vs-davis/checkpoints/best_model.zip \
  --opponent-model runs/difficult-ppo/checkpoints/best_model.zip \
  --episodes 64 --seed-base 92000 --json henry-vs-davis.json
```

The first checkpoint supplies the arena, frame limit, and action repeat for both fighters. The evaluation JSON records both checkpoints and characters. Swap `--model` and `--opponent-model` to reverse the teams.

## Play Against PPO

For keyboard play against the model, run the following in separate terminals from `lf2-rl/`. The asset pack and repository should be siblings named `LF2_19` and `LF2-RL`.

```bash
python serve.py --port 8000
python pve.py --model runs/difficult-ppo/checkpoints/best_model.zip
```

Open `http://localhost:8000/LF2-RL/game/game.html?pve`. Control Dennis with `U/H/M/K` for movement, `I` to jump, `J` to attack, and `,` to defend.

See [TRAINING_FIXES.md](TRAINING_FIXES.md) for historical notes on the browser bridge.

## Acknowledgements

This project builds on LF2-RL by ChristinaPuth and [F.LF](https://github.com/Project-F/F.LF) by Project-F.
