# LF2-RL 训练环境修复说明（2026-10-25）

本文记录这次为了让 `train_DQN.py` 正常跑起来、以及在**不看浏览器窗口时也能稳定训练**所做的全部改动。
每一节都按「现象 → 原因 → 改了什么 → 怎么验证」组织，方便出问题时对照排查。

---

## 0. 一句话结论

| 原来的问题 | 根本原因 | 现在的状态 |
|---|---|---|
| `python train_DQN.py` 直接报错 / 跑到第 1000 步崩 | 4 处 API 与 `env.py` 对不上 | 已修，可连续训练 |
| 浏览器最小化 / 切到别的软件后就不动了 | 浏览器对后台页面节流 + 帧循环停摆 | 无头模式或防节流参数，实测最小化仍有 45 帧/秒 |
| `[结果] 🔄 帧超时`，下一局一直「等待游戏连接」 | 页面因“新局”判定过严而反复重开失败 | 用页面 `session` 判定新局，重开一次即成功 |
| 页面回到主菜单 `F.LF v0.9.9`，再也收不到状态 | 游戏结束后注入按键触发了 F4 退出比赛 | 比赛结束时停止注入按键 |
| 改了 `LF/match.js` 刷新后没效果 | `python -m http.server` 不带缓存头，Chrome 用启发式缓存留旧 JS | 新增 `serve.py`（`Cache-Control: no-store`） |
| 本地页面手动玩，方向键按住只动一下 | RL 每帧注入「全键抬起」，清掉了玩家按住的键 | 只在训练端连着时才注入，手动玩手感与官方一致（第 7 节） |

---

## 1. 怎么跑（三个终端）

> 本机 8000 端口被 `godot_ai` MCP 服务占用（PID 18816），下面统一用 **8010**。
> 你自己的机器如果 8000 空着，把 `--port 8010` 都去掉即可。

```bat
REM 终端 A：静态服务（禁用浏览器缓存）
cd "D:\港城莞\课程\DSC6007强化学习\group_project\LF2-RL\lf2-rl"
python serve.py --port 8010

REM 终端 B：浏览器（无头，不需要一直看着）
cd "D:\港城莞\课程\DSC6007强化学习\group_project\LF2-RL\lf2-rl"
python launch_browser.py --port 8010

REM 终端 C：训练
cd "D:\港城莞\课程\DSC6007强化学习\group_project\LF2-RL\lf2-rl"
python train_DQN.py
```

> **训练脚本要用 `lf2rl` 环境**：`conda activate lf2rl`（Python 3.10 + torch 2.5.1 + stable-baselines3 2.5.0）。
> 系统默认的 Python 3.9 里 pandas 是坏的，`import stable_baselines3` 会直接报 `ImportError: C extension: None not built`。
> `serve.py` / `launch_browser.py` 只用标准库，任何 Python 3 都可以。

PPO（`train.py`）跑法相同：

```bat
conda activate lf2rl
python train.py                              REM 默认 500k 步

REM 只验证能不能跑通（约 1 分钟）
set LF2_TIMESTEPS=2048
set LF2_PORT=8010
python train.py
```

> `set` 写在单独一行：`set X=Y && ...` 会把行尾空格也算进变量值。
> 本机 8000 被 `godot_ai` MCP 占用，所以要和 `serve.py --port 8010` 保持一致。

> `train.py` **本身不监听 HTTP 端口**。它只起 `env.py` 里写死的 `ws://localhost:8765`，
> `8000`/`8010` 是「页面从哪下载」的端口，属于 `serve.py` / `launch_browser.py`。
> `LF2_PORT` 只影响 `train.py` 打印的提示地址，三个终端里的端口保持一致即可。

想**看着它打**（会明显变慢，因为要渲染画面）：

```bat
python launch_browser.py --port 8010 --visible --path "/LF2-RL/game/game.html?debug"
```

`launch_browser.py` 支持：

| 参数 | 作用 |
|---|---|
| `--visible` | 显示窗口（仍然禁用后台节流）；不加则无头 |
| `--port` | 静态服务端口，默认 8000 |
| `--path` | 页面路径，默认 `/LF2-RL/game/game.html?debug&headless` |
| `--browser` | 指定 chrome.exe / msedge.exe，也可用环境变量 `LF2_BROWSER` |
| `--devtools-port` | 远程调试端口，默认 9222；设为 0 让浏览器自选 |

---

## 2. 问题一：`train_DQN.py` 跑不起来

`train_DQN.py` 是按 Gymnasium 之前的旧接口写的，而仓库里的 `env.py` 是 Gymnasium 5 元组接口，两边对不上。

| # | 位置 | 原来 | 现在 | 不改会怎样 |
|---|---|---|---|---|
| 1 | `train_DQN.py:15` | `from rl_env import LF2Env` | `from env import LF2Env` | `ModuleNotFoundError: No module named 'rl_env'`（仓库里没有 `rl_env.py`） |
| 2 | `train_DQN.py:21` | `OBS_DIM = 12` | `OBS_DIM = 13` | 第一层 `nn.Linear(12,128)` 收到 13 维输入直接 RuntimeError |
| 3 | `train_DQN.py:143,192` | `obs = env.reset()` | `obs, _ = env.reset()` | `obs` 变成 `(obs, info)` 元组，1000 步后第一次 `torch.FloatTensor(obs)` 抛 `ValueError: setting an array element with a sequence` |
| 4 | `train_DQN.py:149,203` | `obs, r, done, info = env.step(a)` | `obs, r, terminated, truncated, info = env.step(a)`；`done = terminated or truncated` | `ValueError: too many values to unpack` |
| 5 | `train_DQN.py:153` | `if info == 'win'` | `if info.get('result') == 'win'` | `info` 是 dict，比较永远 False，评估胜率恒为 0% |
| 6 | `train_DQN.py:224` | `dict.get(info, info)` | `dict.get(info.get('result'), '❓ 中斷')` | 同上，且非终止步 `info={}` 时 `.get({})` 抛 `TypeError: unhashable type: 'dict'` |

第 3 条是最隐蔽的：因为 `EPSILON_START=1.0`，前 1000 步都是随机动作，**根本不会调用网络**，所以直到 warmup 结束才崩，看起来像“跑了一段时间才出错”。

---

## 3. 问题二：不看浏览器就不训练

### 3.1 机制

1. 游戏主循环是浏览器的 `setInterval`（`LF/match.js` → `network.setInterval(..., 1000/30/1.5)`，约 22 ms 一帧；`LF/network.js:190`）。
2. 页面被最小化 / 被其它窗口完全遮挡 / 切到后台时，Chrome 会：
   - 降低后台定时器频率（`--disable-background-timer-throttling` 控制）；
   - 隐藏超过 5 分钟后进入 Intensive Wake Up Throttling（**每分钟**才跑一次定时器）；
   - 在某些情况下直接冻结页面（Page Lifecycle）。
3. 帧循环一停，`rl_ws.send(state)` 也停 → 后端 `env.step()` 等 3 秒拿不到新帧 → `[结果] 🔄 帧超时`；如果页面之后回到主菜单，就再也没有状态可收 → `[Reset] 一直等待游戏连接`。

### 3.2 三处改动

**① `lf2-rl/launch_browser.py`（新文件）** —— 用无头模式启动，并禁用节流：

```
--headless=new                            # 根本没有窗口，“看不见”不成立
--disable-background-timer-throttling     # 不降低后台定时器频率
--disable-backgrounding-occluded-windows  # 被遮挡也不当后台
--disable-renderer-backgrounding          # 渲染进程不被降级
--disable-features=IntensiveWakeUpThrottling,CalculateNativeWinOcclusion
--user-data-dir=%TEMP%\lf2_rl_chrome      # 独立配置目录，不碰你日常的 Chrome
```

**② `LF/match.js` 顶部** —— 持有 Web Lock，页面不会被 Page Lifecycle 冻结：

```js
if (navigator.locks && navigator.locks.request) {
  navigator.locks.request('lf2_rl_keepalive', function () {
    return new Promise(function () { /* 永不 resolve，锁一直持有 */ });
  });
}
```

**③ `env.py` 的自愈逻辑** —— 即便页面真的卡住，也不会让训练傻等：

> 补充（第 3 次测试时发现）：还需要一个**每局步数上限**。策略一旦学到"站着不动"（例如未训练时 argmax 恒定选同一个动作），Dennis 可能永远打不死它，一局能跑几万帧。现在 `MAX_EPISODE_STEPS = 3000`（约 66 秒），到上限就 `truncated=True` 并打印 `⏱ 到达步数上限`，下一局由 reset 自动重开。可用环境变量覆盖：`set LF2_MAX_STEPS=1000`。

- `_wait_for_state(5)` 连上了却 5 秒收不到状态 → 认为页面停在主菜单，发 `{"cmd":"reload"}`；
- `reset()` 最多发 4 次重开请求，每次等 30 秒；
- `step()` 等帧超时 → 返回 `truncated=True` 并**主动请求重开**，下一局不会从中局开始；
- 等待连接上限 `CONNECT_TIMEOUT=120` 秒，超时抛 `TimeoutError` 并说明该检查什么。

### 3.3 实测

| 场景 | 结果 |
|---|---|
| 无头（`launch_browser.py`） | 45.7 步/秒 |
| 有窗口 + **最小化**（`--visible`） | 45.2 步/秒 |
| 有窗口 + 最小化，去掉防节流参数 | 44.3 帧/秒（40 秒内，Web Lock 单独也能顶住） |
| 有窗口 + 最小化满 **7 分钟** | 45.43 ~ 45.48 帧/秒，无衰减（逐 30 秒采样，见下） |
| 理论上限 | `match.js` 的 22 ms/帧 ≈ 45 帧/秒 |

```
[  30s] 1364 帧 / 30.0 秒 = 45.46 帧/秒
[  60s] 1364 帧 / 30.0 秒 = 45.45 帧/秒
[  90s] 1364 帧 / 30.0 秒 = 45.46 帧/秒
[ 120s] 1363 帧 / 30.0 秒 = 45.44 帧/秒
[ 150s] 1364 帧 / 30.0 秒 = 45.46 帧/秒
[ 180s] 1364 帧 / 30.0 秒 = 45.46 帧/秒
[ 210s] 1364 帧 / 30.0 秒 = 45.46 帧/秒
[ 240s] 1364 帧 / 30.0 秒 = 45.46 帧/秒
[ 270s] 1364 帧 / 30.0 秒 = 45.44 帧/秒
[ 300s] 1364 帧 / 30.0 秒 = 45.46 帧/秒   ← 超过 Chrome 的 5 分钟阈值仍然满速
[ 330s] 1365 帧 / 30.0 秒 = 45.48 帧/秒
[ 360s] 1363 帧 / 30.0 秒 = 45.43 帧/秒
[ 390s] 1364 帧 / 30.0 秒 = 45.46 帧/秒
[ 420s] 1365 帧 / 30.0 秒 = 45.48 帧/秒
```

> 实测表明 Web Lock 和启动参数是互补的两层保险：Web Lock 防冻结/防强节流，启动参数防定时器降频。
> 值得注意的是，即使只去掉启动参数（保留 Web Lock），最小化 40 秒也是 44 帧/秒 —— 说明 Web Lock 这一层确实在起作用；
> 但它不能覆盖所有情况（例如标签页被浏览器直接丢弃/崩溃），所以启动参数和 `env.py` 的重开兜底都保留。

---

## 4. 问题三：页面回到主菜单，诊断出「连上了但零状态」

### 现象
Python 打印 `游戏已连接！`，但 120 秒内一条状态都收不到。用 CDP 查看 `document.body.innerText` 是 `F.LF v0.9.9 ◼ Project F player1 Random Join? ...` —— 页面停在**主菜单**。

### 原因
`setup_controller` 把 match 塞进了 human controller 的 `child` 数组，所以注入的按键最终会走到 `match.prototype.key`（F1/F2/F4 的处理函数）。比赛结束后（`gameover_state`）再过 60 tick，一个 att/jump 就会触发 `F4()` → `manager.match_end()` → 回主菜单。之后没有 match 也就没有 `game_state()` 消息。

### 修法（`LF/match.js:29-34`，`window.rl_match` 在 `:120`）

```js
function rl_inject_action(action) {
  if (!rl_controller || !rl_controller.buf) return;   // 触屏控制器没有 buf
  if (typeof rl_match !== 'undefined' && rl_match && rl_match.gameover_state) return;  // 比赛结束就不再注入
  ...
}
```

配套：`match.prototype.create` 开头加 `window.rl_match = $`，让上面的判断能拿到当前比赛。
另外后端在“连上了却 5 秒无状态”时会主动要求重开，形成兜底。

---

## 5. 问题四：新局判定太苛刻（这次的主要修复）

### 现象
```
[Reset] 当前不是新局（time=8923），请求浏览器重开（第 1 次）...
游戏断开连接
游戏已连接！
[Reset] 30 秒内没等到新局（此刻 time=1358）     ← 明明重开了，却判定失败
```

### 排查过程
用 CDP 直接观察页面，发现两件事：

1. `window.location.reload()` **是生效的**（`performance.timeOrigin` 变了，`time` 归零）；
2. 页面从载入跑到 `time=20` 只要约 0.5 秒（45 帧/秒）。

原来 `env.py` 用 `time < 10` 判断“新局”，也就是 **0.22 秒**的窗口；而页面重载 + WebSocket 重连往返往往超过这个时间，拿到的第一帧已经 `time > 10`，于是永远判定“不是新局”，一直重开到超时。

### 修法
让页面自带一个“这次载入的身份证”，后端据此判断**页面是否真的重开过**，不再靠帧数猜时间。

- `LF/match.js:7`：`var rl_session_id = Date.now().toString(36) + '-' + Math.random().toString(36).slice(2, 8);`
- `LF/match.js:427`：`game_state()` 的返回值里加上 `session: rl_session_id`
- `lf2-rl/env.py:185` `_is_new_round()`：

```
同一 session 且 time 很小      → 新局（页面内重开）
session 变了                   → 新局（页面确实重载过）
已否决过的 session             → 永远不算新局（避免旧连接残留消息干扰）
```

- `env.py:252`：否决某个 session 时把它记进 `_rejected_sessions`，重载后的新 session 不会被误判。
- `FRESH_MATCH_TIME` 从 10 帧放宽到 60 帧（约 1.3 秒），仅在同 session 时作为兜底。

### 效果
第一次 reset 只需**一次**重开（约 8 秒，含 5 秒等待状态），之后每局结束时页面自己刷新，`reset()` 立刻返回（日志里 `开始第 N 局（t=1）`）。

---

## 6. 问题五：改了 JS 不生效（浏览器缓存）

`python -m http.server` 不发送任何缓存头，Chrome 对这类响应使用**启发式缓存**，`LF/match.js` 可能长期使用旧副本。这次调试中就出现过：改完 `match.js` 后页面里根本没有新代码，`rl_session_id` 是 `undefined`。

**新增 `lf2-rl/serve.py`**：与 `http.server` 用法相同，但回包带 `Cache-Control: no-store, max-age=0` 和 `Pragma: no-cache`，只打印 4xx/5xx（避免素材请求刷屏）。

```bat
python serve.py --port 8010            REM 默认服务 group_project 目录
python serve.py --dir D:\path --port 8010
```

如果仍怀疑是缓存：强制刷新 `Ctrl+Shift+R`，或在 `http://127.0.0.1:9222` 里勾选 Disable cache 后重载。

---

## 7. 问题六：本地页面手动玩「键位稀烂」

### 现象

用这份仓库的页面**手动**玩（不跑训练），方向键按住只动一下，连招基本出不来。官方 F.LF 页面没有这个问题。

### 原因：RL 注入和人类输入写的是同一个控制器缓冲

上游 `Project-F/F.LF` 的 `LF/match.js` 里**没有任何 `rl_` 代码**，`frame()` 开头直接 `$.control.fetch()`，所以官方手感是正常的。

本仓库的 RL 补丁多了两步，而且**不分训练还是手动玩**：

| 位置 | 做了什么 |
|---|---|
| `LF/match.js:173` | 每次建局把 `rl_controller` 设成第一个角色（Davis）的控制器 |
| `LF/match.js:309` | 每帧先注入「除当前动作外的键全部抬起」，再 `$.control.fetch()` |

而游戏的按键是**电平**语义（看「现在是否按住」），不是边沿：

- `LF/character.js:218` `var dx = $.con.state.left !== $.con.state.right` —— 读按住状态决定移动
- `core/combodec.js:119` `if (this.con.state[K]) { push = false }` —— 读按住状态判断连招

于是人按住的键在第 N 帧生效，第 N+1 帧就被注入的「全部抬起」清掉 → 角色只动一帧。
**这不是作者为了训练做的优化，是注入没有和人类输入隔离。**

### 修法

`LF/match.js` 的 `rl_inject_action()` 开头加一道判断——只有训练端（`env.py`）连着时才注入：

```js
// 没有训练端连着时完全不干预，手动玩的手感与官方页面一致
if (!rl_ws || rl_ws.readyState !== 1) return;
```

`match.prototype.frame` 里也加了同样的条件，避免不训练时每 30 帧往 Console 刷调试日志。

训练侧行为不变：`LF2Env` 一连上就是 `readyState === 1`，注入照旧。
唯一的变化是训练中途断开连接的瞬间注入会停，但那时环境也收不到状态，本来就在等重连。

### 训练需不需要考虑键位

**不需要。** 智能体不用键盘，每帧由 `env._send_action()` 直接下发 11 个宏动作之一；
「电平」语义对它反而正合适（一直选 `right` 就一直右移）。
决定训练效果的是动作空间（`env.ACTION_KEYS`）和奖励函数（`_compute_reward`），与手感无关。

### 实测对照

用 CDP 向页面发**一次** `keydown d`（模拟人手按住不放），再读位移和 `con.state.right`：

| 场景 | 按住 0.5s | 再按住 0.5s | 松开后 | `con.state.right` |
|---|---|---|---|---|
| 没有训练端（修复后） | **+110px** | **+115px** | +10px（惯性） | 1 |
| 8765 上有客户端连着（等价于修复前） | 0px | 0px | 0px | 0 |

第二行就是原来「按住不动」的复现：注入把按住的键每帧清掉。

---

## 8. 文件改动清单

| 文件 | 状态 | 改动 |
|---|---|---|
| `lf2-rl/train_DQN.py` | 修改 | import、`OBS_DIM`、reset/step 5 元组、`info['result']` 取值（6 处，见第 2 节） |
| `lf2-rl/env.py` | 修改 | 回合管理常量与 `MAX_EPISODE_STEPS`（步数上限）；`_handler` 只认最新连接并在断开时清空；`_send_raw` / `_request_reload`；`_is_new_round` / `_wait_new_round` / `_ensure_fresh`；`reset()` 记录 session、清零步数；`step()` 帧超时后请求重开并返回 `info['result']='timeout'`；终止时返回 `info['result']='win'/'lose'`；到达上限时 `truncated=True`、`info['result']='max_steps'` |
| `LF/match.js` | 修改 | `rl_session_id`；`game_state()` 增加 `session`；`rl_inject_action` 三道防护（含「只在 ws 连着时注入」，见第 7 节）；`frame()` 同步加连接判断；WS 支持 `{"cmd":"reload"}`；Web Lock 保活；`window.rl_match` |
| `LF/util.js` | 修改 | `setup_resourcemap` 增加本地判断：`127.0.0.1` / `localhost` / `[::1]` 下只用本地兜底映射，不再套用素材包指向 Google 的 112 条资源映射（见第 14 节） |
| `lf2-rl/train.py` | 修改 | 修正过时的页面地址（`/F.LF/...` → `/LF2-RL/...`）；训练步数可用 `LF2_TIMESTEPS` 覆盖，便于冒烟测试；打印的端口可用 `LF2_PORT` 覆盖（默认 8000，只影响提示文字）；挂了 `CheckpointCallback`，每 `LF2_CKPT_FREQ` 步（默认 20000）存 `checkpoints/ppo_*_steps.zip`，避免中途 Ctrl+C 整场白跑 |
| `lf2-rl/launch_browser.py` | 新增 | 无头/防节流启动 Chrome |
| `lf2-rl/serve.py` | 新增 | 禁用缓存的静态服务器 |
| `lf2-rl/eval.py` | 新增 | 用训好的模型打 N 局，统计胜率（含 Wilson 置信区间）、出招占比、平均距离、打掉/挨打血量；`--stochastic` 切换 argmax / 采样选动作（见第 15 节）。只读调用 `env.py`，不改它 |
| `lf2-rl/pve.py` | 新增 | 真人对模型：模型操控左边 Davis、人操控右边 Dennis，不管回合、不 reload（见第 16 节） |
| `LF/manager.js` | 修改 | `start_debug(p2_type)` 增加可选参数（不传仍为 `'computer'`）；URL 分发增加 `?pve` → `start_debug('human')`，供真人对打使用（见第 16 节） |
| `.gitignore` | 修改 | 忽略 `__pycache__/`、`.claude/` |

### 前后端协议（排查时有用）

| 方向 | 消息 |
|---|---|
| 游戏 → Python | `{"session": "...", "time": 123, "davis": {x,y,hp,mp,frame,dir}, "dennis": {...}}` |
| Python → 游戏 | `{"action": 5, "keys": ["att"]}` |
| Python → 游戏 | `{"cmd": "reload"}` → 页面 `location.reload()` |

`action` 0–10：不动、左、右、跳、蹲、攻、防、左+攻、右+攻、跳+攻、下+攻。

---

## 9. 验收记录（本机实测）

环境：Windows 10 19045 / Python 3.10.22（conda `lf2rl`）/ Chrome（最新）/ CPU 训练。

| 检查项 | 结果 |
|---|---|
| `serve.py` 返回 `Cache-Control: no-store` | ✅ |
| 无头浏览器下 `env.reset()` | ✅ 8.8 秒（一次重开），`t=1` 开局 |
| 无头下的步速 | ✅ 50 步 / 1.1 秒 = **45.7 步/秒** |
| 浏览器窗口**最小化**下的步速 | ✅ 50 步 / 1.1 秒 = **45.2 步/秒** |
| 窗口最小化持续 7 分钟 | ✅ 45.43~45.48 帧/秒，无衰减 |
| 连续多局训练（每局自然结束 → 自动重开） | ✅ 先后三段共 13 局连续完成（6+4+3），日志见下 |
| 是否回到主菜单 | ✅ 未再出现 |
| 失败/截断判定 | ✅ `💀 輸`（阵亡）、`⏱ 到达步数上限`（截断）都正确打印 |
| `🏆 贏` 分支 | ⚠️ 未实际触发：未训练的智能体还没赢过，代码路径与“输”共用同一段逻辑 |
| 每局步数上限 | ✅ `LF2_MAX_STEPS=400` 时两局都在 400 步截断，随后自动重开新局 |
| `evaluate()` + checkpoint | ✅ 胜率/平均 reward 正常输出；`torch.save` 写出 81,574 字节 |
| 8000 端口被占用时 | ✅ `serve.py` 打印友好提示并给出 `--port 8010` 建议 |
| 无训练端时手动按住方向键 | ✅ 只发一次 `keydown d`，0.5s 位移 110px、再 0.5s 115px（持续移动） |
| 有训练端连着时注入是否仍生效 | ✅ 同场景位移 0px、`state.right=0`，与修复前一致（第 7 节对照表） |
| 加守卫后动作是否真的下发到角色 | ✅ 随机动作 120 步：Davis 位移 444px、起跳 `y=-45.9`、MP 201→41 |
| PPO（`train.py`）短程 | ✅ `LF2_TIMESTEPS=2048`：1 次迭代 / 2048 步 / 49 秒 / 40 fps，生成 `lf2_ppo.zip` |
| `train.py` 地址打印 | ✅ 不带变量打印 8000；`LF2_PORT=8010` 时 serve/浏览器命令与 URL 全部变成 8010 |

连续训练日志节选（窗口最小化，走 `serve.py` + `launch_browser.py --visible`）：

```
[Reset] 第 1 局开始，等待游戏连接...
游戏已连接！
[Reset] 当前不是新局（time=1038），请求浏览器重开（第 1 次）...
游戏断开连接
游戏已连接！
[Reset] 已连接，开始第 1 局（t=1）
[结果] 💀 Davis 输！davis hp=-27 dennis hp=321
[Ep    1] 💀 輸 | reward=702.21 | steps=3170 | ε=0.940 | loss=9.8346
[Reset] 第 2 局开始，等待游戏连接...
[Reset] 已连接，开始第 2 局（t=1）
[结果] 💀 Davis 输！davis hp=-55 dennis hp=332
[Ep    2] 💀 輸 | reward=634.19 | steps=2750 | ε=0.888 | loss=9.6924
```

> 前几局全输是正常的：ε≈0.94，基本在随机乱按。要看到胜率上升需要几万步。

---

## 10. 排查手册

| 症状 | 最可能的原因 | 处理 |
|---|---|---|
| 卡在 `[Reset] 第 N 局开始，等待游戏连接...` | 浏览器没开 / 页面没打开 / 8765 被占用 | 跑 `launch_browser.py`；确认没有另一个训练脚本占着 8765 |
| `[Reset] 30 秒内没等到新局` | 页面被冻结、页面是旧缓存 JS、或游戏加载失败 | 用 `serve.py`；打开 `http://127.0.0.1:9222` 看 `rl_match.time.t` 是否在涨 |
| `[结果] 🔄 帧超时` 偶发 | 页面正在刷新（正常现象） | 下一局会自动重开，不用管；连续出现才需要查 |
| `[结果] 🔄 帧超时` 每步都出现 | 帧循环停了（节流/冻结） | 用无头模式；检查是不是用普通 `http.server` 打开了旧页面 |
| 页面显示主菜单 `F.LF v0.9.9` | 比赛结束后被 F4 退出（旧 JS） | 确认加载的是新 `match.js`（`serve.py` + 强刷） |
| 改了 JS 没反应 | 浏览器缓存 | 用 `serve.py`；`Ctrl+Shift+R`；或 9222 里 Disable cache |
| `ModuleNotFoundError: No module named 'env'` | 不在 `lf2-rl` 目录下运行 | `cd lf2-rl` 后再 `python train_DQN.py` |
| `ImportError: C extension: None not built`（pandas） | 用错了解释器（系统 Python 3.9，pandas 坏了） | `conda activate lf2rl` 再跑训练 |
| `Address already in use` | 端口被占（本机 8000 被 godot_ai 占用） | `--port 8010` |
| 训练中途浏览器崩了 | —— | 程序会在 reset 等 120 秒后抛 `TimeoutError` 退出；重启浏览器再重跑 |
| 一局打太久（双方僵住） | 策略学到“站着不动”，没人死 | 已默认 3000 帧截断；想更快就 `set LF2_MAX_STEPS=1000` 再启动训练 |
| 想确认浏览器是不是无头 | —— | 任务管理器里看不到窗口，但 `http://127.0.0.1:9222` 能访问到页面 |

### 常用检查命令

```bat
REM 页面是否活着、跑到第几帧
curl http://127.0.0.1:9222/json/list

REM 训练端 8765 是否被占
netstat -ano | findstr :8765
```

---

## 11. 已知限制（这次没做）

1. ~~**没有断点续训**~~ —— **已补（§18）**：`python train_DQN.py --resume`，见 §18.3。
2. **评估仍然占掉约 10% 的墙钟时间**：每 100 局跑 10 局（`EVAL_EPISODES=10`，原来是 20），按每局 45 秒算约 7 分钟一次，整轮 5000 局下来约 6 小时。`time/eval_frac` 曲线会实测这个比例。
3. ~~**截断被当成终止**~~ —— **已修（§17.2）**：`buffer.push(..., float(terminated))`，截断的下一步照样 bootstrap。`done = terminated or truncated` 只剩循环控制的作用。
4. **Replay buffer 不进 checkpoint**：`last.pt` / `ep{N}.pt` 只存网络 + optimizer + 步数。续训后前 64 步不更新（`update()` 有 `len(buffer) < BATCH_SIZE` 守卫），之后正常。要连经验池一起存会多几十 MB，目前没做。
5. **浏览器崩溃不会自动拉起**：环境只会报 `TimeoutError`（`train_DQN.py` 会先把现场存成 `checkpoints/interrupted.pt` 再抛）。训练前先确认 `http://127.0.0.1:9222` 能打开。
6. **`--visible` 且 URL 带 `headless` 时画面是空的**（跳过渲染）。想看画面要去掉 `?headless`，代价是帧率下降。
7. **`?headless` 的意义**是跳过 canvas 渲染提速（`match.js:354`）。无头模式下渲染本来也没人看，所以默认就带上了。
8. **PPO 的断点续训没做**：SB3 的 checkpoint 只存策略，`train.py` 也没有 `--resume`。DQN 那条路线已经能续（§18.3），PPO 要对齐得自己写 resume 逻辑。

---

## 12. 回滚

```bat
cd "D:\港城莞\课程\DSC6007强化学习\group_project\LF2-RL"
git checkout -- LF/match.js lf2-rl/env.py lf2-rl/train_DQN.py lf2-rl/train.py .gitignore
del lf2-rl\launch_browser.py lf2-rl\serve.py
```

> 回滚后：`train_DQN.py` 不能运行，浏览器最小化会停摆，页面可能回到主菜单后收不到状态。

---

## 13. 使用手册（怎么跑、怎么训练、怎么看）

这一节是给「已经修好、现在要真正用」的人看的。前面 1–12 节都是**为什么这么改**。

### 13.1 目录结构（必须是这样）

```
group_project\
├── LF2-RL\            <- 本仓库（游戏引擎 + 训练代码）
│   ├── LF\ core\ game\ ...   引擎
│   └── lf2-rl\              训练代码
└── LF2_19\            <- 游戏素材包，约 15 MB，必须是同级的兄弟目录
    ├── manifest.js
    └── data\ sprite\ sound\ bg\ AI\ UI\
```

原因：`game/game.html` 里写的是 `{"root":"../","package":"../LF2_19"}`，
所以 **HTTP 服务的根目录必须是 `group_project`**，URL 里带 `LF2-RL/` 前缀。

`LF2_19` 没有的话：`git clone https://github.com/Project-F/LF2_19.git LF2_19`
（素材版权归原作者，**不要提交进作业仓库**）。

### 13.2 环境

| 用途 | 解释器 | 说明 |
|---|---|---|
| 训练（`train.py` / `train_DQN.py`） | **conda `lf2rl`** | Python 3.10、torch 2.5.1+cpu、sb3 2.5.0、gymnasium 0.29.1、websockets |
| 服务 / 浏览器（`serve.py` / `launch_browser.py`） | 任意 Python 3 | 只用标准库 |

```bat
conda activate lf2rl
```

系统默认的 Python 3.9 里 pandas 是坏的，`import stable_baselines3` 会报
`ImportError: C extension: None not built`——看到这个就是环境选错了。

### 13.3 启动顺序（每次训练都要做）

**终端 A：静态服务（带 no-store，避免浏览器缓存旧 JS）**

```bat
cd "D:\港城莞\课程\DSC6007强化学习\group_project\LF2-RL\lf2-rl"
python serve.py --port 8010
```

**终端 B：浏览器（无头，训练期间不用管它）**

```bat
python launch_browser.py --port 8010
```

**终端 C：训练脚本**

```bat
conda activate lf2rl
set LF2_PORT=8010
python train_DQN.py        REM 或 python train.py
```

三个终端的端口必须一致（本机 8000 被 `godot_ai` MCP 占用，所以统一 8010；
8000 空着的话 `--port` 和 `LF2_PORT` 都可以不写）。

怎么确认页面真的连上了：

- 终端 C 打印 `游戏已连接！`，随后 `[Reset] 已连接，开始第 1 局（t=...）`
- 浏览器 Console 有 `RL bridge connected`
- `http://127.0.0.1:9222` 能看到页面；`rl_match.time.t` 在涨
- 终端 B 的输出里没有 `404`

> 顺序不强制：训练脚本可以**先**跑，它会等最多 120 秒（`CONNECT_TIMEOUT`）；
> 页面先开也行，`reset()` 发现不是新局会自动请求页面重开。

#### 三个终端各自干什么

```
终端 A  serve.py          提供游戏文件  ──HTTP:8010──▶  终端 B
终端 B  浏览器里的 JS 游戏 每帧推送状态  ──ws:8765───▶  终端 C
终端 C  train_DQN.py      下发动作      ──ws:8765───▶  终端 B
```

| 终端 | 做什么 | 为什么必须一直开着 | 关掉会怎样 |
|---|---|---|---|
| A `serve.py` | 把 `group_project` 挂成 `127.0.0.1:8010`，回包带 `Cache-Control: no-store` | 每局结束 `env.py` 会让页面 `location.reload()`，等于把上千个素材重新下载一遍 | 下一局重开出来是空页面 → 收不到状态 → 训练失败退出 |
| B `launch_browser.py` | 拉起独立 Chrome（无头 + 4 个防节流参数 + 独立配置目录），并阻塞等待 | 游戏本体就是这个浏览器里的 JS，状态由它每帧推给 Python | 游戏没了 → 训练端再也收不到任何一帧 |
| C 训练脚本 | 起 8765 的 WebSocket 服务、算动作、更新网络 | 它就是大脑 | 停止训练（A/B 可以留着，但没意义） |

为什么不能直接用 `file://` 打开 `game.html`：页面要用 RequireJS 加载 `game.js`，
再按 `{"root":"../","package":"../LF2_19"}` 去 XHR 拉素材，`file://` 下这些请求会被跨域拦掉。

为什么不用自己开的 Chrome：那个实例没有防节流参数，窗口最小化约 5 分钟后定时器被降频、
页面被冻结，训练会停摆（第 3 节）。也正因为如此，`launch_browser.py` 用的是**独立**配置目录，
和你日常用的浏览器互不影响。

少开 / 多开时的现象：

| 情况 | 现象 |
|---|---|
| 只有 A | 什么都不会发生——没人打开页面 |
| A + B，没有 C | 游戏在跑，Davis 站着不动（没有客户端连着，注入不生效），被打死后自动重开，循环 |
| A + C，没有 B | C 打印「等待游戏连接」，120 秒后 `TimeoutError` |
| B 没有 A | 浏览器打印 `[警告] 无法访问 ...`，页面 404 |

### 13.4 训练路线 A：Double DQN（`train_DQN.py`）

```bat
conda activate lf2rl
python train_DQN.py
```

| 参数 | 值 | 含义 |
|---|---|---|
| `EPISODES` | 5000 | 总局数 |
| `GAMMA` / `LR` | 0.99 / 1e-3 | 折扣、学习率 |
| `BUFFER_SIZE` / `BATCH_SIZE` | 100k / 64 | 经验池、批大小 |
| `WARMUP_STEPS` | 1000 | 攒够这么多步才开始更新 |
| `EPSILON_START → END` | 1.0 → 0.05，5 万步内衰减 | 探索率 |
| `EVAL_EVERY` / `EVAL_EPISODES` | 100 / 20 | 每 100 局评估 20 局（关探索） |
| `LF2_MAX_STEPS` | 3000（帧） | 每局上限，约 66 秒；可用环境变量改 |

每局一行输出，看懂这行就够：

```
[Ep   12] 💀 輸 | reward=634.19 | steps=2750 | ε=0.888 | loss=9.6924
```

- `🏆 贏 / 💀 輸 / ⏱ 超時`：本局结果（`max_steps` 也显示为超时）
- `reward`：本局**累计**奖励。输了也可能是较大的正数（命中 Dennis +2/点、距离越近每帧最多 +0.5、
  每帧 −0.01），所以别用正负判断打得好不好，看 `🏆/💀`
- `ε`：当前探索率，降到 0.05 后基本在利用已有策略
- `loss`：本局平均 TD loss（前 1000 步是 warmup，显示 0）

模型保存位置（`lf2-rl/checkpoints/`，目录是空的就说明还没训练到）：

| 文件 | 何时写 |
|---|---|
| `best_model.pt` | DQN：每次评估胜率创新高（`(胜率, 平均奖励)` 字典序破纪录） |
| `best_model.zip` | PPO：每 10 万步贪婪评估 5 局，同样按 `(胜率, 平均奖励)` 破纪录（`LF2_BEST_EVERY=0` 可关） |
| `ep500.pt` 等 | 每 500 局 |
| `final_model.pt` | 训练结束 |

**耗时要有心理准备**：约 45 步/秒，每局 2000–3000 步 ≈ 45–70 秒，
5000 局就是 **3–5 天不停**。每 100 局那次评估还要额外跑 20 局（约 15 分钟）。
想先看趋势就把 `EPISODES` 调小、或 `set LF2_MAX_STEPS=600` 让每局短一点。

### 13.5 训练路线 B：PPO（`train.py`）

```bat
conda activate lf2rl
set LF2_TIMESTEPS=500000     REM 想先验证就说小一点，例如 2048
set LF2_CKPT_FREQ=20000      REM 每多少步存一个 checkpoint，不设就是 20000
python train.py
```

| 参数 | 值 |
|---|---|
| 策略 | `MlpPolicy`（2 层 MLP） |
| `n_steps` / `batch_size` | 2048 / 64 |
| `learning_rate` / `n_epochs` | 1e-4 / 5 |
| `ent_coef` / `gamma` | 0.01 / 0.99 |
| 总步数 | 500k（≈ 3 小时 @ 45 步/秒） |

- 日志：`lf2-rl/logs/PPO_N/`（`PPO_1..PPO_8` 是原作者的历史记录，`PPO_9` 是这次冒烟测试）
- 模型：`lf2-rl/lf2_ppo.zip`（**只有整场训练正常结束才会写**）
- 中途存盘：`lf2-rl/checkpoints/ppo_{步数}_steps.zip`，默认每 20000 步一个（40fps 下约 8 分钟）
- 每轮结束打印 `fps / iterations / total_timesteps`；`ep_rew_mean` 要等有回合结束才会出现

**中途 Ctrl+C 会怎样**：`train.py` 只在 `model.learn()` 正常返回后才 `save("lf2_ppo")`，
所以 `lf2_ppo.zip` **不会**产生；但已经存下的 `checkpoints/ppo_*_steps.zip` 都还在，
最多丢 `LF2_CKPT_FREQ` 步。想接着练就手动加载（脚本不自动续训）：

```python
from stable_baselines3 import PPO
from env import LF2Env

model = PPO.load("checkpoints/ppo_20000_steps", env=LF2Env())
model.learn(total_timesteps=480000, callback=...)   # callback 见 train.py 里的 CheckpointCallback
```

> **两条路线不能同时跑**：`env.py:85` 把 WebSocket 端口写死成 8765，
> 第二个进程会报端口占用（Windows 上是 `WinError 10048`）。
> 真要同时跑，得同时改 `env.py:85` 和 `LF/match.js:52` 两边。

### 13.6 看它打（三种方式）

**① 看内置 AI 对打（不需要 Python，最省事）**

```bat
python launch_browser.py --port 8010 --visible --path "/LF2-RL/game/game.html?demo"
```

`?demo` 是两个内置 AI 互打，按 `Esc`（或点页面上的 here）回到主菜单；
`?demo_display` 是纯展示、不能退出。这条只用来看引擎是否正常、素材是否齐全。

**② 看训练实时画面**

把 B 终端的地址去掉 `?headless` 再加 `--visible`：

```bat
python launch_browser.py --port 8010 --visible --path "/LF2-RL/game/game.html?debug"
```

`?headless` 是跳过 canvas 渲染（提速），带上它画面就是空的。看着画面训练会明显变慢。

**③ 用训好的模型打（自己存成 `watch.py` 放 `lf2-rl/` 下）**

PPO：

```python
from stable_baselines3 import PPO
from env import LF2Env

env = LF2Env()
model = PPO.load("lf2_ppo")          # logs 里的 checkpoint 换成对应路径
obs, _ = env.reset()
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(int(action))
    done = terminated or truncated
print("结果:", info.get("result"))    # win / lose / max_steps / timeout
```

> 只想看一局用上面这段就够；要**统计胜率**（跑 N 局、出置信区间、看它出不出招）用
> `python eval.py`，见第 15 节。单局结果不能拿来判断强弱 —— 一局的随机波动太大。

DQN：

```python
import torch
from env import LF2Env
from train_DQN import QNetwork, OBS_DIM, ACT_DIM

net = QNetwork(OBS_DIM, ACT_DIM)
net.load_state_dict(torch.load("checkpoints/best_model.pt", map_location="cpu", weights_only=True))
net.eval()

env = LF2Env()
obs, _ = env.reset()
done = False
with torch.no_grad():
    while not done:
        action = int(net(torch.FloatTensor(obs).unsqueeze(0)).argmax())
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
print("结果:", info.get("result"))
```

跑 ③ 时同样要用 `--visible` + 不带 `?headless` 的地址才看得见画面。

### 13.7 手动玩

**想正常玩（有主菜单，自己选角色 / 场景 / 难度）——用这个地址：**

```bat
python launch_browser.py --port 8010 --visible --path "/LF2-RL/game/game.html"
```

进去是 F.LF 主菜单：`Start Game` → 选角色 → 选场景与难度 → 开打（`manager.js:1442`
的 `switch_UI('character_selection')`）。另外还有 `Network Game`（要 http:// 下才能进）和 `Settings`（改键位）。

**想直接 1v1（跳过菜单，Davis vs Dennis，和训练完全一样的对局）：**

```bat
python launch_browser.py --port 8010 --visible --path "/LF2-RL/game/game.html?debug"
```

`?debug` 的作用就是**跳过菜单直接开局**（`manager.js:250` → `start_debug()`，
固定 Davis vs Dennis），所以带 `?debug` 时不会看到选人界面——这是设计如此，不是坏了。

**难度是换对手 AI 脚本实现的，不是靠 `options.difficulty`。** 那个字段只用来渲染菜单文字，
`start_match()` 根本不读它（只读 `background`），`match.create()` 也没有 `options` 键——
所以以前在 `manager.js` 里看到 `difficulty: 2` 就以为在打"难度 2"，其实一直没有生效，
真正在打的是 `AI_list[0]` = Crusher。现在由 URL 的 `diff=` 选脚本：

| `diff` | 难度 | 对手 AI | 规模 |
|---|---|---|---|
| 缺省 / `0` | Easy | `dumbass` | 108 行，只会追着扇巴掌，不出特技 |
| `1` | Normal | `CHALLANGAR 1.0` | 200 行 |
| `2` | Difficult | `CRUSHER 1.0` | 614 行，连招+闪避（**所有历史数字都是这个**） |

| 想做的事 | `--path` |
|---|---|
| 正常玩，自己选人和场景 | `/LF2-RL/game/game.html` |
| 直接打训练同款对局（Easy） | `/LF2-RL/game/game.html?debug` |
| 打 Difficult（复现旧结果） | `/LF2-RL/game/game.html?debug&diff=2` |
| 看内置 AI 互打 | `/LF2-RL/game/game.html?demo`（按 `Esc` 退出） |
| 训练 / 无人值守 | `/LF2-RL/game/game.html?debug&headless`（默认值） |

不想手写 query string 就用 `launch_browser.py --diff easy|normal|difficult`。
跑起来之后，`train.py` / `train_DQN.py` / `eval.py` 的终端里会打印一次
`[难度] Easy（对手 AI：dumbass）`——**看这一行就够了**，别靠记忆判断在打谁。

键位（玩家 1，见 `LF/manager.js:121`；`Settings` 里可以改）：

| 动作 | 键 | 动作 | 键 |
|---|---|---|---|
| 上 | `W` | 跳 | `Q` |
| 下 | `X` | 攻击 | `S` |
| 左 | `A` | 防御 | `Z` |
| 右 | `D` | | |

> **重要**：只要 8765 上有客户端连着（训练脚本、`test_bridge.py`、`bridge.py` 都算），
> 注入就生效，你的按键会被每帧清掉——**手动玩之前先把终端 C 关掉**。
> 这正是第 7 节修的那个问题。

> **Git Bash 用户注意**：`--path "/LF2-RL/..."` 会被 MSYS 当成路径转换，URL 变成
> `http://127.0.0.1:8010C:/Program Files/Git/LF2-RL/...`，页面打不开。
> 要么改用 cmd / PowerShell，要么加前缀：`MSYS_NO_PATHCONV=1 python launch_browser.py ...`。

### 13.8 调试桥接（不训练，只确认通路）

```bat
python test_bridge.py
```

连上后每 30 帧打印 `t=... davis_hp=... dennis_hp=...`，并且给页面回 `{"action": 0}`（角色站着不动）。
用来区分「页面/WS 坏了」还是「训练脚本的锅」。验证完 Ctrl+C。

### 13.9 看 TensorBoard

```bat
conda activate lf2rl
tensorboard --logdir logs
```

然后打开 `http://localhost:6006`。

两条路线现在都写曲线，目录并列：

| 路线 | 目录 | 写曲线的方式 |
|---|---|---|
| PPO（`train.py`） | `logs/PPO_1`、`logs/PPO_2` … | SB3 自带，`tensorboard_log="./logs/"` |
| DQN（`train_DQN.py`） | `logs/DQN_1`、`logs/DQN_2` … | `SummaryWriter`，tag 命名对齐 SB3（见下） |

PPO 的 `rollout/ep_rew_mean`、`train/value_loss`、`train/entropy_loss` 都在里面。

DQN 的 tag（横轴一律是**训练步数** `total_steps`，所以能和 PPO 的 `total_timesteps` 直接叠图）：

| Tag | 含义 |
|---|---|
| `rollout/ep_rew_mean` / `rollout/ep_len_mean` | 最近 100 局的 reward / 局长滑动均值（和 SB3 同名，可与 PPO 叠在一起看） |
| `rollout/ep_win_rate` | 最近 100 局**训练回合**的胜率（含 ε 探索，不是评估口径） |
| `rollout/ep_timeout_rate` | 最近 100 局被 3000 帧上限截断的比例 —— 僵局预警 |
| `episode/reward` / `episode/length` | 单局原始值，不平滑 |
| `explore/epsilon` | 当前 ε。**这条最值得先看**：它在 5 万步就压到 0.05，而整轮是 1090 万步 |
| `explore/action_entropy` / `explore/action_rate/*` | 动作分布的熵（上限 ln11≈2.4）和 11 个动作各自占比，用来看探索有没有塌 |
| `train/td_loss` / `train/q_mean` / `train/target_mean` / `train/grad_norm` | TD loss、Q 值、target 值、裁剪前梯度范数。`q_mean` 和 `target_mean` 越拉越开就是发散前兆 |
| `eval/win_rate`（+`_ci_low`/`_ci_high`/`_decisive`/`best_win_rate`） | 每 100 局用贪婪策略跑 10 局的胜率，**和 `eval.py` 同口径** |
| `eval/attack_rate` / `eval/mean_dist` / `eval/dealt` / `eval/taken` / `eval/avg_len` | 出招占比（随机基线 45.5%）、平均距离、打掉/挨打血量、局长 |
| `eval/distinct_len` | 10 局里有几种不同局长。一直是 1 说明 deterministic 退化成同一条轨迹，CI 偏乐观 |
| `time/fps` / `time/eval_frac` | 步速；以及评估累计占掉多少墙钟时间（默认约 10%） |

同一个运行目录里还有两份纯文本记录（不装 TensorBoard 也能看，写报告直接引用）：

| 文件 | 内容 |
|---|---|
| `logs/DQN_N/episodes.csv` | 每局一行：result / reward / steps / ε / loss / q_mean / target_mean / grad_norm / attack_rate / action_entropy / fps |
| `logs/DQN_N/eval.csv` | 每轮评估一行：胜率 + Wilson CI、局长、出招占比、距离、打掉/挨打、11 个动作的计数 |
| `logs/DQN_N/history.json` | 上面评估记录的 JSON 版，方便直接抄进报告 |

> 断点续训默认写回**原来那个** `logs/DQN_N`，TensorBoard 会把同目录的多个事件文件按 step 合并成一条连续曲线。想另开一条就用 `--new-run`。

### 13.10 参数速查

| 想改什么 | 改哪里 |
|---|---|
| 每局步数上限 | 环境变量 `LF2_MAX_STEPS`（默认 3000 帧） |
| DQN 总局数 | `train_DQN.py --episodes N` 或环境变量 `LF2_EPISODES`（默认 5000） |
| DQN 评估频率 / 局数 | `--eval-every` / `--eval-episodes` 或 `LF2_EVAL_EVERY`（100）/ `LF2_EVAL_EPISODES`（10） |
| DQN 存档间隔 | `--ckpt-every` 或 `LF2_CKPT_EVERY`（默认每 100 局写 `last.pt`）；`LF2_FULL_CKPT_EVERY` 控制 `ep{N}.pt`（500） |
| DQN 断点续训 | `python train_DQN.py --resume [路径]`，不写路径用 `checkpoints/last.pt` |
| DQN 日志目录 | `--log-dir` 或 `LF2_LOG_DIR`；`--no-tb` 关闭；`LF2_LOG_WINDOW` 调滑动窗口（100） |
| DQN ε 衰减步数 | 环境变量 `LF2_EPSILON_DECAY`（默认 50000；见 §17.6 的风险提示） |
| PPO 训练步数 | 环境变量 `LF2_TIMESTEPS`（`train.py:9` 默认 200000） |
| `train.py` 打印的端口 | 环境变量 `LF2_PORT`（默认 8000，只影响提示文字） |
| 观测（13 维） | `env.py` 的 `_state_to_obs` / `OBS_SCALE` |
| 动作空间（11 个宏动作） | `env.py` 的 `ACTION_KEYS` + `LF/match.js` 的 `ACTION_MAP`（两处要一致） |
| 奖励 | `env.py` 的 `_compute_reward` |
| 对手和难度 | `LF/manager.js:1515` 的 `start_debug`（`selected: 8` 是 Dennis，`difficulty: 2`） |
| 帧率上限 | `LF/global.js:94` 的 `framerate`（现在是 30，实际约 45 帧/秒） |
| 浏览器参数 | `launch_browser.py --help`（`--visible` / `--port` / `--path` / `--browser` / `--devtools-port`） |
| 评估跑几局 | `eval.py --episodes N`（默认 20） |
| 评估用哪个模型 | `eval.py --model checkpoints/ppo_xxxxx_steps.zip`（默认取 checkpoints/ 里步数最大的） |
| 评估 DQN 模型 | `eval.py --algo dqn --model checkpoints/best_model.pt`（`--algo auto` 会按 `.zip`/`.pt` 自动判断） |
| 评估时是否采样 | `eval.py --stochastic`（不加 = argmax，加 = 按策略分布采样；DQN 用 `softmax(Q/T)`，`--temperature` 调 T） |
| 评估结果存 JSON | `eval.py --json out.json` |

### 13.11 收尾

```bat
REM 关掉终端 A / B / C
REM 想让浏览器也彻底干净：
rmdir /s /q "%TEMP%\lf2_rl_chrome"
```

`launch_browser.py` 用的是独立配置目录 `%TEMP%\lf2_rl_chrome`，
**不会**动你日常 Chrome 的书签、登录态和设置；删掉这个目录就等于把训练用的浏览器环境清空。

### 13.12 一页速查

```bat
REM ===== 一次完整的训练 =====
cd "D:\港城莞\课程\DSC6007强化学习\group_project\LF2-RL\lf2-rl"
python serve.py --port 8010                          REM 终端 A，一直开着
python launch_browser.py --port 8010                 REM 终端 B，一直开着
conda activate lf2rl && set LF2_PORT=8010 && python train_DQN.py   REM 终端 C

REM ===== 只想看看效果 =====
python launch_browser.py --port 8010 --visible --path "/LF2-RL/game/game.html?demo"

REM ===== 手动玩（带菜单，自己选人）=====
python launch_browser.py --port 8010 --visible --path "/LF2-RL/game/game.html"

REM ===== 手动玩（跳过菜单，直接 Davis vs Dennis）=====
python launch_browser.py --port 8010 --visible --path "/LF2-RL/game/game.html?debug"
```

---

## 14. 问题七：本地页面每局都会去拉 Google 的图（已修）

### 现象

每局结束页面重开时（见第 13.6 节：重置 = 整页 `location.reload()`），浏览器除了向
`127.0.0.1:8010` 取本地文件，还会向 Google 的图床发请求。实测（CDP 读运行中页面的
`performance.getEntriesByType('resource')` 和 `document.images`）：

| 去向 | 改前 | 改后 |
|---|---|---|
| `127.0.0.1:8010`（本地） | 101 | 121 |
| `lh3/4/5/6.googleusercontent.com`（18）+ `1/2/3.bp.blogspot.com`（4） | **22** | **0** |
| `clarity.ms`（上游统计） | 3 | 3 |

改前这 22 张图是**真的加载成功**的：页面里 59 个 `<img>` 指向 Google/Blogspot，
`naturalWidth` 全部 > 0，没有一个触发回退。

### 原因

`LF2_19/resourcemap.js` 开头：

```js
condition: 'location contain http at 0',
resource: {
  'sprite/davis_0.png': '//lh3.googleusercontent.com/...',
  ...   // 共 112 条：67 个 sprite + 42 个背景 + 3 个 UI
}
```

`LF/util.js` 的 `setup_resourcemap()` 把这句字符串编译成
`window.location.href.indexOf('http') === 0`。这个判断分不清「公网上的 http」和
`http://127.0.0.1:8010` —— 后者也命中，于是本地开发同样用外网图。
上游的本意应该是「网页版优先用网上的图」，本地 dev 页被顺带命中了。

### 为什么不至于崩

加载失败时 `core/sprite-dom.js:211-214`（以及 `core/sprite-canvas.js:142`）的
`img.onerror` 会回退到 `pack.location + res`（素材包里的本地文件）。所以断网也能跑，
**但每张被映射的图都要先等一次外网超时**，每局重开会被拖慢。

### 修法

`LF/util.js` 的 `setup_resourcemap()` 开头加本地判断，本地直接返回只含本地路径的兜底映射：

```js
const local_dev = window.location.hostname === '127.0.0.1' ||
  window.location.hostname === 'localhost' ||
  window.location.hostname === '[::1]'
if (local_dev) {
  return [{
    get: function (res) {
      return pack.location + res
    }
  }]
}
```

**必须返回数组，不能返回 `undefined`**：`manager.js:198` 会 `new Fresourcemap(...)`，
而 `core/util.js:46` 的 `make_array` 把 falsy 变成 `[]`；空映射的 `get()` 会原样返回
`sprite/davis_0.png` 这种相对路径，浏览器会相对 `game/` 去解析，路径就错了。
部署到真实域名时这段不生效，行为与本改动前一致。

### 实测结果

改动后等训练自然走到下一局重开（页面自动刷新，不需要手工干预）：

```
本地请求: 121
外网请求: 3 个 -> ['l.clarity.ms', 'scripts.clarity.ms', 'www.clarity.ms']
图片: 127 个, 加载失败 0 个, 仍指向外网 0 个
局内时间: 435 -> 571（游戏循环正常推进）
```

### 没有动的地方

`game/game.js:37-39` 是上游 F.LF 自带的微软 Clarity 统计脚本，触发条件和上面的资源映射
一模一样（`location.href.indexOf('http') === 0`），所以**每局重开仍有 3 个 `clarity.ms`
请求发出去**。这次只处理了图片资源映射，统计脚本没动 —— 改完不等于「零外联」。

---

## 15. 评估：量一下真实胜率（`lf2-rl/eval.py`，新增）

### 为什么不能只看训练曲线

`train.py` 只打 `ep_rew_mean`。这个指标在本项目里**基本没有参考价值**，因为它被
「距离奖励」灌满了：`env.py:167` 的 `max(0,(300-dx)/300)*0.5` 每帧最多给 0.5 分，
3000 帧就封顶 1500 分，而**「赢」只值 200 分**。模型站在中距离不动也能拿 400 多分。

200k 步这一轮（`logs/PPO_11`）的完整曲线：

| 指标 | 起点 4k | 终点 200k |
|---|---|---|
| `rollout/ep_rew_mean` | 473.2 | 509.9 |
| `rollout/ep_len_mean` | 3000（每局都撞上限） | 2022 |
| `train/entropy_loss` | −2.397 | −2.307 |
| `train/clip_fraction` | 0.000 | 0.001 |
| `train/explained_variance` | −0.00 | 0.61 |
| `time/fps` | 45 | 44 |

`train/entropy_loss` 是**熵取负**，所以 −2.307 表示熵 2.307 nat，而 11 个动作的熵上限是
`ln 11 = 2.398` —— 熵只降了 0.09 nat。**这个读法是对的**：实测按策略分布**采样**时，
20 局全败、出招占比 49.4%（随机基线 45.5%）—— 分布上它确实还是一条随机策略。

**但从这个曲线再往下推一步就错了。** 部署用的是 **argmax**，不是采样，而实测 argmax
20 局全胜。一个分布上接近均匀的策略，它的 argmax 仍然可以是状态的确定函数；每局初始
条件完全相同，同样的状态反复出现，同一串动作就被反复执行，于是呈现出高度稳定的打法。

准确的结论是：**这个策略的「分布」几乎没离开初始值，但它的「argmax」被 PPO 雕成了
一个有效的确定性脚本。** 训练曲线只能告诉你前者；后者必须真的去打才知道。

### 用法

```bat
REM 前提：serve.py + launch_browser.py 已经开着（见 13.3），且 train.py 已停（8765 是独占的）
cd "D:\港城莞\课程\DSC6007强化学习\group_project\LF2-RL\lf2-rl"
conda activate lf2rl

python eval.py                              REM 默认读 checkpoints/ 里步数最大的模型，跑 20 局
python eval.py --model lf2_ppo.zip          REM 指定模型
python eval.py --episodes 50                REM 跑 50 局
python eval.py --episodes 20 --stochastic   REM 按策略分布采样选动作（和训练时同一口径）
set LF2_MAX_STEPS=400 && python eval.py --episodes 2    REM 短局冒烟，几分钟就能跑完
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `--model` | checkpoints/ 里步数最大的 | 路径，`.zip` 可省略 |
| `--episodes` | 20 | 跑几局 |
| `--device` | cpu | 13 维 MLP，用不上 GPU |
| `--stochastic` | 关 | 打开则按策略分布**采样**，否则取 **argmax** |

耗时（本机实测）：argmax 模式一局约 12 秒，20 局 4 分 13 秒；`--stochastic` 下动作接近
随机，一局 41 秒左右，20 局 13 分 44 秒。

### 实测结果（本机，2026-10-25，`checkpoints/ppo_200000_steps.zip`）

评估条件：对手是 `?debug` 页面里的 Dennis（`LF/manager.js` 的 `start_debug`，
`selected_AI: 0`、`difficulty: 2`），和训练时完全一致；每局都是页面 `location.reload()`
之后的同一个固定初始站位。**换了对手或难度，下面的数字不能直接沿用。**

#### deterministic（默认，argmax）

```
评估结果：checkpoints\ppo_200000_steps.zip，共 20 局
胜 20 / 负 0 / 超时 0
胜率        100.0%   (95% Wilson CI 83.9% – 100.0%)
决胜局胜率  100.0%   (20/20，不含 0 局超时)

平均局长        540 帧（约 12 秒/局）
平均距离         99 px
出招占比       86.6%   <- 随机策略基线 45.5%
平均打掉        516 hp / 平均挨打 24 hp

动作分布（全部 10809 步）：
   8 ['right', 'att']   6954   64.3%
   7 ['left', 'att']    2406   22.3%
   4 ['down']           1210   11.2%
   2 ['right']           239    2.2%
```

它实际上只用 4 个动作（其余 7 个一次都没选），策略是「贴近 + 一直右攻/左攻」，
平均 12 秒把 Dennis 打穿，自己只掉 24 血。

#### stochastic（`--stochastic`，按策略分布采样）

```
评估结果：checkpoints\ppo_200000_steps.zip，共 20 局，采样 stochastic
胜 0 / 负 20 / 超时 0
胜率          0.0%   (95% Wilson CI 0.0% – 16.1%)

平均局长       1838 帧（约 41 秒/局）
平均距离        178 px
出招占比       49.4%   <- 随机策略基线 45.5%
平均打掉        149 hp / 平均挨打 513 hp

动作分布（全部 36759 步）：
   8 ['right', 'att']   15.4%     6 ['def']        10.4%     1 ['left']      5.5%
   4 ['down']           13.3%     5 ['att']         9.4%     9 ['jump','att'] 4.5%
   7 ['left', 'att']    12.8%     0 []              8.6%     3 ['jump']      4.4%
                                  2 ['right']       8.5%
                                 10 ['down','att']  7.3%
```

11 个动作全部被用到，出招占比 49.4%，和随机策略的 45.5% 几乎没区别。

### 同一份权重，两种口径差多少

| | argmax（部署口径） | 采样（训练口径） |
|---|---|---|
| 20 局胜负 | **20 胜 0 负** | **0 胜 20 负** |
| 胜率 | 100%（CI 83.9–100%） | 0%（CI 0–16.1%） |
| 局均步数 | 540（12 秒） | 1838（41 秒） |
| 出招占比 | 86.6% | 49.4% |
| 打掉 / 挨打 | 516 / 24 | 149 / 513 |
| 用到的动作数 | 4 / 11 | 11 / 11 |

**这张表是这一节最重要的东西。** 同一份 `.zip`，换个选动作的方式就从全胜变成全败 ——
说明 PPO 这 200k 步几乎没有改变策略的**分布**（所以采样时仍是随机打法），只是把
**argmax** 挪到了一个能打赢的位置。这种策略**没有余量**：只要对手、难度或初始站位
变一点，argmax 就可能翻到别的动作上去，表现会突然崩掉。

写报告时建议明确注明评估口径是 argmax，否则「胜率 100%」和「胜率 0%」两个数字
都能从同一个模型上诚实测得，读者会以为搞错了。

### 怎么读这些数

**① 出招占比要和 45.5% 对比，不是和 0 对比。**
11 个动作里含 `att` 的有 5 个（5 攻、7 左攻、8 右攻、9 跳攻、10 下攻），所以**随机策略
的出招占比就是 45.5%**。实测 86.6% 说明它是主动进攻的；如果测出来是 10% 以下，那才是
「学会了不出招」。

**② 100% 不等于 20 个独立样本。**
deterministic 模式下 20 局的步数只在 **529** 和 **645** 两个值附近出现，也就是说这不
是 20 次独立试验，而是两三条轨迹重复了 20 次。因此 `83.9% – 100%` 这个区间是**偏乐观**的，
它只反映了「初始条件固定时」的表现 —— 换句话说，它衡量的是「这条脚本跑得有多稳」，
不是「这个策略有多强」。真正的方差看 `--stochastic`：同样是 20 局，那边是 0 胜 20 负。

**③ 一局定不了强弱。** 只看一局的话，赢一次就是 100%、输一次就是 0%，信息量约等于零。
要下「胜率 80%」这种结论至少跑 100 局，并且报置信区间 —— 20 局赢 16 局，真实胜率可能
只有 57%。

**④「决胜局胜率」是另一个口径。** 超时（3000 帧内没人死）不算输也不算赢。如果一大堆
超时，说明模型在打太极；这时「决胜局胜率」看起来可能很高，但没有任何意义，要看超时占比。

### 两个坑

**坑一：GBK 控制台会 `UnicodeEncodeError`。**
`env.py:339` 在到达步数上限时会打印 `⏱`（U+23F1），GBK 代码页打不出来，直接崩在
`step()` 里。报错长这样：

```
UnicodeEncodeError: 'gbk' codec can't encode character '\u23f1'
```

解决办法（任选一个）：

```bat
chcp 65001                                     REM 把当前窗口切成 UTF-8
set PYTHONIOENCODING=utf-8                      REM 或者只让 Python 用 UTF-8 输出
```

Windows Terminal / VS Code 的终端默认就是 UTF-8，一般碰不到；用 `cmd.exe` 老窗口才有。

**坑二：`train.py` 还在跑会抢 8765。**
`env.py:85` 把端口写死成 8765，两个进程不能同时用。`eval.py` 启动时会先试 bind 一次
8765，被占用就直接报错退出，而不是傻等 120 秒再超时。确认端口空闲：

```bat
netstat -ano | findstr :8765      REM 没有输出 = 空闲
```

---

## 16. 真人对模型 PVE（`?pve` + `lf2-rl/pve.py`，新增）

### 为什么模型只能站在左边

两条硬约束，都在游戏侧：

1. **RL 桥接的注入目标是写死的** —— `LF/match.js:174-178`：
   ```js
   // 设置 RL controller（Player 0 = Davis）
   var char_keys = Object.keys($.character);
   rl_controller = $.character[char_keys[0]].con;
   ```
   模型的按键只能打进 0 号角色，也就是 Davis。
2. **Dennis 是电脑** —— `LF/manager.js:1486`：
   ```js
   controller: players[i].type === 'human'
     ? session.control[i]
     : { type: 'AIscript', id: AI_list[players[i].selected_AI].id }
   ```
   `type: 'computer'` 的角色拿到的是 AIscript 控制器，往它的 `buf` 里塞按键事件毫无作用。

所以要一人一机，只能让**人**换到右边：Dennis 改成 `type: 'human'`（拿到 `session.control[1]`），
模型继续操控左边的 Davis。反过来把模型放到右边的话，它学到的「一直按右 + 攻击」会变成
「一直后退 + 打空气」，除非把观测做水平镜像、把左右动作互换，风险大且不保证可用。

### 改动

| 文件 | 改动 |
|---|---|
| `LF/manager.js` | `start_debug(p2_type)` 增加可选参数，只影响 Dennis 的 `type`（不传 = `'computer'`，训练/评估路径行为不变）；URL 分发增加 `?pve` → `start_debug('human')` |
| `lf2-rl/env.py` | 把 `_state_to_obs` 的方法体提成模块级 `state_to_obs(state)`，方法改为一行委托。行为等价，只是为了让 `pve.py` 能复用而不复制 `OBS_SCALE`/`OBS_OFFSET` |
| `lf2-rl/pve.py` | 新增。WS 服务在 8765，收到一帧状态就当场算动作回发；不管回合、不 reload、不计步数 |

`pve.py` 和 `eval.py` 的关键区别：**eval 管回合，pve 不管**。`eval.py` 会 reset、会超时截断，
你正打着它就把页面重开了；`pve.py` 只做「读状态 → 算动作 → 发回去」，直到 Ctrl+C。

### 用法（三个终端）

```powershell
REM 终端 A：静态服务（已经开着就不用重开）
cd "D:\港城莞\课程\DSC6007强化学习\group_project\LF2-RL\lf2-rl"
python serve.py --port 8010

REM 终端 B：有窗口的浏览器，页面是 ?pve
python launch_browser.py --port 8010 --visible --path "/LF2-RL/game/game.html?pve"

REM 终端 C：模型驱动
conda activate lf2rl
python pve.py               # 默认 argmax，和 eval.py 一致
python pve.py --stochastic  # 改成每帧采样，模型不会卡在墙角（见 16.7）
```

三个要点，少一个就看不到东西：

- **必须 `--visible`**，否则没有窗口；
- 路径里**不能带 `&headless`** —— 那个参数会跳过 canvas 渲染，画面就是空的；
- 顺序上 `pve.py` 先起后起都行，页面没连上会每 2 秒重试一次（`LF/match.js:69-72`）。

### 键位

模型占左边，所以左边那套 W/A/S/D 你别碰（按了也会被模型每帧清掉）。
你操作右边的 Dennis：

| 动作 | 上 | 下 | 左 | 右 | 跳 | 防 | 攻 |
|---|---|---|---|---|---|---|---|
| 键 | `U` | `M` | `H` | `K` | `I` | `,` | `J` |

键位出自 `LF/manager.js:118-126` 的 `settings.control`。

### 打完一局

有人倒下后比赛进入 gameover，`rl_inject_action` 会自己停止注入（`LF/match.js:38`），
所以不会出现「按键把游戏顶回主菜单」的问题。**按 F5 就开新一局**，`pve.py` 不用重启 ——
页面重载后会自己重连 ws。

### 和 `?debug` 的唯一差别

`?debug` 和 `?pve` 走的是同一个 `start_debug()`，唯一的区别是传给它的
`p2_type` 参数：`?debug` 不传（Dennis = `computer`），`?pve` 传 `'human'`。
地形、难度（`difficulty: 2`）、角色（Davis vs Dennis）全部相同。

### 16.6 实测验证了什么、没验证什么

用 CDP 往页面里注入真实键盘事件（`Input.dispatchKeyEvent`，走的是
`core/controller.js:9-12` 的 `document` 级 `keydown`/`keyup`）逐项测过：

| 项目 | 结果 |
|---|---|
| `?pve` 把 Dennis 设成 human | ✅ 控制器 `type='keyboard'`，`config` 是第二套键位，`att` = keycode 74 |
| 人操作 Dennis 移动 | ✅ 按住 `H`，Dennis 从 x=2347 走到 x=1502 |
| 人操作 Dennis 出招 | ✅ 按 `J`，Dennis 帧 61→63→回到站立 |
| 模型在驱动 Davis | ✅ 帧在 61-68 之间循环（`punch` 动画），会转身、会追击 |
| `?debug` 训练/评估路径没坏 | ✅ `LF2_MAX_STEPS=1200 python eval.py --episodes 1` → 529 帧、86.4% 出招、Davis 赢，与改动前一致 |

**但是打不成一场像样的对局。** 实测到的现象：

```
t= 1800  Davis x=2400(右墙) dir=right 帧 62-68 循环出拳  hp=500
         Dennis x=2347 hp=500 站着一动不动
         → 连续 1200+ 帧，Davis 一拳没打中，因为他背对着 Dennis 朝墙出拳

t= 3101  Davis x=2290 dir=left   hp=500
         Dennis x=1502 hp=500
         → 连续 900 帧两人都不动；页面 buf 显示模型输出的是 ["down",1] = 蹲下
```

两种卡死的形态不同，成因是同一个：**观测饱和 + 确定性策略的不动点**。

**成因一：观测里的坐标有一半地图是瞎的。**

`env.py` 的 `OBS_SCALE[0] = OBS_SCALE[6] = 400`、`OBS_OFFSET = 1`，
所以 `obs_x = x/400 - 1`，再 `clip(-2, 2)`。反解出来，可观测范围是
`x ∈ [-400, 1200]` —— 而地图是 0~2400（`LF/mechanics.js:334-340` 把 `ps.x` 夹在
`[0, bg.width]`，实测右墙正好停在 2400），左边还用不满。

于是 **x ≥ 1200 的整个右半场全部被压平成 2.0**。两个出生点 x=2175 / x=2347
从第一帧就已经是 `2.0`，Davis 和 Dennis 在观测里位置完全一样。
模型能用的相对位置信息只剩 `obs[12] = dx/400`（dx=53 只有 0.13，dx=788 是 1.97）。

**成因二：确定性策略一旦进入不动点就永远出不来。**

`pve.py` 用 `deterministic=True`。当动作不改变观测（蹲下、对着墙出拳），
下一帧观测和这一帧一模一样，模型必然再输出同一个动作 —— 状态机锁死。
在 `eval.py` 里不会暴露，因为 AI 控制的 Dennis 会主动贴上来，观测每帧都在变。

**为什么 eval 还是 20/20 赢：** 训练和评估时 Dennis 是 AIscript，会主动进攻，
Davis 永远处在「对手就在旁边、dx 在变」的分布里；`?pve` 里人站着不动，
Davis 走出了训练分布。这不是回归，是策略泛化不足 —— 训练分布外的行为没人教过。

**结论：`?pve` 的接线是对的，人确实能操作 Dennis 和模型对打，但模型目前
会跑到右墙卡住，不会主动找人。** 想让它真能打，需要改 `OBS_SCALE` 的坐标那一维
（`OBS_SCALE[0]` 和 `OBS_SCALE[6]` 从 400 改成 600、对应的 `OBS_OFFSET` 从 1 改成 2，
才能把 0~2400 映射到 -2~2 而不截断）并重新训练，属于训练侧改动，未实施。

### 16.7 `--stochastic` 实验：打破了不动点，但打不死人

`pve.py` 加了 `--stochastic`（每帧从策略分布采样，而不是取 argmax），
用来分离「确定性不动点」和「观测饱和」两个成因。同样的权重、同样的被动 Dennis：

| 策略 | Dennis 挨打 | Davis 活动范围 | 卡在右墙的时间 |
|---|---|---|---|
| 默认 argmax | 4000+ 帧内 **0 伤害** | 基本只有 2400 | 100% |
| `--stochastic` | 60 秒内 **1 次命中、共 14 伤害** | x = 2038~2400 | 40% |

结论：**采样确实打破了死锁** —— Davis 会动了、会打到人了。但命中率低到没有意义，
而这里还有个更要命的东西：

**这游戏有血量自然回复。** `LF/character.js:124-129`：

```js
if ($.match.time.t % 12 === 0) {
  if ($.health.hp >= 0 && $.health.hp < $.health.hp_bound) $.health.hp++
}
```

每 12 帧 +1 hp，按 44 fps 算约 **+3.7 hp/秒**。也就是说模型必须持续输出
大于 3.7 hp/s 的净伤害才可能赢。实测 `--stochastic` 是 14 hp / 60 s = **0.23 hp/s**，
连回复速度的零头都不到 —— 打不动，不是「打得慢」，是**永远赢不了**。

对比一下 `eval.py` 里赢的那局：529 帧（约 12 秒）打掉 517 hp = **43 hp/s**，
是回复速度的 11 倍，所以赢得干脆。差别不在策略的确定性，而在于
**对手会不会主动送上门** —— 训练分布里的 Dennis 是一直往 Davis 身上扑的。

所以 `--stochastic` 是个有用的诊断开关（能说明卡死是确定性造成的），
但**不是修复**。真正的修复还是 16.6 结尾那条：把坐标归一化改对，重新训练。

---

## 17. DQN 能不能跑、要多久：截断 bug 修复 + 10 分钟实测标定（2026-11-07）

### 17.1 现象

PPO 重训到 1.2M 步（`logs/PPO_15`）终于第一次打出伤害（打掉 311 / 挨打 522，
0 胜），但要判断换 `train_DQN.py` 值不值得，得先回答两件事：**它能不能跑对**、
**5000 局到底要几天**。原来的「约 4 天」里，「平均局长」是估的、「DQN 步速」是
从一次间接测量推的，两个都是误差源。

### 17.2 原因 / 修法：`train_DQN.py` 把「截断」当成了「死亡」

```python
next_obs, reward, terminated, truncated, info = env.step(action)
done = terminated or truncated
buffer.push(obs, action, reward, next_obs, float(done))   # ← 原来
```

`update()` 里 `target = r + GAMMA * next_q * (1 - done)`。把 `truncated`
（3000 帧步数上限、以及 `env.step()` 的帧超时）也写成 `done=1`，等于告诉网络
「走到 3000 帧之后世界就结束了，后面没有任何收益」。后果是**所有需要拖时间的
策略（躲开等破绽、后手反击）被系统性低估，速攻换血被高估** —— 正好和 PPO
现在卡住的那个坑是同一种病。

改法只动一行（循环里的 `done` 保持不变，它还要用来退出 while）：

```python
buffer.push(obs, action, reward, next_obs, float(terminated))
```

`evaluate()` 里的 `done = terminated or truncated` 是循环控制，不动。

### 17.3 标定方法：`lf2-rl/dqn_bench.py`（新增）

复用 `train_DQN.py` 的 `QNetwork / ReplayBuffer / select_action / update /
get_epsilon`，主循环与其真正的主循环逐行一致（reset → ε-greedy → step → push →
过 warmup 后每步 update → 每 1000 步同步 target）。三处不同：不做周期评估、
不写任何文件、跑满 `--minutes` 就停。分项计时等帧 / 前向 / 反向 / 页面重载。

```bash
python dqn_bench.py              # 10 分钟
python dqn_bench.py --minutes 3  # 冒烟
```

### 17.4 实测（本机，10.0 分钟，15 局，26,541 步）

| 量 | 实测值 |
|---|---|
| 端到端步速 `R_e2e` | **44.23 步/秒** |
| 纯步速 `R_step`（扣掉重载） | 45.37 步/秒 |
| PPO_15 步速（checkpoint 时间戳反推） | 41.70 步/秒 |
| 平均局长（14 局完整 + 1 局截半 = 15） | 1769 帧（14 局完整时 1823 帧） |
| 局长分布 | 最短 636 / 中位 1610 / 最长 3000 |
| 平均每次 `update()` | 2.90 毫秒 |
| 梯度更新次数 | 25,542（≈每步一次） |

时间构成：

| 项 | 秒 | 占比 |
|---|---|---|
| 等帧 `env.step()` | 509.0 | 84.8% |
| 更新 `update()` | 74.1 | 12.4% |
| 重载 `env.reset()` | 15.1 | 2.5% |
| 选动作 `select_action()` | 1.5 | 0.2% |

**关键读数**：一次循环 = 等帧 19.2 ms + 更新 2.9 ms + 选动作 0.06 ms = **22.2 ms**，
正好等于浏览器的一帧。也就是说**梯度更新被等帧的余量完整吃掉了** ——
把 update 去掉，步速仍然是 45 步/秒；就算 update 慢到 15 ms 也一样。
所以 DQN 慢不慢**和 CPU 线程数无关，瓶颈完全在游戏自身的帧率**。
实测 DQN 甚至比 PPO 快 6%（44.2 vs 41.7），因为 SB3 每 2048 步要做 5 个 epoch、
32 个 minibatch 的更新，平均下来比 DQN 的单步更新更重。

### 17.5 工期推算

```
总步数 = 5000 局 × L + (5000/100) 次评估 × 20 局 × L = 6000 × L
L = 1823（14 局完整实测平均）→ 10,935,000 步
总时长 = 10,935,000 / 44.23 ≈ 247,000 秒 ≈ 68.7 小时 ≈ 2.86 天
```

敏感度（局长随学到的策略变化，这是最大的不确定项）：

| 学习结果 | 平均局长 | 总时长 |
|---|---|---|
| 学会赢（一局 ~13 秒） | ~600 | **0.9 天** |
| 和随机策略差不多（实测） | ~1820 | **2.9 天** |
| 学会僵持（每局跑满上限） | 3000 | **4.7 天** |

注意这个标定用的策略 ε 从 1.0 只降到 0.62（10 分钟 ≈ 2.6 万步），**基本上是随机
乱按**，15 局输了 14 局、Dennis 血几乎不掉（大多剩 300–500）。所以它测的是
**速度和局长**，不是学习效果。

### 17.6 结论

1. **管道是通的**，修完 bootstrap 之后 10 分钟跑了 15 完整局，没有崩、没有卡死。
2. **工期 ≈ 2.9 天**（区间 0.9–4.7 天），不是 4 天。
3. **步速不是瓶颈**，和 PPO 同量级；瓶颈是游戏 45 帧/秒的实时时钟。
4. 5000 局 ≈ 1090 万步，是 PPO 这轮 120 万步的 **8.8 倍经验**；
   DQN 还是 off-policy（每条经验回放约 85 次），且 Q-learning 优化的
   本身就是 greedy 策略 —— 部署时取 argmax 和训练目标一致，没有 PPO
   那个「训练随机策略、部署取 argmax」的结构性错位。
5. **风险**：`EPSILON_DECAY = 50_000` 在整个 1090 万步里只占 0.46%，
   ε 在第 5 万步就压到 0.05，之后几乎没有探索。这是一次 3 天的单次下注，
   中途发现锁死就白费。真要跑，建议先把 `EPSILON_DECAY` 提到总量的
   10%–20%（约 1–2M 步），但那属于超参改动，本次未实施。


---

## 18. DQN 的训练曲线、离线评估、断点续训（2026-11-08）

### 18.1 现象

PPO 有曲线（`logs/PPO_N`）也有 `eval.py`，DQN 两样都没有：
`train_DQN.py` 只能 `print()` 到终端；`evaluate()` 埋在训练循环里只返回两个数；
`eval.py` 只认 `PPO.load()` 的 `.zip`，`pve.py:117` 还显式拒绝 `.pt`；
checkpoint 只存裸 `state_dict`，**2.9 天的长跑崩一次就得从头再来**。

于是「DQN 到底学到什么程度」这个问题没有答案：既没有趋势，也没有绝对水平，还不能中途续。

### 18.2 改了什么

| 文件 | 状态 | 改动 |
|---|---|---|
| `lf2-rl/rl_metrics.py` | 新增 | 观测下标（`IDX_D_HP/IDX_E_HP/IDX_DX`）、`ATTACK_ACTIONS`、`RANDOM_ATTACK_RATE`、`wilson()`、`action_entropy()`、`action_name()`。**训练内评估和 `eval.py` 共用这一份**，两边口径不会漂 |
| `lf2-rl/train_DQN.py` | 大改 | 接 `SummaryWriter` 写 `logs/DQN_N`；`evaluate()` 返回富指标 dict；checkpoint 改富格式（含 optimizer/total_steps/episode）并兼容旧裸 `state_dict`；`--resume` 续训；`Ctrl+C` / 异常时先存 `interrupted.pt`；`episodes.csv` / `eval.csv` / `history.json` 二次记录；`best_model.pt` 仍是裸 `state_dict`（§13.6 的加载片段保持可用） |
| `lf2-rl/eval.py` | 改 | `--algo {auto,ppo,dqn}`（默认 auto，按 `.zip`/`.pt` 判断）；`QNetworkPolicy` / `PPOPolicy` 适配层，跑局统计那段代码一行没动；`--temperature`、`--json`；`latest_dqn_checkpoint()` |
| `lf2-rl/pve.py` | 改 | 删掉「只支持 PPO 的 .zip」，改用同一个适配层，DQN 的 `.pt` 也能真人对打 |
| `lf2-rl/smoke_dqn.py` | 新增 | **不开浏览器的**端到端冒烟：假环境跑真训练循环 + 真 TensorBoard + 真续训，30 秒出结果 |
| `lf2-rl/dqn_bench.py` | 不动 | 它的导入契约（`QNetwork/ReplayBuffer/update/EPISODES/EVAL_EPISODES` 等）一条没破；`EVAL_EPISODES` 默认 20→10 会让它的工期推算同步变小，这是预期 |

几个设计取舍：

- **TensorBoard 用 `SummaryWriter` 而不是 SB3 的 `configure()`**：DQN 和 SB3 没有耦合，
  `SummaryWriter` 三行就能用，而且能显式控制 `global_step`、`flush()`、`close()`。
  所有标量的横轴统一是 `total_steps`，才能和 PPO 的 `total_timesteps` 叠在一张图上。
- **每局写一次，不逐步写**：1090 万步 × 30 个 tag 会产生几亿个点，事件文件上 GB、CPU 也有开销。
  按局聚合后每 tag 5000 个点，画三天的曲线足够了。
- **评估从 20 局降到 10 局**（`LF2_EVAL_EPISODES` 可改回）：评估会实打实占用浏览器墙钟，
  20 局要 14 小时（总工期的 20%），10 局省一半。代价是 CI 变宽（p=0.1 时半宽 0.137 → 0.193），
  所以**报告里引用绝对胜率时用 `eval.py --episodes 100` 单独跑**，训练曲线只看趋势。
  另外胜率相同时改为比 `avg_reward`，避免 best 停在 10% 的噪声高点。
- **`best_win_rate` 初值陷阱**：原来严格 `>` 且初值 0.0，第一轮评估如果一局没赢，
  `best_model.pt` 根本不会生成（`eval.py` 默认找模型就落空）。现在改成 `(win_rate, avg_reward)` 元组比较，
  `avg_reward` 初值 `-inf`，所以 0% 胜率也会落一个 best。
- **存档先写 `.tmp` 再 `os.replace`**：三天训练里 `best_model.pt` 是唯一值得留的东西，
  中途被 Ctrl+C 打断不能留下半截文件。
- **和评估节奏解耦**：原来 `ep{N}.pt` 嵌在 `if episode % EVAL_EVERY == 0` 里面，
  把 `LF2_EVAL_EVERY` 调大（比如 200）会让 500 局的存档静默失效。现在存档独立判断。

### 18.3 断点续训怎么用

```bat
REM 中途 Ctrl+C 后（会存 checkpoints/interrupted.pt）
python train_DQN.py --resume checkpoints/interrupted.pt

REM 或者直接接着最近一次的 last.pt（每 100 局自动写）
python train_DQN.py --resume

REM 想另开一条 TensorBoard 曲线
python train_DQN.py --resume checkpoints/last.pt --new-run
```

- 恢复：网络权重、target 网络、optimizer 状态、`total_steps`、`episode`、`best_win_rate`。
- **ε 不用存**：`get_epsilon(total_steps)` 是纯函数，恢复步数就自动接着衰减 —— 这是续训代价极低的原因。
- **经验池不存**：续训后前 64 步不更新，之后恢复正常。
- 默认写回**原来那个** `logs/DQN_N`，所以曲线是连续的（同目录多个事件文件按 step 合并）。
- 从旧的裸 `state_dict`（比如 `best_model.pt`）续训也能跑，但 optimizer 动量重新开始累计，会打印提示。

### 18.4 验收记录（本机实测，2026-11-08）

**A. 无浏览器冒烟**（`python smoke_dqn.py`，30 秒）

| 检查项 | 结果 |
|---|---|
| 真训练循环 + 真 TB 写入器（假环境每步瞬时） | ✅ 两轮共 10 局，51 个标量，22 个 `action_rate` |
| `last.pt` / `best_model.pt` / `ep{N}.pt` / `final_model.pt` 全部落盘 | ✅ |
| `best_model.pt` 仍是裸 `state_dict` 且能被 `load_state_dict` 吃下 | ✅ |
| `episodes.csv` / `eval.csv` / `history.json` | ✅ |
| 续训接着局数跑、`total_steps` 不清零、事件文件落在同一目录 | ✅（6 局 → 10 局，558 → 934 步，2 个事件文件） |
| 旧格式裸 `state_dict` 被识别成 `format=1` | ✅ |

**B. 真游戏短程**（`serve.py:8010` + `launch_browser.py:8010`，`LF2_MAX_STEPS=400`）

| 检查项 | 结果 |
|---|---|
| `train_DQN.py --episodes 2 --eval-every 2` | ✅ 2 局各 400 帧截断，33 fps，评估 2 局 |
| 训练曲线 | ✅ 51 个标量；`explore/epsilon` 0.9924→0.9848；`rollout/ep_win_rate` 0；`time/fps` 33 |
| 训练期出招占比 | ✅ 0.445（随机基线 0.455）—— 未训练时确实接近随机 |
| `eval.py --algo dqn --model checkpoints/best_model.pt` | ✅ 完整报告 + `--json`；动作分布 `down` 75.4% / `right` 24.6% |
| `eval.py`（不带 `--algo`，auto→PPO）回归 | ✅ 行为与改动前一致（出招 89.8%，距离 215px） |
| `train_DQN.py --resume` 真续训 | ✅ 「已完成 2 局 / 800 步，从第 3 场开始」，写回同一个 `DQN_2`，`episodes.csv` 从 2 行续到 4 行，`total_steps` 800→1600 |
| `dqn_bench.py` 导入契约 | ✅ `import train_DQN, dqn_bench` 正常，`EPISODES=5000 / EVAL_EPISODES=10` |

**C. 顺手修掉的一个坑**

| 现象 | 原因 | 修法 |
|---|---|---|
| `python train_DQN.py > log.txt` 直接 `UnicodeEncodeError: 'gbk' codec can't encode '🏆'` | Windows 重定向时 stdout 用 GBK，`🏆` 编不出来。三天长跑基本都会重定向，等于一启动就崩 | `train()` / `eval.py` 启动时 `sys.stdout.reconfigure(encoding='utf-8', errors='replace')` |

### 18.5 还没做的

1. **PPO 没有断点续训**（SB3 的 `save` 只存策略，`train.py` 没有 `--resume`）。
2. **Replay buffer 不进存档**（多几十 MB 且慢）。真要的话在 `save_checkpoint` 里把
   `deque` 拼成张量存进 `ep{N}.pt` 即可，`update()` 的守卫已经能容忍空池。
3. **`EPSILON_DECAY=50000` 没动**（§17.6 指出它只占总量 0.46%，是个真风险）。
   这次只加了 `LF2_EPSILON_DECAY` 覆盖开关和 `explore/epsilon` 曲线，把问题**变得可见**，
   超参本身该不该改是实验设计决定，不夹在日志改动里偷改。
