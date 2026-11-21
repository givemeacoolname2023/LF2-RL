// RL Bridge
var rl_ws = null;
var rl_action = 0;
var rl_controller = null; // 将在 match 创建后设置
// 每次页面载入生成一个新的 session id。env.py 用它判断“页面真的重开过”，
// 不再依赖 time 很小（页面从载入跑到 time=10 只要 0.2 秒，重连根本来不及）
var rl_session_id = Date.now().toString(36) + '-' + Math.random().toString(36).slice(2, 8);
var rl_last_damage_t = 0;  // 上次造成伤害的时间
var rl_last_dennis_hp = 500;  // 上次 dennis 的 hp
var rl_reloading = false;  // 防止重复 reload

// Headless mode: ?headless 参数跳过 canvas 渲染，大幅提速
var rl_headless = (window.location.search.indexOf('headless') !== -1);
var rl_step_mode = new URLSearchParams(window.location.search).has('rlstep');

// 把页面 URL 里的 diff / seed 换成指定值，用来响应 env.py 的切换难度 / 换种子请求。
// 只改指定的那个 token：不能用 URLSearchParams，因为它会输出 debug= / headless=，
// 而 manager.js:254 的 `else if (param.debug)` 靠真值判断，空串会让页面退回主菜单，
// 训练端就再也收不到状态了。
function rl_set_url_param(base, name, value) {
  var re = new RegExp('([?&])' + name + '(?:=[^&]*)?(?=&|$)')
  if (re.test(base)) {
    return base.replace(re, '$1' + name + '=' + value)
  }
  return base + (base.indexOf('?') >= 0 ? '&' : '?') + name + '=' + value
}

// level 为 null/非法 → 不动 diff；seed / bg 同理。
function rl_url_with_nav(level, seed, bg) {
  var href = window.location.href
  var hashAt = href.indexOf('#')
  var hash = hashAt >= 0 ? href.slice(hashAt) : ''
  var base = hashAt >= 0 ? href.slice(0, hashAt) : href
  if (level === 0 || level === 1 || level === 2) {
    base = rl_set_url_param(base, 'diff', level)
  }
  if (typeof seed === 'number' && isFinite(seed)) {
    base = rl_set_url_param(base, 'seed', seed)
  }
  if (typeof bg === 'number' && isFinite(bg)) {
    base = rl_set_url_param(base, 'bg', bg)
  }
  return base + hash
}

function rl_url_with_diff(level) {
  return rl_url_with_nav(level, null, null)
}

var ACTION_MAP = [
  [],                        // 0: 不动
  ['left'],                  // 1: 左
  ['right'],                 // 2: 右
  ['jump'],                  // 3: 跳
  ['down'],                  // 4: 纵深向下（朝镜头），不是蹲
  ['att'],                   // 5: 攻击
  ['def'],                   // 6: 防御
  ['left', 'att'],           // 7: 左+攻击
  ['right', 'att'],          // 8: 右+攻击
  ['jump', 'att'],           // 9: 跳+攻击(Shoryuken方向)
  ['down', 'att'],           // 10: 下+攻击（combo 优先级下等同普通攻击）
  ['up'],                    // 11: 纵深向上（远离镜头）。必须与 env.py 的 ACTION_KEYS 一致
  ['left', 'down'],          // 12: 左+纵深向下（斜走；con.state 两个轴同时生效）
  ['right', 'down'],         // 13: 右+纵深向下
  ['left', 'up'],            // 14: 左+纵深向上
  ['right', 'up'],           // 15: 右+纵深向上
];

function rl_combat_keys(action, match, player) {
  const agent = match?.rl_characters?.[player]
  const opponent = match?.rl_characters?.[1 - player]
  if (!opponent) return []
  const delta_x = opponent.ps.x - agent.ps.x
  const delta_z = opponent.ps.z - agent.ps.z
  const toward = delta_x < 0 ? 'left' : 'right'
  switch (action) {
    case 16: return ['def', toward, 'att']
    case 17: return agent.id === 4 ? ['def', toward, 'jump'] : [toward, 'def', 'down', 'att']
    case 18: return agent.id === 4 ? [toward, 'def', 'jump', 'att'] : [toward, 'def', 'up', 'att']
    case 19: return [toward, 'def', 'up', 'jump']
    case 20: {
      const keys = []
      if (Math.abs(delta_x) > 30) keys.push(toward)
      if (Math.abs(delta_z) > 4) keys.push(delta_z < 0 ? 'up' : 'down')
      return keys
    }
    case 21: return [toward, 'att']
    default: return []
  }
}

function rl_inject_action(action, match, player = 0) {
  // 触屏控制器没有 buf，直接跳过，避免每帧抛异常把整个游戏循环卡死
  var controller = player === 1 ? match.rl_characters[1].con : (match?.simulation ? match.rl_controller : rl_controller);
  if (!controller || !controller.buf) return;
  // 只有训练端（env.py）连着时才注入。注入的「全键抬起」会清掉玩家按住的方向键
  // （角色的移动/连招读的是 con.state 电平，见 LF/character.js 与 core/combodec.js），
  // 没有训练连接时完全不干预，手动玩的手感与官方页面一致。
  if (!rl_step_mode && (!rl_ws || rl_ws.readyState !== 1)) return;
  // 比赛已结束：不再注入。注入的 att/jump 会经 controller.child 传到 match.key，
  // 在 gameover 状态下触发 F4() 退回主菜单，导致训练端再也收不到状态。
  var active = match || window.rl_match;
  if (active && active.gameover_state) return;
  var keys = ACTION_MAP[action] || rl_combat_keys(action, active, player);
  var all_keys = ['left', 'right', 'up', 'down', 'jump', 'att', 'def'];

  // 正确方式：往 buf 里 push 事件，fetch() 时才会被游戏读到
  // 先 push 所有键的 up 事件（清除之前的状态）
  for (var i = 0; i < all_keys.length; i++) {
    controller.buf.push([all_keys[i], 0]);
  }
  // 再 push 当前动作的 down 事件
  for (var i = 0; i < keys.length; i++) {
    controller.buf.push([keys[i], 1]);
  }
}


(function () {
  if (rl_step_mode) return;
  function connect() {
    rl_ws = new WebSocket('ws://localhost:8765');
    rl_ws.onopen = function () { console.log('RL bridge connected'); };
    rl_ws.onmessage = function (e) {
      var msg;
      try { msg = JSON.parse(e.data); } catch (err) { return; }
      // 后端要求重开一局（回合结束/超时/页面停在主菜单时由 env.py 主动请求）
      if (msg.cmd === 'reload') {
        // 带 diff / seed 时顺带改写 URL 再导航：新页面 start_debug() 读 param.diff 选对手 AI，
        // manager.js 读 param.seed 播种（决定出生点和 AI 的随机序列）。
        // 两个都不带时行为与原来完全一致（纯 reload，沿用页面上的 URL）。
        var level = (msg.diff === undefined || msg.diff === null)
          ? NaN : parseInt(msg.diff, 10)
        var seed = (msg.seed === undefined || msg.seed === null)
          ? NaN : parseInt(msg.seed, 10)
        var bg = (msg.bg === undefined || msg.bg === null)
          ? NaN : parseInt(msg.bg, 10)
        var hasLevel = (level === 0 || level === 1 || level === 2)
        var hasSeed = isFinite(seed)
        var hasBg = isFinite(bg)
        if (hasLevel || hasSeed || hasBg) {
          var target = rl_url_with_nav(hasLevel ? level : null,
            hasSeed ? seed : null, hasBg ? bg : null)
          console.log('[RL] 重开一局 diff=' + (hasLevel ? level : '-')
            + ' seed=' + (hasSeed ? seed : '-')
            + ' bg=' + (hasBg ? bg : '-') + ' -> ' + target)
          window.location.replace(target);   // 不留 history，效果等同一次 reload
        } else {
          window.location.reload();
        }
        return;
      }
      rl_action = msg.action || 0;
      rl_inject_action(rl_action);
    };
    rl_ws.onclose = function () {
      console.log('RL bridge disconnected, retrying...');
      setTimeout(connect, 2000);
    };
  }
  connect();
})();

// 保活：持有 Web Lock 的页面不会被浏览器的 Page Lifecycle 冻结。
// 这只能防止“冻结”，后台标签页的定时器降频仍需浏览器参数配合
// （见 lf2-rl/launch_browser.py 的 --disable-background-timer-throttling）。
if (!rl_step_mode && navigator.locks && navigator.locks.request) {
  navigator.locks.request('lf2_rl_keepalive', function () {
    return new Promise(function () { /* 永不 resolve，锁一直持有 */ });
  });
}



/*\
 * match
 * a match hosts a game.
 * a match is a generalization above game modes (e.g. VSmode, stagemode, battlemode)
\*/


define(['core/util', 'core/controller', 'LF/sprite-select',
  'LF/network', 'LF/factories', 'LF/scene', 'LF/background', 'LF/AI', 'third_party/random', 'LF/util',
  'LF/global'],
  function (Futil, Fcontroller, Fsprite,
    network, factory, Scene, Background, AI, Random, util,
    Global) {
    const GA = Global.application
    /*\
     * match
     [ class ]
     |  config =
     |  {
     |  manager,//the game manager
     |  state,  //the state machine handling various events in a match
     |  package //the content package
     |  }
    \*/
    function match(config) {
      const $ = this
      $.manager = config.manager
      $.state = config.state
      $.data = config.package.data
      $.simulation = config.simulation
      $.sound = config.manager.sound
      $.spec = $.data.properties.data
      $.time
    }

    match.prototype.create = function (setting) {
      const $ = this
      if (!$.simulation) window.rl_match = $
      let object_ids = []
      const AI_ids = []
      for (var i = 0; i < setting.player.length; i++) {
        // (lazy) now load all characters and associated data files
        object_ids.push(setting.player[i].id)
        object_ids = object_ids.concat(Futil.extract_array(util.select_from($.data.object, { id: setting.player[i].id }).pack, 'id').id)
        for (const index in $.data.object) {
          if ($.data.object[index].id == setting.player[i].id && $.data.object[index].AI) {
            AI_ids.push($.data.object[index].AI)
          }
        }
        if (setting.player[i].controller.type === 'AIscript') {
          AI_ids.push(setting.player[i].controller.id)
        }
      }
      if (!setting.set) { setting.set = {} }

      $.gameover_state = false
      $.randomseed = $.new_randomseed()
      $.create_scenegraph()
      $.control = $.create_controller(setting.control)
      $.functionkey_control = setting.control
      if ($.functionkey_control &&
        $.functionkey_control.restart) {
        $.functionkey_control.restart()
      }
      if ($.manager.panel_layer) {
        $.panel = []
        for (var i = 0; i < 8; i++) $.panel[i] = {}
      }
      $.overlay_message('loading')
      $.tasks = [] // pending tasks
      $.AIscript = []
      if ($.manager.summary) {
        $.manager.summary.hide()
      }
      $.manager.canvas.render()

      let already = false
      this.data.load({
        object: object_ids,
        background: setting.background ? [setting.background.id] : [],
        AI: AI_ids
      }, function () {  // when all necessary data files are loaded
        $.create_background(setting.background)
        $.create_effects()
        if (setting.player) {
          // $.create_characters(setting.player, { pane: true })
          $.create_characters(setting.player, { pane: true })
          // 设置 RL controller（Player 0 = Davis）
          const characters = Object.values($.character)
          $.rl_characters = characters
          if ($.simulation || rl_step_mode) {
            $.rl_damage = 0
            $.rl_center_damage = 0
          }
          if ($.simulation) {
            $.rl_controller = characters[0].con
          } else if (characters.length > 0) {
            rl_controller = characters[0].con
          }





        }
        if (setting.set.weapon) {
          $.drop_weapons(setting.set.weapon)
        }

        if (rl_step_mode) {
          onready()
        } else {
          Fsprite.masterconfig_set('onready', onready)
          setTimeout(function () { onready() }, 8000) // assume it is ready after 8 seconds
        }
      })
      function onready() {
        if (!already) { // all loading finished
          already = true
          if ($.manager.overlay_mess) {
            $.manager.overlay_mess.hide()
          }
          if (setting.set.demo_mode) {
            $.demo_mode = true
            $.overlay_message('demo')
          }
          $.create_timer()
        }
      }
    }

    match.prototype.destroy = function () {
      const $ = this
      $.time.paused = true
      $.destroyed = true
      if ($.time.timer !== null) network.clearInterval($.time.timer)

      // destroy all objects
      $.for_all('destroy')
      $.background.destroy()
      if ($.panel) {
        for (let i = 0; i < $.panel.length; i++) {
          if ($.panel[i].hp) {
            $.panel[i].hp.remove()
            $.panel[i].hp_bound.remove()
            $.panel[i].mp.remove()
            $.panel[i].mp_bound.remove()
            $.panel[i].spic.remove()
          }
        }
      }
    }

    match.prototype.log = function (mes) {
      console.log(this.time.t + ': ' + mes)
    }

    match.prototype.create_non_player_characters = function (players) {
      const $ = this
      $.tasks.push({
        task: 'create_non_player_characters',
        players,
      })
    }

    match.prototype.create_transform_character = function (player) {
      const $ = this
      $.tasks.push({
        task: 'create_transform_character',
        player,
      })
    }

    match.prototype.create_multiple_objects = function (opoint, parent, number, vz) {
      const $ = this
      $.tasks.push({
        task: 'create_multiple_objects',
        number: number,
        parent: parent,
        opoint: opoint,
        team: parent.team,
        pos: parent.mech.make_point(opoint),
        z: parent.ps.z,
        dir: parent.ps.dir,
        dvz: parent.dirv() * 2,
        vz: vz
      })
    }

    match.prototype.create_object = function (opoint, parent) {
      const $ = this
      $.tasks.push({
        task: 'create_object',
        parent: parent,
        opoint: opoint,
        team: parent.team,
        pos: parent.mech.make_point(opoint),
        z: parent.ps.z,
        dir: parent.ps.dir,
        dvz: parent.dirv() * 2
      })
    }

    match.prototype.destroy_object = function (obj) {
      const $ = this
      $.tasks.push({
        task: 'destroy_object',
        obj: obj
      })
    }

    // all methods below are considered private

    match.prototype.create_scenegraph = function () {
      const $ = this
      $.scene = new Scene()
      for (const objecttype in factory) {
        $[objecttype] = {}
      }
    }

    match.prototype.create_timer = function () {
      const $ = this
      $.time =
      {
        t: 0,
        paused: false,
        timer: null,
        $fps: rl_step_mode ? null : util.div('fps')
      }
      if (!$.time.$fps) $.calculate_fps = function () { }
      if (!rl_step_mode) {
        $.time.timer = network.setInterval(function () { return $.frame() }, 1000 / Global.gameplay.framerate / 1.5)
      }
    }

    match.prototype.rl_step = function (action, frames, opponent_action = null) {
      this.rl_opponent_action = opponent_action
      if (this.simulation) {
        this.rl_action = action
        for (let index = 0; index < frames && !this.gameover_state; index++) this.frame()
        return
      }
      rl_action = action
      let state
      for (let index = 0; index < frames; index++) {
        state = this.frame()
        if (state.davis.hp <= 0 || state.dennis.hp <= 0) break
      }
      return state
    }

    match.prototype.frame = function () {
      const $ = this
      const [agent, opponent] = $.rl_damage === undefined ? [] : $.rl_characters
      const previous_hp = opponent && Math.max(0, opponent.health.hp)
      // 每帧在 fetch 之前注入动作，保证攻击键被正确检测
      // 只在训练端连着时注入（否则连调试日志也会刷屏，且会干扰手动操作）
      if ($.simulation) {
        rl_inject_action($.rl_action, $)
      } else if (rl_step_mode) {
        rl_inject_action(rl_action)
      } else if (rl_controller && rl_ws && rl_ws.readyState === 1) {
        rl_inject_action(rl_action)
        // 每30帧打印一次调试信息
        if ($.time.t % 30 === 0) {
          var action_names = ['idle', 'left', 'right', 'jump', 'down', 'att', 'def', 'left+att', 'right+att', 'jump+att', 'down+att']
          var dbg_state = $.game_state ? $.game_state() : null
          if (dbg_state) {
            console.log(
              '[t=' + $.time.t + '] action=' + action_names[rl_action] +
              ' | davis hp=' + dbg_state.davis.hp +
              ' | dennis hp=' + dbg_state.dennis.hp +
              ' | dx=' + Math.round(dbg_state.davis.x - dbg_state.dennis.x) +
              ' | att_key=' + (rl_controller.buf.length > 0 ? '1' : '0')
            )
            // 超时检测已关闭，依靠 env.py 的 MAX_EPISODE_STEPS 控制每局长度
            // 帧率检测已关闭
          }
        }
      }
      if (($.simulation || rl_step_mode) && $.rl_opponent_action != null) {
        rl_inject_action($.rl_opponent_action, $, 1)
      }
      if ($.control) { $.control.fetch() }
      if (!$.time.paused || $.time.paused === 'F2') {
        for (const i in $.character) {
          $.character[i].con.fetch()
          $.character[i].combodec.frame()
        }
        if ($.destroyed) {
          return
        }
        // RL controller 자동 설정
        if (!$.simulation && !rl_controller) {
          for (var k in $.character) {
            // if ($.character[k].data.id === 11) {
            if ($.character[k].data.bmp.name === 'Davis') {
              rl_controller = $.character[k].con;
              console.log('RL controller set to Davis');
              break;
            }
          }
        }

        $.TU_trans()
        if (opponent) {
          const damage = previous_hp - Math.max(0, opponent.health.hp)
          if (damage > 0) {
            const width = $.background.width
            const midpoint = (agent.ps.x + opponent.ps.x) / 2
            const distance = Math.min(1, Math.abs(midpoint - width / 2) / (width / 2))
            $.rl_damage += damage
            $.rl_center_damage += damage * (1 - distance ** 2)
          }
        }
        if ($.time.t === 0) {
          $.match_event('start')
        }
        $.time.t++
        if (!rl_headless) { $.manager.canvas.render() }
        $.calculate_fps()

        if ($.time.paused === 'F2') {
          $.time.paused = true
        }
      } else {
        if ($.time.$fps) {
          $.time.$fps.value = 'paused'
        }
      }
      // return $.game_state()
      if ($.simulation) return
      const state = $.game_state()
      if (rl_ws && rl_ws.readyState === 1 && state) {
        rl_ws.send(JSON.stringify(state))
        // 检测游戏结束，自动重启
        // 超时检测：超过200帧没有造成伤害就重开
        if (!window.rl_last_dennis_hp) window.rl_last_dennis_hp = state.dennis.hp
        if (!window.rl_no_damage_frames) window.rl_no_damage_frames = 0
        if (state.dennis.hp < window.rl_last_dennis_hp) {
          window.rl_no_damage_frames = 0  // dennis掉血了，重置计数
          window.rl_last_dennis_hp = state.dennis.hp
        } else {
          window.rl_no_damage_frames++
        }
        var timeout = false  // 关闭超时重开，让每局打完

        if ((state.davis.hp <= 0 || state.dennis.hp <= 0 || timeout) && !window.rl_reloading) {
          if (timeout) console.log('超时重开！frames=' + window.rl_no_damage_frames)
          window.rl_reloading = true
          window.rl_no_damage_frames = 0
          window.rl_last_dennis_hp = null
          setTimeout(function () {
            rl_controller = null
            window.rl_reloading = false
            window.location.reload()
          }, 500)
        }
      }
      return state
    }

    // match.prototype.game_state = function () {
    //   const $ = this
    //   const d = {}
    //   d.time = $.time.t
    //   for (const i in $.character) {
    //     const c = $.character[i]
    //     d[i] = [c.ps.x, c.ps.y, c.ps.z, c.health.hp, c.health.mp]
    //   }
    //   return d
    // }
    match.prototype.game_state = function () {
      const $ = this
      const chars = Object.values($.character)
      if (chars.length < 2) return null

      // 第一帧打印所有角色名字，方便调试
      if (!rl_step_mode && $.time.t === 1) {
        chars.forEach(function (c) {
          console.log('角色 bmp.name:', c.data.bmp.name, '| team:', c.team)
        })
      }

      // 按队伍区分：team=1 是 Davis(RL), team=2 是 Dennis(AI)
      let davis = null, dennis = null
      for (var i = 0; i < chars.length; i++) {
        if (chars[i].team === 1) davis = chars[i]
        if (chars[i].team === 2) dennis = chars[i]
      }
      if (!davis || !dennis) return null

      return {
        session: rl_session_id,
        time: $.time.t,
        // 由 manager.js 的 start_debug 写进 window，用来让训练/评估日志能写明在打谁
        difficulty: window.rl_difficulty,
        ai: window.rl_ai_name,
        // 本局实际生效的随机种子（manager.js 从 ?seed= 读，缺省 824163532）；
        // env.py 用它校验「要的种子」和「页面真的用了的种子」是否一致
        seed: window.rl_seed,
        // 本局的背景场地在 bg_list 里的下标（?bg= 可钉住）；各场地宽度差 4 倍，
        // 观测缩放只对 ~800 宽的场地成立，所以评估时必须记录/校验它
        bg: window.rl_bg,
        damage: $.rl_damage,
        center_damage: $.rl_center_damage,
        davis: {
          character: davis.data.bmp.name,
          x: davis.ps.x, y: davis.ps.y, z: davis.ps.z,
          hp: davis.health.hp, mp: davis.health.mp,
          frame: davis.frame.N, dir: davis.ps.dir
        },
        dennis: {
          character: dennis.data.bmp.name,
          x: dennis.ps.x, y: dennis.ps.y, z: dennis.ps.z,
          hp: dennis.health.hp, mp: dennis.health.mp,
          frame: dennis.frame.N, dir: dennis.ps.dir
        }
      }
    }

    match.prototype.TU_trans = function () {
      const $ = this
      $.emit_event('transit')
      $.process_tasks()
      $.emit_event('TU')
      $.background.TU()
      $.sound.TU()
      $.show_hp()
      $.check_gameover()
      if (rl_step_mode && $.gameover_state) return
      const AI_frameskip = 3 // AI script runs at a lower framerate, and is still very reactive
      if ($.time.t % AI_frameskip === 0) {
        for (let i = 0; i < $.AIscript.length; i++) {
          $.AIscript[i].TU()
        }
      }
    }

    match.prototype.match_event = function (E) {
      const $ = this
      if ($.state && $.state.event) $.state.event.call(this, E)
    }

    match.prototype.emit_event = function (E) {
      const $ = this
      $.match_event(E)
      $.for_all(E)
    }

    match.prototype.for_all = function (oper) {
      const $ = this
      for (const objecttype in factory) {
        for (const i in $[objecttype]) {
          $[objecttype][i][oper]()
        }
      }
    }

    match.prototype.process_tasks = function () {
      const $ = this
      for (let i = 0; i < $.tasks.length; i++) {
        $.process_task($.tasks[i])
      }
      $.tasks.length = 0
    }
    match.prototype.process_task = function (T) {
      const $ = this
      switch (T.task) {
        case 'create_object':
          if (T.opoint.oid) {
            const OBJ = util.select_from($.data.object, { id: T.opoint.oid })
            if (!OBJ) {
              console.error('Object', T.opoint.oid, 'not exists')
              break
            }
            const config =
            {
              match: $,
              team: T.team
            }
            var obj = new factory[OBJ.type](config, OBJ.data, T.opoint.oid)
            obj.init(T)
            var uid = $.scene.add(obj)
            $[obj.type][uid] = obj
          }
          break
        case 'create_multiple_objects':
          let vz_array = []
          let max_number = 0
          max_number = Math.floor(T.number / 2)
          if (T.number % 2 == 1) { // calculate array of vz for ajusting ninja star or arrow's z coord
            for (let temp1 = -1 * max_number; temp1 <= max_number; temp1++) {
              vz_array.push(temp1 * T.vz)
            }
          } else {
            for (let temp1 = -1 * max_number; temp1 <= max_number; temp1++) {
              if (temp1 != 0) {
                vz_array.push(temp1 * T.vz)
              }
            }
          }
          for (vz of vz_array) {
            if (T.opoint.oid) {
              const OBJ = util.select_from($.data.object, { id: T.opoint.oid })
              if (!OBJ) {
                console.error('Object', T.opoint.oid, 'not exists')
                break
              }
              const config =
              {
                match: $,
                team: T.team
              }
              var obj = new factory[OBJ.type](config, OBJ.data, T.opoint.oid)
              obj.init(T)
              obj.ps.vz = vz
              if (T.dir === 'left') {
                obj.ps.vx += Math.abs(vz)
              } else {
                obj.ps.vx -= Math.abs(vz)
              }
              var uid = $.scene.add(obj)
              $[obj.type][uid] = obj
            }
          }
          break
        case 'create_non_player_characters':
          $.create_characters(T.players, { pane: false })
          break
        case 'create_transform_character':
          $.create_characters([T.player], { replace: true })
          break
        case 'destroy_object':
          var obj = T.obj
          obj.destroy()
          var uid = $.scene.remove(obj)
          delete $[obj.type][uid]
          break
      }
    }

    match.prototype.calculate_fps = function () {
      const $ = this
      const mul = 10
      if ($.time.t % mul === 0) {
        const ot = $.time.time
        $.time.time = new Date().getTime()
        const diff = $.time.time - ot
        $.time.$fps.value = Math.round(1000 / diff * mul) + 'fps'
      }
    }

    match.prototype.transform_panel = function (from_uid, to_uid) {
      const $ = this
      // ==========panel==========
      let from_index = -1
      let to_index = -1
      for (index in $.panel) {
        if (from_uid) {
          if ($.panel[index].uid === from_uid) {
            from_index = index
          }
        }
        if (to_uid) {
          if ($.panel[index].uid === to_uid) {
            to_index = index
          }
        }
      }
      if (from_index == -1) { return }
      if (to_index != -1) {
        $.panel[from_index].spic.temp_img = { 0: $.panel[from_index].spic.img[0] }
        $.panel[from_index].spic.img[0] = $.panel[to_index].spic.img[0]
      } else {
        $.panel[from_index].spic.img = $.panel[from_index].spic.temp_img
      }
    }

    match.prototype.create_characters = function (players, option) {
      const $ = this
      const char_config =
      {
        match: $,
        controller: null,
        team: 0
      }
      for (let i = 0; i < players.length; i++) {
        var player = players[i]
        const player_obj = util.select_from($.data.object, { id: player.id })
        var pdata = player_obj.data
        preload_pack_images(player_obj)
        if (option.replace) {
          player.controller.child.length = 0
        }
        const controller = setup_controller(player)
        // create character
        const char = new factory.character(char_config, pdata, player.id)
        if (controller.type === 'AIcontroller') {
          const AIcontroller = util.select_from($.data.AI, { id: player.controller.id }).data
          $.AIscript.push(new AIcontroller(char, $, controller))
        }
        // spec
        if (player.spec) {
          for (let I in player.spec) { // assign each spec into character
            assign_character_spec(char, player.spec, I)
          }
        }
        // outside spec
        // positioning
        if (player.pos) {
          char.set_pos(player.pos.x, player.pos.y, player.pos.z)
        } else {
          const pos = $.background.get_pos($.random(), $.random())
          char.set_pos(pos.x, pos.y, pos.z)
        }
        // option
        var uid
        if (option.replace) {
          uid = $.scene.replace(player.spec.replace_from, char)
          char.uid = uid
          player.spec.replace_from.destroy()
        } else {
          uid = $.scene.add(char)
        }

        $.character[uid] = char
        // pane
        if ($.panel && option.pane) {
          create_pane(i)
        }
      }
      function preload_pack_images(char) {
        for (let j = 0; j < char.pack.length; j++) {
          const obj = char.pack[j].data
          if (obj.bmp && obj.bmp.file) {
            for (let k = 0; k < obj.bmp.file.length; k++) {
              const file = obj.bmp.file[k]
              for (const m in file) {
                if (typeof file[m] === 'string' && m.indexOf('file') === 0) {
                  Fsprite.preload_image(file[m])
                }
              }
            }
          }
        }
      }
      function setup_controller(player) {
        let controller
        switch (player.controller.type) {
          case 'AIscript':
            controller = new AI.controller()
            break
          default:
            controller = player.controller
            controller.child.push($)
        }
        char_config.controller = controller
        char_config.team = player.team
        controller.sync = true
        return controller
      }
      function create_pane(i) {
        const X = $.data.UI.data.panel.pane_width * (i % 4)
        const Y = $.data.UI.data.panel.pane_height * Math.floor(i / 4)
        const spic = new Fsprite({
          canvas: $.manager.panel_layer,
          img: pdata.bmp.small,
          xy: { x: X + $.data.UI.data.panel.x, y: Y + $.data.UI.data.panel.y },
          wh: 'fit'
        })
        $.panel[i].uid = uid
        $.panel[i].name = player.name
        $.panel[i].spic = spic
        $.panel[i].hp_bound = new Fsprite({ canvas: $.manager.panel_layer })
        $.panel[i].hp_bound.set_x_y(X + $.data.UI.data.panel.hpx, Y + $.data.UI.data.panel.hpy)
        $.panel[i].hp_bound.set_w_h($.data.UI.data.panel.hpw, $.data.UI.data.panel.hph)
        $.panel[i].hp_bound.set_bgcolor($.data.UI.data.panel.hp_dark)
        $.panel[i].hp = new Fsprite({ canvas: $.manager.panel_layer })
        $.panel[i].hp.set_x_y(X + $.data.UI.data.panel.hpx, Y + $.data.UI.data.panel.hpy)
        $.panel[i].hp.set_w_h($.data.UI.data.panel.hpw, $.data.UI.data.panel.hph)
        $.panel[i].hp.set_bgcolor($.data.UI.data.panel.hp_bright)
        $.panel[i].mp_bound = new Fsprite({ canvas: $.manager.panel_layer })
        $.panel[i].mp_bound.set_x_y(X + $.data.UI.data.panel.mpx, Y + $.data.UI.data.panel.mpy)
        $.panel[i].mp_bound.set_w_h($.data.UI.data.panel.mpw, $.data.UI.data.panel.mph)
        $.panel[i].mp_bound.set_bgcolor($.data.UI.data.panel.mp_dark)
        $.panel[i].mp = new Fsprite({ canvas: $.manager.panel_layer })
        $.panel[i].mp.set_x_y(X + $.data.UI.data.panel.mpx, Y + $.data.UI.data.panel.mpy)
        $.panel[i].mp.set_w_h($.data.UI.data.panel.mpw, $.data.UI.data.panel.mph)
        $.panel[i].mp.set_bgcolor($.data.UI.data.panel.mp_bright)
      }
      function assign_character_spec(char, spec, index) {
        switch (index) {
          case 'is_npc':
            char.is_npc = spec[index]
            break
          case 'health':
            for (var I in spec[index]) {
              char.health[I] = spec[index][I]
            }
            break
          case 'dir':
            char.switch_dir(spec[index])
            break
          case 'stat':
            for (var J in spec[index]) {
              char.stat[J] = spec[index][J]
            }
            break
          case 'parent':
            char.parent = spec[index]
            break
          case 'transform_character':
            if (!char.transform_character) {
              char.transform_character = {}
            }
            for (var L in spec[index]) {
              char.transform_character[L] = spec[index][L]
            }
            break
        }
      }
    }

    match.prototype.show_hp = function () {
      const $ = this
      if ($.panel) {
        for (let i = 0; i < $.panel.length; i++) {
          if ($.panel[i].uid !== undefined) {
            const ch = $.character[$.panel[i].uid]
            let hp = Math.floor(ch.health.hp / ch.health.hp_full * $.data.UI.data.panel.hpw)
            hp_bound = Math.floor(ch.health.hp_bound / ch.health.hp_full * $.data.UI.data.panel.hpw)
            if (hp < 0) { hp = 0 }
            if (hp_bound < 0) { hp_bound = 0 }
            $.panel[i].hp.set_w(hp)
            $.panel[i].hp_bound.set_w(hp_bound)
            $.panel[i].mp.set_w(Math.floor(ch.health.mp / ch.health.mp_full * $.data.UI.data.panel.mpw))
            if (ch.effect.heal && ch.effect.heal > 0 && $.time.t % 3 == 0) {
              $.panel[i].hp.set_bgcolor($.data.UI.data.panel.hp_light)
            } else {
              $.panel[i].hp.set_bgcolor($.data.UI.data.panel.hp_bright)
            }
          }
        }
      }
    }

    match.prototype.check_gameover = function () {
      const $ = this
      if ($.simulation) {
        if ($.rl_characters.some(character => character.health.hp <= 0)) $.gameover_state = true
        return
      }
      const teams = {}
      if (!$.panel) {
        return
      }
      for (let i = 0; i < $.panel.length; i++) {
        if ($.panel[i].uid !== undefined) {
          const ch = $.character[$.panel[i].uid]
          if (ch.health.hp > 0) {
            teams[ch.team] = true
          }
        }
      }
      if (Object.keys(teams).length < 2) {
        if (!$.gameover_state) {
          $.gameover_state = $.time.t
        } else {
          if ($.time.t == $.gameover_state + 30) {
            $.gameover()
          }
        }
      } else {
        if ($.gameover_state) {
          $.gameover_state = false
          $.gameover()
        }
      }
    }

    match.prototype.gameover = function () {
      const $ = this
      if ($.gameover_state) {
        const info = []
        const teams = {}
        for (var i = 0; i < $.panel.length; i++) {
          if ($.panel[i].uid !== undefined) {
            var ch = $.character[$.panel[i].uid]
            if (ch.health.hp > 0) {
              teams[ch.team] = true
            }
          }
        }
        for (var i = 0; i < $.panel.length; i++) {
          if ($.panel[i].uid !== undefined) {
            var ch = $.character[$.panel[i].uid]
            const alive = ch.health.hp > 0
            const win = teams[ch.team]
            // [ Icon, Name, Kill, Attack, HP Lost, MP Usage, Picking, Status ]
            info.push([ch.data.bmp.small, $.panel[i].name, ch.stat.kill, ch.stat.attack, ch.health.hp_lost, ch.health.mp_usage, ch.stat.picking, (win ? 'Win' : 'Lose') + ' (' + (alive ? 'Alive' : 'Dead') + ')'])
          }
        }
        $.manager.summary.set_info(info)
        const dur = $.time.t / Global.gameplay.framerate
        $.manager.summary.set_time(new Date(dur * 1000).toISOString().substr(14, 5))
        $.manager.summary.show()
        $.manager.sound.play('1/m_end')
      } else {
        $.manager.summary.hide()
      }
    }

    match.prototype.key = function (K, down) {
      const $ = this
      if ($.gameover_state) {
        if (down) {
          if ($.time.t > $.gameover_state + 60) {
            if (K === 'att' || K === 'jump') {
              $.F4()
            }
          }
        }
      }
    }

    match.prototype.create_effects = function (config) {
      const $ = this
      const effects = Futil.extract_array(util.selectA_from($.data.object, { type: 'effect' }), ['data', 'id'])
      const broken = util.select_from($.data.object, { type: 'broken' })
      $.broken_list = Futil.group_elements(broken.data.broken_list, 'id')
      $.visualeffect = $.effect[0] = new factory.effect({ match: $, stage: $.stage }, effects.data, effects.id)
      $.brokeneffect = $.effect[1] = new factory.effect({ match: $, stage: $.stage, broken_list: $.broken_list }, broken.data, broken.id)
    }

    match.prototype.drop_weapons = function (setup) {
      const $ = this
      const num = 5
      const weapon_list =
        util.selectA_from($.data.object, function (o) {
          return o.id >= 100 && o.id < 200
        })
      for (let i = 0; i < num; i++) {
        const O = $.background.get_pos($.random(), $.random())
        O.y = -800
        $.create_weapon(weapon_list[Math.floor(weapon_list.length * $.random())].id, O)
      }
    }

    match.prototype.destroy_weapons = function () {
      const $ = this
      for (var i in $.lightweapon) {
        $.lightweapon[i].health.hp = 0
      }
      for (var i in $.heavyweapon) {
        $.heavyweapon[i].health.hp = 0
      }
    }

    match.prototype.create_weapon = function (id, pos) {
      const $ = this
      const weapon = id < 150 ? 'lightweapon' : 'heavyweapon'
      const wea_config =
      {
        match: $
      }
      const object = util.select_from($.data.object, { id: id })
      const wea = new factory[weapon](wea_config, object.data, object.id)
      wea.set_pos(pos.x, pos.y, pos.z)
      const uid = $.scene.add(wea)
      $[weapon][uid] = wea
    }

    match.prototype.create_background = function (bg) {
      const $ = this
      if (bg) {
        const bgdata = util.select_from($.data.background, { id: bg.id }).data
        $.background = new Background({
          layers: $.manager.background_layer,
          scrollbar: $.manager.gameplay,
          camerachase: { character: $.character },
          onscroll: function () { $.manager.canvas.render() }
        }, bgdata, bg.id)
        $.stage = $.background.floor
      } else {
        $.background = new Background(null) // create an empty background
        $.stage = $.manager.canvas
      }
    }

    match.prototype.F4 = function () {
      const $ = this
      $.destroy()
      $.manager.match_end()
    }

    match.prototype.F7 = function () {
      const $ = this
      for (const i in $.character) {
        const ch = $.character[i]
        ch.health.hp = ch.health.hp_full = ch.health.hp_bound = ch.proper('hp') || Global.gameplay.default.health.hp_full
        ch.health.mp = ch.health.mp_full
      }
    }

    match.prototype.new_randomseed = function () {
      const rand = new Random()
      rand.seed(this.manager.random())
      return rand
    }

    match.prototype.random = function () {
      return this.randomseed.next()
    }

    match.prototype.overlay_message = function (mess) {
      const $ = this
      if ($.manager.overlay_mess) {
        $.manager.overlay_mess.show()
        const item = $.data.UI.data.message_overlay[mess]
        $.manager.overlay_mess.set_img_x_y(-item[0], -item[1])
        $.manager.overlay_mess.set_w_h(item[2], item[3])
      }
    }

    match.prototype.create_controller = function (funcon) {
      const $ = this
      function show_pause() {
        if (!$) return
        if ($.time.paused) {
          $.overlay_message('pause')
        }
      }
      if (funcon) {
        funcon.sync = true
        funcon.child.push({
          key: function (I, down) {
            const opaused = $.time.paused // original pause state
            if (down) {
              switch (I) {
                case 'F1':
                  if (!$.time.paused) { $.time.paused = true } else { $.time.paused = false }
                  break

                case 'F2':
                  $.time.paused = 'F2'
                  break

                case 'esc':
                case 'F4':
                  $.F4()
                  break

                case 'F6':
                  if (!$.F6_mode) { $.F6_mode = true } else { $.F6_mode = false }
                  break

                case 'F7':
                  $.F7()
                  break

                case 'F8':
                  $.drop_weapons()
                  break

                case 'F9':
                  $.destroy_weapons()
                  break
              }
              if ((I === 'F1' || I === 'F2') && $.time.paused) {
                $.manager.overlay_mess.hide()
                setTimeout(show_pause, 4) // so that the 'pause' message blinks
              } else if (!$.time.paused) {
                $.manager.overlay_mess.hide()
              }
              if (opaused !== $.time.paused) {  // state change
                if ($.time.paused) {
                  if (funcon.paused) {
                    funcon.paused(true)
                  }
                } else {
                  if (funcon.paused) {
                    funcon.paused(false)
                  }
                }
              }
            }
          }
        })
        return funcon
      }
    }

    match.prototype.get_living_object = function () {
      const $ = this
      var temp = {}
      for (a in $.scene.live) {
        if ($.scene.live[a].health.hp > 0) {
          if ($.scene.live[a].type == 'character') {
            if ($.scene.live[a].counter.disappear_count == -1) {
              temp[a] = $.scene.live[a]
            }
          } else {
            temp[a] = $.scene.live[a]
          }
        }
      }
      return temp
    }

    return match
  })
