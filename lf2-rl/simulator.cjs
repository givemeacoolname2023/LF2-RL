const fs = require('node:fs')
const path = require('node:path')
const readline = require('node:readline')
const vm = require('node:vm')

const root = path.resolve(__dirname, '..')
const assets = path.resolve(process.argv[2])
const noop = () => {}

class HeadlessSprite {
  constructor() {
    this.img = [{ addEventListener: noop }]
  }
}

for (const method of ['add_img', 'set_w_h', 'set_w', 'set_x_y', 'set_z', 'set_bgcolor', 'set_alpha',
  'set_flipx', 'switch_img', 'set_img_x_y', 'remove', 'remove_all', 'show', 'hide']) {
  HeadlessSprite.prototype[method] = noop
}
HeadlessSprite.preload_image = noop

const modules = new Map()
const stubs = new Map([
  ['LF/sprite-select', HeadlessSprite],
  ['core/support', {}],
  ['LF/network', {}]
])
const window = { location: { search: '?rlstep&headless' } }
const context = vm.createContext({ window, URLSearchParams, navigator: {},
  document: { addEventListener: noop },
  console: new console.Console(process.stderr)
})

function load(directory, name) {
  if (directory === root && stubs.has(name)) return stubs.get(name)
  const filename = path.resolve(directory, name.endsWith('.js') ? name : `${name}.js`)
  if (modules.has(filename)) return modules.get(filename)
  let definition
  context.define = (...args) => { definition = args }
  vm.runInContext(fs.readFileSync(filename, 'utf8'), context, { filename })
  if (!definition) throw new Error(`Expected AMD module in ${filename}`)
  const [dependencies, factory] = definition.length === 1 ? [[], definition[0]] : definition
  const value = typeof factory === 'function' ? factory(...dependencies.map(dep => load(root, dep))) : factory
  modules.set(filename, value)
  return value
}

const Match = load(root, 'LF/match')
const Controller = load(root, 'core/controller')
const Random = load(root, 'third_party/random')
const manifest = load(assets, 'manifest')
const data = load(assets, manifest.data)
for (const group of [data.object, data.AI, data.background, [data.properties]]) {
  for (const entry of group) if (entry.file) entry.data = load(assets, entry.file)
}
data.load = (sets, ready) => ready()
const pack = { data }
load(root, 'LF/util').organize_package(pack)

const sound = { TU: noop, play: noop }
const canvas = { render: noop }
const keys = { left: 37, right: 39, up: 38, down: 40, jump: 74, att: 65, def: 68 }
const opponents = [2, 1, 0].map(index => data.AI[index])

function create(seed, difficulty, background, character = 'davis', opponent) {
  const random = new Random()
  random.seed(seed)
  const match = new Match({ manager: { sound, canvas, random: () => random.next() },
    package: pack, simulation: true })
  function policyController() {
    const controller = new Controller(keys)
    controller.destroy()
    return controller
  }
  const characterIds = { davis: 11, henry: 4 }
  match.create({
    player: [
      { id: characterIds[character], team: 1, controller: policyController() },
      { id: opponent ? characterIds[opponent] : 9, team: 2,
        controller: opponent ? policyController() : { type: 'AIscript', id: opponents[difficulty].id } }
    ],
    background: { id: data.background[background].id },
    set: { weapon: true }
  })
  return match
}

function state(match) {
  const row = [match.time.t]
  for (const character of match.rl_characters) {
    const { x, y, z, dir } = character.ps
    row.push(x, y, character.health.hp, character.health.mp, character.frame.N, dir === 'right' ? 1 : -1, z)
  }
  row.push(match.rl_damage, match.rl_center_damage)
  return row
}

let matches = []
const input = readline.createInterface({ input: process.stdin, crlfDelay: Infinity })
input.on('line', line => {
  try {
    const request = JSON.parse(line)
    const newMatch = seed => create(seed, request.difficulty, request.background, request.character, request.opponent)
    let output
    if (request.actions) {
      const final = []
      output = matches.map((match, index) => {
        match.rl_step(request.actions[index], Math.min(request.repeat, request.limit - match.time.t),
          request.opponent_actions?.[index])
        if (request.restart || match.gameover_state || match.time.t >= request.limit) {
          final.push([index, state(match)])
          match.destroy()
          match = matches[index] = newMatch(request.seeds[index])
        }
        return state(match)
      })
      output = [output, final]
    } else {
      matches.forEach(match => match.destroy())
      matches = request.seeds.map(newMatch)
      output = matches.map(state)
    }
    process.stdout.write(`${JSON.stringify(output)}\n`)
  } catch (error) {
    process.stdout.write(`${JSON.stringify({ error: error.stack })}\n`)
    input.close()
  }
})
