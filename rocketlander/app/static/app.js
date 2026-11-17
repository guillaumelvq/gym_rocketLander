// Rocket RL Lab - front end (no build step, no external dependency)

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
const fmt = (v, d = 2) => (v == null || Number.isNaN(v) ? "–" : Number(v).toFixed(d));

const store = {
  get(key, fallback) { try { const v = localStorage.getItem("rrl." + key); return v == null ? fallback : JSON.parse(v); } catch { return fallback; } },
  set(key, value) { try { localStorage.setItem("rrl." + key, JSON.stringify(value)); } catch { /* private mode */ } },
};

const S_KEYS = {
  x: "position horizontale, -1 bord gauche … 1 bord droit de l'écran",
  x_on_ship: "position sur le bateau : 0 centre, ±1 bords",
  altitude: "hauteur au-dessus du pont (m)",
  y: "hauteur normalisée, -1 = pont",
  angle: "inclinaison (×π), 0 = droit",
  rotation: "tours effectués depuis le départ (+ = anti-horaire)",
  velocity_x: "vitesse horizontale (×40 m/s)",
  velocity_y: "vitesse verticale (×40 m/s), < 0 = chute",
  speed: "vitesse totale (×40 m/s)",
  angular_velocity: "vitesse de rotation (rad/s)",
  left_leg_contact: "pied gauche sur le bateau",
  right_leg_contact: "pied droit sur le bateau",
  touch_down: "un pied a déjà touché",
  landed: "posé sur les deux pieds, lentement",
  landed_full: "posé depuis 1 s : réussite",
  game_over: "crash (coque, pied cassé, sortie)",
  broken_leg: "pied cassé (contact trop rapide)",
  outside: "sorti de l'écran",
  timeout: "temps écoulé",
  throttle: "poussée 0…1",
  gimbal: "orientation du moteur -1…1",
  side_thruster: "propulseur latéral -1, 0, 1",
  fuel: "carburant restant 0…1",
  step: "instant (60 par seconde)",
  memory: "dict libre, vidé à chaque épisode",
};

const LEVEL_DESC = (lvl) => {
  if (lvl === 0) return "Les pieds ne cassent jamais : même un crash à pleine vitesse sur le bateau compte.";
  if (lvl < 10) return `Les pieds cassent au-delà de ${(8 - 0.4 * lvl).toFixed(1)} m/s au contact. Départ un peu aléatoire.`;
  if (lvl === 10) return "4,4 m/s max au contact, et les jambes cassent si elles sont trop comprimées.";
  return `Comme le niveau 10, avec des jambes raccourcies (×${Math.max(0.3, 1 - 0.07 * (lvl - 10)).toFixed(2)}).`;
};

let META = null;
let RUNS = [];
const state = {
  level: 1,
  config: {},
  reward: {},
  code: "",
  challenge: "landing",
};

// ===================================================================== sidebar

function rangeField(container, key, label, min, max, step, help, value, onChange, opts = {}) {
  const wrap = document.createElement("div");
  wrap.className = "range";
  const id = `f-${container.id}-${key}`;
  const nullable = opts.nullable;
  wrap.innerHTML = `<label for="${id}">${label}</label><output></output>
    <input id="${id}" type="range" min="${min}" max="${max}" step="${step}">
    ${help ? `<small>${help}</small>` : ""}
    ${nullable ? `<label class="check never"><input type="checkbox"> jamais</label>` : ""}`;
  const input = $("input[type=range]", wrap);
  const out = $("output", wrap);
  const never = nullable ? $(".never input", wrap) : null;
  const decimals = step >= 1 ? 0 : Math.max(0, -Math.floor(Math.log10(step)));
  const set = (v) => {
    if (nullable && v == null) { never.checked = true; input.disabled = true; out.textContent = "jamais"; return; }
    if (never) { never.checked = false; input.disabled = false; }
    input.value = v; out.textContent = Number(v).toFixed(decimals);
  };
  input.addEventListener("input", () => { out.textContent = Number(input.value).toFixed(decimals); onChange(Number(input.value)); });
  never?.addEventListener("change", () => {
    if (never.checked) { set(null); onChange(null); } else { set(Number(input.value) || (min + max) / 2); onChange(Number(input.value)); }
  });
  set(value);
  container.appendChild(wrap);
  return { wrap, set };
}

function checkField(container, key, label, help, value, onChange) {
  const wrap = document.createElement("div");
  wrap.className = "range";
  wrap.innerHTML = `<label class="check"><input type="checkbox"> ${label}</label><output></output>${help ? `<small>${help}</small>` : ""}`;
  const input = $("input", wrap);
  input.checked = !!value;
  input.addEventListener("change", () => onChange(input.checked));
  container.appendChild(wrap);
  return { wrap, set: (v) => { input.checked = !!v; } };
}

const configInputs = {};
function buildConfigFields() {
  const box = $("#config-fields");
  box.innerHTML = "";
  for (const [key, [label, min, max, step, help]] of Object.entries(META.config_doc)) {
    const onChange = (v) => { state.config[key] = v; markChanged(); saveState(); };
    configInputs[key] = min == null
      ? checkField(box, key, label, help, state.config[key], onChange)
      : rangeField(box, key, label, min, max, step, help, state.config[key], onChange,
                   { nullable: key === "max_touchdown_speed" });
  }
  markChanged();
}

function markChanged() {
  const base = META.levels[state.level];
  for (const [key, input] of Object.entries(configInputs)) {
    input.wrap.classList.toggle("changed", JSON.stringify(base[key]) !== JSON.stringify(state.config[key]));
  }
}

const rewardInputs = {};
function buildRewardFields() {
  const box = $("#reward-fields");
  box.innerHTML = "";
  for (const [key, [label, min, max, step]] of Object.entries(META.reward_doc)) {
    rewardInputs[key] = rangeField(box, key, label, min, max, step, "", state.reward[key] ?? META.reward_defaults[key],
      (v) => { state.reward[key] = v; saveState(); });
  }
}

function applyLevel(level, keepOverrides = {}) {
  state.level = Number(level);
  state.config = { ...META.levels[state.level], ...keepOverrides };
  $("#level").value = state.level;
  $("#level-desc").textContent = LEVEL_DESC(state.level);
  for (const [k, input] of Object.entries(configInputs)) input.set(state.config[k]);
  markChanged();
  saveState();
}

function applyChallenge(id) {
  const ch = META.challenges[id];
  state.challenge = id;
  $("#challenge").value = id;
  $("#challenge-goal").textContent = ch.goal;
  applyLevel(ch.level, ch.config);
  state.code = ch.code;
  $("#reward-code").value = ch.code;
  saveState();
}

function saveState() { store.set("state", state); }

function initSidebar() {
  const chSel = $("#challenge");
  for (const [id, ch] of Object.entries(META.challenges)) chSel.add(new Option(ch.title, id));
  chSel.addEventListener("change", () => applyChallenge(chSel.value));

  const lvl = $("#level");
  for (let i = 0; i <= 20; i++) lvl.add(new Option(`Niveau ${i}`, i));
  lvl.addEventListener("change", () => applyLevel(lvl.value));
  $("#reset-level").addEventListener("click", () => applyLevel(state.level));

  const sk = $("#s-keys");
  for (const [k, v] of Object.entries(S_KEYS)) sk.insertAdjacentHTML("beforeend", `<code>${k}</code><span>${v}</span>`);

  const code = $("#reward-code");
  code.addEventListener("input", () => { state.code = code.value; saveState(); });
  code.addEventListener("keydown", (e) => {
    if (e.key !== "Tab") return;
    e.preventDefault();
    const { selectionStart: a, selectionEnd: b, value } = code;
    code.value = value.slice(0, a) + "    " + value.slice(b);
    code.selectionStart = code.selectionEnd = a + 4;
    state.code = code.value; saveState();
  });

  const saved = store.get("state", null);
  if (saved && META.challenges[saved.challenge]) {
    Object.assign(state, saved);
    state.config = { ...META.levels[state.level], ...saved.config };
    chSel.value = state.challenge;
    $("#challenge-goal").textContent = META.challenges[state.challenge].goal;
    lvl.value = state.level;
    $("#level-desc").textContent = LEVEL_DESC(state.level);
    code.value = state.code;
    buildConfigFields();
  } else {
    buildConfigFields();
    applyChallenge("landing");
  }
  buildRewardFields();
}

// ===================================================================== rendering

const COLORS = {
  skyTop: "#0b1230", skyMid: "#2a3a78", skyLow: "#e08a5a",
  sea: "#14306b", seaLight: "#2a5aa8",
  deck: "#2b2f3a", deckLine: "#ffd23f", container: "#d9b600",
  body: "#f2f4f8", bodyShade: "#c9cfdb", band: "#1b1f2a", leg: "#2d3240",
};

class Renderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.world = null;
    this.frame = null;
    this.smoke = [];
    this.trail = [];
    this.stars = Array.from({ length: 70 }, () => [Math.random(), Math.random() * 0.55, Math.random() * 1.2 + 0.3]);
    this.cam = { x: 0, y: 0, zoom: 1 };
    this.camMode = "auto";
  }

  resize() {
    const r = this.canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(1, Math.round(r.width * dpr)), h = Math.max(1, Math.round(r.height * dpr));
    if (this.canvas.width !== w || this.canvas.height !== h) { this.canvas.width = w; this.canvas.height = h; }
    this.dpr = dpr;
  }

  setWorld(world) { this.world = world; this.smoke = []; this.trail = []; this.cam = null; }

  setFrame(f) {
    this.frame = f;
    const r = f.rocket;
    if (r.power > 0.01 && Math.random() < 0.7) {
      const a = r.angle + r.gimbal;
      this.smoke.push({ x: r.x + Math.sin(a) * 2, y: r.y - Math.cos(a) * 2, vx: Math.sin(a) * 6 * r.power + (Math.random() - .5) * 2,
        vy: -Math.cos(a) * 6 * r.power + (Math.random() - .5) * 2, life: 0, max: 50 + 60 * r.power, size: 1 + r.power * 2 });
    }
    if (f.t % 3 === 0) { this.trail.push([r.x, r.y]); if (this.trail.length > 400) this.trail.shift(); }
  }

  updateCamera() {
    const W = this.world.W, H = this.world.H, cw = this.canvas.width, ch = this.canvas.height;
    const fit = Math.min(cw / W, ch / H);
    const r = this.frame?.rocket;
    let target;
    if (!r || this.camMode === "wide") target = { x: W / 2, y: H / 2, zoom: 1 };
    else {
      const alt = this.frame.hud.altitude;
      const z = this.camMode === "follow" ? 4.5 : 1 + 3.5 * Math.min(1, Math.max(0, 1 - alt / 120));
      target = { x: r.x, y: r.y + this.world.rocket_height * 0.4, zoom: z };
    }
    // keep the view inside the world
    const viewW = cw / (fit * target.zoom), viewH = ch / (fit * target.zoom);
    target.x = Math.min(Math.max(target.x, viewW / 2), W - viewW / 2);
    target.y = Math.min(Math.max(target.y, viewH / 2 - 2), H - viewH / 2);
    if (!this.cam) this.cam = { ...target };
    const k = 0.12;
    this.cam.x += (target.x - this.cam.x) * k;
    this.cam.y += (target.y - this.cam.y) * k;
    this.cam.zoom += (target.zoom - this.cam.zoom) * k;
    this.scale = fit * this.cam.zoom;
  }

  tx(x) { return (x - this.cam.x) * this.scale + this.canvas.width / 2; }
  ty(y) { return this.canvas.height / 2 - (y - this.cam.y) * this.scale; }

  poly(points, fill, stroke) {
    const c = this.ctx;
    c.beginPath();
    points.forEach(([x, y], i) => (i ? c.lineTo(this.tx(x), this.ty(y)) : c.moveTo(this.tx(x), this.ty(y))));
    c.closePath();
    if (fill) { c.fillStyle = fill; c.fill(); }
    if (stroke) { c.strokeStyle = stroke; c.stroke(); }
  }

  draw(time) {
    this.resize();
    const c = this.ctx, cw = this.canvas.width, ch = this.canvas.height;
    if (!this.world) {
      c.fillStyle = "#0b1230"; c.fillRect(0, 0, cw, ch);
      c.fillStyle = "#8f9bbd"; c.font = `${14 * this.dpr}px system-ui`; c.textAlign = "center";
      c.fillText("Choisis un pilote puis ▶ Lancer", cw / 2, ch / 2);
      return;
    }
    this.updateCamera();
    const w = this.world;

    // sky
    const g = c.createLinearGradient(0, this.ty(w.H), 0, this.ty(0));
    g.addColorStop(0, COLORS.skyTop); g.addColorStop(0.65, COLORS.skyMid); g.addColorStop(1, COLORS.skyLow);
    c.fillStyle = g; c.fillRect(0, 0, cw, ch);
    for (const [sx, sy, s] of this.stars) {
      const x = this.tx(sx * w.W), y = this.ty(w.H * (1 - sy));
      c.globalAlpha = 0.35 + 0.35 * Math.sin(time / 700 + sx * 40);
      c.fillStyle = "#fff"; c.fillRect(x, y, s * this.dpr, s * this.dpr);
    }
    c.globalAlpha = 1;

    // trail
    if (this.trail.length > 1) {
      c.strokeStyle = "rgba(255,255,255,0.18)"; c.lineWidth = 1.5 * this.dpr; c.setLineDash([3 * this.dpr, 5 * this.dpr]);
      c.beginPath();
      this.trail.forEach(([x, y], i) => (i ? c.lineTo(this.tx(x), this.ty(y)) : c.moveTo(this.tx(x), this.ty(y))));
      c.stroke(); c.setLineDash([]);
    }

    // sea
    const seaTop = w.water[2][1];
    const sg = c.createLinearGradient(0, this.ty(seaTop), 0, this.ty(0));
    sg.addColorStop(0, COLORS.seaLight); sg.addColorStop(1, COLORS.sea);
    c.fillStyle = sg; c.fillRect(0, this.ty(seaTop), cw, ch - this.ty(seaTop));
    c.strokeStyle = "rgba(255,255,255,0.25)"; c.lineWidth = this.dpr;
    for (let i = 0; i < 3; i++) {
      c.beginPath();
      const yy = seaTop - 1.5 - i * 4;
      for (let x = 0; x <= w.W; x += 2) {
        const y = yy + Math.sin(x / 6 + time / 500 + i) * 0.5;
        x ? c.lineTo(this.tx(x), this.ty(y)) : c.moveTo(this.tx(x), this.ty(y));
      }
      c.stroke();
    }

    // ship + landing zone
    this.poly(w.ship, COLORS.deck);
    const dy = w.deck_y, x1 = w.ship_x1, x2 = w.ship_x2, mid = (x1 + x2) / 2;
    c.lineWidth = 2 * this.dpr; c.strokeStyle = COLORS.deckLine;
    c.beginPath(); c.moveTo(this.tx(x1), this.ty(dy)); c.lineTo(this.tx(x2), this.ty(dy)); c.stroke();
    c.fillStyle = "rgba(255,210,63,0.9)";
    for (let k = -1; k <= 1; k += 2) {
      const xm = mid + k * (x2 - x1) * 0.25;
      c.fillRect(this.tx(xm) - this.dpr, this.ty(dy) - 1, 2 * this.dpr, Math.max(2, 0.6 * this.scale));
    }
    c.font = `bold ${Math.max(9, 1.4 * this.scale)}px system-ui`; c.textAlign = "center";
    c.fillText("✕", this.tx(mid), this.ty(dy) - 2);
    for (const cont of w.containers) this.poly(cont, COLORS.container);

    // smoke
    for (const p of this.smoke) {
      p.life++; p.x += p.vx / 60; p.y += p.vy / 60; p.vx *= 0.97; p.vy = p.vy * 0.97 + 0.03;
      const a = 1 - p.life / p.max;
      c.fillStyle = `rgba(230,230,240,${0.22 * a})`;
      c.beginPath(); c.arc(this.tx(p.x), this.ty(p.y), (p.size + p.life * 0.06) * this.scale, 0, Math.PI * 2); c.fill();
    }
    this.smoke = this.smoke.filter((p) => p.life < p.max);

    const f = this.frame;
    if (!f) return;
    const r = f.rocket;

    // struts + legs
    c.strokeStyle = "#5b6275"; c.lineWidth = Math.max(1, 0.25 * this.scale);
    for (const [[ax, ay], [bx, by]] of f.struts) { c.beginPath(); c.moveTo(this.tx(ax), this.ty(ay)); c.lineTo(this.tx(bx), this.ty(by)); c.stroke(); }
    for (const leg of f.legs) this.poly(leg, COLORS.leg);

    // rocket in its local frame
    c.save();
    c.translate(this.tx(r.x), this.ty(r.y));
    c.rotate(-r.angle);
    const s = this.scale, RW = w.rocket_width * s, RH = w.rocket_height * s;
    // flame
    if (r.power > 0.01) {
      c.save(); c.rotate(-r.gimbal);
      const len = (w.engine_height * 7 * r.power * (0.85 + Math.random() * 0.3)) * s;
      const fg = c.createLinearGradient(0, 0, 0, len);
      fg.addColorStop(0, "rgba(255,255,255,0.95)"); fg.addColorStop(0.25, "rgba(255,214,107,0.9)");
      fg.addColorStop(0.7, "rgba(255,120,40,0.6)"); fg.addColorStop(1, "rgba(255,80,20,0)");
      c.fillStyle = fg;
      c.beginPath(); c.moveTo(-RW * 0.32, w.engine_height * s); c.quadraticCurveTo(-RW * 0.9, len * 0.45, 0, len + w.engine_height * s);
      c.quadraticCurveTo(RW * 0.9, len * 0.45, RW * 0.32, w.engine_height * s); c.closePath(); c.fill();
      c.globalCompositeOperation = "lighter";
      c.fillStyle = `rgba(255,150,60,${0.25 * r.power})`;
      c.beginPath(); c.arc(0, len * 0.3, RW * 2.2, 0, Math.PI * 2); c.fill();
      c.restore();
    }
    // engine bell
    c.save(); c.rotate(-r.gimbal);
    c.fillStyle = "#4a4f5c";
    c.beginPath(); c.moveTo(-RW * 0.2, 0); c.lineTo(RW * 0.2, 0); c.lineTo(RW * 0.35, w.engine_height * s); c.lineTo(-RW * 0.35, w.engine_height * s); c.closePath(); c.fill();
    c.restore();
    // body
    const bg = c.createLinearGradient(-RW / 2, 0, RW / 2, 0);
    bg.addColorStop(0, COLORS.bodyShade); bg.addColorStop(0.45, COLORS.body); bg.addColorStop(1, COLORS.bodyShade);
    c.fillStyle = bg; c.fillRect(-RW / 2, -RH, RW, RH);
    c.fillStyle = COLORS.band;
    c.fillRect(-RW / 2, -RH * 0.9, RW, RH * 0.1);   // interstage
    c.fillRect(-RW / 2, -RH * 0.06, RW, RH * 0.06); // engine section
    // grid fins
    c.fillStyle = "#3b4150";
    c.fillRect(-RW * 1.1, -w.thruster_height * s - 0.2 * s, RW * 0.6, 0.4 * s);
    c.fillRect(RW * 0.5, -w.thruster_height * s - 0.2 * s, RW * 0.6, 0.4 * s);
    // side thruster puff
    if (r.side) {
      const dir = r.side;
      c.fillStyle = "rgba(235,240,255,0.85)";
      c.beginPath();
      c.moveTo(dir * RW / 2, -w.thruster_height * s);
      c.lineTo(dir * RW * 3.5, -w.thruster_height * s - RW * 0.8);
      c.lineTo(dir * RW * 3.5, -w.thruster_height * s + RW * 0.8);
      c.closePath(); c.fill();
    }
    c.restore();

    this.drawHud(f);
  }

  drawHud(f) {
    const c = this.ctx, d = this.dpr, h = f.hud;
    const lim = this.config?.max_touchdown_speed;
    const pad = 10 * d;
    c.font = `${11.5 * d}px ui-monospace, Consolas, monospace`;
    c.textAlign = "left";
    const rows = [
      ["ALT", `${fmt(h.altitude, 1)} m`, null],
      ["V↓", `${fmt(-h.vy, 1)} m/s`, lim != null && h.altitude < 30 && -h.vy > lim ? "#ff5d6c" : null],
      ["V→", `${fmt(h.vx, 1)} m/s`, null],
      ["ANG", `${fmt(h.angle_deg, 0)}°`, Math.abs(h.angle_deg) > 20 ? "#ffc857" : null],
      ["TOUR", fmt(h.rotation, 2), null],
      ["T", `${(f.t / 60).toFixed(1)} s`, null],
    ];
    c.fillStyle = "rgba(5,10,25,0.55)";
    c.fillRect(pad - 6 * d, pad - 4 * d, 132 * d, rows.length * 16 * d + 46 * d);
    rows.forEach(([k, v, col], i) => {
      c.fillStyle = "#8f9bbd"; c.fillText(k, pad, pad + 10 * d + i * 16 * d);
      c.fillStyle = col || "#e8edf6"; c.fillText(v, pad + 42 * d, pad + 10 * d + i * 16 * d);
    });
    const by = pad + rows.length * 16 * d + 4 * d;
    const bar = (label, v, y, color) => {
      c.fillStyle = "#8f9bbd"; c.fillText(label, pad, y + 8 * d);
      c.fillStyle = "rgba(255,255,255,0.12)"; c.fillRect(pad + 42 * d, y, 76 * d, 8 * d);
      c.fillStyle = color; c.fillRect(pad + 42 * d, y, 76 * d * Math.max(0, Math.min(1, v)), 8 * d);
    };
    bar("GAZ", h.throttle, by, "#ff8a3d");
    bar("FUEL", h.fuel, by + 16 * d, h.fuel < 0.2 ? "#ff5d6c" : "#3ddc97");
    // legs
    const lx = this.canvas.width - pad - 40 * d, ly = pad + 6 * d;
    c.fillStyle = "rgba(5,10,25,0.55)"; c.fillRect(lx - 8 * d, pad - 4 * d, 54 * d, 26 * d);
    [[h.left, 0], [h.right, 22]].forEach(([on, dx]) => {
      c.fillStyle = on ? "#3ddc97" : "rgba(255,255,255,0.2)";
      c.beginPath(); c.arc(lx + dx * d + 6 * d, ly + 3 * d, 6 * d, 0, Math.PI * 2); c.fill();
    });
    if (h.landed_ticks > 0) {
      c.fillStyle = "#3ddc97"; c.fillRect(lx - 8 * d, pad + 24 * d, 54 * d * Math.min(1, h.landed_ticks / 60), 3 * d);
    }
  }
}

// ===================================================================== arenas

const OUTCOMES = {
  success: ["Atterrissage réussi", "success", "✅"],
  crash: ["Crash", "fail", "💥"],
  broken_leg: ["Pied cassé", "fail", "🦵"],
  outside: ["Sortie de l'écran", "fail", "↗️"],
  timeout: ["Temps écoulé", "timeout", "⏱️"],
};
const outcomeIcon = (o) => (OUTCOMES[o] || ["", "", "❔"])[2];
const outcomeLabel = (o) => (OUTCOMES[o] || [o])[0];
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

const arenas = [];
let focusedHuman = null;
let BOARD = [];          // leaderboard rows of the current scenario
const REPLAY_LABELS = {}; // attempt id -> label, for arenas showing a replay

class Arena {
  constructor(agent = "autopilot") {
    const node = $("#arena-tpl").content.firstElementChild.cloneNode(true);
    $("#arenas").appendChild(node);
    this.node = node;
    this.renderer = new Renderer($("canvas", node));
    this.renderer.camMode = $("#camera").value;
    this.select = $(".agent", node);
    this.banner = $(".banner", node);
    this.countdownEl = $(".countdown", node);
    this.note = $(".arena-note", node);
    this.wins = 0; this.total = 0;
    this.ws = null;
    this.running = false;
    this.scenario = null; // per-arena scenario override {level, config, label} (comparisons)
    this.fillAgents(agent);
    this.select.addEventListener("change", () => {
      this.scenario = null;
      this.wins = this.total = 0; this.updateTally(); this.updateNote(); saveArenas(); this.start();
    });
    $(".remove", node).addEventListener("click", () => this.destroy());
    node.addEventListener("pointerdown", () => { if (this.select.value === "human") setFocusedHuman(this); });
    this.updateNote();
    this.connect();
  }

  fillAgents(selected) {
    const prev = selected ?? this.select.value;
    this.select.innerHTML = "";
    this.select.add(new Option("🧑 Humain (clavier)", "human"));
    this.select.add(new Option("🤖 Autopilote (programmé, sans IA)", "autopilot"));
    this.select.add(new Option("🎲 Aléatoire", "random"));
    const models = RUNS.filter((r) => r.has_best || r.has_last);
    if (models.length) {
      const g = document.createElement("optgroup"); g.label = "Agents entraînés";
      for (const r of models) g.appendChild(new Option(`🧠 ${r.name}`, "model:" + r.name));
      this.select.appendChild(g);
    }
    const replays = BOARD.slice(0, 15).map((a) => [a.id, `🎬 ${a.name} ${outcomeIcon(a.outcome)} ${fmt(a.score, 1)}`]);
    if (prev?.startsWith("replay:") && !replays.some(([id]) => "replay:" + id === prev)) {
      const id = prev.slice(7);
      replays.unshift([id, REPLAY_LABELS[id] || `🎬 tentative ${id}`]);
    }
    if (replays.length) {
      const g = document.createElement("optgroup"); g.label = "Tentatives humaines (replay)";
      for (const [id, label] of replays) g.appendChild(new Option(label, "replay:" + id));
      this.select.appendChild(g);
    }
    this.select.value = [...this.select.options].some((o) => o.value === prev) ? prev : "autopilot";
  }

  updateNote() {
    const v = this.select.value;
    this.note.classList.remove("error");
    let text = "";
    if (v.startsWith("model:")) {
      const r = RUNS.find((x) => x.name === v.slice(6));
      text = r ? `Scénario d'entraînement : niveau ${r.level}${Object.keys(r.config || {}).length ? " + réglages" : ""}` : "";
    } else if (v.startsWith("replay:")) text = "Replay exact d'une tentative (sa graine et son scénario).";
    else if (v === "human") text = focusedHuman === this ? "C'est toi qui pilotes ici." : "Clique ici pour piloter cette arène.";
    if (this.scenario && !v.startsWith("replay:")) text = `Scénario de ${this.scenario.label}`;
    this.note.textContent = text;
  }

  connect() {
    const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/sim`);
    this.ws = ws;
    ws.onmessage = (ev) => this.onMessage(JSON.parse(ev.data));
    ws.onclose = () => { if (this.ws === ws && !this.destroyed) setTimeout(() => this.connect(), 1000); };
  }

  send(msg) { if (this.ws?.readyState === 1) this.ws.send(JSON.stringify(msg)); }

  start(seed = Number($("#seed").value)) {
    if (this.ws?.readyState !== 1) { setTimeout(() => this.start(seed), 200); return; }
    const v = this.select.value;
    const scen = this.scenario || { level: state.level, config: state.config };
    const msg = { type: "start", seed, speed: Number($("#speed").value), level: scen.level, config: scen.config,
                  reward: state.reward, reward_code: state.code };
    if (v.startsWith("model:")) Object.assign(msg, { agent: "model", run: v.slice(6), which: "best", override_scenario: !!this.scenario });
    else if (v.startsWith("replay:")) Object.assign(msg, { agent: "replay", attempt: v.slice(7) });
    else msg.agent = v;
    this.awaitingGo = false;
    if (v === "human") {
      setFocusedHuman(this);
      Object.assign(msg, { controls: $("#controls").value, player: playerName(), countdown: true, speed: 1 });
      this.awaitingGo = true;
      human.keys.clear();
    }
    clearInterval(this.countdownTimer);
    this.countdownEl.hidden = true;
    this.banner.hidden = true;
    this.updateNote();
    this.running = true;
    this.send(msg);
  }

  runCountdown() {
    let n = 3;
    const el = this.countdownEl;
    el.hidden = false; el.textContent = n;
    clearInterval(this.countdownTimer);
    this.countdownTimer = setInterval(() => {
      n -= 1;
      if (n > 0) { el.textContent = n; return; }
      clearInterval(this.countdownTimer);
      el.textContent = "Go !";
      this.send({ type: "keys", keys: human.serverKeys() });
      this.send({ type: "go" });
      setTimeout(() => { el.hidden = true; }, 500);
    }, 700);
  }

  onMessage(m) {
    if (m.type === "init") {
      this.renderer.setWorld(m.world); this.renderer.config = m.config; this.frameSkip = m.frame_skip;
      if (m.attempt) this.note.textContent = `Replay : ${m.attempt.name} · ${outcomeLabel(m.attempt.outcome)} · graine ${m.attempt.seed}`;
      if (this.awaitingGo) { this.awaitingGo = false; this.runCountdown(); }
    } else if (m.type === "frame") {
      this.renderer.setFrame(m);
      $(".ret", this.node).textContent = fmt(m.return);
      $(".cret", this.node).textContent = m.custom_return == null ? "–" : fmt(m.custom_return);
    } else if (m.type === "end") {
      this.running = false;
      this.total++; if (m.outcome === "success") this.wins++;
      this.updateTally();
      const [label, cls] = OUTCOMES[m.outcome] || [m.outcome, "fail"];
      const extra = m.outcome === "success" ? `position ${fmt(m.x_on_ship, 2)} · carburant ${Math.round(m.fuel * 100)}%` : `après ${(m.steps / 60).toFixed(1)} s`;
      this.banner.className = `banner ${cls}`;
      this.banner.innerHTML = `${label}<small>${extra}</small>`;
      this.banner.hidden = false;
      if (m.attempt) { showResult(m); refreshBoard(); }
      if ($("#loop").checked && this.select.value !== "human") {
        setTimeout(() => { if (!this.running) this.start(Number($("#seed").value) + this.total); }, 1200);
      }
    } else if (m.type === "error") {
      this.note.textContent = m.message; this.note.classList.add("error");
    }
  }

  updateTally() { $(".tally", this.node).textContent = `${this.wins} / ${this.total}`; }

  destroy() {
    this.destroyed = true;
    clearInterval(this.countdownTimer);
    this.ws?.close();
    this.node.remove();
    arenas.splice(arenas.indexOf(this), 1);
    saveArenas();
  }
}

function setFocusedHuman(arena) {
  focusedHuman = arena;
  arenas.forEach((a) => { a.node.classList.toggle("focused", a === arena); a.updateNote(); });
}

function saveArenas() { store.set("arenas", arenas.map((a) => a.select.value)); }

function addArena(agent) { const a = new Arena(agent); arenas.push(a); saveArenas(); return a; }

function playerName() { return ($("#player-name").value || "").trim() || "Anonyme"; }

// keyboard: the page only sends which keys are held, the server turns them into actions
// (frame by frame, so attempts are recorded and replayed exactly)
const KEYMAP = { ArrowUp: "up", ArrowDown: "down", ArrowLeft: "left", ArrowRight: "right", KeyA: "side_left", KeyQ: "side_left", KeyD: "side_right" };
const human = {
  keys: new Set(),
  serverKeys() { return [...new Set([...this.keys].map((k) => KEYMAP[k]).filter(Boolean))]; },
};
function humanTarget() { return focusedHuman && arenas.includes(focusedHuman) ? focusedHuman : arenas.find((a) => a.select.value === "human"); }
function sendKeys() { const t = humanTarget(); if (t?.running) t.send({ type: "keys", keys: human.serverKeys() }); }
setInterval(sendKeys, 150); // resend regularly in case a message was lost
window.addEventListener("keydown", (e) => {
  if (e.target.matches("textarea, input, select")) return;
  if (e.code === "Space") { e.preventDefault(); startAll(); return; }
  if (!KEYMAP[e.code]) return;
  e.preventDefault();
  if (!human.keys.has(e.code)) { human.keys.add(e.code); sendKeys(); }
});
window.addEventListener("keyup", (e) => { if (human.keys.delete(e.code)) sendKeys(); });
window.addEventListener("blur", () => { human.keys.clear(); sendKeys(); });

const CONTROLS_HELP = {
  simple: "Mode simple : <kbd>↑</kbd> moteur (relâcher = coupé) · <kbd>←</kbd><kbd>→</kbd> tourner la fusée (elle se stabilise seule quand tu lâches) · <kbd>Espace</kbd> relancer",
  expert: "Mode expert : <kbd>↑</kbd><kbd>↓</kbd> monter / baisser la poussée · <kbd>←</kbd><kbd>→</kbd> orienter le moteur · <kbd>Q</kbd><kbd>D</kbd> propulseurs latéraux · <kbd>Espace</kbd> relancer",
};
function updateControlsHelp() { $("#keys-help").innerHTML = CONTROLS_HELP[$("#controls").value]; }

function startAll() { for (const a of arenas) { a.wins = a.total = 0; a.updateTally(); a.start(); } }
window.rrl = { arenas, human }; // handy from the browser console (and for automated UI checks)

// ---------------------------------------------------------------- leaderboard + result

function boardScenario() { return { level: state.level, config: state.config }; }

async function refreshBoard() {
  const scen = boardScenario();
  const params = new URLSearchParams({ level: scen.level, config: JSON.stringify(scen.config) });
  let data;
  try { data = await (await fetch(`/api/attempts?${params}`)).json(); } catch { return; }
  BOARD = data.attempts || [];
  BOARD.forEach((a) => { REPLAY_LABELS[a.id] = `🎬 ${a.name} ${outcomeIcon(a.outcome)} ${fmt(a.score, 1)}`; });
  $("#board-count").textContent = data.count ? `${data.count} tentative${data.count > 1 ? "s" : ""}` : "";
  const body = $("#board tbody");
  if (!BOARD.length) {
    body.innerHTML = `<tr><td colspan="8" class="empty-row">Personne n'a encore tenté ce scénario. Choisis « Humain » dans une arène et lance !</td></tr>`;
  } else {
    body.innerHTML = BOARD.map((a) => `
      <tr data-id="${a.id}">
        <td class="rank">${a.rank}</td>
        <td class="who">${esc(a.name)}<small>${a.controls === "expert" ? "expert" : "simple"}</small></td>
        <td title="${outcomeLabel(a.outcome)}">${outcomeIcon(a.outcome)}</td>
        <td class="num">${fmt(a.score, 1)}</td>
        <td class="num">${fmt(a.duration, 1)} s</td>
        <td class="num">${Math.round(a.fuel * 100)}%</td>
        <td class="num">${a.outcome === "success" ? fmt(Math.abs(a.x_on_ship), 2) : "–"}</td>
        <td class="acts">
          <button class="small ghost" data-act="replay" title="Revoir cette tentative">▶ Revoir</button>
          <button class="small ghost" data-act="compare" title="Rejouer à côté de l'autopilote et du meilleur agent">⚔ IA</button>
          <button class="small ghost icon" data-act="delete" title="Supprimer" aria-label="Supprimer">✕</button>
        </td>
      </tr>`).join("");
  }
  arenas.forEach((a) => a.fillAgents());
}

async function boardAction(e) {
  const btn = e.target.closest("button[data-act]");
  if (!btn) return;
  const id = btn.closest("tr").dataset.id;
  const act = btn.dataset.act;
  if (act === "replay") watchReplay(id);
  else if (act === "compare") compareWithAI(id);
  else if (act === "delete") {
    const a = BOARD.find((x) => x.id === id);
    if (!confirm(`Supprimer la tentative de ${a?.name || id} ?`)) return;
    await fetch(`/api/attempts/${id}`, { method: "DELETE" });
    refreshBoard();
  }
}

function watchReplay(id) {
  switchTab("arena");
  const a = arenas[0] || addArena();
  a.scenario = null;
  a.fillAgents("replay:" + id);
  a.wins = a.total = 0; a.updateTally(); saveArenas();
  a.start();
}

async function compareWithAI(id) {
  switchTab("arena");
  const attempt = await (await fetch(`/api/attempts/${id}`)).json();
  const scenario = { level: attempt.level, config: attempt.config, label: `la tentative de ${attempt.name}` };
  const best = [...RUNS].filter((r) => r.has_best || r.has_last)
    .sort((a, b) => (b.eval?.success_rate ?? -1) - (a.eval?.success_rate ?? -1))[0];
  const wanted = ["replay:" + id, "autopilot", ...(best ? ["model:" + best.name] : [])];
  while (arenas.length < wanted.length) addArena("autopilot");
  wanted.forEach((agent, i) => {
    const a = arenas[i];
    a.fillAgents(agent);
    a.scenario = agent.startsWith("replay:") ? null : scenario;
    a.wins = a.total = 0; a.updateTally(); a.updateNote();
  });
  saveArenas();
  $("#seed").value = attempt.seed;
  wanted.forEach((_, i) => arenas[i].start(attempt.seed));
}

function showResult(m) {
  const a = m.attempt;
  const card = $("#last-result");
  card.hidden = false;
  const [label, cls] = OUTCOMES[a.outcome] || [a.outcome, "fail"];
  $(".result-title", card).innerHTML = `<span class="pill ${cls === "success" ? "done" : cls === "timeout" ? "stopped" : "failed"}">${outcomeIcon(a.outcome)} ${label}</span> ${esc(a.name)}`;
  const td = a.touchdown_vy == null ? "–" : `${fmt(-a.touchdown_vy, 1)} m/s`;
  const lim = a.touchdown_limit == null ? "pieds incassables" : `limite ${fmt(a.touchdown_limit, 1)} m/s`;
  const tdBad = a.touchdown_vy != null && a.touchdown_limit != null && -a.touchdown_vy > a.touchdown_limit;
  const ap = m.autopilot;
  const stat = (v, l, bad) => `<span class="stat"><b${bad ? ' class="bad"' : ""}>${v}</b><small>${l}</small></span>`;
  $(".result-stats", card).innerHTML =
    stat(`${a.rank} / ${a.total}`, "classement") + stat(fmt(a.score, 1), "score") +
    stat(`${fmt(a.duration, 1)} s`, "durée") + stat(`${Math.round(a.fuel * 100)}%`, "carburant restant") +
    stat(a.outcome === "success" ? fmt(Math.abs(a.x_on_ship), 2) : "–", "écart au centre (0 = parfait)") +
    stat(td, `vitesse au contact (${lim})`, tdBad);
  $(".result-ref", card).textContent = ap
    ? `Autopilote sur la même graine : ${outcomeIcon(ap.outcome)} ${outcomeLabel(ap.outcome)}, score ${fmt(ap.score, 1)}.`
    : "";
  card.dataset.id = a.id;
  drawTrajectories($("canvas.alt", card), a.trajectory, ap?.trajectory, 1, "Altitude (m)");
  drawTrajectories($("canvas.vy", card), a.trajectory, ap?.trajectory, 2, "Vitesse verticale (m/s)", a.touchdown_limit);
}

function drawTrajectories(canvas, mine, ref, idx, title, limit) {
  const dpr = window.devicePixelRatio || 1;
  const r = canvas.getBoundingClientRect();
  canvas.width = Math.round(r.width * dpr); canvas.height = Math.round(r.height * dpr);
  const c = canvas.getContext("2d"), w = canvas.width, h = canvas.height;
  const css = getComputedStyle(document.documentElement);
  const col = (n) => css.getPropertyValue(n).trim();
  const pad = { l: 34 * dpr, r: 8 * dpr, t: 16 * dpr, b: 16 * dpr };
  c.clearRect(0, 0, w, h);
  const all = [...(mine || []), ...(ref || [])];
  if (!all.length) return;
  const tMax = Math.max(...all.map((p) => p[0]));
  let vals = all.map((p) => p[idx]);
  if (limit != null) vals = vals.concat([-limit]);
  let lo = Math.min(0, ...vals), hi = Math.max(0, ...vals);
  if (hi - lo < 1) hi = lo + 1;
  const X = (t) => pad.l + (t / tMax) * (w - pad.l - pad.r);
  const Y = (v) => pad.t + (1 - (v - lo) / (hi - lo)) * (h - pad.t - pad.b);
  c.font = `${10 * dpr}px system-ui`; c.fillStyle = col("--muted"); c.textAlign = "left";
  c.fillText(title, pad.l, 11 * dpr);
  c.textAlign = "right";
  c.fillText(Math.round(hi), pad.l - 4 * dpr, Y(hi) + 4 * dpr);
  c.fillText(Math.round(lo), pad.l - 4 * dpr, Y(lo));
  c.textAlign = "center"; c.fillText(`${(tMax / 60).toFixed(0)} s`, w - 14 * dpr, h - 3 * dpr);
  c.strokeStyle = col("--line"); c.lineWidth = dpr;
  c.beginPath(); c.moveTo(pad.l, Y(0)); c.lineTo(w - pad.r, Y(0)); c.stroke();
  if (limit != null) {
    c.strokeStyle = col("--bad"); c.setLineDash([4 * dpr, 4 * dpr]);
    c.beginPath(); c.moveTo(pad.l, Y(-limit)); c.lineTo(w - pad.r, Y(-limit)); c.stroke(); c.setLineDash([]);
  }
  const line = (pts, color, dash) => {
    if (!pts?.length) return;
    c.strokeStyle = color; c.lineWidth = 2 * dpr; c.setLineDash(dash ? [5 * dpr, 4 * dpr] : []);
    c.beginPath(); pts.forEach((p, i) => (i ? c.lineTo(X(p[0]), Y(p[idx])) : c.moveTo(X(p[0]), Y(p[idx])))); c.stroke();
    c.setLineDash([]);
  };
  line(ref, col("--info"), true);
  line(mine, col("--accent"), false);
}

function initArena() {
  $("#play-all").addEventListener("click", startAll);
  $("#add-arena").addEventListener("click", () => addArena("autopilot"));
  $("#seed-random").addEventListener("click", () => { $("#seed").value = Math.floor(Math.random() * 10000); });
  $("#speed").addEventListener("change", () => arenas.forEach((a) => a.send({ type: "speed", value: Number($("#speed").value) })));
  $("#camera").addEventListener("change", () => arenas.forEach((a) => { a.renderer.camMode = $("#camera").value; }));
  const name = $("#player-name");
  name.value = store.get("player", "");
  name.addEventListener("input", () => store.set("player", name.value));
  const controls = $("#controls");
  controls.value = store.get("controls", "simple");
  controls.addEventListener("change", () => { store.set("controls", controls.value); updateControlsHelp(); });
  updateControlsHelp();
  $("#board").addEventListener("click", boardAction);
  $("#last-result .replay").addEventListener("click", () => watchReplay($("#last-result").dataset.id));
  $("#last-result .compare").addEventListener("click", () => compareWithAI($("#last-result").dataset.id));
  $("#last-result .again").addEventListener("click", () => { const a = humanTarget(); if (a) a.start(); });
  const saved = store.get("arenas", null);
  (saved?.length ? saved : ["autopilot", "human"]).forEach((v) => addArena(v));
  refreshBoard();
  // the scenario may change in the sidebar: keep the leaderboard in sync
  let lastScen = "";
  setInterval(() => {
    const s = JSON.stringify(boardScenario());
    if (s !== lastScen) { lastScen = s; refreshBoard(); }
  }, 1000);
  const loop = (t) => { arenas.forEach((a) => a.renderer.draw(t)); requestAnimationFrame(loop); };
  requestAnimationFrame(loop);
}

// ===================================================================== trainings

function drawChart(canvas, history) {
  const dpr = window.devicePixelRatio || 1;
  const r = canvas.getBoundingClientRect();
  canvas.width = Math.round(r.width * dpr); canvas.height = Math.round(r.height * dpr);
  const c = canvas.getContext("2d"), w = canvas.width, h = canvas.height;
  const css = getComputedStyle(document.documentElement);
  const padL = 30 * dpr, padB = 18 * dpr, padT = 6 * dpr, padR = 8 * dpr;
  c.clearRect(0, 0, w, h);
  c.strokeStyle = css.getPropertyValue("--line"); c.lineWidth = dpr;
  c.fillStyle = css.getPropertyValue("--muted"); c.font = `${10 * dpr}px system-ui`;
  for (const v of [0, 0.5, 1]) {
    const y = padT + (1 - v) * (h - padT - padB);
    c.beginPath(); c.moveTo(padL, y); c.lineTo(w - padR, y); c.stroke();
    c.textAlign = "right"; c.fillText(`${v * 100}%`, padL - 4 * dpr, y + 3 * dpr);
  }
  if (!history?.length) { c.textAlign = "center"; c.fillText("en attente des premiers épisodes…", w / 2, h / 2); return; }
  const tMax = history[history.length - 1].t;
  c.textAlign = "center";
  c.fillText(`${Math.round(tMax / 1000)}k pas`, w - 30 * dpr, h - 4 * dpr);
  const series = [["success_rate", css.getPropertyValue("--ok")], ["crash_rate", css.getPropertyValue("--bad")], ["timeout_rate", css.getPropertyValue("--warn")]];
  for (const [key, color] of series) {
    c.strokeStyle = color; c.lineWidth = 2 * dpr; c.beginPath();
    history.forEach((p, i) => {
      const x = padL + (p.t / tMax) * (w - padL - padR), y = padT + (1 - (p[key] ?? 0)) * (h - padT - padB);
      i ? c.lineTo(x, y) : c.moveTo(x, y);
    });
    c.stroke();
  }
}

const runCards = new Map();
const histories = new Map();

function runCard(r) {
  let card = runCards.get(r.name);
  if (!card) {
    card = document.createElement("article");
    card.className = "run";
    card.innerHTML = `
      <div class="run-head"><span class="run-title"></span><span class="pill"></span></div>
      <div class="run-meta"></div>
      <div class="progress"><div></div></div>
      <canvas class="chart"></canvas>
      <div class="legend"><span><i style="background:var(--ok)"></i>atterrissages</span><span><i style="background:var(--bad)"></i>crashs</span><span><i style="background:var(--warn)"></i>temps écoulé</span></div>
      <div class="run-stats"></div>
      <div class="run-actions">
        <button class="small watch">Voir dans l'arène</button>
        <button class="small ghost load">Charger sa reward</button>
        <button class="small ghost logbtn">Log</button>
        <button class="small ghost stop">Arrêter</button>
        <button class="small ghost del">Supprimer</button>
      </div>
      <pre class="log" hidden></pre>`;
    $(".watch", card).addEventListener("click", () => watchRun(r.name));
    $(".load", card).addEventListener("click", () => loadRunSettings(r.name));
    $(".stop", card).addEventListener("click", async () => { await fetch(`/api/runs/${r.name}/stop`, { method: "POST" }); refreshRuns(); });
    $(".del", card).addEventListener("click", async () => {
      if (!confirm(`Supprimer ${r.name} ?`)) return;
      const res = await fetch(`/api/runs/${r.name}`, { method: "DELETE" });
      if (res.ok) { card.remove(); runCards.delete(r.name); refreshRuns(); } else alert((await res.json()).detail);
    });
    $(".logbtn", card).addEventListener("click", async () => {
      const pre = $(".log", card); pre.hidden = !pre.hidden;
      if (!pre.hidden) { pre.textContent = (await (await fetch(`/api/runs/${r.name}/log`)).json()).log || "(vide)"; pre.scrollTop = 1e9; }
    });
    runCards.set(r.name, card);
  }
  $(".run-title", card).textContent = r.name;
  $(".run-title", card).title = r.name;
  const pill = $(".pill", card); pill.className = `pill ${r.status}`;
  pill.textContent = { running: "en cours", starting: "démarrage", done: "terminé", stopped: "arrêté", failed: "erreur" }[r.status] || r.status;
  const p = r.progress || {};
  const ch = r.custom_reward ? "reward perso" : "reward de base";
  $(".run-meta", card).textContent = `${r.algo || "PPO"} · niveau ${r.level} · ${r.continuous === false ? "discret" : "continu"} · ${ch} · ${Math.round((p.timesteps || 0) / 1000)}k / ${Math.round((r.timesteps || 0) / 1000)}k pas`;
  $(".progress > div", card).style.width = `${Math.min(100, 100 * (p.timesteps || 0) / (r.timesteps || 1))}%`;
  const l = p.latest || {}, ev = r.eval;
  const stat = (v, label) => `<span class="stat"><b>${v}</b><small>${label}</small></span>`;
  $(".run-stats", card).innerHTML = ev
    ? stat(`${Math.round(ev.success_rate * 100)}%`, "réussite (éval)") + stat(`${Math.round(ev.crash_rate * 100)}%`, "crash (éval)") +
      stat(fmt(ev.mean_return, 1), "reward moy.") + stat(ev.mean_abs_x_when_landed == null ? "–" : fmt(ev.mean_abs_x_when_landed, 3), "écart au centre")
    : stat(l.success_rate == null ? "–" : `${Math.round(l.success_rate * 100)}%`, "réussite") + stat(l.crash_rate == null ? "–" : `${Math.round(l.crash_rate * 100)}%`, "crash") +
      stat(fmt(l.episode_return, 1), "reward moy.") + stat(p.episodes ?? 0, "épisodes");
  $(".stop", card).hidden = r.status !== "running" && r.status !== "starting";
  $(".watch", card).disabled = !(r.has_best || r.has_last);
  $(".load", card).hidden = !r.reward_code;
  return card;
}

async function refreshRuns() {
  try { RUNS = await (await fetch("/api/runs")).json(); } catch { return; }
  const box = $("#runs");
  if (!RUNS.length) box.innerHTML = `<div class="empty">Aucun entraînement pour l'instant. Choisis un défi à gauche puis « Lancer l'entraînement ».</div>`;
  else $(".empty", box)?.remove();
  RUNS.forEach((r, i) => {
    const card = runCard(r);
    if (box.children[i] !== card) box.insertBefore(card, box.children[i] || null);
  });
  for (const [name, card] of runCards) if (!RUNS.some((r) => r.name === name)) { card.remove(); runCards.delete(name); }
  const active = RUNS.filter((r) => r.status === "running" || r.status === "starting");
  const badge = $("#runs-badge"); badge.hidden = !active.length; badge.textContent = active.length;
  // fetch learning curves of visible runs (all running ones, others once)
  for (const r of RUNS) {
    if (r.status === "running" || !histories.has(r.name) || histories.get(r.name).t !== r.progress?.timesteps) {
      fetch(`/api/runs/${r.name}`).then((x) => x.json()).then((full) => {
        histories.set(r.name, { t: r.progress?.timesteps, h: full.progress?.history || [] });
        const card = runCards.get(r.name); if (card) drawChart($("canvas.chart", card), full.progress?.history);
      });
    } else {
      const card = runCards.get(r.name); if (card) drawChart($("canvas.chart", card), histories.get(r.name).h);
    }
  }
  arenas.forEach((a) => a.fillAgents());
}

function watchRun(name) {
  switchTab("arena");
  const a = arenas[0] || addArena();
  a.scenario = null;
  a.fillAgents("model:" + name);
  a.select.value = "model:" + name;
  a.wins = a.total = 0; a.updateTally(); a.updateNote(); saveArenas();
  a.start();
}

async function loadRunSettings(name) {
  const r = await (await fetch(`/api/runs/${name}`)).json();
  if (r.reward_code) { state.code = r.reward_code; $("#reward-code").value = r.reward_code; }
  applyLevel(r.level, r.config || {});
}

async function startTraining() {
  const msg = $("#train-msg");
  const body = {
    name: $("#run-name").value || META.challenges[state.challenge]?.title || "run",
    level: state.level, continuous: $("#mode").value === "cont", algo: $("#algo").value,
    timesteps: Number($("#timesteps").value), gamma: Number($("#gamma").value),
    config: diffFromLevel(), reward: state.reward, reward_code: state.code,
  };
  $("#train").disabled = true;
  try {
    const res = await fetch("/api/train", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    const data = await res.json();
    msg.hidden = false;
    if (!res.ok) { msg.className = "msg error"; msg.textContent = data.detail; }
    else { msg.className = "msg ok"; msg.textContent = `Entraînement « ${data.name} » lancé.`; refreshRuns(); }
  } finally { $("#train").disabled = false; }
}

function diffFromLevel() {
  const base = META.levels[state.level], out = {};
  for (const [k, v] of Object.entries(state.config)) if (JSON.stringify(base[k]) !== JSON.stringify(v)) out[k] = v;
  return out;
}

// ===================================================================== tabs, theme, boot

function switchTab(name) {
  $$(".tab").forEach((t) => t.classList.toggle("active", t.dataset.tab === name));
  $$(".tab-page").forEach((p) => p.classList.toggle("active", p.id === `tab-${name}`));
  store.set("tab", name);
  if (name === "runs") refreshRuns();
}

function initTheme() {
  const saved = store.get("theme", null);
  if (saved) document.documentElement.dataset.theme = saved;
  $("#theme-toggle").addEventListener("click", () => {
    const cur = document.documentElement.dataset.theme || (matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark");
    const next = cur === "light" ? "dark" : "light";
    document.documentElement.dataset.theme = next; store.set("theme", next);
    runCards.forEach((card, name) => drawChart($("canvas.chart", card), histories.get(name)?.h));
  });
}

async function boot() {
  initTheme();
  META = await (await fetch("/api/meta")).json();
  initSidebar();
  await refreshRuns();
  initArena();
  $$(".tab").forEach((t) => t.addEventListener("click", () => switchTab(t.dataset.tab)));
  $("#train").addEventListener("click", startTraining);
  const params = new URLSearchParams(location.search);
  switchTab(params.get("tab") || store.get("tab", "arena"));
  setInterval(refreshRuns, 2500);
  // ?autoplay&speed=2&camera=follow : handy for demos and screenshots
  if (params.has("speed")) $("#speed").value = params.get("speed");
  if (params.has("camera")) { $("#camera").value = params.get("camera"); arenas.forEach((a) => { a.renderer.camMode = params.get("camera"); }); }
  if (params.has("autoplay")) setTimeout(startAll, 300);
}

boot();
