// "Mark problems": paint over the 3D hero and pin written notes, then save
// one image (the scene with your marks and numbered notes) plus a .json with
// the notes, the camera position and every setting. Part of `?tune`.

const COLORS = { red: "#ff3b30", yellow: "#ffd60a", cyan: "#32d4ff", white: "#ffffff" };
const STORE = "pannalin-hero3d-notes";

export default function addNotes(gui, { SETTINGS, TEST, invalidate, snapshot, cameraOffset }) {
  // --- the drawing layer, over the hero, under the panel --------------------
  const layer = document.createElement("canvas");
  Object.assign(layer.style, { position: "fixed", inset: "0", width: "100vw", height: "100vh",
    zIndex: "999", pointerEvents: "none", touchAction: "none" });
  document.body.append(layer);
  const ctx = layer.getContext("2d");

  // strokes and pins are kept in viewport fractions (0..1) so they survive resizes
  const state = load() || { strokes: [], notes: [] };
  const save = () => {
    try { localStorage.setItem(STORE, JSON.stringify(state)); } catch { /* storage blocked */ }
  };
  const tool = { mode: "off", color: "red", size: 8 };
  let drawing = null;

  function resize() {
    const dpr = window.devicePixelRatio || 1;
    layer.width = Math.round(innerWidth * dpr);
    layer.height = Math.round(innerHeight * dpr);
    redraw();
  }
  window.addEventListener("resize", resize);

  function redraw(target = ctx, w = layer.width, h = layer.height, clear = true) {
    if (clear) target.clearRect(0, 0, w, h);
    const k = w / innerWidth;                         // CSS px -> target px
    for (const s of state.strokes) {
      target.strokeStyle = COLORS[s.color];
      target.globalAlpha = 0.55;
      target.lineWidth = s.size * k;
      target.lineCap = target.lineJoin = "round";
      target.beginPath();
      s.points.forEach(([x, y], i) => (i ? target.lineTo(x * w, y * h) : target.moveTo(x * w, y * h)));
      if (s.points.length === 1) target.lineTo(s.points[0][0] * w + 0.1, s.points[0][1] * h);
      target.stroke();
    }
    target.globalAlpha = 1;
    state.notes.forEach((n, i) => pin(target, n, i + 1, w, h, k));
  }

  function pin(target, n, num, w, h, k) {
    const x = n.x * w, y = n.y * h, r = 13 * k;
    target.fillStyle = COLORS[n.color];
    target.beginPath(); target.arc(x, y, r, 0, Math.PI * 2); target.fill();
    target.fillStyle = "#000";
    target.font = `bold ${14 * k}px system-ui, sans-serif`;
    target.textAlign = "center"; target.textBaseline = "middle";
    target.fillText(String(num), x, y + 0.5 * k);
    // the note text in a dark box beside the pin
    target.font = `${13 * k}px system-ui, sans-serif`;
    target.textAlign = "left";
    const lines = wrap(target, n.text, 260 * k);
    const lh = 17 * k, pad = 7 * k, bw = Math.max(...lines.map((l) => target.measureText(l).width)) + pad * 2;
    let bx = x + r + 6 * k, by = y - r;
    if (bx + bw > w) bx = x - r - 6 * k - bw;
    target.fillStyle = "rgba(0,0,0,0.78)";
    target.fillRect(bx, by, bw, lines.length * lh + pad * 2);
    target.fillStyle = "#fff";
    lines.forEach((l, i) => target.fillText(l, bx + pad, by + pad + lh * (i + 0.5)));
  }

  function wrap(target, text, max) {
    const out = []; let line = "";
    for (const word of String(text).split(/\s+/)) {
      const test = line ? line + " " + word : word;
      if (target.measureText(test).width > max && line) { out.push(line); line = word; } else line = test;
    }
    out.push(line);
    return out;
  }

  // --- input -------------------------------------------------------------------
  const at = (e) => [e.clientX / innerWidth, e.clientY / innerHeight];
  layer.addEventListener("pointerdown", (e) => {
    if (tool.mode === "draw") {
      drawing = { color: tool.color, size: tool.size, points: [at(e)] };
      state.strokes.push(drawing);
      layer.setPointerCapture(e.pointerId);
      redraw();
    } else if (tool.mode === "note") {
      const text = prompt("What's wrong here?");
      if (text && text.trim()) {
        const [x, y] = at(e);
        state.notes.push({ x, y, color: tool.color, text: text.trim() });
        save(); redraw(); refreshList();
      }
    } else if (tool.mode === "erase") {
      const [x, y] = at(e);
      const near = (px, py) => Math.hypot((px - x) * innerWidth, (py - y) * innerHeight) < 24;
      const before = state.strokes.length + state.notes.length;
      state.strokes = state.strokes.filter((s) => !s.points.some(([px, py]) => near(px, py)));
      state.notes = state.notes.filter((n) => !near(n.x, n.y));
      if (state.strokes.length + state.notes.length !== before) { save(); redraw(); refreshList(); }
    }
  });
  layer.addEventListener("pointermove", (e) => {
    if (!drawing) return;
    drawing.points.push(at(e));
    redraw();
  });
  const end = () => { if (drawing) { drawing = null; save(); } };
  layer.addEventListener("pointerup", end);
  layer.addEventListener("pointercancel", end);

  // Drawing freezes the camera where it is, so the marks stay on what they point at.
  let held = null;
  function setMode(mode) {
    layer.style.pointerEvents = mode === "off" ? "none" : "auto";
    layer.style.cursor = { draw: "crosshair", note: "copy", erase: "not-allowed" }[mode] || "";
    if (mode !== "off" && !held) {
      held = { motion: TEST.motion, peekX: TEST.peekX, peekY: TEST.peekY };
      const o = cameraOffset();
      Object.assign(TEST, { motion: "hold", peekX: o.x, peekY: o.y });
      invalidate();
      gui.controllersRecursive().forEach((c) => c.updateDisplay());
    }
  }

  // --- panel ---------------------------------------------------------------------
  const f = gui.addFolder("Mark problems");
  f.add(tool, "mode", { "off (move the camera)": "off", "paint": "draw", "add a note": "note", "erase": "erase" })
    .name("tool").onChange(setMode);
  f.add(tool, "color", Object.keys(COLORS)).name("colour");
  f.add(tool, "size", 2, 40, 1).name("brush size");
  const list = document.createElement("div");
  list.className = "tune-note";
  f.$children.append(list);
  function refreshList() {
    list.textContent = "";
    if (!state.notes.length) { list.textContent = "No notes yet. Pick “add a note” and click the scene."; return; }
    state.notes.forEach((n, i) => {
      const row = document.createElement("div");
      row.textContent = `${i + 1}. ${n.text}`;
      list.append(row);
    });
  }
  refreshList();
  const actions = {
    undo() {
      // undo the most recent mark, whichever kind it was
      if (state.strokes.length) state.strokes.pop(); else state.notes.pop();
      save(); redraw(); refreshList();
    },
    clear() {
      if (!confirm("Remove all marks and notes?")) return;
      state.strokes = []; state.notes = [];
      save(); redraw(); refreshList();
    },
    async download() { await exportForClaude(); },
  };
  f.add(actions, "undo").name("Undo last mark");
  f.add(actions, "clear").name("Clear all marks");
  f.add(actions, "download").name("Save image + notes for Claude");

  async function exportForClaude() {
    const shot = await loadImage(snapshot());
    const out = document.createElement("canvas");
    out.width = shot.width; out.height = shot.height;
    const o = out.getContext("2d");
    o.drawImage(shot, 0, 0);
    redraw(o, out.width, out.height, false);        // marks and notes on top, at full size
    redraw();
    const stamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
    const name = `hero-notes-${stamp}`;
    // One file only: browsers often block a second download in a row, so the
    // image travels inside the .json (Claude extracts it).
    const info = {
      image: out.toDataURL("image/jpeg", 0.9),
      saved: new Date().toISOString(),
      viewport: { width: innerWidth, height: innerHeight },
      camera: { ...cameraOffset(), motion: TEST.motion, orbitBoost: TEST.orbitBoost },
      notes: state.notes.map((n, i) => ({ number: i + 1, text: n.text, color: n.color,
        at: { x: +n.x.toFixed(3), y: +n.y.toFixed(3) } })),
      painted: state.strokes.map((s) => ({ color: s.color, points: s.points.length,
        box: box(s.points) })),
      settings: SETTINGS,
    };
    // Send it straight to the dev server (tools/serve.py saves it in notes/,
    // where Claude reads it); download it instead if that is not available.
    const body = JSON.stringify(info, null, 2);
    try {
      const res = await fetch("/__notes", { method: "POST", body, headers: { "X-Name": name } });
      if (!res.ok) throw new Error(res.status);
      const { saved } = await res.json();
      alert(`Saved to ${saved}. Tell Claude "check my notes".`);
    } catch {
      saveFile(new Blob([body], { type: "application/json" }), `${name}.json`);
    }
  }

  resize();
}

function box(points) {
  const xs = points.map((p) => p[0]), ys = points.map((p) => p[1]);
  const r = (v) => +v.toFixed(3);
  return { left: r(Math.min(...xs)), top: r(Math.min(...ys)), right: r(Math.max(...xs)), bottom: r(Math.max(...ys)) };
}

function loadImage(src) {
  return new Promise((res, rej) => { const i = new Image(); i.onload = () => res(i); i.onerror = rej; i.src = src; });
}

function saveFile(blob, name) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = name;
  document.body.append(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 2000);
}

function load() {
  try { return JSON.parse(localStorage.getItem(STORE)); } catch { return null; }
}
