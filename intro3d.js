// Intro: a catalog envelope floating in the middle of a sheet of gridded paper,
// slowly turning, drawn in fine black ink dots like a stippled specimen plate.
// Its back has a string-and-button closure and a 3D red wax seal over the
// flap's edge; the string's loose end hangs and swings with the float.
//
// The scene is rendered once to a buffer holding, per pixel, how lit the
// surface is and a random number fixed to the surface itself; a second pass
// turns that into ink dots on the paper. Because the random numbers ride on
// the surfaces, the dots stay put as it turns: no crawling, no flicker.
//
// Tapping it: the envelope turns to face you and settles, the string is
// unwound off the buttons, the flap is lifted open, and it drifts down and
// away while the intro fades.
//
// main.js calls window.__envelope.open(onEnter) on the tap; without WebGL, or
// with reduced motion, the flat seal stays and the tap enters straight away.

import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

const intro = document.getElementById("intro");
const button = intro?.querySelector(".intro__seal");
const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;

// The envelope in its own units: 0.7 wide, 1 tall, its back facing +z.
const EW = 0.7, EH = 1, TH = 0.022;
const BZ = TH / 2;                         // the back's surface
const HINGE = EH / 2, DF = 0.3, EDGE_Y = HINGE - DF;
const FZ = BZ + 0.0016;                    // the closed flap
const BACK_TOP = HINGE - 0.026;            // the back panel stops short of the fold: the mouth
const N = 30;                              // flap particles, fold to edge
const RB = 0.036;                          // button radius
const YT = EDGE_Y + 0.072, YB = EDGE_Y - 0.084;
const RR = 0.0042;                         // string radius
const M = 150;                             // string particles
const SEAL = 0.19, SX = 0.205, SY = EDGE_Y - 0.004, SEAL_H = 0.014;
const SPIN = 12;                           // seconds a turn
const G_LOCAL = 30;                        // gravity, in envelope heights a second²

if (intro && button && !reduce) {
  init().catch((error) => {
    console.warn("Envelope unavailable, keeping the flat seal:", error);
    intro.classList.remove("is-3d", "is-live");
  });
}

async function init() {
  const canvas = document.createElement("canvas");
  canvas.className = "intro__envelope";
  canvas.setAttribute("aria-hidden", "true");
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: false, alpha: false, powerPreference: "high-performance" });
  renderer.toneMapping = THREE.NoToneMapping;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;

  const gltf = await new GLTFLoader().loadAsync("assets/3d/wax-seal.glb");

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(28, 1, 0.05, 20);
  const D = 0.5 / Math.tan(THREE.MathUtils.degToRad(14));
  const EL = 0.12;                                           // seen a little from above
  camera.position.set(0, D * Math.sin(EL), D * Math.cos(EL));
  camera.lookAt(0, 0, 0);

  scene.add(new THREE.HemisphereLight(0xffffff, 0x8a8a8a, 0.7));
  // the key: a lamp up and to the left, near enough that its light falls off
  // across the paper
  const sun = new THREE.SpotLight(0xffffff, 9, 0, 0.75, 1, 2);
  const SUN = new THREE.Vector3(-0.9, 1.2, 1.5).normalize();
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  Object.assign(sun.shadow.camera, { near: 0.4, far: 4 });
  sun.shadow.bias = -0.0003;
  sun.shadow.normalBias = 0.0012;
  scene.add(sun, sun.target);

  // every surface's dot size, kept at about 1.3 device pixels
  const cells = [];
  const ink = (mat, k = 1) => { const u = { value: 0.002 }; cells.push([u, k]); return inkify(mat, u); };

  const E = envelope({ gltf, ink, renderer });
  scene.add(E.env, E.ropeMesh);

  // the buffer the scene is drawn to, and the pass that draws it in ink
  const target = new THREE.WebGLRenderTarget(1, 1, { type: THREE.HalfFloatType, minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter });
  const post = new THREE.ShaderMaterial({
    uniforms: { tG: { value: target.texture }, uRes: { value: new THREE.Vector2(1, 1) }, uDpr: { value: 1 }, uExp: { value: 0.95 } },
    vertexShader: "varying vec2 vUv; void main() { vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }",
    fragmentShader: POST,
    depthTest: false, depthWrite: false,
  });
  const quad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), post);
  quad.frustumCulled = false;
  const postScene = new THREE.Scene();
  postScene.add(quad);
  const postCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
  const CLEAR = new THREE.Color(1, 0, 0);

  // ---- everything that depends on the screen's shape -------------------
  let A = 1, vw = 1, vh = 1, scale = 0.4;
  function build() {
    const ir = intro.getBoundingClientRect();
    vw = Math.max(ir.width, 1); vh = Math.max(ir.height, 1); A = vw / vh;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    renderer.setPixelRatio(dpr);
    renderer.setSize(vw, vh, false);
    const W = Math.round(vw * dpr), H = Math.round(vh * dpr);
    target.setSize(W, H);
    post.uniforms.uRes.value.set(W, H);
    post.uniforms.uDpr.value = dpr;
    camera.aspect = A;
    camera.updateProjectionMatrix();
    scale = Math.min(0.46, 0.86 * A);                         // its height, as a share of the screen's
    E.setScale(scale);
    const cellWorld = 1.3 / H;                                // the screen is 1 world unit high at the envelope
    for (const [u, k] of cells) u.value = cellWorld / (k === "world" ? 1 : scale * k);
    // the tap area: the envelope
    const w = EW * scale * vh * 1.08, h = EH * scale * vh * 1.06;
    Object.assign(button.style, { left: `${vw / 2}px`, top: `${vh / 2}px`, width: `${w}px`, height: `${h}px` });
  }

  // ---- the float -------------------------------------------------------
  const clamp01 = (x) => Math.min(Math.max(x, 0), 1);
  const ease = (x) => 0.5 - 0.5 * Math.cos(Math.PI * clamp01(x));
  const easeOut = (x) => 1 - Math.pow(1 - clamp01(x), 3);
  const pull = (x) => { x = clamp01(x); return x * x * (2.2 - 1.2 * x); };
  const OMEGA = (2 * Math.PI) / SPIN;
  const face = { y0: 0, delta: 0, dur: 1 };                   // turning to face you, once tapped
  let T = null;                                               // the opening's timeline, set on the tap

  function pose(tau, t) {
    const { env } = E;
    let yaw = OMEGA * tau, f = 1, drop = 0;
    if (t >= 0) {
      const s = clamp01(t / face.dur);
      // a cubic that leaves at the turning speed and arrives at rest
      const h10 = s * s * s - 2 * s * s + s, h01 = -2 * s * s * s + 3 * s * s;
      yaw = face.y0 + h10 * OMEGA * face.dur + h01 * face.delta;
      f = 1 - ease(t / face.dur);
      drop = pull((t - T.slide) / (T.slideEnd - T.slide));
    }
    const bob = 0.014 * Math.sin(0.8 * tau) + 0.006 * Math.sin(1.9 * tau + 1.3);
    env.position.set(0.006 * Math.sin(0.37 * tau) * f, bob * f - 1.5 * drop, 0);
    env.rotation.set(
      (0.07 * Math.sin(0.55 * tau + 0.5) + 0.03 * Math.sin(1.3 * tau)) * f + 0.35 * drop,
      yaw,
      (0.05 * Math.sin(0.43 * tau + 2) + 0.02 * Math.sin(1.1 * tau + 0.3)) * f - 0.12 * drop,
      "YXZ",
    );
    env.updateMatrixWorld(true);
  }

  // hands: unwind the string, then lift the flap by its edge
  function hands(t) {
    const S = E.sim, R = E.rope;
    R.closed = t < 0 || t < T.lift;
    if (t < 0 || t < T.u0) { R.k = E.K_TAIL; R.finger = null; }
    else if (t < T.u1) {
      const a = ease((t - T.u0) / (T.u1 - T.u0));
      R.k = Math.max(1, Math.round((E.K_TAIL - 1) * (1 - a)));
      R.finger = R.unwindTarget(R.k, easeOut((t - T.u0) / 0.35));
    } else { R.k = 1; R.finger = null; }
    if (t >= T?.lift) {
      const a = clamp01((t - T.lift) / (T.held - T.lift));
      const peel = a < 0.18 ? 0.1 * easeOut(a / 0.18) : 0.1 + 2.75 * ease((a - 0.18) / 0.82);
      const r = S.LF * (1 - 0.08 * Math.sin(Math.PI * clamp01(a * 1.1)));
      S.pin = { u: -r * Math.cos(peel), z: r * Math.sin(peel), w: 0.5 };
      S.open = true;
    } else { S.pin = null; S.open = false; }
    S.lag = t > T?.lift ? 0.5 * Math.sin(Math.PI * clamp01((t - T.lift) / (T.held - T.lift + 0.25))) : 0;
  }

  // ---- loop -----------------------------------------------------------
  const H_STEP = 1 / 240;
  let raf = 0, last = performance.now(), acc = 0, tau = 0;
  let opened = false, tOpen = 0, onEnter = null, entered = false, frozen = false;
  const tNow = () => (opened ? tau - tOpen : -1);
  function draw() {
    const t = tNow();
    E.place();
    sun.target.position.copy(E.env.position);
    sun.position.copy(E.env.position).addScaledVector(SUN, 1.25);
    renderer.setRenderTarget(target);
    renderer.setClearColor(CLEAR, 0);
    renderer.clear();
    renderer.render(scene, camera);
    renderer.setRenderTarget(null);
    renderer.render(postScene, postCam);
  }
  function tick() { tau += H_STEP; const t = tNow(); pose(tau, t); hands(t); E.step(H_STEP); }
  function frame(now) {
    raf = requestAnimationFrame(frame);
    if (frozen) return;
    acc += Math.min((now - last) / 1000, 1 / 8);
    last = now;
    while (acc >= H_STEP) { tick(); acc -= H_STEP; }
    draw();
    const t = tNow();
    if (opened && !entered && t >= T.enter) { entered = true; onEnter?.(); }
    if (opened && t > T.end) stop();
  }
  function stop() {
    cancelAnimationFrame(raf);
    removeEventListener("resize", onResize);
    E.dispose();
    target.dispose();
    renderer.dispose();
    canvas.remove();
  }
  let resizeTimer = 0;
  const onResize = () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(build, 150); };

  function startOpening() {
    // turn on to face you, the short way round unless that means stopping hard
    const y0 = OMEGA * tau, two = Math.PI * 2;
    let delta = two - (((y0 % two) + two) % two);
    if (delta > 1.3 * Math.PI) delta -= two;
    face.y0 = y0; face.delta = delta;
    face.dur = 0.9 + 0.35 * Math.abs(delta) / Math.PI;
    const f = face.dur;
    T = { u0: f - 0.15, u1: f + 0.75, lift: f + 0.85, held: f + 1.8, enter: f + 1.85, slide: f + 1.9, slideEnd: f + 2.9, end: f + 3.3 };
    E.rope.grabFrom = null;
    opened = true; tOpen = tau;
  }

  build();
  pose(0, -1); E.reset();
  for (let i = 0; i < 240; i++) tick();                       // let the string's end settle first
  intro.appendChild(canvas);
  draw();
  canvas.getBoundingClientRect();
  intro.classList.add("is-3d");
  const live = () => intro.classList.add("is-live");
  canvas.addEventListener("transitionend", live, { once: true });
  addEventListener("resize", onResize);
  raf = requestAnimationFrame(frame);

  window.__envelope = {
    open(cb) {
      if (opened) return;
      onEnter = cb; startOpening();
      canvas.style.transition = "none"; canvas.style.opacity = "1"; live();
    },
    // for checking frames: the float at tau seconds, or the opening t seconds after a tap at tau
    preview(at, t = -1) {
      frozen = true; opened = false; tau = 0; pose(0, -1); E.reset();
      while (tau < at) tick();
      if (t >= 0) { startOpening(); while (tau < tOpen + t) tick(); }
      draw();
    },
  };
}

// ---------------------------------------------------------------------------
// The ink pass: paper, its faint grid and dust, and the dots.

const POST = /* glsl */ `
precision highp float;
uniform sampler2D tG;
uniform vec2 uRes;
uniform float uDpr, uExp;
varying vec2 vUv;

float h12(vec2 p) { vec3 q = fract(vec3(p.xyx) * 0.1031); q += dot(q, q.yzx + 33.33); return fract((q.x + q.y) * q.z); }
float vn(vec2 p) {
  vec2 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);
  return mix(mix(h12(i), h12(i + vec2(1, 0)), f.x), mix(h12(i + vec2(0, 1)), h12(i + vec2(1, 1)), f.x), f.y);
}
float rep(float x, float o, float P) { return mod(x - o + 0.5 * P, P) - 0.5 * P; }
float line(float d) { return 1.0 - smoothstep(0.2, 0.9, abs(d)); }

void main() {
  vec2 css = gl_FragCoord.xy / uDpr;
  vec2 size = uRes / uDpr;
  vec2 c = css - 0.5 * size;
  float S = min(size.x, size.y);

  // the sheet: off-white, faintly mottled
  vec3 paper = vec3(0.941, 0.949, 0.937);
  paper *= 0.989 + 0.016 * (vn(css / 230.0) * 0.6 + vn(css / 70.0) * 0.4);

  // its printed grid, faint and a little uneven: single uprights, paired rules
  float P = 0.78 * S, Q = 0.373 * S, g = 0.0;
  g = max(g, line(rep(c.x, 0.13 * S, P)));
  g = max(g, line(rep(c.x, 0.32 * S, P)));
  g = max(g, line(rep(c.x, 0.39 * S, P)));
  g = max(g, line(rep(c.x, 0.46 * S, P)));
  g = max(g, line(rep(c.x, 0.65 * S, P)));
  g = max(g, line(rep(c.y, -0.02 * S, Q)));
  g = max(g, line(rep(c.y, 0.011 * S, Q)));
  paper *= 1.0 - g * 0.05 * (0.55 + 0.9 * vn(css / 45.0));

  // dust
  float dust = step(0.99935, h12(floor(css) + 91.0));
  paper = mix(paper, vec3(0.3), dust * 0.45);

  // the object: lighter surfaces take fewer dots, a soft spray past its outline
  vec4 G = texture2D(tG, vUv);
  vec2 tx = 1.0 / uRes;
  float r = 2.2 * uDpr, ab = 0.0;
  for (int i = 0; i < 8; i++) {
    float a = float(i) * 0.785398;
    ab += texture2D(tG, vUv + vec2(cos(a), sin(a)) * r * tx).a;
  }
  ab *= 0.125;
  float lum = pow(clamp(G.r * uExp, 0.0, 1.0), 0.4545);
  float d = 1.0 - lum;
  float p = 0.03 + 0.97 * pow(clamp((d - 0.04) / 0.86, 0.0, 1.0), 1.3);
  p = min(1.0, p + (1.0 - ab) * 0.55);
  float inkIn = G.a * step(G.g, p);
  float inkOut = (1.0 - G.a) * step(h12(gl_FragCoord.xy + 7.0), ab * 0.4);
  float k = max(inkIn, inkOut);
  gl_FragColor = vec4(mix(paper, vec3(0.085, 0.085, 0.08), k * 0.94), 1.0);
}`;

// A lit material that writes, instead of its colour, its brightness and a
// random number fixed to the surface (from the "rest" attribute: where that
// point of the surface sits when nothing has moved).
function inkify(mat, cell) {
  mat.onBeforeCompile = (shader) => {
    shader.uniforms.uCell = cell;
    shader.vertexShader = shader.vertexShader
      .replace("#include <common>", "#include <common>\nattribute vec3 rest;\nvarying vec3 vRest;")
      .replace("#include <begin_vertex>", "#include <begin_vertex>\nvRest = rest;");
    shader.fragmentShader = shader.fragmentShader
      .replace("#include <common>", `#include <common>
uniform float uCell;
varying vec3 vRest;
float inkHash(vec3 p) { p = fract(p * 0.1031); p += dot(p, p.zyx + 31.32); return fract((p.x + p.y) * p.z); }`)
      .replace("#include <dithering_fragment>", `#include <dithering_fragment>
float inkLum = dot(gl_FragColor.rgb, vec3(0.2126, 0.7152, 0.0722));
inkLum *= mix(0.5, 1.0, pow(clamp(abs(dot(normal, normalize(vViewPosition))), 0.0, 1.0), 0.6));
gl_FragColor = vec4(inkLum, inkHash(floor(vRest / uCell) + 0.5), 0.0, 1.0);`);
  };
  mat.customProgramCacheKey = () => "ink";
  return mat;
}
const withRest = (geo) => (geo.setAttribute("rest", geo.attributes.position.clone()), geo);

// ---------------------------------------------------------------------------
// The envelope.

function envelope({ gltf, ink, renderer }) {
  const rand = mulberry(7);
  const disposables = [];
  const keep = (x) => (disposables.push(x), x);
  const env = new THREE.Group();
  const aniso = renderer.capabilities.getMaxAnisotropy();
  const tex = (c) => { const t = keep(new THREE.CanvasTexture(c)); t.colorSpace = THREE.SRGBColorSpace; t.anisotropy = aniso; return t; };
  const fibre = keep(normalMap(512, mulberry(3)));
  fibre.wrapS = fibre.wrapT = THREE.RepeatWrapping;
  fibre.repeat.set(3, 3 * EH / EW);
  const paper = (map, extra = {}) => ink(keep(new THREE.MeshStandardMaterial({
    map, normalMap: fibre, normalScale: new THREE.Vector2(0.35, 0.35), roughness: 0.92, metalness: 0, ...extra,
  })));
  const sheetFor = (w) => {
    const h = Math.round(w * EH / EW);
    return { w, h, X: (x) => (x / EW + 0.5) * w, Y: (y) => (0.5 - y / EH) * h, k: w / EW };
  };
  const PAPER = "#ebe8e0";

  // ---- the body: a slightly filled envelope, its back with a centre seam
  const BK = sheetFor(1024);
  const backC = sheet(BK.w, BK.h, PAPER, rand, 9, 500);
  {
    const g = backC.getContext("2d"), { X, Y } = BK, px = BK.k * 0.0012;
    // the centre seam: the right panel laps over the left
    const sx = X(0.034);
    g.fillStyle = "rgba(255,255,255,0.16)"; g.fillRect(sx, Y(BACK_TOP), BK.w - sx, BK.h);
    g.save(); g.filter = `blur(${px * 2}px)`; g.fillStyle = "rgba(60,54,40,0.55)"; g.fillRect(sx - px * 3, 0, px * 3, BK.h); g.restore();
    g.fillStyle = "rgba(255,255,255,0.8)"; g.fillRect(sx, 0, px, BK.h);
    // the bottom fold
    const bf = Y(-EH / 2 + 0.11);
    g.fillStyle = "rgba(255,255,255,0.14)"; g.fillRect(0, bf, BK.w, BK.h - bf);
    g.save(); g.filter = `blur(${px * 2}px)`; g.fillStyle = "rgba(60,54,40,0.5)"; g.fillRect(0, bf - px * 3, BK.w, px * 3); g.restore();
    g.fillStyle = "rgba(255,255,255,0.8)"; g.fillRect(0, bf, BK.w, px);
    // the mouth: inside, in shadow
    const grad = g.createLinearGradient(0, 0, 0, Y(BACK_TOP));
    grad.addColorStop(0, "rgba(40,36,28,0.55)"); grad.addColorStop(1, "rgba(40,36,28,0.8)");
    g.fillStyle = grad; g.fillRect(0, 0, BK.w, Y(BACK_TOP));
    g.fillStyle = "rgba(255,255,255,0.85)"; g.fillRect(0, Y(BACK_TOP), BK.w, px * 1.2);
    edgeWear(g, BK.w, BK.h, rand);
  }
  const frontC = sheet(BK.w, BK.h, PAPER, rand, 9, 500);
  edgeWear(frontC.getContext("2d"), BK.w, BK.h, rand);
  const sideC = sheet(64, 64, "#e2ded4", rand, 8, 0);

  const bow = (x, y) => 0.02 * (x / (EW / 2)) ** 2 + 0.009 * (y / (EH / 2)) ** 2 - 0.006;
  const bodyGeo = keep(new THREE.BoxGeometry(EW, EH, TH, 28, 40, 1));
  {
    const pos = bodyGeo.attributes.position;
    for (let i = 0; i < pos.count; i++) {
      const x = pos.getX(i) / (EW / 2), y = pos.getY(i) / (EH / 2);
      const fill = (1 - x ** 6) * (1 - Math.max(y, -1) ** 6) * (y > 0.6 ? 1 - (y - 0.6) / 0.4 * 0.7 : 1);
      pos.setZ(i, pos.getZ(i) * (1 + 0.45 * fill) + bow(pos.getX(i), pos.getY(i)));
    }
    bodyGeo.computeVertexNormals();
  }
  withRest(bodyGeo);
  const sideMat = paper(tex(sideC));
  const body = new THREE.Mesh(bodyGeo, [sideMat, sideMat, sideMat, sideMat, paper(tex(backC)), paper(tex(frontC))]);
  body.castShadow = body.receiveShadow = true;
  env.add(body);
  // where the back's surface sits (it is filled out a little in the middle,
  // and the whole envelope has a gentle bow from being handled)
  const backZ = (x, y) => thick(x, y) + bow(x, y);
  const frontZ = (x, y) => -thick(x, y) + bow(x, y);
  function thick(x, y) {
    const u = x / (EW / 2), v = y / (EH / 2);
    const fill = (1 - Math.min(Math.abs(u), 1) ** 6) * (1 - Math.min(Math.abs(v), 1) ** 6) * (v > 0.6 ? 1 - (v - 0.6) / 0.4 * 0.7 : 1);
    return BZ * (1 + 0.45 * fill);
  }

  // ---- the flap -----------------------------------------------------------
  const TAPER = 0.03, RC = 0.035;
  const flapHalf = (s) => {
    let h = EW / 2 - TAPER * (s / DF);
    const r0 = DF - RC;
    if (s > r0) h -= RC - Math.sqrt(Math.max(RC * RC - (s - r0) ** 2, 0));
    return Math.max(h, 0.002);
  };
  const outline = (g, S, inset = 0) => {
    g.beginPath();
    const steps = 48;
    for (let i = 0; i <= steps; i++) { const s = DF * i / steps; g.lineTo(S.X(-flapHalf(s) + inset), S.Y(HINGE - Math.min(s, DF - inset))); }
    for (let i = steps; i >= 0; i--) { const s = DF * i / steps; g.lineTo(S.X(flapHalf(s) - inset), S.Y(HINGE - Math.min(s, DF - inset))); }
    g.closePath();
  };
  const FO = sheetFor(1024);
  const flapC = sheet(FO.w, FO.h, PAPER, rand, 9, 500);
  {
    const g = flapC.getContext("2d");
    const grad = g.createLinearGradient(0, 0, 0, FO.k * 0.03);
    grad.addColorStop(0, "rgba(60,54,40,0.3)"); grad.addColorStop(1, "rgba(60,54,40,0)");
    g.fillStyle = grad; g.fillRect(0, 0, FO.w, FO.k * 0.03);       // the fold
    g.strokeStyle = "rgba(255,255,255,0.9)"; g.lineWidth = FO.k * 0.002;
    outline(g, FO, 0.004); g.stroke();                               // its pressed edge
    g.strokeStyle = "rgba(50,44,32,0.6)"; g.lineWidth = FO.k * 0.003;
    outline(g, FO, 0); g.stroke();                                   // and its cut edge, in shadow
  }
  const flapInC = sheet(FO.w / 2, FO.h / 2, "#e3dfd5", rand, 8, 150);
  {
    const S2 = sheetFor(FO.w / 2), g = flapInC.getContext("2d");
    g.save(); g.beginPath(); g.rect(0, 0, S2.w, S2.h); outline(g, S2, 0.03); g.clip("evenodd");
    g.fillStyle = "rgba(190,150,80,0.3)"; outline(g, S2, 0.004); g.fill();   // the band of gum
    g.restore();
  }

  const LF = DF, SEG = LF / N, FC = 40;
  const flapGeo = keep(new THREE.BufferGeometry());
  const fpos = new Float32Array((N + 1) * (FC + 1) * 3), fuv = new Float32Array((N + 1) * (FC + 1) * 2);
  const frest = new Float32Array((N + 1) * (FC + 1) * 3), fidx = [];
  const across = new Float32Array((N + 1) * (FC + 1));
  for (let j = 0; j <= N; j++) {
    const s = j * SEG, half = flapHalf(s);
    for (let i = 0; i <= FC; i++) {
      const x = (i / FC * 2 - 1) * half, k = j * (FC + 1) + i;
      across[k] = x;
      fuv[k * 2] = x / EW + 0.5;
      fuv[k * 2 + 1] = (HINGE - s) / EH + 0.5;
      frest[k * 3] = x; frest[k * 3 + 1] = HINGE - s; frest[k * 3 + 2] = FZ;
      if (j < N && i < FC) { const a = k, b = k + 1, c = k + FC + 1, d = c + 1; fidx.push(a, c, b, b, c, d); }
    }
  }
  flapGeo.setAttribute("position", new THREE.BufferAttribute(fpos, 3));
  flapGeo.setAttribute("uv", new THREE.BufferAttribute(fuv, 2));
  flapGeo.setAttribute("rest", new THREE.BufferAttribute(frest, 3));
  flapGeo.setIndex(fidx);
  const flapOut = new THREE.Mesh(flapGeo, paper(tex(flapC), { side: THREE.FrontSide }));
  const flapIn = new THREE.Mesh(flapGeo, paper(tex(flapInC), { side: THREE.BackSide }));
  for (const m of [flapOut, flapIn]) { m.castShadow = m.receiveShadow = true; m.frustumCulled = false; env.add(m); }
  flapOut.customDepthMaterial = keep(new THREE.MeshDepthMaterial({ depthPacking: THREE.RGBADepthPacking, side: THREE.DoubleSide }));

  // ---- the buttons: board discs on brass rivets ---------------------------
  const disc = keep(new THREE.CylinderGeometry(RB, RB * 0.97, 0.003, 48));
  disc.rotateX(Math.PI / 2);
  withRest(disc);
  const rivet = keep(new THREE.CylinderGeometry(0.0068, 0.0074, 0.0024, 24));
  rivet.rotateX(Math.PI / 2);
  withRest(rivet);
  const stem = keep(new THREE.CylinderGeometry(0.004, 0.004, 0.012, 12));
  stem.rotateX(Math.PI / 2);
  withRest(stem);
  const boardMat = ink(keep(new THREE.MeshStandardMaterial({ map: tex(boardTexture(rand)), roughness: 0.8, metalness: 0 })));
  const brassMat = ink(keep(new THREE.MeshStandardMaterial({ color: 0xc9ad6c, roughness: 0.32, metalness: 0.7 })));
  const DISC_Z = 0.0115;                       // raised on its stem, the string runs beneath
  const makeButton = () => {
    const b = new THREE.Group();
    const d = new THREE.Mesh(disc, boardMat); d.position.z = DISC_Z;
    const r = new THREE.Mesh(rivet, brassMat); r.position.z = DISC_Z + 0.0025;
    const st = new THREE.Mesh(stem, brassMat); st.position.z = DISC_Z / 2;
    for (const m of [d, r, st]) { m.castShadow = m.receiveShadow = true; b.add(m); }
    return b;
  };
  const bottomButton = makeButton();
  bottomButton.position.set(0, YB, backZ(0, YB));
  env.add(bottomButton);
  const TOP_ROW = Math.round((HINGE - YT) / SEG);
  const topOffY = -((HINGE - TOP_ROW * SEG) - YT);
  const topPivot = new THREE.Object3D();
  const topButton = makeButton();
  topButton.position.set(0, topOffY, 0);
  topPivot.add(topButton);
  env.add(topPivot);

  // ---- the wax seal, in relief, over the flap's edge right of the string --
  let sealSrc = null;
  gltf.scene.traverse((o) => { if (o.isMesh && !sealSrc) sealSrc = o; });
  const sealGeo = sealSrc.geometry;
  sealGeo.computeBoundingBox();
  const bb = sealGeo.boundingBox, size = new THREE.Vector3(), mid = new THREE.Vector3();
  bb.getSize(size); bb.getCenter(mid);
  sealGeo.translate(-mid.x, -mid.y, -bb.min.z);
  withRest(sealGeo);
  const sealK = SEAL / Math.max(size.x, size.y);
  const sealMat = sealSrc.material;
  sealMat.roughness = 0.3;
  sealMat.color.setScalar(3);                                 // lifted, so its relief reads in the dots
  if (sealMat.normalMap) sealMat.normalScale.multiplyScalar(2.5);
  ink(sealMat, sealK);
  const sealMesh = new THREE.Mesh(sealGeo, sealMat);
  sealMesh.scale.set(sealK, sealK, (SEAL_H / size.z));
  sealMesh.castShadow = sealMesh.receiveShadow = true;
  keep(sealGeo); keep(sealMat);
  const SEAL_ROW = Math.round((HINGE - (EDGE_Y + 0.03)) / SEG);
  const sealPivot = new THREE.Object3D();
  sealMesh.position.set(0, -((HINGE - SEAL_ROW * SEG) - SY), -0.0004);
  sealMesh.rotation.z = -0.18;
  sealPivot.add(sealMesh);
  env.add(sealPivot);

  // ---- flap simulation (position based, along its depth) ------------------
  const p = new Float32Array((N + 1) * 2), q = new Float32Array((N + 1) * 2);
  const sim = { LF, pin: null, lag: 0, open: false, gu: -1, gz: 0 };
  const floor = (u) => (u < 0 ? 0 : -1);
  function flapReset() { for (let i = 0; i <= N; i++) { p[i * 2] = q[i * 2] = -i * SEG; p[i * 2 + 1] = q[i * 2 + 1] = 0; } }
  function link(a, b, len, k) {
    const du = p[b * 2] - p[a * 2], dz = p[b * 2 + 1] - p[a * 2 + 1];
    const d = Math.hypot(du, dz) || 1e-6, e = ((d - len) / d) * k;
    if (a === 0) { p[b * 2] -= du * e; p[b * 2 + 1] -= dz * e; return; }
    p[a * 2] += du * e * 0.5; p[a * 2 + 1] += dz * e * 0.5;
    p[b * 2] -= du * e * 0.5; p[b * 2 + 1] -= dz * e * 0.5;
  }
  function flapStep(h) {
    if (!sim.open) return;
    const gu = sim.gu * G_LOCAL * 0.3 * h * h, gz = sim.gz * G_LOCAL * 0.3 * h * h;
    for (let i = 1; i <= N; i++) {
      const u = p[i * 2], z = p[i * 2 + 1];
      const vu = (u - q[i * 2]) * 0.99, vz = (z - q[i * 2 + 1]) * 0.99;
      q[i * 2] = u; q[i * 2 + 1] = z;
      p[i * 2] = u + vu + gu;
      p[i * 2 + 1] = z + vz + gz;
    }
    for (let it = 0; it < 12; it++) {
      const pin = sim.pin;
      if (pin) {
        const w = pin.w;
        p[N * 2] += (pin.u - p[N * 2]) * w;
        p[N * 2 + 1] += (pin.z - p[N * 2 + 1]) * w;
        const du = pin.u - p[(N - 1) * 2], dz = pin.z - p[(N - 1) * 2 + 1], d = Math.hypot(du, dz) || 1;
        p[(N - 1) * 2] += (pin.u - (du / d) * SEG - p[(N - 1) * 2]) * w * 0.5;
        p[(N - 1) * 2 + 1] += (pin.z - (dz / d) * SEG - p[(N - 1) * 2 + 1]) * w * 0.5;
      }
      for (let i = 0; i < N; i++) link(i, i + 1, SEG, 1);
      for (let i = 1; i < N; i++) link(i - 1, i + 1, 2 * SEG, 0.4);
      for (let i = 2; i < N - 1; i++) link(i - 2, i + 2, 4 * SEG, 0.1);
      for (let i = 1; i <= N; i++) { const f = floor(p[i * 2]); if (p[i * 2 + 1] < f) { p[i * 2 + 1] = f; q[i * 2 + 1] += (f - q[i * 2 + 1]) * 0.6; } }
    }
  }
  // a point on the flap: row j, across x; the corners lag the middle, and the
  // paper keeps a little curl at its edge and corners
  const flapPoint = (j, x) => {
    const half = EW / 2, f = 1 - sim.lag * (x / half) ** 2, ru = -j * SEG, a = j / N;
    const curl = 0.0035 * a ** 3 + 0.003 * (x / flapHalf(j * SEG)) ** 4 * a ** 2;
    const zb = backZ(x, HINGE + ru) + FZ - BZ;
    return [HINGE + ru + (p[j * 2] - ru) * f, zb + p[j * 2 + 1] * f + curl];
  };
  const pinTo = (obj, j, x) => {
    const [y, z] = flapPoint(j, x), [y1, z1] = flapPoint(j + 1, x);
    const tu = y1 - y, tz = z1 - z, tl = Math.hypot(tu, tz) || 1;
    obj.position.set(x, y, z);
    obj.rotation.x = Math.atan2(-tz / tl, -tu / tl);
    obj.rotation.y = -Math.atan((flapPoint(j, x + 0.01)[1] - flapPoint(j, x - 0.01)[1]) / 0.02);
  };

  // ---- the string: wound in a figure of eight, its loose end hanging -------
  const flapTop = (x, y) => y > EDGE_Y && Math.abs(x) < flapHalf(Math.min(HINGE - y, DF));
  const onWax = (x, y) => (x - SX) ** 2 + (y - SY) ** 2 < (SEAL * 0.42) ** 2;
  function surfClosed(x, y) {
    if (onWax(x, y)) return backZ(x, y) + FZ - BZ + SEAL_H * 0.8;
    return flapTop(x, y) ? backZ(x, y) + FZ - BZ + 0.0035 * ((HINGE - y) / DF) ** 3 : backZ(x, y);
  }
  function surfOpen(x, y) { return y < BACK_TOP ? backZ(x, y) : y < HINGE ? 0 : -9; }
  const rope = (() => {
    const mid = (YT + YB) / 2, d = (YT - YB) / 2 + RB * 0.55, w = RB * 2.7;
    const dense = [[0, YT, 0]];
    const TURNS = 1.5, steps = 600;
    for (let i = 0; i <= steps; i++) {
      const t = (i / steps) * TURNS * 2 * Math.PI, layer = Math.floor(t / (2 * Math.PI));
      const spread = 1 + 0.1 * layer + 0.025 * Math.sin(t * 3.1);
      dense.push([w * spread * Math.sin(t) * Math.cos(t) + 0.002 * (layer - 1), mid + d * Math.cos(t) * (1 + 0.03 * layer), layer]);
    }
    const tailFrom = dense.length - 1, end = dense[tailFrom];
    for (let i = 1; i <= 60; i++) {
      const a = i / 60;
      dense.push([end[0] - 0.02 * a + 0.01 * Math.sin(a * Math.PI * 1.5), end[1] - 0.2 * a, 2]);
    }
    const cum = [0];
    for (let i = 1; i < dense.length; i++) cum.push(cum[i - 1] + Math.hypot(dense[i][0] - dense[i - 1][0], dense[i][1] - dense[i - 1][1]));
    const L = cum[cum.length - 1], seg = L / (M - 1);
    const path = new Float32Array(M * 3);
    let j = 0;
    for (let i = 0; i < M; i++) {
      const s = i * seg;
      while (j < cum.length - 2 && cum[j + 1] < s) j++;
      const a = (s - cum[j]) / (cum[j + 1] - cum[j] || 1);
      const x = dense[j][0] + (dense[j + 1][0] - dense[j][0]) * a, y = dense[j][1] + (dense[j + 1][1] - dense[j][1]) * a;
      path[i * 3] = x; path[i * 3 + 1] = y; path[i * 3 + 2] = surfClosed(x, y) + RR + Math.min(dense[j][2], 1) * RR * 0.9;
    }
    const K_TAIL = Math.min(M - 2, Math.ceil(cum[tailFrom] / seg) - 2);
    return { path, seg, mid, K_TAIL, P: new Float32Array(M * 3), Q: new Float32Array(M * 3), k: K_TAIL, finger: null, closed: true, anchor: [0, YT, 0], grabFrom: null };
  })();
  const K_TAIL = rope.K_TAIL;

  // envelope space ↔ world, for the string, which lives in the world
  const Mw = new THREE.Matrix4(), Mi = new THREE.Matrix4(), V = new THREE.Vector3();
  const toWorld = (x, y, z, out, o) => { V.set(x, y, z).applyMatrix4(Mw); out[o] = V.x; out[o + 1] = V.y; out[o + 2] = V.z; };
  let S = 0.4;                                                // the envelope's scale
  // while unwinding, the hand circles back along the eight, outside the buttons
  rope.unwindTarget = (k, grab) => {
    const { path, mid } = rope;
    const x = path[k * 3], y = path[k * 3 + 1];
    let dx = x, dy = y - mid; const l = Math.hypot(dx, dy) || 1;
    dx /= l; dy /= l;
    const out = [0, 0, 0];
    toWorld(x + dx * 0.07, y + dy * 0.07, backZ(x, y) + 0.05, out, 0);
    if (!rope.grabFrom) rope.grabFrom = [rope.P[(M - 1) * 3], rope.P[(M - 1) * 3 + 1], rope.P[(M - 1) * 3 + 2]];
    return out.map((v, c) => rope.grabFrom[c] + (v - rope.grabFrom[c]) * grab);
  };
  function ropeReset() {
    Mw.copy(env.matrixWorld); Mi.copy(Mw).invert();
    for (let i = 0; i < M; i++) toWorld(rope.path[i * 3], rope.path[i * 3 + 1], rope.path[i * 3 + 2], rope.P, i * 3);
    rope.Q.set(rope.P); rope.k = K_TAIL; rope.finger = null; rope.closed = true; rope.grabFrom = null;
  }
  function ropeLink(a, b, len, k, wa, wb) {
    const P = rope.P, ws = wa + wb;
    if (!ws) return;
    const dx = P[b * 3] - P[a * 3], dy = P[b * 3 + 1] - P[a * 3 + 1], dz = P[b * 3 + 2] - P[a * 3 + 2];
    const d = Math.hypot(dx, dy, dz) || 1e-6, e = ((d - len) / d) * k / ws;
    P[a * 3] += dx * e * wa; P[a * 3 + 1] += dy * e * wa; P[a * 3 + 2] += dz * e * wa;
    P[b * 3] -= dx * e * wb; P[b * 3 + 1] -= dy * e * wb; P[b * 3 + 2] -= dz * e * wb;
  }
  function ropeStep(h) {
    const { P, Q, path, k, finger } = rope;
    const seg = rope.seg * S, g = G_LOCAL * S * h * h;
    const pinned = (i) => i < k || (finger && i === M - 1);
    for (let i = 0; i < k; i++) {
      if (i === 0 && !rope.closed) toWorld(rope.anchor[0], rope.anchor[1], rope.anchor[2], P, 0);
      else toWorld(path[i * 3], path[i * 3 + 1], path[i * 3 + 2], P, i * 3);
      Q[i * 3] = P[i * 3]; Q[i * 3 + 1] = P[i * 3 + 1]; Q[i * 3 + 2] = P[i * 3 + 2];
    }
    for (let i = k; i < M; i++) {
      if (pinned(i)) continue;
      for (let c = 0; c < 3; c++) {
        const v = (P[i * 3 + c] - Q[i * 3 + c]) * 0.988;
        Q[i * 3 + c] = P[i * 3 + c];
        P[i * 3 + c] += v;
      }
      P[i * 3 + 1] -= g;
    }
    if (finger) for (let c = 0; c < 3; c++) P[(M - 1) * 3 + c] = finger[c];
    const surf = rope.closed ? surfClosed : surfOpen;
    const e = Mi.elements, w = Mw.elements;
    for (let it = 0; it < 10; it++) {
      for (let i = 0; i < M - 1; i++) ropeLink(i, i + 1, seg, 1, pinned(i) ? 0 : 1, pinned(i + 1) ? 0 : 1);
      for (let i = 0; i < M - 2; i++) ropeLink(i, i + 2, seg * 2, 0.22, pinned(i) ? 0 : 1, pinned(i + 2) ? 0 : 1);
      // it rests on the envelope, never through it
      for (let i = k; i < M; i++) {
        if (pinned(i)) continue;
        const X = P[i * 3], Y = P[i * 3 + 1], Zw = P[i * 3 + 2];
        const lx = e[0] * X + e[4] * Y + e[8] * Zw + e[12];
        const ly = e[1] * X + e[5] * Y + e[9] * Zw + e[13];
        let lz = e[2] * X + e[6] * Y + e[10] * Zw + e[14];
        if (Math.abs(lx) > EW / 2 + RR || Math.abs(ly) > EH / 2 + RR) continue;
        const top = surf(lx, ly) + RR, bottom = frontZ(lx, ly) - RR;
        if (lz >= top || lz <= bottom) continue;
        lz = lz > (top + bottom) / 2 ? top : bottom;
        P[i * 3] = w[0] * lx + w[4] * ly + w[8] * lz + w[12];
        P[i * 3 + 1] = w[1] * lx + w[5] * ly + w[9] * lz + w[13];
        P[i * 3 + 2] = w[2] * lx + w[6] * ly + w[10] * lz + w[14];
        for (let c = 0; c < 3; c++) Q[i * 3 + c] += (P[i * 3 + c] - Q[i * 3 + c]) * 0.25;   // friction on the paper
      }
    }
  }

  // the string's mesh: a cord swept along the particles, in world space
  const RS = 8;
  const ropeGeo = keep(new THREE.BufferGeometry());
  const rpos = new Float32Array(M * (RS + 1) * 3), rnor = new Float32Array(M * (RS + 1) * 3), rrest = new Float32Array(M * (RS + 1) * 3), ridx = [];
  for (let i = 0; i < M; i++) for (let r = 0; r <= RS; r++) {
    const k = i * (RS + 1) + r;
    if (i < M - 1 && r < RS) { const a = k, b = k + 1, c = k + RS + 1, d = c + 1; ridx.push(a, b, c, b, d, c); }
  }
  ropeGeo.setAttribute("position", new THREE.BufferAttribute(rpos, 3));
  ropeGeo.setAttribute("normal", new THREE.BufferAttribute(rnor, 3));
  ropeGeo.setAttribute("rest", new THREE.BufferAttribute(rrest, 3));
  ropeGeo.setIndex(ridx);
  const ropeMesh = new THREE.Mesh(ropeGeo, ink(keep(new THREE.MeshStandardMaterial({ color: 0x2b241e, roughness: 0.9, metalness: 0 })), "world"));
  ropeMesh.castShadow = ropeMesh.receiveShadow = true;
  ropeMesh.frustumCulled = false;
  const T_ = new THREE.Vector3(), N_ = new THREE.Vector3(), B_ = new THREE.Vector3(), up = new THREE.Vector3();
  function sweepRope() {
    const P = rope.P, rr = RR * S;
    up.set(Mw.elements[8], Mw.elements[9], Mw.elements[10]).normalize();
    for (let i = 0; i < M; i++) {
      const a = Math.max(i - 1, 0), b = Math.min(i + 1, M - 1);
      T_.set(P[b * 3] - P[a * 3], P[b * 3 + 1] - P[a * 3 + 1], P[b * 3 + 2] - P[a * 3 + 2]).normalize();
      if (i === 0) N_.crossVectors(T_, up);
      if (N_.lengthSq() < 1e-8) N_.set(1, 0, 0);
      N_.addScaledVector(T_, -N_.dot(T_)).normalize();
      B_.crossVectors(T_, N_);
      for (let r = 0; r <= RS; r++) {
        const ang = (r / RS) * Math.PI * 2, c = Math.cos(ang), s = Math.sin(ang), k = i * (RS + 1) + r;
        const nx = N_.x * c + B_.x * s, ny = N_.y * c + B_.y * s, nz = N_.z * c + B_.z * s;
        rpos[k * 3] = P[i * 3] + nx * rr; rpos[k * 3 + 1] = P[i * 3 + 1] + ny * rr; rpos[k * 3 + 2] = P[i * 3 + 2] + nz * rr;
        rnor[k * 3] = nx; rnor[k * 3 + 1] = ny; rnor[k * 3 + 2] = nz;
        rrest[k * 3] = i * rope.seg * S; rrest[k * 3 + 1] = (r % RS) / RS * Math.PI * 2 * rr; rrest[k * 3 + 2] = 0;
      }
    }
    ropeGeo.attributes.position.needsUpdate = true;
    ropeGeo.attributes.normal.needsUpdate = true;
    ropeGeo.attributes.rest.needsUpdate = true;
  }

  function place() {
    for (let j = 0; j <= N; j++) {
      for (let i = 0; i <= FC; i++) {
        const k = j * (FC + 1) + i, x = across[k];
        const [y, z] = flapPoint(j, x);
        fpos[k * 3] = x; fpos[k * 3 + 1] = y; fpos[k * 3 + 2] = z;
      }
    }
    flapGeo.attributes.position.needsUpdate = true;
    flapGeo.computeVertexNormals();
    pinTo(topPivot, TOP_ROW, 0);
    pinTo(sealPivot, SEAL_ROW, SX);
    sweepRope();
  }
  function step(h) {
    Mw.copy(env.matrixWorld); Mi.copy(Mw).invert();
    // gravity, as the envelope feels it, for the flap
    V.set(0, -1, 0).transformDirection(Mi);
    sim.gu = V.y; sim.gz = V.z;
    flapStep(h);
    // the string is tied under the top button, which rides on the flap
    pinTo(topPivot, TOP_ROW, 0);
    const c = Math.cos(topPivot.rotation.x), s = Math.sin(topPivot.rotation.x), oz = RR + 0.0006;
    rope.anchor[0] = 0;
    rope.anchor[1] = topPivot.position.y + topOffY * c - oz * s;
    rope.anchor[2] = topPivot.position.z + topOffY * s + oz * c;
    ropeStep(h);
  }
  function reset() { flapReset(); ropeReset(); }

  return {
    env, ropeMesh, sim, rope, K_TAIL, step, reset, place,
    setScale(s) {
      // keep the string where it was, relative to the envelope
      const old = S;
      S = s; env.scale.setScalar(s); env.updateMatrixWorld(true);
      if (old !== s) {
        const k = s / old, c = env.position;
        for (const A of [rope.P, rope.Q]) for (let i = 0; i < M; i++) for (let a = 0; a < 3; a++) A[i * 3 + a] = c.getComponent(a) + (A[i * 3 + a] - c.getComponent(a)) * k;
        Mw.copy(env.matrixWorld); Mi.copy(Mw).invert();
      }
    },
    dispose() { for (const x of disposables) x.dispose?.(); },
  };
}

// ---------------------------------------------------------------------------
// Drawn textures

function sheet(w, h, base, rand, grain, fibres) {
  const c = document.createElement("canvas"); c.width = w; c.height = h;
  const g = c.getContext("2d");
  g.fillStyle = base; g.fillRect(0, 0, w, h);
  const img = g.getImageData(0, 0, w, h), d = img.data;
  for (let i = 0; i < d.length; i += 4) { const n = (rand() - 0.5) * grain; d[i] += n; d[i + 1] += n; d[i + 2] += n * 0.9; }
  g.putImageData(img, 0, 0);
  g.lineCap = "round";
  const scale = w / 1024;
  for (let k = 0; k < fibres; k++) {
    const x = rand() * w, y = rand() * h, a = rand() * Math.PI, l = (6 + rand() * 22) * scale;
    g.strokeStyle = `rgba(${120 + rand() * 40},${112 + rand() * 30},${92 + rand() * 30},${0.03 + rand() * 0.05})`;
    g.lineWidth = (0.5 + rand() * 0.8) * scale;
    g.beginPath(); g.moveTo(x, y);
    g.quadraticCurveTo(x + Math.cos(a) * l * 0.5 + (rand() - 0.5) * 6 * scale, y + Math.sin(a) * l * 0.5 + (rand() - 0.5) * 6 * scale, x + Math.cos(a) * l, y + Math.sin(a) * l);
    g.stroke();
  }
  return c;
}

// handled paper: its edges a touch darker and soft-worn
function edgeWear(g, w, h, rand) {
  const e = w * 0.03;
  for (const [x, y, gw, gh, x0, y0, x1, y1] of [
    [0, 0, e, h, 0, 0, e, 0], [w - e, 0, e, h, w, 0, w - e, 0],
    [0, 0, w, e, 0, 0, 0, e], [0, h - e, w, e, 0, h, 0, h - e],
  ]) {
    const grad = g.createLinearGradient(x0, y0, x1, y1);
    grad.addColorStop(0, "rgba(70,62,46,0.16)"); grad.addColorStop(1, "rgba(70,62,46,0)");
    g.fillStyle = grad; g.fillRect(x, y, gw, gh);
  }
  for (let i = 0; i < 40; i++) {
    const t = rand(), side = Math.floor(rand() * 4);
    const x = side < 2 ? (side ? w - 2 : 2) : t * w, y = side < 2 ? t * h : (side === 2 ? 2 : h - 2);
    g.strokeStyle = `rgba(70,62,46,${0.06 + rand() * 0.1})`; g.lineWidth = 1 + rand() * 2;
    g.beginPath(); g.moveTo(x, y); g.lineTo(x + (rand() - 0.5) * 30, y + (rand() - 0.5) * 30); g.stroke();
  }
}

// pressed fibre board, oxblood, with a debossed ring
function boardTexture(rand) {
  const S = 256, c = document.createElement("canvas"); c.width = c.height = S;
  const g = c.getContext("2d");
  g.fillStyle = "#8a5a48"; g.fillRect(0, 0, S, S);
  for (let i = 0; i < 1400; i++) { g.fillStyle = `rgba(${rand() < 0.5 ? "20,8,4" : "180,120,90"},${rand() * 0.14})`; g.fillRect(rand() * S, rand() * S, 1 + rand() * 2, 1 + rand() * 2); }
  g.strokeStyle = "rgba(20,8,4,0.45)"; g.lineWidth = 4; g.beginPath(); g.arc(S / 2, S / 2, S * 0.4, 0, Math.PI * 2); g.stroke();
  g.strokeStyle = "rgba(255,200,170,0.2)"; g.lineWidth = 2; g.beginPath(); g.arc(S / 2 - 1, S / 2 - 1, S * 0.4, 0, Math.PI * 2); g.stroke();
  return c;
}

function normalMap(S, rand) {
  const c = document.createElement("canvas"); c.width = c.height = S;
  const g = c.getContext("2d");
  g.fillStyle = "#808080"; g.fillRect(0, 0, S, S);
  for (let k = 0; k < 2600; k++) {
    const x = rand() * S, y = rand() * S, a = rand() * Math.PI, l = 4 + rand() * 18;
    g.strokeStyle = `rgba(${rand() < 0.5 ? "0,0,0" : "255,255,255"},${0.08 + rand() * 0.12})`;
    g.lineWidth = 0.6 + rand();
    g.beginPath(); g.moveTo(x, y); g.lineTo(x + Math.cos(a) * l, y + Math.sin(a) * l); g.stroke();
  }
  const h = g.getImageData(0, 0, S, S).data;
  const out = g.createImageData(S, S), o = out.data;
  const at = (x, y) => h[(((y + S) % S) * S + ((x + S) % S)) * 4] / 255;
  for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) {
    const dx = (at(x + 1, y) - at(x - 1, y)) * 2, dy = (at(x, y + 1) - at(x, y - 1)) * 2;
    const l = Math.hypot(dx, dy, 1), i = (y * S + x) * 4;
    o[i] = (-dx / l * 0.5 + 0.5) * 255; o[i + 1] = (dy / l * 0.5 + 0.5) * 255; o[i + 2] = (1 / l * 0.5 + 0.5) * 255; o[i + 3] = 255;
  }
  g.putImageData(out, 0, 0);
  return new THREE.CanvasTexture(c);
}

function mulberry(a) {
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
