// Intro: the whole screen is the back of an off-white catalog envelope with a
// string-and-button closure, a red wax seal pressed over the flap's edge just
// right of the string. Tapping it: the string is unwound off the two buttons
// (a small rope simulation), the flap peels open and cracks the wax (a paper
// simulation, its corners lagging the middle), and the envelope is drawn
// down and away; the site underneath is what was inside.
//
// main.js calls window.__envelope.open(onEnter) on the tap; without WebGL, or
// with reduced motion, the flat seal stays and the tap enters straight away.

import * as THREE from "three";

const intro = document.getElementById("intro");
const button = intro?.querySelector(".intro__seal");
const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;

// World units: the screen is 1 high and A wide; z is towards the viewer.
const Z = { inner: 0.0012, back: 0.0062, hinge: 0.0088 };
const PAPER = "#f2f1ec";
const EDGE_Y = 0.04;                 // the flap's lower edge (0 = middle of the screen)
const N = 36;                        // flap particles, hinge to edge
const RB = 0.032;                    // button radius
const YT = EDGE_Y + 0.078, YB = EDGE_Y - 0.088;   // the two buttons
const RR = 0.0023;                   // string radius
const M = 170;                       // string particles
const SEAL = 0.135, SX = 0.118, SY = EDGE_Y - 0.008;

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
  const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true, powerPreference: "high-performance" });
  renderer.setClearColor(0x000000, 0);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFShadowMap;

  const seal = await loadImage("assets/img/wax-seal.png");

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(28, 1, 0.01, 50);
  const D0 = 0.5 / Math.tan(THREE.MathUtils.degToRad(14));

  scene.add(new THREE.HemisphereLight(0xffffff, 0xe4e1d8, 1.95));
  const sun = new THREE.DirectionalLight(0xfffaf2, 1.3);
  sun.position.set(-0.6, 0.9, 2.2);
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  sun.shadow.radius = 4;
  sun.shadow.bias = -0.0003;
  sun.shadow.normalBias = 0.0015;
  scene.add(sun, sun.target);

  // the table under it all (the site): only the envelope's shadow shows
  const ground = new THREE.Mesh(new THREE.PlaneGeometry(12, 12), new THREE.ShadowMaterial({ opacity: 0.26 }));
  ground.receiveShadow = true;
  scene.add(ground);

  const fibre = normalMap(512, mulberry(3));
  fibre.wrapS = fibre.wrapT = THREE.RepeatWrapping;
  const twine = twineTexture();

  // ---- everything that depends on the screen's shape -------------------
  let E = null;
  let A = 1, vw = 1, vh = 1;
  function build() {
    if (E) { scene.remove(E.env); E.dispose(); }
    const ir = intro.getBoundingClientRect();
    vw = Math.max(ir.width, 1); vh = Math.max(ir.height, 1); A = vw / vh;
    E = envelope({ A, vw, seal, fibre, twine, renderer });
    scene.add(E.env);
    Object.assign(sun.shadow.camera, { left: -A / 2 - 0.4, right: A / 2 + 0.4, top: 1.2, bottom: -2.4, near: 0.5, far: 6 });
    sun.shadow.camera.updateProjectionMatrix();
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setSize(vw, vh, false);
    camera.aspect = A;
    camera.updateProjectionMatrix();
    // the tap area: the string closure and the wax
    const x0 = -0.075, x1 = SX + SEAL / 2 + 0.01, y0 = YB - RB - 0.012, y1 = YT + RB + 0.012;
    Object.assign(button.style, {
      left: `${vw / 2 + ((x0 + x1) / 2) * vh}px`, top: `${(0.5 - (y0 + y1) / 2) * vh}px`,
      width: `${(x1 - x0) * vh}px`, height: `${(y1 - y0) * vh}px`,
    });
  }

  // ---- the opening ------------------------------------------------------
  const T = { u0: 0.05, u1: 0.95, drop: 1.2, lift: 1.22, release: 1.98, enter: 2.0, slide: 2.05, slideEnd: 3.0, end: 3.5 };
  const clamp01 = (x) => Math.min(Math.max(x, 0), 1);
  const ease = (x) => 0.5 - 0.5 * Math.cos(Math.PI * clamp01(x));
  const easeOut = (x) => 1 - Math.pow(1 - clamp01(x), 3);
  const pull = (x) => { x = clamp01(x); return x * x * (2.2 - 1.2 * x); };   // a hand pulling: slow, then away

  let opened = false, t0 = 0, onEnter = null, entered = false, frozen = false;

  // hands: one unwinds the string, then lifts the flap by its edge
  function hands(t) {
    const S = E.sim, R = E.rope;
    if (t < T.u0) { R.k = M; R.finger = null; }
    else if (t < T.u1) {
      const a = ease((t - T.u0) / (T.u1 - T.u0));
      R.k = Math.max(1, Math.round((M - 1) * (1 - a)));
      R.finger = R.unwindTarget(R.k);
    } else if (t < T.drop) {
      R.k = 1;
      R.finger = R.dropTarget(ease((t - T.u1) / (T.drop - T.u1)));
    } else { R.k = 1; R.finger = null; }
    R.closed = t < T.lift;
    if (t >= T.lift && t < T.release) {
      const a = (t - T.lift) / (T.release - T.lift);
      const peel = a < 0.18 ? 0.1 * easeOut(a / 0.18) : 0.1 + 1.85 * ease((a - 0.18) / 0.82);
      const r = S.LF * (1 - 0.08 * Math.sin(Math.PI * clamp01(a * 1.1)));
      S.pin = { u: -r * Math.cos(peel), z: r * Math.sin(peel), w: 0.5 };
    } else S.pin = null;
    S.lag = t > T.lift ? 0.5 * Math.sin(Math.PI * clamp01((t - T.lift) / (T.release - T.lift + 0.25))) : 0;
  }

  const shown = { dist: D0 };
  function pose(t) {
    const { env, sealLow, sealHigh } = E;
    const c = T.lift + 0.06;                                   // the wax gives as the flap pulls
    const jolt = t > c ? Math.exp(-(t - c) * 9) * Math.sin((t - c) * 46) : 0;
    sealLow.rotation.z = (t > c ? -0.03 : 0) + jolt * 0.02;
    sealHigh.rotation.z = (t > c ? 0.025 : 0) - jolt * 0.02;
    shown.dist = D0 * (1 + 0.16 * ease((t - T.lift + 0.1) / 0.9));
    const s = pull((t - T.slide) / (T.slideEnd - T.slide));
    env.position.y = -2.1 * s;
    env.rotation.z = -0.03 * s + jolt * 0.003;
    E.flapShadow.material.opacity = 1 - clamp01((t - T.lift) / 0.35);
  }

  // ---- loop -----------------------------------------------------------
  let raf = 0, last = performance.now(), acc = 0;
  const H_STEP = 1 / 240;
  function draw(t) {
    pose(t);
    E.place();
    camera.position.set(0, 0, shown.dist);
    renderer.render(scene, camera);
  }
  function frame(now) {
    raf = requestAnimationFrame(frame);
    if (frozen) return;
    const dt = Math.min((now - last) / 1000, 1 / 8);          // slow frames still keep the paper in time
    last = now;
    const t = opened ? (now - t0) / 1000 : -1;
    acc += dt;
    while (acc >= H_STEP) { hands(t); E.step(H_STEP); acc -= H_STEP; }
    draw(t);
    if (opened && !entered && t >= T.enter) { entered = true; onEnter?.(); }
    if (opened && t > T.end) stop();
  }
  function stop() {
    cancelAnimationFrame(raf);
    removeEventListener("resize", onResize);
    E.dispose();
    renderer.dispose();
    canvas.remove();
  }
  let resizeTimer = 0;
  const onResize = () => { if (opened) return; clearTimeout(resizeTimer); resizeTimer = setTimeout(build, 150); };

  build();
  intro.appendChild(canvas);
  draw(-1);
  canvas.getBoundingClientRect();                              // start the fade from opacity 0
  intro.classList.add("is-3d");                                // fade the canvas in over the flat paper
  const live = () => intro.classList.add("is-live");           // once it covers the screen, the site may show through
  canvas.addEventListener("transitionend", live, { once: true });
  addEventListener("resize", onResize);
  raf = requestAnimationFrame(frame);

  window.__envelope = {
    open(cb) {
      if (opened) return;
      opened = true; onEnter = cb; t0 = performance.now();
      canvas.style.transition = "none"; canvas.style.opacity = "1"; live();   // tapped before the fade ended
    },
    // for checking frames: the opening as it is t seconds after the tap
    preview(t) {
      frozen = true; opened = true; E.reset();
      for (let s = 0; s < t; s += H_STEP) { hands(s); E.step(H_STEP); }
      draw(t);
    },
  };
}

// ---------------------------------------------------------------------------
// One envelope for one screen shape.

function envelope({ A, vw, seal: sealImg, fibre, twine, renderer }) {
  const rand = mulberry(7);
  const EW = A + 0.04, EH = 1.04;            // a little bleed past every edge
  const HINGE = EH / 2;
  const DF = HINGE - EDGE_Y;                 // flap depth
  const TAPER = 0.035, RC = 0.03;            // the flap narrows a little and has rounded corners
  const BACK_TOP = HINGE - 0.03;             // the back stops short of the hinge
  const disposables = [];
  const keep = (x) => (disposables.push(x), x);

  const env = new THREE.Group();
  fibre.repeat.set(5 * EW, 5 * EH);
  const paper = (map, extra = {}) => keep(new THREE.MeshStandardMaterial({
    map, normalMap: fibre, normalScale: new THREE.Vector2(0.2, 0.2), roughness: 0.94, metalness: 0, ...extra,
  }));
  const aniso = renderer.capabilities.getMaxAnisotropy();
  const tex = (c) => { const t = keep(new THREE.CanvasTexture(c)); t.colorSpace = THREE.SRGBColorSpace; t.anisotropy = aniso; return t; };
  const sheetFor = (w) => {
    const h = Math.round(w * EH / EW);
    return { w, h, X: (x) => (x / EW + 0.5) * w, Y: (y) => (0.5 - y / EH) * h, k: w / EW };
  };
  // the flap's half width at depth s below the hinge
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

  // ---- inside: the paper's own inner face, deep in shadow -----------------
  const IN = sheetFor(256);
  const innerC = sheet(IN.w, IN.h, "#d9d6cd", rand, 8, 30);
  {
    const g = innerC.getContext("2d");
    const grad = g.createLinearGradient(0, 0, 0, IN.h * 0.12);
    grad.addColorStop(0, "rgba(30,26,18,0.1)"); grad.addColorStop(1, "rgba(30,26,18,0.4)");
    g.fillStyle = grad; g.fillRect(0, 0, IN.w, IN.h);
  }
  const inner = new THREE.Mesh(keep(new THREE.PlaneGeometry(EW * 0.995, EH * 0.995)), paper(tex(innerC)));
  inner.position.z = Z.inner;
  inner.receiveShadow = true;
  env.add(inner);

  // ---- the back: side seams, the bottom fold ----------------------------
  const BK = sheetFor(Math.min(2048, Math.round(vw * 1.2)));
  const backC = sheet(BK.w, BK.h, PAPER, rand, 7, Math.round(700 * EW));
  {
    const g = backC.getContext("2d"), { X, Y } = BK;
    const px = BK.k * 0.0014;
    for (const side of [-1, 1]) {
      const sx = X(side * (EW / 2 - 0.045));
      g.fillStyle = "rgba(60,54,40,0.035)";
      side < 0 ? g.fillRect(0, 0, sx, BK.h) : g.fillRect(sx, 0, BK.w - sx, BK.h);
      g.save(); g.filter = `blur(${px * 2.5}px)`; g.fillStyle = "rgba(60,54,40,0.13)";
      g.fillRect(side < 0 ? sx + px : sx - px * 4, 0, px * 3, BK.h); g.restore();
      g.fillStyle = "rgba(255,255,255,0.7)"; g.fillRect(sx - px * 0.5, 0, px, BK.h);
    }
    const bf = -EH / 2 + 0.14;
    g.fillStyle = "rgba(255,255,255,0.18)"; g.fillRect(0, Y(bf), BK.w, BK.h);
    g.save(); g.filter = `blur(${px * 2}px)`; g.fillStyle = "rgba(60,54,40,0.12)"; g.fillRect(0, Y(bf) - px * 3, BK.w, px * 3); g.restore();
    g.fillStyle = "rgba(255,255,255,0.75)"; g.fillRect(0, Y(bf), BK.w, px);
    g.fillStyle = "rgba(255,255,255,0.7)"; g.fillRect(0, Y(BACK_TOP), BK.w, px * 1.2);
  }
  const backGeo = keep(new THREE.PlaneGeometry(EW, BACK_TOP + EH / 2));
  backGeo.translate(0, (BACK_TOP - EH / 2) / 2, 0);
  rectUV(backGeo, EW, EH);
  const back = new THREE.Mesh(backGeo, paper(tex(backC)));
  back.position.z = Z.back;
  back.castShadow = back.receiveShadow = true;
  env.add(back);

  // the soft shadow the closed flap's edge lays on the back (fades as it lifts)
  const FS = sheetFor(1024);
  const fsC = document.createElement("canvas"); fsC.width = FS.w; fsC.height = FS.h;
  {
    const g = fsC.getContext("2d"), px = FS.k * 0.0014;
    g.filter = `blur(${px * 4}px)`; g.fillStyle = "rgba(50,44,32,0.28)";
    g.translate(px * 2, px * 3); outline(g, FS); g.fill();
    g.setTransform(1, 0, 0, 1, 0, 0); g.filter = "none"; g.globalCompositeOperation = "destination-out";
    outline(g, FS); g.fill();
  }
  const flapShadow = new THREE.Mesh(keep(new THREE.PlaneGeometry(EW, EH)), keep(new THREE.MeshBasicMaterial({ map: tex(fsC), transparent: true, depthWrite: false })));
  flapShadow.position.z = Z.back + 0.0004;
  env.add(flapShadow);

  // ---- the flap -----------------------------------------------------------
  const FO = sheetFor(Math.min(2048, Math.round(vw * 1.2)));
  const flapC = sheet(FO.w, FO.h, PAPER, rand, 7, Math.round(700 * EW));
  {
    const g = flapC.getContext("2d");
    const grad = g.createLinearGradient(0, 0, 0, FO.k * 0.025);
    grad.addColorStop(0, "rgba(60,54,40,0.16)"); grad.addColorStop(1, "rgba(60,54,40,0)");
    g.fillStyle = grad; g.fillRect(0, 0, FO.w, FO.k * 0.025);               // the fold at the hinge
    g.strokeStyle = "rgba(255,255,255,0.85)"; g.lineWidth = FO.k * 0.0016;
    outline(g, FO, 0.0012); g.stroke();                                        // its pressed edge
  }
  const FI = sheetFor(1024);
  const flapInC = sheet(FI.w, FI.h, "#e9e7e0", rand, 7, 200);
  {
    const g = flapInC.getContext("2d");
    g.save(); g.beginPath(); g.rect(0, 0, FI.w, FI.h); outline(g, FI, 0.028); g.clip("evenodd");
    g.fillStyle = "rgba(205,170,100,0.22)"; outline(g, FI, 0.004); g.fill();  // the band of gum
    g.restore();
  }

  const LF = DF, SEG = LF / N, FC = 44;
  const flapGeo = keep(new THREE.BufferGeometry());
  const fpos = new Float32Array((N + 1) * (FC + 1) * 3), fuv = new Float32Array((N + 1) * (FC + 1) * 2), fidx = [];
  const across = new Float32Array((N + 1) * (FC + 1));
  for (let j = 0; j <= N; j++) {
    const s = j * SEG, half = flapHalf(s);
    for (let i = 0; i <= FC; i++) {
      const x = (i / FC * 2 - 1) * half, k = j * (FC + 1) + i;
      across[k] = x;
      fuv[k * 2] = x / EW + 0.5;
      fuv[k * 2 + 1] = (HINGE - s) / EH + 0.5;
      if (j < N && i < FC) { const a = k, b = k + 1, c = k + FC + 1, d = c + 1; fidx.push(a, c, b, b, c, d); }
    }
  }
  flapGeo.setAttribute("position", new THREE.BufferAttribute(fpos, 3));
  flapGeo.setAttribute("uv", new THREE.BufferAttribute(fuv, 2));
  flapGeo.setIndex(fidx);
  const flapOut = new THREE.Mesh(flapGeo, paper(tex(flapC), { side: THREE.FrontSide }));
  const flapIn = new THREE.Mesh(flapGeo, paper(tex(flapInC), { side: THREE.BackSide }));
  for (const m of [flapOut, flapIn]) { m.castShadow = m.receiveShadow = true; env.add(m); }
  flapOut.customDepthMaterial = keep(new THREE.MeshDepthMaterial({ depthPacking: THREE.RGBADepthPacking, side: THREE.DoubleSide }));

  // ---- the buttons: board discs on brass rivets ---------------------------
  const disc = keep(new THREE.CylinderGeometry(RB, RB * 0.985, 0.0026, 56));
  disc.rotateX(Math.PI / 2);
  const rivet = keep(new THREE.CylinderGeometry(0.0062, 0.0068, 0.0018, 28));
  rivet.rotateX(Math.PI / 2);
  const boardMat = keep(new THREE.MeshStandardMaterial({ map: tex(boardTexture(rand)), roughness: 0.85, metalness: 0 }));
  const brassMat = keep(new THREE.MeshStandardMaterial({ map: tex(rivetTexture()), roughness: 0.38, metalness: 0.35 }));
  const makeButton = () => {
    const b = new THREE.Group();
    const d = new THREE.Mesh(disc, boardMat); d.castShadow = d.receiveShadow = true;
    const r = new THREE.Mesh(rivet, brassMat); r.position.z = 0.0021; r.castShadow = true;
    b.add(d, r);
    return b;
  };
  const DISC_Z = 0.0078;                     // raised on its rivet, the string runs beneath
  const bottomButton = makeButton();
  bottomButton.position.set(0, YB, Z.back + DISC_Z);
  env.add(bottomButton);
  const topPivot = new THREE.Object3D();
  const topButton = makeButton();
  const TOP_ROW = Math.round((HINGE - YT) / SEG);
  const topOffY = -((HINGE - TOP_ROW * SEG) - YT);
  topButton.position.set(0, topOffY, DISC_Z - (Z.hinge - Z.back));
  topPivot.add(topButton);
  env.add(topPivot);

  // ---- the seal over the edge, right of the string -------------------------
  const cut = sealCut(sealImg, -(EDGE_Y - SY) / SEAL, rand);
  const sealMat = (map) => keep(new THREE.MeshStandardMaterial({ map, transparent: true, alphaTest: 0.35, roughness: 0.42, metalness: 0, emissive: 0xffffff, emissiveMap: map, emissiveIntensity: 0.34 }));
  const sealDepth = (map) => keep(new THREE.MeshDepthMaterial({ depthPacking: THREE.RGBADepthPacking, map, alphaTest: 0.35 }));
  const sealGeo = keep(new THREE.PlaneGeometry(SEAL, SEAL));
  const lowTex = tex(cut.low), highTex = tex(cut.high);
  const sealLow = new THREE.Mesh(sealGeo, sealMat(lowTex));
  sealLow.customDepthMaterial = sealDepth(lowTex);
  sealLow.position.set(SX, SY, Z.back + 0.0042);
  sealLow.castShadow = true;
  env.add(sealLow);
  const SEAL_ROW = Math.round((HINGE - (EDGE_Y + 0.02)) / SEG);
  const sealPivot = new THREE.Object3D();
  const sealHigh = new THREE.Mesh(sealGeo, sealMat(highTex));
  sealHigh.customDepthMaterial = sealDepth(highTex);
  sealHigh.castShadow = true;
  sealHigh.position.set(0, -((HINGE - SEAL_ROW * SEG) - SY), 0.0016);
  sealPivot.add(sealHigh);
  env.add(sealPivot);

  // a soft contact shadow past the envelope's edges, for when it moves
  const ccv = document.createElement("canvas"); ccv.width = 256; ccv.height = Math.round(256 * EH / EW);
  {
    const g = ccv.getContext("2d");
    g.filter = "blur(8px)"; g.fillStyle = "rgba(30,26,18,0.36)";
    g.fillRect(ccv.width * 0.06, ccv.height * 0.06, ccv.width * 0.88, ccv.height * 0.88);
  }
  const contact = new THREE.Mesh(keep(new THREE.PlaneGeometry(EW * 1.12, EH * 1.12)), keep(new THREE.MeshBasicMaterial({ map: tex(ccv), transparent: true, depthWrite: false })));
  contact.position.set(0.01, -0.012, 0.0002);
  env.add(contact);

  // ---- flap simulation (position based, along its depth) ------------------
  const p = new Float32Array((N + 1) * 2), q = new Float32Array((N + 1) * 2);
  const G = 20;
  const floor = (u) => (u < 0 ? 0 : -Z.hinge + 0.0009);
  const sim = { LF, pin: null, lag: 0 };
  function flapReset() { for (let i = 0; i <= N; i++) { p[i * 2] = q[i * 2] = -i * SEG; p[i * 2 + 1] = q[i * 2 + 1] = 0; } }
  function link(a, b, len, k) {
    const du = p[b * 2] - p[a * 2], dz = p[b * 2 + 1] - p[a * 2 + 1];
    const d = Math.hypot(du, dz) || 1e-6, e = ((d - len) / d) * k;
    if (a === 0) { p[b * 2] -= du * e; p[b * 2 + 1] -= dz * e; return; }
    p[a * 2] += du * e * 0.5; p[a * 2 + 1] += dz * e * 0.5;
    p[b * 2] -= du * e * 0.5; p[b * 2 + 1] -= dz * e * 0.5;
  }
  function flapStep(h) {
    for (let i = 1; i <= N; i++) {
      const u = p[i * 2], z = p[i * 2 + 1];
      const vu = (u - q[i * 2]) * 0.99, vz = (z - q[i * 2 + 1]) * 0.99;
      q[i * 2] = u; q[i * 2 + 1] = z;
      p[i * 2] = u + vu;
      p[i * 2 + 1] = z + vz - G * h * h;
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
  // a point on the flap: row j, across x; the corners lag the middle
  const flapPoint = (j, x) => {
    const half = EW / 2, f = 1 - sim.lag * (x / half) ** 2, ru = -j * SEG;
    return [HINGE + ru + (p[j * 2] - ru) * f, Z.hinge + p[j * 2 + 1] * f];
  };
  const pinTo = (obj, j, x) => {
    const [y, z] = flapPoint(j, x), [y1, z1] = flapPoint(j + 1, x);
    const tu = y1 - y, tz = z1 - z, tl = Math.hypot(tu, tz) || 1;
    obj.position.set(x, y, z);
    obj.rotation.x = Math.atan2(-tz / tl, -tu / tl);
  };

  // ---- the string: wound in a figure of eight, then a rope ----------------
  // what the string rests on: paper, the closed flap, or over the wax
  const onWax = (x, y) => (x - SX) ** 2 + (y - SY) ** 2 < (SEAL * 0.42) ** 2;
  function surfClosed(x, y) { return onWax(x, y) ? Z.hinge + 0.005 : y > EDGE_Y && Math.abs(x) < flapHalf(DF) ? Z.hinge : Z.back; }
  function surfOpen(x, y) { return y < BACK_TOP ? (onWax(x, y) ? Z.back + 0.006 : Z.back) : y < HINGE ? Z.inner : 0; }
  const rope = (() => {
    // the wound path: from under the top button, one and a half turns, a tail
    const mid = (YT + YB) / 2, d = (YT - YB) / 2 + RB * 0.55, w = RB * 2.7;
    const dense = [[0, YT, 0]];
    const TURNS = 1.5, steps = 600;
    for (let i = 0; i <= steps; i++) {
      const t = (i / steps) * TURNS * 2 * Math.PI, layer = Math.floor(t / (2 * Math.PI));
      const spread = 1 + 0.1 * layer + 0.025 * Math.sin(t * 3.1);
      dense.push([w * spread * Math.sin(t) * Math.cos(t) + 0.002 * (layer - 1), mid + d * Math.cos(t) * (1 + 0.03 * layer), layer]);
    }
    const end = dense[dense.length - 1];
    for (let i = 1; i <= 40; i++) {
      const a = i / 40;
      dense.push([end[0] - 0.07 * a - 0.02 * Math.sin(a * Math.PI), end[1] - 0.075 * a + 0.012 * Math.sin(a * Math.PI * 1.5), 3]);
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
      path[i * 3] = x; path[i * 3 + 1] = y; path[i * 3 + 2] = surfClosed(x, y) + RR + dense[j][2] * RR * 0.9;
    }
    return { path, seg, mid, P: new Float32Array(M * 3), Q: new Float32Array(M * 3), k: M, finger: null, closed: true, anchor: [0, YT, 0] };
  })();
  // while unwinding, the hand circles back along the eight, outside the buttons
  rope.unwindTarget = (k) => {
    const { path, mid } = rope;
    const x = path[k * 3], y = path[k * 3 + 1];
    let dx = x, dy = y - mid; const l = Math.hypot(dx, dy) || 1;
    dx /= l; dy /= l;
    // the hand keeps what it has unwound almost taut, so the loops come off
    const reach = (M - 1 - k) * rope.seg * 0.9;
    return [x + dx * reach, y + dy * reach, Z.hinge + 0.012 + Math.min(reach, 0.2) * 0.25];
  };
  let dropFrom = null;
  rope.dropTarget = (a) => {
    if (!dropFrom) dropFrom = rope.unwindTarget(1);
    // swung round by the left, away from the wax, and laid down to the lower left
    // at nearly the string's full length, so it is laid down, not bunched up
    const L = rope.seg * (M - 1) * 0.85;
    const via = [-0.97 * L, YT + 0.2 * L, Z.hinge + 0.05], to = [-0.42 * L, YT - 0.9 * L, Z.back + 0.015];
    const b = 1 - a;
    return [0, 1, 2].map((c) => b * b * dropFrom[c] + 2 * a * b * via[c] + a * a * to[c]);
  };
  function ropeReset() { rope.P.set(rope.path); rope.Q.set(rope.path); rope.k = M; rope.finger = null; rope.closed = true; dropFrom = null; }
  const RG = 7;
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
    if (k >= M) { P.set(path); Q.set(path); return; }
    const pinned = (i) => i < k || (finger && i === M - 1);
    for (let i = 0; i < M; i++) {
      if (pinned(i)) continue;
      for (let c = 0; c < 3; c++) {
        const v = (P[i * 3 + c] - Q[i * 3 + c]) * 0.985;
        Q[i * 3 + c] = P[i * 3 + c];
        P[i * 3 + c] += v;
      }
      P[i * 3 + 2] -= RG * h * h;
    }
    for (let i = 0; i < k; i++) {
      if (i === 0 && !rope.closed) { P[0] = rope.anchor[0]; P[1] = rope.anchor[1]; P[2] = rope.anchor[2]; }
      else for (let c = 0; c < 3; c++) P[i * 3 + c] = Q[i * 3 + c] = path[i * 3 + c];
    }
    if (finger) for (let c = 0; c < 3; c++) P[(M - 1) * 3 + c] = finger[c];
    const surf = rope.closed ? surfClosed : surfOpen;
    for (let it = 0; it < 10; it++) {
      for (let i = 0; i < M - 1; i++) ropeLink(i, i + 1, rope.seg, 1, pinned(i) ? 0 : 1, pinned(i + 1) ? 0 : 1);
      for (let i = 0; i < M - 2; i++) ropeLink(i, i + 2, rope.seg * 2, 0.25, pinned(i) ? 0 : 1, pinned(i + 2) ? 0 : 1);   // cord stiffness: soft curves, no kinks
      for (let i = 0; i < M; i++) {
        if (pinned(i)) continue;
        const f = surf(P[i * 3], P[i * 3 + 1]) + RR;
        if (P[i * 3 + 2] < f) {
          P[i * 3 + 2] = f;
          Q[i * 3] += (P[i * 3] - Q[i * 3]) * 0.35;                       // friction on the paper
          Q[i * 3 + 1] += (P[i * 3 + 1] - Q[i * 3 + 1]) * 0.35;
        }
      }
    }
  }

  // the string's mesh: a twisted cord swept along the particles
  const RS = 7;
  const ropeGeo = keep(new THREE.BufferGeometry());
  const rpos = new Float32Array(M * (RS + 1) * 3), rnor = new Float32Array(M * (RS + 1) * 3), ruv = new Float32Array(M * (RS + 1) * 2), ridx = [];
  for (let i = 0; i < M; i++) for (let r = 0; r <= RS; r++) {
    const k = i * (RS + 1) + r;
    ruv[k * 2] = r / RS; ruv[k * 2 + 1] = i * 0.9;
    if (i < M - 1 && r < RS) { const a = k, b = k + 1, c = k + RS + 1, d = c + 1; ridx.push(a, b, c, b, d, c); }
  }
  ropeGeo.setAttribute("position", new THREE.BufferAttribute(rpos, 3));
  ropeGeo.setAttribute("normal", new THREE.BufferAttribute(rnor, 3));
  ropeGeo.setAttribute("uv", new THREE.BufferAttribute(ruv, 2));
  ropeGeo.setIndex(ridx);
  const ropeMesh = new THREE.Mesh(ropeGeo, keep(new THREE.MeshStandardMaterial({ map: twine, roughness: 0.95, metalness: 0 })));
  ropeMesh.castShadow = true; ropeMesh.receiveShadow = true;
  ropeMesh.frustumCulled = false;
  env.add(ropeMesh);
  const T_ = new THREE.Vector3(), N_ = new THREE.Vector3(), B_ = new THREE.Vector3(), up = new THREE.Vector3(0, 0, 1);
  function sweepRope() {
    const P = rope.P;
    for (let i = 0; i < M; i++) {
      const a = Math.max(i - 1, 0), b = Math.min(i + 1, M - 1);
      T_.set(P[b * 3] - P[a * 3], P[b * 3 + 1] - P[a * 3 + 1], P[b * 3 + 2] - P[a * 3 + 2]).normalize();
      if (i === 0) N_.crossVectors(T_, up);
      if (N_.lengthSq() < 1e-8) N_.set(1, 0, 0);
      N_.addScaledVector(T_, -N_.dot(T_)).normalize();                  // parallel transport
      B_.crossVectors(T_, N_);
      for (let r = 0; r <= RS; r++) {
        const ang = (r / RS) * Math.PI * 2, c = Math.cos(ang), s = Math.sin(ang), k = i * (RS + 1) + r;
        const nx = N_.x * c + B_.x * s, ny = N_.y * c + B_.y * s, nz = N_.z * c + B_.z * s;
        rpos[k * 3] = P[i * 3] + nx * RR; rpos[k * 3 + 1] = P[i * 3 + 1] + ny * RR; rpos[k * 3 + 2] = P[i * 3 + 2] + nz * RR;
        rnor[k * 3] = nx; rnor[k * 3 + 1] = ny; rnor[k * 3 + 2] = nz;
      }
    }
    ropeGeo.attributes.position.needsUpdate = true;
    ropeGeo.attributes.normal.needsUpdate = true;
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
    flapGeo.computeBoundingSphere();
    pinTo(topPivot, TOP_ROW, 0);
    pinTo(sealPivot, SEAL_ROW, SX);
    sweepRope();
  }
  function step(h) {
    flapStep(h);
    // the string is tied under the top button, which rides on the flap
    pinTo(topPivot, TOP_ROW, 0);
    const c = Math.cos(topPivot.rotation.x), s = Math.sin(topPivot.rotation.x), oz = RR + 0.0004;
    rope.anchor[0] = 0;
    rope.anchor[1] = topPivot.position.y + topOffY * c - oz * s;
    rope.anchor[2] = topPivot.position.z + topOffY * s + oz * c;
    ropeStep(h);
  }
  function reset() { flapReset(); ropeReset(); }
  reset();

  return {
    env, sealLow, sealHigh, flapShadow, sim, rope, step, reset, place,
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
    g.strokeStyle = `rgba(${120 + rand() * 40},${112 + rand() * 30},${92 + rand() * 30},${0.02 + rand() * 0.035})`;
    g.lineWidth = (0.4 + rand() * 0.6) * scale;
    g.beginPath(); g.moveTo(x, y);
    g.quadraticCurveTo(x + Math.cos(a) * l * 0.5 + (rand() - 0.5) * 6 * scale, y + Math.sin(a) * l * 0.5 + (rand() - 0.5) * 6 * scale, x + Math.cos(a) * l, y + Math.sin(a) * l);
    g.stroke();
  }
  return c;
}

// waxed cotton twine: a warm natural cord with its twist
function twineTexture() {
  const c = document.createElement("canvas"); c.width = 64; c.height = 64;
  const g = c.getContext("2d");
  g.fillStyle = "#c9b18a"; g.fillRect(0, 0, 64, 64);
  g.save(); g.transform(1, 0, -0.9, 1, 0, 0);
  for (let i = -64; i < 128; i += 10) {
    const grad = g.createLinearGradient(i, 0, i + 10, 0);
    grad.addColorStop(0, "rgba(90,66,36,0.38)"); grad.addColorStop(0.5, "rgba(255,240,210,0.25)"); grad.addColorStop(1, "rgba(90,66,36,0.38)");
    g.fillStyle = grad; g.fillRect(i, 0, 10, 64);
  }
  g.restore();
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  return t;
}

// pressed fibre board, oxblood, with a debossed ring
function boardTexture(rand) {
  const S = 256, c = document.createElement("canvas"); c.width = c.height = S;
  const g = c.getContext("2d");
  g.fillStyle = "#6e3a2a"; g.fillRect(0, 0, S, S);
  for (let i = 0; i < 1400; i++) { g.fillStyle = `rgba(${rand() < 0.5 ? "20,8,4" : "180,120,90"},${rand() * 0.12})`; g.fillRect(rand() * S, rand() * S, 1 + rand() * 2, 1 + rand() * 2); }
  g.strokeStyle = "rgba(20,8,4,0.35)"; g.lineWidth = 3; g.beginPath(); g.arc(S / 2, S / 2, S * 0.4, 0, Math.PI * 2); g.stroke();
  g.strokeStyle = "rgba(255,200,170,0.12)"; g.lineWidth = 2; g.beginPath(); g.arc(S / 2 - 1, S / 2 - 1, S * 0.4, 0, Math.PI * 2); g.stroke();
  return c;
}

function rivetTexture() {
  const S = 64, c = document.createElement("canvas"); c.width = c.height = S;
  const g = c.getContext("2d");
  const grad = g.createRadialGradient(S * 0.4, S * 0.4, 2, S / 2, S / 2, S / 2);
  grad.addColorStop(0, "#efe2bd"); grad.addColorStop(0.55, "#bfa66d"); grad.addColorStop(1, "#7a6236");
  g.fillStyle = grad; g.fillRect(0, 0, S, S);
  g.strokeStyle = "rgba(60,40,10,0.5)"; g.lineWidth = 3; g.beginPath(); g.arc(S / 2, S / 2, S * 0.18, 0, Math.PI * 2); g.stroke();
  return c;
}

// Two canvases of the seal: above the crack (stuck to the flap) and below it.
// crackV is how far below the seal's centre the crack runs, as a share of its size.
function sealCut(img, crackV, rand) {
  const S = 768;
  const pts = [];
  let y = S * (0.5 + crackV);
  for (let x = -10; x <= S + 10; x += 14 + rand() * 22) {
    y += (rand() - 0.5) * 16;
    y += (S * (0.5 + crackV) - y) * 0.25;
    pts.push([x, y]);
  }
  const half = (top) => {
    const c = document.createElement("canvas"); c.width = c.height = S;
    const g = c.getContext("2d");
    g.drawImage(img, 0, 0, S, S);
    g.globalCompositeOperation = "destination-in";
    g.beginPath();
    g.moveTo(-10, top ? -10 : S + 10);
    for (const [x, yy] of pts) g.lineTo(x, yy);
    g.lineTo(S + 10, top ? -10 : S + 10);
    g.closePath(); g.fill();
    g.globalCompositeOperation = "source-atop";
    g.strokeStyle = "rgba(40,6,4,0.5)"; g.lineWidth = 3;
    g.beginPath(); pts.forEach(([x, yy], i) => (i ? g.lineTo(x, yy) : g.moveTo(x, yy))); g.stroke();
    return c;
  };
  return { high: half(true), low: half(false) };
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

function rectUV(geo, W, H) {
  const pos = geo.attributes.position, uv = geo.attributes.uv;
  for (let i = 0; i < pos.count; i++) uv.setXY(i, pos.getX(i) / W + 0.5, pos.getY(i) / H + 0.5);
  uv.needsUpdate = true;
}

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = src;
  });
}

function mulberry(a) {
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
