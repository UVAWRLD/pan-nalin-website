// The photo as a 3D scene.
//
// assets/scene/ holds the photo cut into layers, far to near (the room beyond
// the frame, wall and floor, trunk and cans, the boy, his near hand and the
// film in front of him). Each layer is a colour image with transparency and a
// depth map (tools/scene3d/build.py). Every layer becomes a relief mesh placed
// where the photo's camera saw it, so from that camera the layers add up to
// the photo exactly, and moving the camera gives true parallax, with real
// picture behind everything that moves aside.
//
// Colours stay in the photo's own (sRGB) numbers end to end: the textures are
// uploaded untouched and the shader writes them straight to the screen.
//
//   const scene = await createScene(canvas, options)
//   scene.start()
//
// The camera orbits a point at the boy's distance. The pointer steers it
// (drag on touch screens); left alone it drifts slowly. `dolly` moves it
// toward the boy.

import * as THREE from "three";

const VERT = /* glsl */ `
uniform sampler2D uDepth;
uniform vec2 uInv;        // 1/depth at depth-map value 0 and 65535
uniform vec4 uCam;        // focal, cx, cy (photo px), photo px per depth-map cell
uniform vec4 uRect;       // the layer image's rectangle in photo px: x, y, width, height
in vec2 ij;               // which depth-map cell this vertex sits on
out vec2 vUv;
out vec2 vPx;
void main() {
  vec4 t = texelFetch(uDepth, ivec2(ij), 0);
  float q = (floor(t.r * 255.0 + 0.5) * 256.0 + floor(t.g * 255.0 + 0.5)) / 65535.0;
  float z = 1.0 / mix(uInv.x, uInv.y, q);
  vec2 p = uRect.xy + min(ij * uCam.w, uRect.zw);     // depth is sampled at cell corners, edge to edge
  vec3 pos = vec3((p.x - uCam.y) / uCam.x * z, -(p.y - uCam.z) / uCam.x * z, -z);
  vUv = (p - uRect.xy) / uRect.zw;
  vPx = p;
  gl_Position = projectionMatrix * viewMatrix * vec4(pos, 1.0);
}`;

const FRAG = /* glsl */ `
precision highp float;
uniform sampler2D uColor;   // premultiplied colour
uniform sampler2D uAlpha;   // its transparency, kept in a separate lossless image
uniform float uOpaque;
uniform vec2 uGuard;      // stretch at which a surface starts to fade, and is gone (0 = off)
in vec2 vUv;
in vec2 vPx;
out vec4 outColor;
void main() {
  vec4 c = vec4(texture(uColor, vUv).rgb, uOpaque > 0.5 ? 1.0 : texture(uAlpha, vUv).r);
  if (uGuard.x > 0.0) {
    // Seen from the photo's camera every surface maps photo pixels to screen
    // pixels evenly. A part of the mesh that bridges a jump in depth is pulled
    // long and thin from anywhere else: fade it out, so the layer behind shows
    // instead of a smear.
    vec2 dx = dFdx(vPx), dy = dFdy(vPx);
    float a = dot(dx, dx), b = dot(dx, dy), d = dot(dy, dy);
    float h = 0.5 * (a + d), r = sqrt(max(h * h - (a * d - b * b), 0.0));
    float stretch = sqrt((h + r) / max(h - r, 1e-9));
    c *= 1.0 - smoothstep(uGuard.x, uGuard.y, stretch);
  }
  if (c.a < 0.004) discard;
  outColor = c;            // premultiplied
}`;

const loader = new THREE.TextureLoader();
const loadTexture = (url) => new Promise((resolve, reject) => loader.load(url, resolve, undefined, reject));

// A grid of vertices, one every `step` depth-map cells (the last row and column always included).
function gridGeometry(gw, gh, step) {
  const xs = [], ys = [];
  for (let i = 0; i < gw - 1; i += step) xs.push(i);
  xs.push(gw - 1);
  for (let j = 0; j < gh - 1; j += step) ys.push(j);
  ys.push(gh - 1);
  const nx = xs.length, ny = ys.length;
  const ij = new Float32Array(nx * ny * 2);
  for (let j = 0, k = 0; j < ny; j++) for (let i = 0; i < nx; i++) { ij[k++] = xs[i]; ij[k++] = ys[j]; }
  const index = new Uint32Array((nx - 1) * (ny - 1) * 6);
  for (let j = 0, k = 0; j < ny - 1; j++) {
    for (let i = 0; i < nx - 1; i++) {
      const a = j * nx + i, b = a + 1, c = a + nx, d = c + 1;
      index[k++] = a; index[k++] = c; index[k++] = b;
      index[k++] = b; index[k++] = c; index[k++] = d;
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("ij", new THREE.BufferAttribute(ij, 2));
  g.setIndex(new THREE.BufferAttribute(index, 1));
  return g;
}

const clamp = THREE.MathUtils.clamp;

export async function createScene(canvas, options = {}) {
  const o = {
    base: "assets/scene/",
    half: false,            // half-resolution pictures (phones)
    meshStep: 1,            // one vertex every this many depth-map cells (2 on phones)
    zoom: 1.06,             // >1 crops in a little, leaving the camera room to move
    focusX: 0.53, focusY: 0.4,  // which part of the photo stays in view when it is cropped (like background-position)
    maxYaw: 6, maxPitch: 3.5,   // how far the pointer swings the camera, degrees
    maxDolly: 0.35,         // how far `dolly` 1 moves the camera toward the boy, metres
    follow: 3.2,            // how quickly the camera follows (higher = snappier)
    drift: 0.4,             // idle sway, as a fraction of the pointer range (0 = none)
    driftSpeed: 1,          // how fast the idle sway goes (1 = about half a minute there and back)
    maxPixelRatio: 2,
    still: false,           // no pointer, no drift: the camera only moves through setView()
    only: null,             // for checking: draw only these layers
    ...options,
  };
  const meta = await (await fetch(o.base + "scene.json")).json();

  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false, powerPreference: "high-performance" });
  renderer.setClearColor(0x000000, 1);
  const maxAniso = renderer.capabilities.getMaxAnisotropy();
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(30, 1, 0.1, 100);

  await Promise.all(meta.layers.map(async (layer, i) => {
    const pick = (full, half) => o.base + (o.half && layer[half] ? layer[half] : layer[full]);
    const [color, depth, alpha] = await Promise.all([
      loadTexture(pick("color", "color_half")), loadTexture(o.base + layer.depth),
      layer.alpha ? loadTexture(pick("alpha", "alpha_half")) : null,
    ]);
    for (const t of [color, depth, alpha]) {
      if (t) { t.flipY = false; t.colorSpace = THREE.NoColorSpace; t.premultiplyAlpha = false; }
    }
    for (const t of [color, alpha]) {
      if (!t) continue;
      t.generateMipmaps = true;
      t.minFilter = THREE.LinearMipmapLinearFilter;
      t.magFilter = THREE.LinearFilter;
      t.anisotropy = maxAniso;
    }
    if (alpha) { alpha.format = THREE.RedFormat; alpha.internalFormat = "R8"; }
    depth.generateMipmaps = false;
    depth.minFilter = depth.magFilter = THREE.NearestFilter;

    const [gw, gh] = layer.grid;
    const material = new THREE.ShaderMaterial({
      glslVersion: THREE.GLSL3,
      vertexShader: VERT,
      fragmentShader: FRAG,
      uniforms: {
        uDepth: { value: depth },
        uColor: { value: color },
        uAlpha: { value: alpha },
        uInv: { value: new THREE.Vector2(layer.inv_min, layer.inv_max) },
        uCam: { value: new THREE.Vector4(meta.focal, meta.cx, meta.cy, layer.cell) },
        uRect: { value: new THREE.Vector4(...layer.rect) },
        uOpaque: { value: layer.opaque ? 1 : 0 },
        uGuard: { value: new THREE.Vector2(...(layer.guard || [0, 0])) },
      },
      transparent: !layer.opaque,
      premultipliedAlpha: true,
      blending: layer.opaque ? THREE.NoBlending : THREE.CustomBlending,
      blendSrc: THREE.OneFactor,
      blendDst: THREE.OneMinusSrcAlphaFactor,
      blendSrcAlpha: THREE.OneFactor,
      blendDstAlpha: THREE.OneMinusSrcAlphaFactor,
      depthTest: true,
      depthWrite: true,
      side: THREE.DoubleSide,
    });
    const mesh = new THREE.Mesh(gridGeometry(gw, gh, o.meshStep), material);
    mesh.name = layer.name;
    mesh.renderOrder = i;
    mesh.frustumCulled = false;
    // Layers are painted far to near; each starts with a clean depth buffer so
    // it only hides itself, never fights the layer behind where they touch.
    mesh.onBeforeRender = (r) => r.clearDepth();
    if (o.only && !o.only.includes(layer.name)) mesh.visible = false;
    scene.add(mesh);
  }));

  // ---- camera ------------------------------------------------------------------
  const view = { yaw: 0, pitch: 0, dolly: 0 };              // where the camera is (degrees, degrees, 0..1)
  const target = { yaw: 0, pitch: 0, dolly: 0 };            // where it is heading
  const pivot = new THREE.Vector3(0, 0, -meta.pivot_depth);
  let dirty = true;

  function resize() {
    const w = canvas.clientWidth || 1, h = canvas.clientHeight || 1;
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, o.maxPixelRatio));
    renderer.setSize(w, h, false);
    // "cover": the photo fills the viewport, cropped on one axis, then zoomed
    const tanY = meta.height / 2 / meta.focal, tanX = meta.width / 2 / meta.focal;
    const aspect = w / h;
    const halfTan = Math.min(tanY, tanX / aspect) / o.zoom;
    camera.fov = THREE.MathUtils.radToDeg(2 * Math.atan(halfTan));
    camera.aspect = aspect;
    // slide the crop like CSS background-position
    const spareX = tanX - halfTan * aspect, spareY = tanY - halfTan;
    const offX = (o.focusX - 0.5) * 2 * spareX / (halfTan * aspect) * (w / 2);
    const offY = (o.focusY - 0.5) * 2 * spareY / halfTan * (h / 2);
    camera.setViewOffset(w, h, offX, offY, w, h);
    camera.updateProjectionMatrix();
    dirty = true;
  }

  const dir = new THREE.Vector3();
  function render() {
    const yaw = THREE.MathUtils.degToRad(view.yaw), pitch = THREE.MathUtils.degToRad(view.pitch);
    dir.set(Math.sin(yaw) * Math.cos(pitch), Math.sin(pitch), Math.cos(yaw) * Math.cos(pitch));
    camera.position.copy(pivot).addScaledVector(dir, meta.pivot_depth - view.dolly * o.maxDolly);
    camera.up.set(0, 1, 0);
    camera.lookAt(pivot);
    renderer.render(scene, camera);
    dirty = false;
  }

  // ---- input ---------------------------------------------------------------------
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const steer = { yaw: 0, pitch: 0, lastInput: -1e9, drag: null };
  let onScreen = true, paused = false, running = false, raf = 0, last = 0;

  function onPointerMove(e) {
    if (e.pointerType === "mouse") {
      const r = canvas.getBoundingClientRect();
      steer.yaw = -clamp(((e.clientX - r.left) / r.width) * 2 - 1, -1, 1);
      steer.pitch = -clamp(((e.clientY - r.top) / r.height) * 2 - 1, -1, 1);
      steer.lastInput = performance.now();
    } else if (steer.drag && e.pointerId === steer.drag.id) {
      // a finger dragging sideways swings the camera; up and down is left to the page's scrolling
      const dx = (e.clientX - steer.drag.x) / Math.max(canvas.clientWidth, 1);
      steer.yaw = clamp(steer.drag.yaw + dx * 2.2, -1, 1);
      steer.lastInput = performance.now();
    }
  }
  function onPointerDown(e) {
    if (e.pointerType !== "mouse") steer.drag = { id: e.pointerId, x: e.clientX, yaw: target.yaw / o.maxYaw };
  }
  function onPointerUp(e) {
    if (steer.drag && e.pointerId === steer.drag.id) steer.drag = null;
  }

  function tick(now) {
    raf = requestAnimationFrame(tick);
    const dt = Math.min(0.05, (now - last) / 1000 || 0.016);
    last = now;
    if (!onScreen || paused || document.hidden) return;
    if (!o.still && !reduceMotion) {
      // Left alone for a while, the camera eases into a slow sway.
      const idle = clamp((now - steer.lastInput - 2500) / 3000, 0, 1) * o.drift;
      const t = now / 1000 * o.driftSpeed;
      target.yaw = (steer.yaw * (1 - idle) + Math.sin(t * 0.21) * idle) * o.maxYaw;
      target.pitch = (steer.pitch * (1 - idle) + Math.sin(t * 0.13 + 1.3) * idle * 0.7) * o.maxPitch;
    }
    const k = 1 - Math.exp(-dt * o.follow);
    for (const key of ["yaw", "pitch", "dolly"]) {
      const d = target[key] - view[key];
      if (Math.abs(d) > 1e-5) { view[key] += d * k; dirty = true; }
    }
    if (dirty) render();
  }

  const io = "IntersectionObserver" in window
    ? new IntersectionObserver((entries) => { onScreen = entries[0].isIntersecting; }) : null;
  const onResize = () => resize();

  resize();
  render();

  return {
    meta, view, target, options: o, renderer, scene, camera,
    /** Put the camera somewhere at once (degrees, degrees, 0..1). */
    setView(yaw, pitch, dolly = 0) {
      view.yaw = target.yaw = yaw; view.pitch = target.pitch = pitch; view.dolly = target.dolly = dolly;
      render();
    },
    /** Ease the camera toward (0) or away from (1) the boy. */
    setDolly(d) { target.dolly = clamp(d, 0, 1); },
    setZoom(z) { o.zoom = z; resize(); render(); },
    /** Stop drawing while the scene is covered by something else. */
    setPaused(p) { paused = !!p; },
    resize() { resize(); render(); },
    start() {
      if (running) return;
      running = true;
      window.addEventListener("resize", onResize);
      if (!o.still) {
        window.addEventListener("pointermove", onPointerMove, { passive: true });
        window.addEventListener("pointerdown", onPointerDown, { passive: true });
        window.addEventListener("pointerup", onPointerUp, { passive: true });
        window.addEventListener("pointercancel", onPointerUp, { passive: true });
      }
      io?.observe(canvas);
      raf = requestAnimationFrame(tick);
    },
    stop() {
      running = false;
      cancelAnimationFrame(raf);
      io?.disconnect();
      window.removeEventListener("resize", onResize);
      window.removeEventListener("pointermove", onPointerMove);
      window.removeEventListener("pointerdown", onPointerDown);
      window.removeEventListener("pointerup", onPointerUp);
      window.removeEventListener("pointercancel", onPointerUp);
    },
  };
}
