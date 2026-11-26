// The boys and the sky, in 3D (data: assets/sky2/, made by tools/sky3d/build2.py).
//
//   sky dome  a sphere around the camera showing the photo's own sky, clouds
//             and far shore, as seen from the photo's camera (so from there it
//             is the photo exactly); the clouds drift very slowly
//   lake      a real horizontal water surface showing the photo's water, with
//             slow ripples
//   boys      each boy a relief of his body, at his own distance
//
// The camera circles the boys a little (±10° by default), steered by the
// pointer, drifting slowly on its own when left alone.
//
//   const s = await createSkyScene(canvas, options); s.start();

import * as THREE from "three";

const PROJECT = /* glsl */ `
uniform float uF, uCx, uCy;
uniform vec4 uRect;                     // the sky picture's rectangle in photo px
vec2 photoPx(vec3 d) {                  // where a direction from the photo's camera lands in the photo
  float z = max(-d.z, 1e-3);
  return vec2(uCx + uF * d.x / z, uCy - uF * d.y / z);
}
vec2 skyUv(vec2 px) { return clamp((px - uRect.xy) / uRect.zw, vec2(0.0005), vec2(0.9995)); }
`;

// value noise and its sum over octaves, for the procedural detail
const NOISE = /* glsl */ `
float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
float noise(vec2 p) {
  vec2 i = floor(p), f = fract(p), u = f * f * (3.0 - 2.0 * f);
  return mix(mix(hash(i), hash(i + vec2(1, 0)), u.x), mix(hash(i + vec2(0, 1)), hash(i + vec2(1, 1)), u.x), u.y);
}
float fbm(vec2 p) {
  float s = 0.0, a = 0.5;
  for (int i = 0; i < 5; i++) { s += a * noise(p); p = p * 2.03 + 17.0; a *= 0.5; }
  return s;
}
`;

const DOME_VERT = /* glsl */ `
out vec3 vDir;
void main() {
  vDir = position;
  vec4 p = projectionMatrix * viewMatrix * vec4(position, 1.0);
  gl_Position = p.xyww;                // always at the far plane
}`;
const DOME_FRAG = /* glsl */ `
precision highp float;
uniform sampler2D uSky;
uniform float uTime, uHorizon, uDrift;
uniform vec2 uSun;                      // the brightest part of the sky (photo px)
in vec3 vDir;
out vec4 outColor;
${PROJECT}
${NOISE}
void main() {
  vec2 px = photoPx(normalize(vDir));
  // the clouds drift, gently, more the higher they are; the shore stays put
  float k = smoothstep(uHorizon - 40.0, uHorizon - 400.0, px.y);
  px.x += k * uDrift * sin(uTime * 0.021) + k * uDrift * 0.35 * sin(uTime * 0.013 + px.y * 0.002);
  vec3 c = texture(uSky, skyUv(px)).rgb;
  outColor = vec4(c, 1.0);
}`;

const WATER_VERT = /* glsl */ `
out vec3 vPos;
void main() {
  vec4 w = modelMatrix * vec4(position, 1.0);
  vPos = w.xyz;
  gl_Position = projectionMatrix * viewMatrix * w;
}`;
const WATER_FRAG = /* glsl */ `
precision highp float;
uniform sampler2D uSky;
uniform float uTime;
uniform vec3 uUp, uLight;
uniform float uBottom;                  // the picture's bottom row (photo px)
in vec3 vPos;
out vec4 outColor;
${PROJECT}
${NOISE}
float waves(vec2 q) {                   // the lake's surface, metres
  return fbm(q * 0.55 + vec2(uTime * 0.05, uTime * 0.03)) * 0.6 + fbm(q * 2.1 - vec2(uTime * 0.11, -uTime * 0.07)) * 0.25;
}
void main() {
  vec3 e1 = normalize(cross(uUp, vec3(0.0, 0.0, 1.0)));
  vec3 e2 = normalize(cross(e1, uUp));
  vec2 q = vec2(dot(vPos, e1), dot(vPos, e2));
  float a = q.x, b = q.y;
  float w1 = sin(a * 0.9 + uTime * 0.55) + sin(b * 1.3 - uTime * 0.4) * 0.6;
  float w2 = sin(b * 0.7 + uTime * 0.35) + sin(a * 1.7 + uTime * 0.6) * 0.5;
  vec3 p = vPos + e1 * w1 * 0.06 + e2 * w2 * 0.10;
  vec2 px = photoPx(p);
  vec3 c = texture(uSky, skyUv(px)).rgb;
  vec3 view = normalize(vPos - cameraPosition);
  vec3 n = normalize(uUp + (e1 * w1 + e2 * w2) * 0.02);
  // below the picture (only if the view ever goes there): the sky reflected
  float below = smoothstep(uBottom - 140.0, uBottom - 10.0, px.y);
  if (below > 0.0) {
    vec3 refl = texture(uSky, skyUv(photoPx(reflect(view, n)))).rgb;
    vec3 deep = textureLod(uSky, skyUv(vec2(px.x, uBottom - 60.0)), 5.0).rgb;
    float fres = 0.25 + 0.55 * pow(1.0 - max(dot(-view, uUp), 0.0), 4.0);
    c = mix(c, mix(deep, refl, fres), below);
  }
  c *= 1.0 + 0.035 * sin(a * 2.3 + b * 1.1 + uTime * 0.8);
  outColor = vec4(c, 1.0);
}`;

const BOY_VERT = /* glsl */ `
uniform sampler2D uDepth;
uniform vec2 uInv;
uniform vec4 uCam;                      // focal, cx, cy, photo px per depth cell
uniform vec4 uLayer;                    // this boy's rectangle in photo px
in vec2 ij;
out vec2 vUv;
out vec3 vPos;
out float vShade;
void main() {
  vec4 t = texelFetch(uDepth, ivec2(ij), 0);
  float q = (floor(t.r * 255.0 + 0.5) * 256.0 + floor(t.g * 255.0 + 0.5)) / 65535.0;
  float z = 1.0 / mix(uInv.x, uInv.y, q);
  vec2 p = uLayer.xy + min(ij * uCam.w, uLayer.zw);
  vec3 pos = vec3((p.x - uCam.y) / uCam.x * z, -(p.y - uCam.z) / uCam.x * z, -z);
  vUv = (p - uLayer.xy) / uLayer.zw;
  vPos = pos;
  vShade = t.b;                         // the shadow nearer boys cast here (1 = none)
  gl_Position = projectionMatrix * viewMatrix * vec4(pos, 1.0);
}`;
const BOY_FRAG = /* glsl */ `
precision highp float;
uniform sampler2D uColor, uAlpha;
uniform vec3 uLight;
in vec2 vUv;
in vec3 vPos;
in float vShade;
out vec4 outColor;
void main() {
  vec4 c = vec4(texture(uColor, vUv).rgb, texture(uAlpha, vUv).r);
  if (c.a < 0.004) discard;
  // shadow from the boys in front, only on parts the photo never showed
  c.rgb *= vShade;
  outColor = vec4(max(c.rgb, 0.0), c.a);   // premultiplied
}`;

// The scroll transition (home page), after saintdsgn's shader transition: a
// broad, bright wave rises from the bottom of the screen (its middle leading).
// Inside it the picture smears upward with its colours split into ghosts, and
// pale iridescent bands (pink, lilac, cyan, mint) wash over everything; behind
// it, the blurred sky is left.
const POST_VERT = /* glsl */ `
out vec2 vUv;
void main() { vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }`;
const POST_FRAG = /* glsl */ `
precision highp float;
uniform sampler2D uA, uB;               // the scene, the blurred sky
uniform float uT, uTime;
uniform vec4 uBRect;                    // the view's rectangle on the blurred sky (uv)
in vec2 vUv;
out vec4 outColor;
vec3 sceneAt(vec2 uv, float lod) { return textureLod(uA, clamp(uv, 0.001, 0.999), lod).rgb; }
vec3 skyAt(vec2 uv) { return texture(uB, clamp(uBRect.xy + vec2(uv.x, 1.0 - uv.y) * uBRect.zw, 0.001, 0.999)).rgb; }
// one colour channel, smeared upward over 'len' (the trail of the wave)
vec3 smear(vec2 uv, float len, float lod, float m) {
  vec3 c = vec3(0.0);
  for (int i = 0; i < 8; i++) {
    vec2 q = uv - vec2(0.0, len * float(i) / 7.0);
    c += mix(sceneAt(q, lod), skyAt(q), m);
  }
  return c / 8.0;
}
void main() {
  float t = clamp(uT, 0.0, 1.0);
  float x = vUv.x * 2.0 - 1.0;
  float front = mix(-0.7, 1.9, t) + 0.3 * (1.0 - x * x);          // a dome: the middle leads, the corners trail
  float d = vUv.y - front;                                          // below 0: already through
  float k = exp(-d * d / 0.14);                                     // the wave (same width in hero3d.js)
  float wash = sin(3.14159265 * t);                                 // ...and the glow it brings
  float m = 1.0 - smoothstep(-0.12, 0.12, d);
  float len = 0.12 * k, ca = 0.02 * k, lod = 6.0 * k;
  vec3 c = vec3(smear(vUv + vec2(0.0, ca), len, lod, m).r,
                smear(vUv, len, lod, m).g,
                smear(vUv - vec2(0.0, ca), len, lod, m).b);
  // iridescent bands running across, travelling up with the wave
  float ph = d * 1.3 - t * 0.8 + 0.05 * sin(vUv.x * 4.0 + uTime * 0.4);   // bands curved like the front
  vec3 iri = 0.5 + 0.5 * cos(6.28318 * (ph + vec3(0.0, 0.3, 0.62)));
  iri = mix(vec3(1.0), iri, 0.38);
  // two soft white blooms drifting with it
  float bloom = exp(-dot(vUv - vec2(0.3, front + 0.05), vUv - vec2(0.3, front + 0.05)) / 0.05)
              + exp(-dot(vUv - vec2(0.75, front - 0.25), vUv - vec2(0.75, front - 0.25)) / 0.04);
  float a = clamp(k * 0.85 + wash * 0.25, 0.0, 1.0);
  c = mix(c, 1.0 - (1.0 - c) * (1.0 - iri * 0.85), a);              // screen
  c = mix(c, vec3(0.98, 0.97, 1.0), clamp(k * 0.25 + bloom * k * 0.5, 0.0, 0.8));
  outColor = vec4(clamp(c, 0.0, 1.0), 1.0);
}`;

const loader = new THREE.TextureLoader();
const load = (url) => new Promise((res, rej) => loader.load(url, res, undefined, rej));
const clamp = THREE.MathUtils.clamp;

function grid(gw, gh, step) {
  const xs = [], ys = [];
  for (let i = 0; i < gw - 1; i += step) xs.push(i);
  xs.push(gw - 1);
  for (let j = 0; j < gh - 1; j += step) ys.push(j);
  ys.push(gh - 1);
  const ij = new Float32Array(xs.length * ys.length * 2);
  let k = 0;
  for (const y of ys) for (const x of xs) { ij[k++] = x; ij[k++] = y; }
  const idx = new Uint32Array((xs.length - 1) * (ys.length - 1) * 6);
  k = 0;
  for (let j = 0; j < ys.length - 1; j++) for (let i = 0; i < xs.length - 1; i++) {
    const a = j * xs.length + i, b = a + 1, c = a + xs.length, d = c + 1;
    idx.set([a, c, b, b, c, d], k); k += 6;
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("ij", new THREE.BufferAttribute(ij, 2));
  g.setIndex(new THREE.BufferAttribute(idx, 1));
  return g;
}

function prep(t, mips = true) {
  t.flipY = false; t.colorSpace = THREE.NoColorSpace; t.premultiplyAlpha = false;
  t.generateMipmaps = mips;
  t.minFilter = mips ? THREE.LinearMipmapLinearFilter : THREE.NearestFilter;
  t.magFilter = mips ? THREE.LinearFilter : THREE.NearestFilter;
  return t;
}

export async function createSkyScene(canvas, options = {}) {
  const o = {
    base: "assets/sky2/", half: false, meshStep: 1, zoom: 1.06,
    maxYaw: 5, maxPitch: 1.5, maxDolly: 0.4, follow: 1.4, drift: 1, driftSpeed: 0.3,
    cloudDrift: 90,                     // how far (photo px) the clouds wander
    lift: 0.22,                         // how much lake shows below the boys (fraction of the view)
    maxPixelRatio: 2, still: false, only: null,
    // Home page: two framings, blended by pull (0 = close on the boys, 1 = wide
    // on the sky). The lake's edge sits `horizonAt` down the view in both.
    pullFrames: false, blurImage: null, blurFollow: 4, horizonAt: 0.885, closeFill: 0.8, closeMax: 0.8, pullFollow: 5,
    ...options,
  };
  const meta = await (await fetch(o.base + "scene.json")).json();
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: "high-performance" });
  renderer.autoClear = true;
  const aniso = renderer.capabilities.getMaxAnisotropy();
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(30, 1, 0.05, 2000);

  // the photo camera's frame: it looks a little up, so the world's "up" is tilted
  const f = meta.focal, cx = meta.cx, cy = meta.cy;
  const a = -(meta.horizon - cy) / f;
  const up = new THREE.Vector3(0, 1, a).normalize();
  const common = {
    uF: { value: f }, uCx: { value: cx }, uCy: { value: cy },
    uRect: { value: new THREE.Vector4(...meta.sky.rect) }, uTime: { value: 0 },
  };
  // the light: from the brightest part of the sky (left and above, in the photo)
  const sunPx = meta.sun || [cx - f * 0.3, cy - f * 0.3];
  const light = new THREE.Vector3((sunPx[0] - cx) / f, -(sunPx[1] - cy) / f, -1).normalize();
  light.z = Math.abs(light.z) * 0.6;                          // falling on the boys from in front of them
  light.normalize();
  const sky = prep(await load(o.base + (o.half && meta.sky.image_half ? meta.sky.image_half : meta.sky.image)));
  sky.anisotropy = aniso;

  const dome = new THREE.Mesh(new THREE.SphereGeometry(1000, 96, 48), new THREE.ShaderMaterial({
    glslVersion: THREE.GLSL3, vertexShader: DOME_VERT, fragmentShader: DOME_FRAG, side: THREE.BackSide,
    depthWrite: false, depthTest: false,
    uniforms: { ...common, uSky: { value: sky }, uHorizon: { value: meta.horizon }, uDrift: { value: o.cloudDrift },
                uSun: { value: new THREE.Vector2(...sunPx) } },
  }));
  dome.renderOrder = -2; dome.frustumCulled = false;
  scene.add(dome);

  const water = new THREE.Mesh(new THREE.PlaneGeometry(3000, 3000, 1, 1), new THREE.ShaderMaterial({
    glslVersion: THREE.GLSL3, vertexShader: WATER_VERT, fragmentShader: WATER_FRAG,
    depthWrite: false, depthTest: true, side: THREE.DoubleSide,
    transparent: true,                  // (drawn with the boys, after them, so it can cover them)
    uniforms: { ...common, uSky: { value: sky }, uUp: { value: up }, uBottom: { value: meta.height },
                uLight: { value: new THREE.Vector3((sunPx[0] - cx) / f, -(sunPx[1] - cy) / f, -1).normalize() } },
  }));
  // The lake's surface: level, behind the boys (they stand in front of it),
  // its nearest visible part at the picture's bottom edge.
  let waterH = meta.camera_height;
  if (meta.group) {
    const D = meta.pivot_depth * 1.5;
    const pAt = new THREE.Vector3(0, -(meta.height - cy) / f * D, -D);
    waterH = -pAt.dot(up);
  }
  water.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), up);
  water.position.copy(up).multiplyScalar(-waterH);
  water.renderOrder = 1000; water.frustumCulled = false;
  scene.add(water);

  await Promise.all(meta.layers.map(async (L, i) => {
    if (o.only && !o.only.includes(L.name)) return;
    const pick = (k) => o.base + (o.half && L[k + "_half"] ? L[k + "_half"] : L[k]);
    const [col, al, dep] = await Promise.all([load(pick("color")), load(pick("alpha")), load(o.base + L.depth)]);
    prep(col); prep(al); prep(dep, false);
    col.anisotropy = aniso;
    al.format = THREE.RedFormat; al.internalFormat = "R8";
    const m = new THREE.Mesh(grid(L.grid[0], L.grid[1], o.meshStep), new THREE.ShaderMaterial({
      glslVersion: THREE.GLSL3, vertexShader: BOY_VERT, fragmentShader: BOY_FRAG, transparent: true,
      premultipliedAlpha: true, blending: THREE.CustomBlending, blendSrc: THREE.OneFactor,
      blendDst: THREE.OneMinusSrcAlphaFactor, blendSrcAlpha: THREE.OneFactor, blendDstAlpha: THREE.OneMinusSrcAlphaFactor,
      side: THREE.DoubleSide,
      uniforms: {
        uDepth: { value: dep }, uColor: { value: col }, uAlpha: { value: al }, uLight: { value: light },
        uInv: { value: new THREE.Vector2(L.inv_min, L.inv_max) },
        uCam: { value: new THREE.Vector4(f, cx, cy, L.cell) }, uLayer: { value: new THREE.Vector4(...L.rect) },
      },
    }));
    m.renderOrder = i; m.frustumCulled = false;   // far to near; the depth test sorts out the rest (and the water)
    scene.add(m);
  }));

  // ---- camera: circles a point at the boys' distance, around the world's up ----
  const view = { yaw: 0, pitch: 0, dolly: 0, pull: 0, blur: 0 }, target = { yaw: 0, pitch: 0, dolly: 0, pull: 0, blur: 0 };

  // ---- the scroll transition to the blurred sky (only if a blurred sky is given) ----
  let post = null;
  if (o.blurImage) {
    const tex = prep(await load(o.blurImage));
    const rt = new THREE.WebGLRenderTarget(1, 1, { depthBuffer: true });
    rt.texture.generateMipmaps = true;
    rt.texture.minFilter = THREE.LinearMipmapLinearFilter;
    const mat = new THREE.ShaderMaterial({
      glslVersion: THREE.GLSL3, vertexShader: POST_VERT, fragmentShader: POST_FRAG, depthTest: false, depthWrite: false,
      uniforms: { uA: { value: rt.texture }, uB: { value: tex }, uT: { value: 0 }, uTime: common.uTime,
                  uBRect: { value: new THREE.Vector4(0, 0, 1, 1) } },
    });
    const quad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), mat);
    quad.frustumCulled = false;
    const ps = new THREE.Scene(); ps.add(quad);
    post = { rt, mat, scene: ps, cam: new THREE.Camera() };
  }
  const pivot = new THREE.Vector3(0, 0, -meta.pivot_depth);
  const right = new THREE.Vector3().crossVectors(new THREE.Vector3(0, 0, -1), up).normalize();
  let dirty = true;

  // Close: all the boys across closeFill of the view's width, the lake's
  // edge just above the card. Wide: as much sky as the picture holds. The
  // zoom between them is even (log scale), so the pull feels steady.
  function pullFrame(aspect, p) {
    const [gx0, , gx1] = meta.group;
    const edgeY = f * Math.tan(THREE.MathUtils.degToRad(o.maxPitch + 0.8));
    const edgeX = f * Math.tan(THREE.MathUtils.degToRad(o.maxYaw + 1));
    const fit = (h) => Math.min(h, (meta.width - 2 * edgeX) / aspect);
    const wideH = fit((meta.horizon - edgeY) / o.horizonAt);
    const closeH = Math.min(fit((gx1 - gx0) / o.closeFill / aspect), wideH * o.closeMax);
    const e = p * p * (3 - 2 * p);
    const visH = Math.exp(Math.log(closeH) + (Math.log(wideH) - Math.log(closeH)) * e);
    const visW = visH * aspect;
    const vcx = clamp((gx0 + gx1) / 2, edgeX + visW / 2, meta.width - edgeX - visW / 2);
    return { half: visH / 2 / f, vcx, vcy: meta.horizon - o.horizonAt * visH + visH / 2 };
  }

  function resize() {
    const w = canvas.clientWidth || 1, h = canvas.clientHeight || 1;
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, o.maxPixelRatio));
    renderer.setSize(w, h, false);
    if (post) { const v = renderer.getDrawingBufferSize(new THREE.Vector2()); post.rt.setSize(v.x, v.y); }
    project();
  }

  function project() {
    const w = canvas.clientWidth || 1, h = canvas.clientHeight || 1;
    const aspect = w / h;
    let half, vcx, vcy;
    if (meta.group && o.pullFrames) {
      ({ half, vcx, vcy } = pullFrame(aspect, view.pull));
    } else if (meta.group) {
      // Any screen shape: all the boys fit across (with a margin), their
      // shorts at the bottom edge, and the sky fills whatever is left above.
      // Margins keep the slow pan from ever reaching the picture's edges.
      const [gx0, gy0, gx1, gy1] = meta.group;
      const edgeY = f * Math.tan(THREE.MathUtils.degToRad(o.maxPitch + 0.8));
      const edgeX = f * Math.tan(THREE.MathUtils.degToRad(o.maxYaw + 1));
      let visW = (gx1 - gx0) * 1.12, visH = visW / aspect;
      const bottom = meta.height - edgeY;                  // down to the picture's bottom edge
      const needTop = gy0 - 0.12 * (gy1 - gy0);            // room above the raised hands
      if (bottom - visH > needTop) { visH = bottom - needTop; visW = visH * aspect; }      // wide screens
      if (bottom - visH < edgeY) { visH = bottom - edgeY; visW = visH * aspect; }          // very tall screens
      if (visW > meta.width - 2 * edgeX) { visW = meta.width - 2 * edgeX; visH = visW / aspect; }
      half = visH / 2 / f;
      vcx = (gx0 + gx1) / 2; vcy = bottom - visH / 2;
    } else {
      const fr = aspect < 1 ? meta.frames.tall : meta.frames.wide;   // [x, y, w, h] in photo px
      half = Math.min(fr[3] / 2, fr[2] / 2 / aspect) / f / o.zoom;
      vcx = fr[0] + fr[2] / 2; vcy = fr[1] + fr[3] / 2;
    }
    camera.fov = THREE.MathUtils.radToDeg(2 * Math.atan(half));
    camera.aspect = aspect;
    const sx = (vcx - cx) / f, sy = (vcy - cy) / f;
    camera.setViewOffset(w, h, sx / (2 * half * aspect) * w, sy / (2 * half) * h, w, h);
    camera.updateProjectionMatrix();
    if (post) {
      const visH = 2 * half * f, visW = visH * aspect;
      post.mat.uniforms.uBRect.value.set((vcx - visW / 2) / meta.width, (vcy - visH / 2) / meta.height, visW / meta.width, visH / meta.height);
    }
    dirty = true;
  }

  const off = new THREE.Vector3(), q1 = new THREE.Quaternion(), q2 = new THREE.Quaternion();
  function render(t) {
    off.set(0, 0, meta.pivot_depth - view.dolly * o.maxDolly);
    q1.setFromAxisAngle(up, THREE.MathUtils.degToRad(view.yaw));
    q2.setFromAxisAngle(right, -THREE.MathUtils.degToRad(view.pitch));
    off.applyQuaternion(q2).applyQuaternion(q1);
    camera.position.copy(pivot).add(off);
    camera.up.copy(up);
    camera.lookAt(pivot);
    common.uTime.value = t;
    if (post && view.blur > 0.001) {
      post.mat.uniforms.uT.value = view.blur;
      if (view.blur < 0.999) {                 // once fully through, the scene is not needed
        renderer.setRenderTarget(post.rt);
        renderer.render(scene, camera);
        renderer.setRenderTarget(null);
      }
      renderer.render(post.scene, post.cam);
    } else {
      renderer.render(scene, camera);
    }
    dirty = false;
  }

  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const steer = { yaw: 0, pitch: 0, last: -1e9, drag: null };
  let raf = 0, last = 0, onScreen = true, paused = false;
  const onMove = (e) => {
    if (e.pointerType === "mouse") {
      const r = canvas.getBoundingClientRect();
      steer.yaw = -clamp(((e.clientX - r.left) / r.width) * 2 - 1, -1, 1);
      steer.pitch = -clamp(((e.clientY - r.top) / r.height) * 2 - 1, -1, 1);
      steer.last = performance.now();
    } else if (steer.drag && e.pointerId === steer.drag.id) {
      steer.yaw = clamp(steer.drag.yaw + (e.clientX - steer.drag.x) / Math.max(canvas.clientWidth, 1) * 2.2, -1, 1);
      steer.last = performance.now();
    }
  };
  // Tilt (phones): the camera follows how the phone is held, measured from a
  // rest pose that slowly re-centres itself, so any way of holding it works
  // and a quick turn of the wrist looks around. Tilting right looks right,
  // tipping the top away looks up into the sky.
  const tilt = { on: false, x0: null, y0: null, angle: 0 };
  const vel = { yaw: 0, pitch: 0 };
  const onTilt = (e) => {
    if (e.beta == null || e.gamma == null) return;
    const a = (screen.orientation?.angle ?? window.orientation ?? 0) % 360;
    let lr = e.gamma, ud = e.beta;                          // portrait
    if (a === 90) { lr = e.beta; ud = -e.gamma; }
    else if (a === 270 || a === -90) { lr = -e.beta; ud = e.gamma; }
    else if (a === 180) { lr = -e.gamma; ud = -e.beta; }
    if (tilt.x0 === null || a !== tilt.angle) { tilt.x0 = lr; tilt.y0 = ud; tilt.angle = a; }
    tilt.x0 += (lr - tilt.x0) * 0.004;
    tilt.y0 += (ud - tilt.y0) * 0.004;
    // Soft limits (no hard stop at the edge) and a light low-pass that takes
    // the sensor's shiver out; the spring in tick() does the rest.
    const yaw = -Math.tanh((lr - tilt.x0) / 18), pitch = Math.tanh((ud - tilt.y0) / 14);
    steer.yaw += (yaw - steer.yaw) * 0.25;
    steer.pitch += (pitch - steer.pitch) * 0.25;
    steer.last = performance.now();
  };
  const onDown = (e) => { if (e.pointerType !== "mouse" && !tilt.on) steer.drag = { id: e.pointerId, x: e.clientX, yaw: target.yaw / o.maxYaw }; };
  const onUp = (e) => { if (steer.drag && e.pointerId === steer.drag.id) steer.drag = null; };

  function tick(now) {
    raf = requestAnimationFrame(tick);
    const dt = Math.min(0.05, (now - last) / 1000 || 0.016);
    last = now;
    if (!onScreen || paused || document.hidden) return;
    const t = now / 1000;
    if (!o.still && !reduce) {
      const idle = clamp((now - steer.last - 2500) / 4000, 0, 1) * o.drift;
      const s = t * o.driftSpeed;
      target.yaw = (steer.yaw * (1 - idle) + Math.sin(s * 0.21) * idle) * o.maxYaw;
      target.pitch = (steer.pitch * (1 - idle) + Math.sin(s * 0.13 + 1.3) * idle * 0.7) * o.maxPitch;
    }
    const k = 1 - Math.exp(-dt * o.follow);
    if (tilt.on) {
      // A hand-held look: a critically damped spring eases into each turn of
      // the wrist and settles without the lag of the mouse follow.
      const w = 4.2;
      for (const key of ["yaw", "pitch"]) {
        vel[key] += ((target[key] - view[key]) * w * w - 2 * w * vel[key]) * dt;
        view[key] += vel[key] * dt;
      }
      view.dolly += (target.dolly - view.dolly) * k;
    } else for (const key of ["yaw", "pitch", "dolly"]) view[key] += (target[key] - view[key]) * k;
    if (Math.abs(target.blur - view.blur) > 1e-4) {
      view.blur += (target.blur - view.blur) * (1 - Math.exp(-dt * o.blurFollow));
      if (Math.abs(target.blur - view.blur) < 2e-3) view.blur = target.blur;
    }
    if (Math.abs(target.pull - view.pull) > 1e-4) {
      view.pull += (target.pull - view.pull) * (1 - Math.exp(-dt * o.pullFollow));
      project();
    }
    if (!reduce || dirty) render(o.still ? 0 : t);            // the water and clouds keep moving
  }

  const io = "IntersectionObserver" in window ? new IntersectionObserver((e) => { onScreen = e[0].isIntersecting; }) : null;
  resize();
  render(0);
  return {
    meta, view, target, options: o, renderer, camera,
    setView(yaw, pitch, dolly = 0) { Object.assign(view, { yaw, pitch, dolly }); Object.assign(target, { yaw, pitch, dolly }); render(0); },
    setDolly(d) { target.dolly = clamp(d, 0, 1); },
    setBlur(b, now = false) { target.blur = clamp(b, 0, 1); if (now) { view.blur = target.blur; dirty = true; } },
    setPull(p, now = false) { target.pull = clamp(p, 0, 1); if (now) { view.pull = target.pull; project(); render(0); } },
    setPaused(p) { paused = !!p; },
    // Steer by tilting the phone (call after any permission prompt).
    enableTilt() {
      if (tilt.on || o.still || reduce || !("DeviceOrientationEvent" in window)) return;
      tilt.on = true; steer.drag = null;
      addEventListener("deviceorientation", onTilt, { passive: true });
    },
    resize() { resize(); render(0); },
    start() {
      addEventListener("resize", resize);
      if (!o.still) {
        addEventListener("pointermove", onMove, { passive: true });
        addEventListener("pointerdown", onDown, { passive: true });
        addEventListener("pointerup", onUp, { passive: true });
        addEventListener("pointercancel", onUp, { passive: true });
      }
      io?.observe(canvas);
      raf = requestAnimationFrame(tick);
    },
    stop() {
      cancelAnimationFrame(raf); io?.disconnect();
      removeEventListener("resize", resize); removeEventListener("pointermove", onMove);
      removeEventListener("pointerdown", onDown); removeEventListener("pointerup", onUp); removeEventListener("pointercancel", onUp);
      removeEventListener("deviceorientation", onTilt);
    },
  };
}
