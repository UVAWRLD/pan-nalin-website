// Pan Nalin — 3D hero
//
// The boys and the sky as a 3D scene (sky3d.js). The camera drifts slowly
// around them on its own and the pointer steers it (a finger drags it on touch
// screens). Pulling the biography card up pulls the camera back: from close
// on the boys (Figma 150:181) to the wide sky with the boys behind the card
// (Figma 153:225). Scrolling on, the scene smears into a blurred sky.
//
// Until the scene is ready, without WebGL 2, or with "reduce motion" on, the
// page shows the flat photo (.hero__image).
//
// `?flat` keeps the flat photo. sky3.html shows the scene on its own.

import { createSkyScene } from "./sky3d.js";

const hero = document.querySelector(".hero");
const canvas = document.querySelector(".hero__canvas");
const params = new URLSearchParams(location.search);
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

function hasWebGL2() {
  try {
    return !!document.createElement("canvas").getContext("webgl2");
  } catch {
    return false;
  }
}

// the film grain: a small tile of soft noise, made once
{
  const tile = document.createElement("canvas");
  tile.width = tile.height = 192;
  const g = tile.getContext("2d"), d = g.createImageData(192, 192);
  for (let i = 0; i < d.data.length; i += 4) {
    const v = 128 + (Math.random() + Math.random() - 1) * 90;
    d.data[i] = d.data[i + 1] = d.data[i + 2] = v; d.data[i + 3] = 255;
  }
  g.putImageData(d, 0, 0);
  const grain = document.querySelector(".hero__grain");
  if (grain) grain.style.backgroundImage = `url(${tile.toDataURL()})`;
}

if (hero && canvas && !params.has("flat") && !reduceMotion && hasWebGL2()) start();

async function start() {
  // Phones and tablets get half-size pictures and a coarser mesh: a small
  // screen cannot show the difference, and it is a quarter of the memory.
  const small = window.matchMedia("(max-width: 1099px), (pointer: coarse)").matches;
  let scene;
  try {
    scene = await createSkyScene(canvas, {
      base: "assets/sky3/", half: small, meshStep: small ? 2 : 1,
      pullFrames: true, closeFill: 0.62, closeMax: 0.66, blurImage: "assets/sky3/sky-blur.webp", maxYaw: 5, maxPitch: 1.5, drift: 1, driftSpeed: 0.3, follow: 1.4,
    });
  } catch (error) {
    console.warn("3D hero unavailable, keeping the photo:", error);
    return;
  }

  // The pull: in the design the card rises 384 px (of 1920) from where it
  // peeks to where the camera is fully back; on tall, narrow screens a quarter
  // of the window's height.
  const pullDistance = () => Math.max((384 * window.innerWidth) / 1920, window.innerHeight * 0.25);
  const onScroll = () => scene.setPull(window.scrollY / pullDistance());
  window.addEventListener("scroll", onScroll, { passive: true });
  scene.setPull(window.scrollY / pullDistance(), true);
  skyTransition(scene, pullDistance);

  canvas.addEventListener("webglcontextlost", () => hero.classList.remove("is-3d"));
  canvas.addEventListener("webglcontextrestored", () => location.reload());

  glassWave(scene);
  scene.start();
  if (matchMedia("(pointer: coarse)").matches) tiltSteering(scene);
  hero.classList.add("is-3d");
  window.__hero = scene;
}

// The wave of the scroll transition (sky3d.js) passes over the page too: as
// it crosses each part of the card and the header, that part warps as if
// seen through rippled glass, smears upward and splits into red and blue
// ghosts, then settles. Each part has its own SVG filter, strongest where the
// wave is. The card's glass itself is only stretched (a filter on it would
// switch its backdrop blur off).
function glassWave(scene) {
  const els = [...document.querySelectorAll(
    ".brand, .nav, .bio__portrait, .bio__meta, .bio__name, .bio__role, .bio__mark, .bio__tag, .bio__text p, .bio__films, .bio__card")];
  const NS = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(NS, "svg");
  svg.setAttribute("aria-hidden", "true");
  svg.style.cssText = "position:absolute;width:0;height:0;overflow:hidden";
  const only = (r, g, b) => `${r} 0 0 0 0  0 ${g} 0 0 0  0 0 ${b} 0 0  0 0 0 1 0`;
  svg.innerHTML = els.map((_, i) => `
    <filter id="glass${i}" x="-25%" y="-150%" width="150%" height="400%" color-interpolation-filters="sRGB">
      <feTurbulence type="fractalNoise" baseFrequency="0.0035 0.02" numOctaves="2" seed="${i * 7 + 3}" result="n"/>
      <feDisplacementMap in="SourceGraphic" in2="n" scale="0" xChannelSelector="R" yChannelSelector="G" result="d"/>
      <feGaussianBlur in="d" stdDeviation="0 0" result="s"/>
      <feColorMatrix in="s" type="matrix" values="${only(1, 0, 0)}" result="r"/>
      <feOffset in="r" dy="0" result="ro"/>
      <feColorMatrix in="s" type="matrix" values="${only(0, 1, 0)}" result="g"/>
      <feColorMatrix in="s" type="matrix" values="${only(0, 0, 1)}" result="b"/>
      <feOffset in="b" dy="0" result="bo"/>
      <feBlend in="ro" in2="g" mode="screen" result="rg"/>
      <feBlend in="rg" in2="bo" mode="screen"/>
    </filter>`).join("");
  document.body.appendChild(svg);

  const parts = els.map((el, i) => {
    const f = svg.querySelector(`#glass${i}`);
    const base = getComputedStyle(el).transform;
    return {
      el, glass: !el.classList.contains("bio__card"), id: `url(#glass${i})`,
      base: base === "none" ? "" : base + " ",
      disp: f.querySelector("feDisplacementMap"), blur: f.querySelector("feGaussianBlur"),
      red: f.querySelectorAll("feOffset")[0], blue: f.querySelectorAll("feOffset")[1],
      on: false,
    };
  });
  const off = (p) => {
    if (!p.on) return;
    p.el.style.filter = ""; p.el.style.transform = ""; p.on = false;
  };

  (function loop() {
    requestAnimationFrame(loop);
    const t = scene.view.blur;
    if (t <= 0.001 || t >= 0.999) { parts.forEach(off); return; }
    const W = window.innerWidth, H = window.innerHeight, s = H / 1000;
    for (const p of parts) {
      const r = p.el.getBoundingClientRect();
      if (r.bottom < 0 || r.top > H) { off(p); continue; }
      const y = (Math.max(r.top, 0) + Math.min(r.bottom, H)) / 2;     // the part of it on screen
      const x = ((r.left + r.width / 2) / W) * 2 - 1;
      const d = 1 - y / H - (-0.7 + 2.6 * t + 0.3 * (1 - x * x));   // as in sky3d.js
      const k = Math.exp((-d * d) / 0.14);
      if (k < 0.02) { off(p); continue; }
      if (p.glass) {
        p.disp.setAttribute("scale", (70 * k * s).toFixed(1));
        p.blur.setAttribute("stdDeviation", `${(0.6 * k * s).toFixed(2)} ${(8 * k * s).toFixed(2)}`);
        p.red.setAttribute("dy", (-11 * k * s).toFixed(1));
        p.blue.setAttribute("dy", (11 * k * s).toFixed(1));
        p.el.style.filter = p.id;
        p.el.style.transform = `${p.base}translateY(${(-20 * k * s).toFixed(1)}px) skewX(${(-x * 7 * k).toFixed(2)}deg) scaleY(${(1 + 0.07 * k).toFixed(3)})`;
      } else {
        p.el.style.transform = `${p.base}translateY(${(-12 * k * s).toFixed(1)}px) scaleY(${(1 + 0.025 * k).toFixed(3)})`;
      }
      p.on = true;
    }
  })();
}

// The transition to the blurred sky (image 35) does not follow the scroll.
// Past the pull, scrolling meets resistance: the card moves at a third of the
// speed and the wave only peeks in at the bottom, springing back if let go.
// Push past the threshold and it plays through on its own, and the card
// glides back into step. Scrolling back up to where it began plays it back.
function skyTransition(scene, pullDistance) {
  // Rubber band: once the pull-back is done the card stretches against the
  // scroll (it moves less and less the further you drag), the sky wave peeks
  // in, and nothing gives until the first paragraph of the card has been
  // scrolled through. Past that line the transition plays on its own and the
  // card springs back into place with a little overshoot.
  const root = document.documentElement;
  const first = document.querySelector(".bio__lede");
  const DURATION = 1600, BACK = 0.25, PEEK = 0.16, STRETCH = 0.92, AVG = 0.62;
  const s = { state: "idle", v: 0, from: 0, to: 0, start: 0, hold: 0, hv: 0, shown: -1 };
  const docBottom = (el) => {
    let y = el.offsetHeight;
    for (let n = el; n; n = n.offsetParent) y += n.offsetTop;
    return y;
  };
  // The card's own travel to the line where the lede's last line sits at the
  // bottom of the screen (above the dark shadow strip), and the scroll it
  // takes to get there: AVG of the usual speed on average, never less than
  // 22% of the screen.
  const measure = () => {
    const pull = pullDistance();
    const room = root.scrollHeight - window.innerHeight - pull - 2;
    const end = first ? docBottom(first) - window.innerHeight * 0.88 - pull : 0;
    const z = Math.max(Math.min(Math.max(end / AVG, window.innerHeight * 0.22), room), 1);
    const d = Math.min(Math.max(end, 32), STRETCH * z * 0.9);
    // Find c so the card arrives at d exactly at the threshold.
    let lo = d, hi = d * 50;
    for (let i = 0; i < 30; i++) {
      const c = (lo + hi) / 2;
      if (c * Math.tanh((STRETCH * z) / c) < d) lo = c; else hi = c;
    }
    return { z, c: (lo + hi) / 2 };
  };
  // How far the card travels for `x` of scroll: almost freely at first
  // (STRETCH of the scroll), then tighter and tighter, like a band pulling
  // taut, towards a limit of `c`.
  const rubber = (x, c) => c * Math.tanh((STRETCH * x) / c);
  const ease = (x) => (x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2);
  const play = (to, now) => {
    s.from = s.v; s.to = to; s.start = now; s.state = to === 1 ? "on" : "back";
  };
  (function loop(now) {
    requestAnimationFrame(loop);
    const into = window.scrollY - pullDistance(), { z, c } = measure();
    const r = Math.min(Math.max(into / z, 0), 1);
    if (s.state === "idle" && into >= z) play(1, now);
    else if (s.state === "on" && r <= BACK) play(0, now);
    if (s.state === "idle") {
      const peek = PEEK * (1 - Math.exp(-3 * r * r));
      s.v += (peek - s.v) * 0.12;
    } else {
      const p = Math.min((now - s.start) / (DURATION * Math.max(Math.abs(s.to - s.from), 0.3)), 1);
      s.v = s.from + (s.to - s.from) * ease(p);
      if (p >= 1 && s.state === "back") s.state = "idle";
    }
    scene.setBlur(s.v, true);
    // Held back by the scroll the card has not been allowed to follow.
    const x = Math.min(Math.max(into, 0), z);
    const target = s.state === "idle" ? x - rubber(x, c) : 0;
    s.hv = (s.hv + (target - s.hold) * 0.08) * 0.8;               // spring
    s.hold += s.hv;
    if (Math.abs(s.hold - s.shown) > 0.05) {
      s.shown = s.hold;
      root.style.setProperty("--hold", `${s.hold.toFixed(1)}px`);
    }
  })(performance.now());
}

// Phones and tablets: look around the sky by tilting the device. iOS asks
// for motion access on the intro tap (main.js sets window.__tilt); if that
// tap never happened, the first tap on the page asks instead. Without
// access, a finger drag still steers.
function tiltSteering(scene) {
  const go = (state) => { if (state === "granted") scene.enableTilt(); };
  const DOE = window.DeviceOrientationEvent;
  if (!DOE) return;
  if (typeof DOE.requestPermission !== "function") return go("granted");
  if (window.__tilt) return void window.__tilt.then(go);
  const ask = () => {
    removeEventListener("touchend", ask);
    (window.__askTilt ? window.__askTilt() : DOE.requestPermission()).then(go, () => {});
  };
  addEventListener("touchend", ask, { passive: true });
}
