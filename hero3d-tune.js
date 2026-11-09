// Tuning panel for the 3D hero. Loaded only with `?tune` (see hero3d.js).
// Every change applies live and is remembered in this browser; "Copy
// settings" puts the values on the clipboard to paste into SETTINGS.

import GUI from "https://cdn.jsdelivr.net/npm/lil-gui@0.20.0/dist/lil-gui.esm.min.js";
import addNotes from "./hero3d-notes.js";

export default function openTuner(api) {
  const { SETTINGS, TEST, apply, invalidate, stats, boyMode, roomUrl, saveTuned, clearTuned } = api;
  const S = SETTINGS;
  const defaults = JSON.parse(JSON.stringify(S));
  // Skip the intro screen: tuning is about the hero.
  if (document.body.classList.contains("is-loading")) document.querySelector("[data-enter]")?.click();
  const changed = () => { apply(); saveTuned(); };
  const gui = new GUI({ title: "3D hero tuning", width: 320 });
  gui.domElement.style.zIndex = "1000";

  // --- compare with the photo ---------------------------------------------
  const image = document.querySelector(".hero__image");
  const compare = { photo: 0, difference: false, hidePage: true };
  const showCompare = () => {
    image.style.transition = "none";
    image.style.zIndex = "1";
    image.style.opacity = String(compare.photo);
    image.style.mixBlendMode = compare.difference ? "difference" : "normal";
    document.body.classList.toggle("tune-hide-page", compare.hidePage);
  };
  const style = document.createElement("style");
  style.textContent = `
    body.tune-hide-page .bio, body.tune-hide-page .site-header { opacity: 0; pointer-events: none; }
    .lil-gui .tune-note { padding: 6px 8px; font-size: 11px; line-height: 1.4; color: #aaa; }`;
  document.head.append(style);
  const f0 = gui.addFolder("Compare with the photo");
  f0.add(compare, "photo", 0, 1, 0.01).name("photo on top").onChange(showCompare);
  f0.add(compare, "difference").name("difference (black = match)").onChange(showCompare);
  f0.add(compare, "hidePage").name("hide page content").onChange(showCompare);
  showCompare();

  // --- test the camera move -------------------------------------------------
  const f1 = gui.addFolder("Test the camera move");
  f1.add(TEST, "motion", { "follow the mouse": "pointer", "hold at a position": "hold", "no motion": "still" })
    .name("camera").onChange(invalidate);
  f1.add(TEST, "peekX", -1, 1, 0.01).name("hold: left / right").onChange(invalidate);
  f1.add(TEST, "peekY", -1, 1, 0.01).name("hold: up / down").onChange(invalidate);
  f1.add(TEST, "orbitBoost", 1, 10, 0.1).name("exaggerate the move ×").onChange(invalidate);
  const fps = { value: "" };
  f1.add(fps, "value").name("frames per second").disable();
  setInterval(() => { fps.value = String(stats.fps); f1.controllers.at(-1).updateDisplay(); }, 500);

  // --- camera ---------------------------------------------------------------
  const c = gui.addFolder("Camera");
  c.add(S.camera, "fov", 15, 60, 0.01).name("field of view °").onChange(changed);
  c.add(S.camera, "lockBottom").name("keep the photo's bottom edge").onChange(changed);
  c.add(S.camera, "shiftX", -300, 300, 0.5).name("photo centre x (px)").onChange(changed);
  c.add(S.camera, "shiftY", -300, 300, 0.5).name("photo centre y (px)").onChange(changed);
  c.add(S.camera, "frameX", 0, 1, 0.01).name("framing x").onChange(changed);
  c.add(S.camera, "frameY", 0, 1, 0.01).name("framing y").onChange(changed);
  c.add(S.camera, "orbitX", 0, 0.6, 0.005).name("move left / right (m)").onChange(changed);
  c.add(S.camera, "orbitY", 0, 0.6, 0.005).name("move up / down (m)").onChange(changed);
  c.add(S.camera, "pivot", 0.5, 8, 0.05).name("look-at distance (m)").onChange(changed);
  c.add(S.camera, "scrollPush", 0, 1.5, 0.01).name("scroll push-in (m)").onChange(changed);
  c.add(S.camera, "ease", 0.005, 0.3, 0.005).name("follow speed").onChange(changed);
  c.add(S.camera, "driftX", 0, 1, 0.01).name("phone drift x").onChange(changed);
  c.add(S.camera, "driftY", 0, 1, 0.01).name("phone drift y").onChange(changed);

  // --- boy --------------------------------------------------------------------
  const b = gui.addFolder(`Boy (${boyMode})`);
  b.add(S.boy, "visible").onChange(changed);
  if (boyMode === "splat") {
    b.add(S.boy, "distance", 0.5, 1.5, 0.005).name("distance ×").onChange(changed);
  }
  b.add(S.boy, "x", -0.5, 0.5, 0.005).name("nudge x (m)").onChange(changed);
  b.add(S.boy, "y", -0.5, 0.5, 0.005).name("nudge y (m)").onChange(changed);
  b.add(S.boy, "z", -1, 1, 0.005).name("nudge z (m)").onChange(changed);
  if (boyMode === "mesh") {
    b.add(S.mesh, "bias", 0, 0.2, 0.001).name("photo tolerance").onChange(changed);
    b.add(S.mesh.gain, "0", 0, 2, 0.01).name("hidden sides red").onChange(changed);
    b.add(S.mesh.gain, "1", 0, 2, 0.01).name("hidden sides green").onChange(changed);
    b.add(S.mesh.gain, "2", 0, 2, 0.01).name("hidden sides blue").onChange(changed);
    b.add(S.mesh, "card").name("fill outline gaps").onChange(changed);
    b.add(S.mesh, "scale", 0.2, 1.5, 0.01).name("size (reload)").onChange(saveTuned);
  }
  const mode = { boy: boyMode };
  b.add(mode, "boy", { "splats (SHARP)": "splat", "Meshy model": "mesh" }).name("boy type (reloads)")
    .onChange((v) => reloadWith({ boy: v === "mesh" ? "mesh" : null }));

  // --- room -------------------------------------------------------------------
  const r = gui.addFolder("Room");
  r.add(S.room, "visible").onChange(changed);
  r.add(S.room, "distance", 0.5, 1.5, 0.005).name("distance ×").onChange(changed);
  r.add(S.room, "x", -0.5, 0.5, 0.005).name("nudge x (m)").onChange(changed);
  r.add(S.room, "y", -0.5, 0.5, 0.005).name("nudge y (m)").onChange(changed);
  r.add(S.room, "z", -1, 1, 0.005).name("nudge z (m)").onChange(changed);
  const file = { room: roomUrl.includes("mobile") ? "mobile" : "full" };
  r.add(file, "room", { "full (desktop)": "full", "thinned (phones)": "mobile" }).name("room file (reloads)")
    .onChange((v) => reloadWith({ roomfile: v }));

  // --- render -----------------------------------------------------------------
  const g = gui.addFolder("Render");
  g.add(S.render, "pixelRatio", 0.5, 2, 0.05).name("resolution ×").onChange(changed);
  g.addColor(S.render, "background").onChange(changed);
  g.add(S.render, "vignette", 0, 1, 0.01).name("edge darkening").onChange(changed);
  g.add(S.render, "vignetteSize", 0.2, 1, 0.01).name("clear middle size").onChange(changed);

  // --- save -------------------------------------------------------------------
  const actions = {
    copy() {
      const text = "SETTINGS = " + JSON.stringify(S, null, 2);
      navigator.clipboard?.writeText(text).then(
        () => flash("Copied. Paste it to Claude or into SETTINGS in hero3d.js."),
        () => { console.log(text); flash("Copy blocked: the values are in the console."); });
    },
    reset() {
      clearTuned();
      for (const k of Object.keys(defaults)) Object.assign(S[k], JSON.parse(JSON.stringify(defaults[k])));
      gui.controllersRecursive().forEach((ctl) => ctl.updateDisplay());
      apply();
    },
  };
  gui.add(actions, "copy").name("Copy settings");
  gui.add(actions, "reset").name("Reset to the code's values");

  [f1, c, b, r, g].forEach((f) => f.close());
  addNotes(gui, api);

  function flash(msg) {
    const note = document.createElement("div");
    note.className = "tune-note";
    note.textContent = msg;
    gui.domElement.append(note);
    setTimeout(() => note.remove(), 4000);
  }
}

function reloadWith(extra) {
  const url = new URL(location.href);
  for (const [k, v] of Object.entries(extra)) v == null ? url.searchParams.delete(k) : url.searchParams.set(k, v);
  location.href = url;
}
