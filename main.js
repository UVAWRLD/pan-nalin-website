// Pan Nalin — interactions
(() => {
  const body = document.body;
  const intro = document.getElementById("intro");

  // Motion access for steering the 3D sky by tilt (iOS asks once, in a tap).
  window.__askTilt = () => (window.__tilt ||= (() => {
    const DOE = window.DeviceOrientationEvent;
    return typeof DOE?.requestPermission === "function"
      ? DOE.requestPermission().catch(() => "denied")
      : Promise.resolve("granted");
  })());

  // --- Intro: tap the seal to open the envelope and enter -------------------
  // intro3d.js opens the envelope and calls back as the letter lifts;
  // hero3d.js listens for "pannalin:enter" to start its camera move.
  let opening = false;
  const go = () => {
    window.scrollTo(0, 0);
    body.classList.remove("is-loading");
    document.activeElement?.blur();
    intro.inert = true;
    document.dispatchEvent(new CustomEvent("pannalin:enter"));
  };
  const enter = (event) => {
    event.preventDefault();
    if (!body.classList.contains("is-loading") || opening) return;
    opening = true;
    // iPhone: motion access (to steer the 3D sky by tilting) must be asked
    // for inside the tap, so it is asked here and only here.
    window.__askTilt();
    if (window.__envelope) window.__envelope.open(go);
    else go();
  };
  document.querySelectorAll("[data-enter]").forEach((el) =>
    el.addEventListener("click", enter)
  );

  // --- Header shade once the card starts sliding under it -------------------
  const card = document.querySelector(".bio__card");
  const onScroll = () => {
    const under = card && card.getBoundingClientRect().top < window.innerHeight * 0.3;
    body.classList.toggle("is-scrolled", !!under);
  };
  window.addEventListener("scroll", onScroll, { passive: true });
  window.addEventListener("resize", onScroll);
  onScroll();

  // --- The dark fade at the screen's bottom edge says "more below" ---------
  // As the card's end comes up it melts away: thinner, softer, fainter, over
  // the last third of a screen.
  // Set on the fade itself: on the root they would restyle the whole page each frame.
  const fadeEl = document.querySelector(".bio__fade");
  let last = -1;
  (function fade() {
    requestAnimationFrame(fade);
    if (!card || !fadeEl) return;
    const left = card.getBoundingClientRect().bottom - window.innerHeight;
    const v = Math.min(Math.max(left / (window.innerHeight * 0.33), 0), 1);
    const o = v * v * (3 - 2 * v);
    if (Math.abs(o - last) < 0.002) return;
    last = o;
    fadeEl.style.setProperty("--fade-o", o.toFixed(3));
    fadeEl.style.setProperty("--fade-s", (0.35 + 0.65 * o).toFixed(3));
    fadeEl.style.setProperty("--fade-b", `${((1 - o) * 14).toFixed(1)}px`);
  })();
})();
