// Pan Nalin — interactions
(() => {
  const body = document.body;
  const intro = document.getElementById("intro");

  // --- Intro: click the seal or "Discover" to enter -------------------------
  // hero3d.js listens for "pannalin:enter" to start its camera move.
  const enter = (event) => {
    event.preventDefault();
    if (!body.classList.contains("is-loading")) return;
    window.scrollTo(0, 0);
    body.classList.remove("is-loading");
    intro.querySelector("video")?.pause();
    document.activeElement?.blur();
    intro.inert = true;
    document.querySelector(".brand")?.focus({ preventScroll: true });
    document.dispatchEvent(new CustomEvent("pannalin:enter"));
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
  const root = document.documentElement;
  let last = -1;
  (function fade() {
    requestAnimationFrame(fade);
    if (!card) return;
    const left = card.getBoundingClientRect().bottom - window.innerHeight;
    const v = Math.min(Math.max(left / (window.innerHeight * 0.33), 0), 1);
    const o = v * v * (3 - 2 * v);
    if (Math.abs(o - last) < 0.002) return;
    last = o;
    root.style.setProperty("--fade-o", o.toFixed(3));
    root.style.setProperty("--fade-s", (0.35 + 0.65 * o).toFixed(3));
    root.style.setProperty("--fade-b", `${((1 - o) * 14).toFixed(1)}px`);
  })();
})();
