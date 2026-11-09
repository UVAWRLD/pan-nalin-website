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
})();
