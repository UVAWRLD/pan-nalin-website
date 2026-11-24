// Headless screenshots of the home page hero: the flat photo, the 3D scene at
// rest (they should match), and the scene with the pointer in two corners.
//   node tools/scene3d/hero_shots.mjs <out_dir> [width=1440] [height=900] [mobile]
import { createRequire } from "node:module";
import fs from "node:fs";
import path from "node:path";
const require = createRequire(path.resolve("work/qa/") + "/");
const { chromium } = require("playwright");
const out = process.argv[2] || "work/shots/hero";
const width = Number(process.argv[3] || 1440), height = Number(process.argv[4] || 900);
const mobile = process.argv[5] === "mobile";
fs.mkdirSync(out, { recursive: true });
const browser = await chromium.launch({ args: ["--use-angle=metal", "--enable-gpu", "--ignore-gpu-blocklist"] });
const ctx = await browser.newContext({ viewport: { width, height }, deviceScaleFactor: mobile ? 2 : 1, hasTouch: mobile, isMobile: mobile });
const page = await ctx.newPage();
const errors = [];
page.on("console", (m) => { if (m.type() === "error" || m.type() === "warning") errors.push(m.text()); });
page.on("pageerror", (e) => errors.push(e.message));
await page.goto("http://localhost:8431/index.html");
await page.click("[data-enter]");
await page.waitForFunction("document.querySelector('.hero').classList.contains('is-3d')", null, { timeout: 60000 });
await page.addStyleTag({ content: ".hero__image, .hero__canvas { transition: none !important; }" });
await page.waitForTimeout(3500);   // let the intro screen finish fading out
const shot = async (name) => { await page.waitForTimeout(250); await page.screenshot({ path: path.join(out, name + ".png") }); console.log("shot", name); };
// freeze the camera at rest
await page.evaluate(() => { const s = window.__hero; s.options.drift = 0; s.setView(0, 0, 0); s.setPaused(true); });
await shot("3d-rest");
await page.evaluate(() => document.querySelector(".hero").classList.remove("is-3d"));
await shot("photo");
await page.evaluate(() => { document.querySelector(".hero").classList.add("is-3d"); window.__hero.setPaused(false); });
if (!mobile) {
  await page.mouse.move(20, 20); await page.waitForTimeout(2500); await shot("pointer-top-left");
  await page.mouse.move(width - 20, height - 20); await page.waitForTimeout(2500); await shot("pointer-bottom-right");
}
await page.evaluate(() => window.scrollTo(0, innerHeight * 0.5)); await page.waitForTimeout(2000); await shot("scrolled");
console.log("fps check:", await page.evaluate(() => new Promise((res) => { let n = 0; const t0 = performance.now(); const f = () => { n++; performance.now() - t0 < 1000 ? requestAnimationFrame(f) : res(n); }; requestAnimationFrame(f); })));
if (errors.length) console.log("console:", errors.slice(0, 8));
await browser.close();
