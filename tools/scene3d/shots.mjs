// Headless screenshots of scene.html from a list of camera positions.
//
//   node tools/scene3d/shots.mjs <out_dir> [width=1920] [height=1366] [views.json] [extra query]
//
// views: [{ name, yaw, pitch, dolly }]. Default: home plus the corners of the range.
import { createRequire } from "node:module";
import fs from "node:fs";
import path from "node:path";
const require = createRequire(path.resolve("work/qa/") + "/");
const { chromium } = require("playwright");

const out = process.argv[2] || "work/shots";
const width = Number(process.argv[3] || 1920), height = Number(process.argv[4] || 1366);
const views = process.argv[5] && process.argv[5] !== "-" ? JSON.parse(fs.readFileSync(process.argv[5], "utf8")) : [
  { name: "home", yaw: 0, pitch: 0 },
  { name: "left", yaw: -10, pitch: 0 }, { name: "right", yaw: 10, pitch: 0 },
  { name: "up", yaw: 0, pitch: 6 }, { name: "down", yaw: 0, pitch: -6 },
];
const extra = process.argv[6] || "";
fs.mkdirSync(out, { recursive: true });

const browser = await chromium.launch({ args: ["--use-angle=metal", "--enable-gpu", "--ignore-gpu-blocklist", "--enable-webgl"] });
const page = await browser.newPage({ viewport: { width, height }, deviceScaleFactor: 1 });
page.on("console", (m) => { if (m.type() === "error") console.log("console error:", m.text()); });
page.on("pageerror", (e) => console.log("page error:", e.message));
await page.goto(`${process.env.SCENE_URL || "http://localhost:8431/scene.html"}?still&dpr=1${extra ? "&" + extra : ""}`);
await page.waitForFunction("window.__ready === true", null, { timeout: 60000 });
for (const v of views) {
  await page.evaluate((v) => { if (v.zoom) window.__scene.setZoom(v.zoom); window.__scene.setView(v.yaw || 0, v.pitch || 0, v.dolly || 0); }, v);
  await page.waitForTimeout(150);
  await page.screenshot({ path: path.join(out, v.name + ".png") });
  console.log("shot", v.name);
}
await browser.close();
