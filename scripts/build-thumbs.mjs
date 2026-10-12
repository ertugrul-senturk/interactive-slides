// Takes a picture of each deck's title slide for the index page: public/decks/<deck>.jpg,
// and a small picture of every slide for the viewer's slide list: public/decks/thumbs/<deck>/
// (one .jpg per slide, named by slide id, plus slides.json listing them in order).
//
// Usage: npm run thumbs               (all decks)
//        npm run thumbs -- some-deck.html
// Set CHROME_PATH if Chrome/Edge is not in a standard location.

import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";
import puppeteer from "puppeteer-core";

const DECKS_DIR = path.join(process.cwd(), "public", "decks");

function browserPath() {
  const candidates = [
    process.env.CHROME_PATH,
    "C:/Program Files/Google/Chrome/Application/chrome.exe",
    "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
    "C:/Program Files/Microsoft/Edge/Application/msedge.exe",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
  ];
  const found = candidates.find((p) => p && fs.existsSync(p));
  if (!found) throw new Error("No Chrome or Edge found. Set CHROME_PATH.");
  return found;
}

const only = process.argv.slice(2).filter((a) => a.endsWith(".html"));
const files = fs.readdirSync(DECKS_DIR).filter((f) => f.toLowerCase().endsWith(".html") && (!only.length || only.includes(f)));

const browser = await puppeteer.launch({ executablePath: browserPath(), headless: true });
for (const file of files) {
  const src = path.join(DECKS_DIR, file);
  const page = await browser.newPage();
  // a 1280×720 picture, shown at about 300 px wide, so it stays sharp on high-density screens
  await page.setViewport({ width: 1280, height: 720, deviceScaleFactor: 1 });
  await page.emulateMediaFeatures([{ name: "prefers-reduced-motion", value: "reduce" }]);
  await page.goto(pathToFileURL(src).href + "#1", { waitUntil: "load" });
  await page.addStyleTag({ content: "*,*::before,*::after{transition:none!important;animation:none!important}" });
  await page.evaluate(() => document.fonts.ready);
  await new Promise((r) => setTimeout(r, 500));
  const dest = src.replace(/\.html$/i, ".jpg");
  await page.screenshot({ path: dest, type: "jpeg", quality: 86 });

  // every slide at 384×216, which stays sharp in the ~190 px wide list on high-density screens
  const base = file.replace(/\.html$/i, "");
  const dir = path.join(DECKS_DIR, "thumbs", base);
  fs.rmSync(dir, { recursive: true, force: true });
  fs.mkdirSync(dir, { recursive: true });
  const ids = await page.evaluate(() => [...document.querySelectorAll(".slide")].map((s) => s.id || null));
  const shots = [];
  for (let k = 0; k < ids.length; k++) {
    // the decks follow the slide number in the address
    await page.evaluate((n) => location.replace("#" + n), k + 1);
    await page.waitForFunction((k) => document.querySelectorAll(".slide")[k].classList.contains("active"), { timeout: 5000 }, k);
    await page.evaluate(() => Promise.all([...document.querySelectorAll(".slide.active img")]
      .filter((i) => !i.complete).map((i) => new Promise((r) => { i.onload = i.onerror = r; }))));
    await new Promise((r) => setTimeout(r, 250));
    const name = `${ids[k] ?? String(k + 1).padStart(2, "0")}.jpg`;
    await page.screenshot({ path: path.join(dir, name), type: "jpeg", quality: 80, clip: { x: 0, y: 0, width: 1280, height: 720, scale: 0.3 } });
    shots.push({ id: ids[k], file: name });
  }
  fs.writeFileSync(path.join(dir, "slides.json"), JSON.stringify(shots, null, 1) + "\n");
  await page.close();
  console.log(`${file} → ${path.basename(dest)}, ${shots.length} slide pictures`);
}
await browser.close();
