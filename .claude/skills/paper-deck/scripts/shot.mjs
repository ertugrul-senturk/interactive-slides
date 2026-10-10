// Screenshot slides of a deck in public/decks at the native 1280x720, by section id, with headless Chrome/Edge.
//
// usage (from the repo root):
//   node .claude/skills/paper-deck/scripts/shot.mjs <deck.html> <outDir> all
//   node .claude/skills/paper-deck/scripts/shot.mjs <deck.html> <outDir> s-main s-data [--eval "<js>"] [--base http://localhost:3000/decks/]
//
// <deck.html> is a file name in public/decks (or a path). Each shot is saved as NN-<id>.png, NN = slide number.
// --eval runs JavaScript on the page before each shot, e.g. to click a tab or press the arrow key:
//   --eval "dispatchEvent(new KeyboardEvent('keydown',{key:'ArrowRight'}))"
// --base loads the deck over http instead of file:// (needed for slides that fetch models or data,
//   e.g. the live demo; the Next dev server serves public/decks at http://localhost:3000/decks/).
// It prints the slide order and any page errors. Under file://, fetch() from a demo slide fails with CORS; ignore those.
import { createRequire } from "module";
import { pathToFileURL } from "url";
import fs from "fs";
import path from "path";

const ROOT = process.cwd();
const require = createRequire(path.join(ROOT, "package.json"));
const puppeteer = require("puppeteer-core");

const args = process.argv.slice(2);
const opt = name => { const i = args.indexOf(name); if (i < 0) return null; const v = args[i + 1]; args.splice(i, 2); return v; };
const evalCode = opt("--eval"), base = opt("--base");
const [deckArg, out, ...want] = args;
if (!deckArg || !out || !want.length) { console.error("usage: shot.mjs <deck.html> <outDir> all | <slideId> ... [--eval js] [--base url]"); process.exit(1); }
const deckPath = fs.existsSync(deckArg) ? deckArg : path.join(ROOT, "public", "decks", deckArg);
const url = base ? base.replace(/\/?$/, "/") + encodeURIComponent(path.basename(deckPath)) : pathToFileURL(deckPath).href;
fs.mkdirSync(out, { recursive: true });

const chrome = [process.env.CHROME_PATH, "C:/Program Files/Google/Chrome/Application/chrome.exe",
  "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe", "C:/Program Files/Microsoft/Edge/Application/msedge.exe",
  "/usr/bin/google-chrome", "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"].find(p => p && fs.existsSync(p));
const browser = await puppeteer.launch({ executablePath: chrome, headless: true });
const page = await browser.newPage();
await page.setViewport({ width: 1280, height: 720, deviceScaleFactor: 1 });
const errors = [];
page.on("pageerror", e => errors.push(String(e)));
page.on("console", m => { if (m.type() === "error") errors.push(m.text()); });

await page.goto(url + "#1", { waitUntil: "networkidle0" });
const ids = await page.evaluate(() => [...document.querySelectorAll(".slide")].map(s => s.id));
for (const id of want[0] === "all" ? ids : want) {
  const i = ids.indexOf(id);
  if (i < 0) { console.log("no slide with id", id); continue; }
  await page.goto(url + "#" + (i + 1), { waitUntil: "networkidle0" });
  await page.reload({ waitUntil: "networkidle0" });           // the deck reads the hash only on load
  await page.evaluate(() => document.fonts.ready);
  if (evalCode) await page.evaluate(evalCode);
  await new Promise(r => setTimeout(r, 500));                 // let the .2s slide fade and bar transitions finish
  const f = path.join(out, String(i + 1).padStart(2, "0") + "-" + id + ".png");
  await page.screenshot({ path: f });
  console.log(f);
}
console.log("slide order:", ids.map((d, k) => `${k + 1}:${d}`).join(" "));
const real = errors.filter(e => !/CORS|ERR_FAILED|Failed to fetch/.test(e));
if (real.length) console.log("PAGE ERRORS:\n" + real.join("\n"));
await browser.close();
