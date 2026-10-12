// Records one real run of the VPT live demo, so the PowerPoint and PDF versions can show real
// predictions instead of empty columns (the static export opens the deck from disk, where the
// models cannot be downloaded).
//
// Needs the dev server (npm run dev, http://localhost:3000). It opens the demo slide, picks one
// test image, waits until all three models have answered, once with the models trained on all
// training images and once with the 1,000-image models, and writes both finished demo panels into
// the DEMO_SNAPSHOT block of 2026-10-vpt-visual-prompt-tuning.js (one static page each).
//
//   node scripts/pptx-prep/2026-10-vpt-snapshot-demo.mjs [image file, default dermamnist-4-0.png] [deck .html]
//
// The LoRA deck has the same demo slide: pass 2026-10-lora-low-rank-adaptation.html to record it into that deck's
// prep script instead.
import fs from "node:fs";
import path from "node:path";
import puppeteer from "puppeteer-core";

const args = process.argv.slice(2);
const DECK = args.find((a) => a.endsWith(".html")) || "2026-10-vpt-visual-prompt-tuning.html";
const PREP = path.join(process.cwd(), "scripts", "pptx-prep", DECK.replace(/\.html$/, ".js"));
const image = args.find((a) => a.endsWith(".png")) || "dermamnist-4-0.png";
const task = image.split("-")[0];
const chrome = [process.env.CHROME_PATH, "C:/Program Files/Google/Chrome/Application/chrome.exe",
  "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe", "/usr/bin/google-chrome",
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"].find((p) => p && fs.existsSync(p));

const browser = await puppeteer.launch({ executablePath: chrome, headless: true });
try {
  const page = await browser.newPage();
  await page.setViewport({ width: 1280, height: 720 });
  const n = await (async () => {
    await page.goto(`http://localhost:3000/decks/${DECK}#1`, { waitUntil: "load" });
    return page.evaluate(() => [...document.querySelectorAll(".slide")].findIndex((s) => s.id === "s-demo") + 1);
  })();
  await page.goto(`http://localhost:3000/decks/${DECK}#${n}`, { waitUntil: "load" });
  await page.reload({ waitUntil: "load" });
  const done = () => page.waitForFunction(() => /All [a-z]+ done/.test(document.querySelector("#s-demo .st-text")?.textContent || ""), { timeout: 600000 });
  await page.click(`#task button[data-t="${task}"]`);
  await page.waitForSelector(`#thumbs img[src$="${image}"]`);
  await page.click(`#thumbs button:has(img[src$="${image}"])`);
  const shots = [];
  for (const regime of ["full", "1k"]) {
    if (regime !== "full") {
      await page.evaluate(() => { document.querySelector("#s-demo .st-text").textContent = ""; });
      await page.click(`#regime button[data-r="${regime}"]`);
    }
    await done();
    await new Promise((r) => setTimeout(r, 800));     // let the bars finish their transition
    const html = await page.evaluate(() => document.querySelector("#s-demo .demo").outerHTML);
    if (/blob:/.test(html)) throw new Error("snapshot still refers to a blob: URL");
    shots.push({ regime, html });
  }
  const html = shots;
  const src = fs.readFileSync(PREP, "utf8");
  const re = /\/\* DEMO_SNAPSHOT \*\/[\s\S]*?\/\* END_DEMO_SNAPSHOT \*\//;
  if (!re.test(src)) throw new Error("no DEMO_SNAPSHOT block in " + PREP);
  fs.writeFileSync(PREP, src.replace(re, `/* DEMO_SNAPSHOT */ ${JSON.stringify(html)} /* END_DEMO_SNAPSHOT */`));
  console.log(`recorded ${image} (${task}), all training images and 1,000, into ${path.basename(PREP)}`);
} finally {
  await browser.close();
}
