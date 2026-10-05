// Converts the HTML decks in public/decks into .pptx files next to them.
//
// Each deck is opened in headless Edge/Chrome at 1280×720 and stepped through with the
// right-arrow key, so every slide and every step of an animated slide becomes one
// PowerPoint slide. For each state the visuals (photos, charts, canvases, shapes) are
// captured with the text made transparent and used as the slide background, and the
// text is laid back on top as editable PowerPoint text boxes at the same position,
// font, size, colour and line breaks.
//
// Usage: npm run pptx                 (all decks)
//        npm run pptx -- some-deck.html
// Set CHROME_PATH if Edge/Chrome is not in a standard location.
// scripts/pptx-prep/<deck>.js, if present, runs in the page first to fix interactive
// slides in one state and remove controls that would not work in PowerPoint.

import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";
import puppeteer from "puppeteer-core";
import PptxGenJS from "pptxgenjs";

const DECKS_DIR = path.join(process.cwd(), "public", "decks");
const W = 1280, H = 720, SCALE = 2;
const PX = 1 / 96; // inches per CSS pixel

function browserPath() {
  const candidates = [
    process.env.CHROME_PATH,
    "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
    "C:/Program Files/Microsoft/Edge/Application/msedge.exe",
    "C:/Program Files/Google/Chrome/Application/chrome.exe",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
  ];
  const found = candidates.find((p) => p && fs.existsSync(p));
  if (!found) throw new Error("No Chrome or Edge found. Set CHROME_PATH.");
  return found;
}

// Runs in the page: collects the visible text of the active slide as positioned
// blocks of styled tokens, tags each text-bearing element, and hides the text.
function extractAndHide() {
  const root = document.querySelector(".slide.active") || document.body;
  const EXCLUDE = "script,style,svg,canvas,button,select,textarea,noscript,.ticks,.tok";
  const opacityOf = (el) => { let o = 1; for (let e = el; e && e.nodeType === 1; e = e.parentElement) o *= +getComputedStyle(e).opacity; return o; };
  const blockOf = (el) => { let b = el; while (b !== root && getComputedStyle(b).display === "inline") b = b.parentElement; return b; };

  const blocks = new Map();
  const tagged = [];
  let nextId = 0;
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
  let prevEndsWithSpace = true;
  for (let n = walker.nextNode(); n; n = walker.nextNode()) {
    const value = n.nodeValue;
    if (!value.trim()) { if (value.length) prevEndsWithSpace = true; continue; }
    const p = n.parentElement;
    if (p.closest(EXCLUDE)) continue;
    const cs = getComputedStyle(p);
    if (cs.visibility !== "visible" || cs.display === "none") continue;
    const op = opacityOf(p);
    if (op < 0.05) continue;
    if (!p.dataset.pptxId) { p.dataset.pptxId = String(nextId++); tagged.push(p); }
    const style = {
      el: p.dataset.pptxId,
      size: parseFloat(cs.fontSize),
      weight: parseInt(cs.fontWeight, 10),
      italic: cs.fontStyle === "italic",
      color: cs.color,
      opacity: op,
    };
    const block = blockOf(p);
    if (!blocks.has(block)) blocks.set(block, []);
    const tokens = blocks.get(block);
    const re = /\S+/g;
    let m;
    const rectOf = (a, b) => { const r = document.createRange(); r.setStart(n, a); r.setEnd(n, b); return r.getClientRects(); };
    while ((m = re.exec(value))) {
      const space = m.index > 0 ? /\s/.test(value[m.index - 1]) : prevEndsWithSpace;
      // A word can wrap inside itself at a hyphen ("fine-" / "tuning"): split it where the line changes.
      const pieces = [];
      const rects = rectOf(m.index, m.index + m[0].length);
      if (rects.length > 1) {
        let start = m.index, top = rectOf(start, start + 1)[0]?.top;
        for (let i = start + 1; i < m.index + m[0].length; i++) {
          const t = rectOf(i, i + 1)[0]?.top;
          if (t !== undefined && top !== undefined && t > top + 2) { pieces.push([start, i]); start = i; top = t; }
        }
        pieces.push([start, m.index + m[0].length]);
      } else pieces.push([m.index, m.index + m[0].length]);
      pieces.forEach(([a, b], k) => {
        const rect = rectOf(a, b)[0];
        if (!rect || rect.width === 0) return;
        tokens.push({ text: value.slice(a, b), space: k === 0 ? space : false, style, left: rect.left, right: rect.right, top: rect.top, bottom: rect.bottom });
      });
    }
    prevEndsWithSpace = /\s$/.test(value);
  }

  const out = [];
  for (const [block, tokens] of blocks) {
    if (!tokens.length) continue;
    const bcs = getComputedStyle(block);
    const fs0 = parseFloat(bcs.fontSize);
    const lh = bcs.lineHeight === "normal" ? fs0 * 1.2 : parseFloat(bcs.lineHeight);
    // A new line starts whenever a word lands to the left of, and below, the previous one.
    const lines = [];
    for (const t of tokens) {
      const prev = lines.length ? lines[lines.length - 1].at(-1) : null;
      if (!prev || (t.left < prev.right - 1 && t.top > prev.top + 2)) lines.push([t]);
      else lines[lines.length - 1].push(t);
    }
    out.push({ align: bcs.textAlign, lineHeight: lh, lines });
  }

  for (const el of tagged) {
    el.dataset.pptxPrev = JSON.stringify([
      el.style.getPropertyValue("color"), el.style.getPropertyPriority("color"),
      el.style.getPropertyValue("-webkit-text-fill-color"), el.style.getPropertyPriority("-webkit-text-fill-color"),
    ]);
    el.style.setProperty("color", "transparent", "important");
    el.style.setProperty("-webkit-text-fill-color", "transparent", "important");
  }
  return { blocks: out, ids: tagged.map((e) => e.dataset.pptxId) };
}

function restoreText() {
  document.querySelectorAll("[data-pptx-id]").forEach((el) => {
    if (el.dataset.pptxPrev) {
      const [c, cp, f, fp] = JSON.parse(el.dataset.pptxPrev);
      el.style.removeProperty("color");
      el.style.removeProperty("-webkit-text-fill-color");
      if (c) el.style.setProperty("color", c, cp);
      if (f) el.style.setProperty("-webkit-text-fill-color", f, fp);
    }
    delete el.dataset.pptxId;
    delete el.dataset.pptxPrev;
  });
}

// The font Chrome actually rendered with (not the CSS stack), so PowerPoint matches.
async function renderedFonts(cdp, ids) {
  const fonts = {};
  const { root } = await cdp.send("DOM.getDocument", { depth: -1 });
  for (const id of ids) {
    const { nodeId } = await cdp.send("DOM.querySelector", { nodeId: root.nodeId, selector: `[data-pptx-id="${id}"]` });
    if (!nodeId) continue;
    const { fonts: used } = await cdp.send("CSS.getPlatformFontsForNode", { nodeId });
    if (used.length) fonts[id] = used.slice().sort((a, b) => b.glyphCount - a.glyphCount)[0].familyName;
  }
  return fonts;
}

function parseColor(css, opacity) {
  const m = css.match(/rgba?\(([^)]+)\)/);
  if (!m) return { color: "000000", transparency: 0 };
  const [r, g, b, a = 1] = m[1].split(/[ ,/]+/).filter(Boolean).map(Number);
  const hex = [r, g, b].map((v) => Math.round(v).toString(16).padStart(2, "0")).join("").toUpperCase();
  return { color: hex, transparency: Math.round((1 - a * opacity) * 100) };
}

function addTextBlocks(slide, blocks, fonts) {
  for (const b of blocks) {
    const all = b.lines.flat();
    const left = Math.min(...all.map((t) => t.left));
    const right = Math.max(...all.map((t) => t.right));
    // Place the text box on the CSS line box of the first line, so line i sits at top + i·lineHeight.
    const first = b.lines[0];
    const glyphTop = Math.min(...first.map((t) => t.top));
    const glyphBottom = Math.max(...first.map((t) => t.bottom));
    const top = (glyphTop + glyphBottom) / 2 - b.lineHeight / 2;

    const runs = [];
    b.lines.forEach((line, li) => {
      line.forEach((t, ti) => {
        const s = t.style;
        const { color, transparency } = parseColor(s.color, s.opacity);
        const opts = {
          fontFace: fonts[s.el] || "Segoe UI",
          fontSize: +(s.size * 0.75).toFixed(2),
          bold: s.weight >= 600,
          italic: s.italic,
          color,
          ...(transparency > 0 ? { transparency } : {}),
        };
        const text = (ti > 0 && t.space ? " " : "") + t.text;
        const last = runs.at(-1);
        const startsLine = ti === 0 && li > 0;
        if (last && !startsLine && JSON.stringify(last.options.__style) === JSON.stringify(opts)) {
          last.text += text;
        } else {
          runs.push({ text, options: { ...opts, __style: opts, ...(startsLine ? { softBreakBefore: true } : {}) } });
        }
      });
    });
    runs.forEach((r) => delete r.options.__style);

    const pad = 6 + (right - left) * 0.04;
    const center = b.align === "center";
    const x = center ? left - pad : left;
    const w = right - left + (center ? 2 * pad : pad);
    slide.addText(runs, {
      x: x * PX, y: top * PX, w: w * PX, h: b.lineHeight * b.lines.length * PX,
      margin: 0, wrap: false, fit: "none", valign: "top",
      align: center ? "center" : b.align === "right" || b.align === "end" ? "right" : "left",
      lineSpacing: +(b.lineHeight * 0.75).toFixed(2),
      paraSpaceBefore: 0, paraSpaceAfter: 0,
      isTextBox: true,
    });
  }
}

async function convert(browser, file) {
  const src = path.join(DECKS_DIR, file);
  const dest = src.replace(/\.html$/i, ".pptx");
  const page = await browser.newPage();
  await page.setViewport({ width: W, height: H, deviceScaleFactor: SCALE });
  await page.emulateMediaFeatures([{ name: "prefers-reduced-motion", value: "reduce" }]);
  await page.goto(pathToFileURL(src).href + "#1", { waitUntil: "load" });
  await page.addStyleTag({ content: "*,*::before,*::after{transition:none!important;animation:none!important}" });
  await page.evaluate(() => document.fonts.ready);
  await new Promise((r) => setTimeout(r, 400));
  // Optional per-deck script that fixes interactive slides in one state for the static file.
  const prep = path.join(process.cwd(), "scripts", "pptx-prep", file.replace(/\.html$/i, ".js"));
  if (fs.existsSync(prep)) {
    await page.evaluate(fs.readFileSync(prep, "utf8"));
    await new Promise((r) => setTimeout(r, 400));
  }
  const cdp = await page.createCDPSession();
  await cdp.send("DOM.enable");
  await cdp.send("CSS.enable");

  const title = await page.title();
  const pres = new PptxGenJS();
  pres.layout = "LAYOUT_WIDE"; // 13.333 × 7.5 in = 1280 × 720 CSS px
  pres.title = title;
  pres.subject = "Converted from " + file;

  let prevSig = null;
  let count = 0;
  for (let guard = 0; guard < 300; guard++) {
    const sig = await page.evaluate(() => location.hash + "|" + (document.querySelector(".slide.active")?.innerText ?? document.body.innerText));
    if (sig === prevSig) break;
    prevSig = sig;

    const { blocks, ids } = await page.evaluate(extractAndHide);
    const fonts = await renderedFonts(cdp, ids);
    const shot = await page.screenshot({ type: "jpeg", quality: 90, encoding: "base64" });
    await page.evaluate(restoreText);

    const slide = pres.addSlide();
    slide.background = { data: "image/jpeg;base64," + shot };
    addTextBlocks(slide, blocks, fonts);
    count++;

    await page.keyboard.press("ArrowRight");
    await new Promise((r) => setTimeout(r, 250));
  }

  await page.close();
  await pres.writeFile({ fileName: dest });
  console.log(`${file} → ${path.basename(dest)} (${count} slides)`);
}

const wanted = process.argv.slice(2);
const files = wanted.length ? wanted.map((f) => path.basename(f)) : fs.readdirSync(DECKS_DIR).filter((f) => f.toLowerCase().endsWith(".html"));
const browser = await puppeteer.launch({ executablePath: browserPath(), headless: true });
try {
  for (const f of files) await convert(browser, f);
} finally {
  await browser.close();
}
