---
name: paper-deck
description: How to design, build and revise the interactive paper-reading slide decks in this repo (public/decks/*.html, e.g. the VPT deck and the LoRA deck that follows) so they match Ertugrul's style — academic print look, plain human titles, short full-sentence text, charts redrawn in code from the paper's real numbers, arrow-key-stepped tabs, auto page numbers, and a live in-browser model demo. Use it whenever you create a new deck, rework or review slides, redraw a paper figure, add an interactive, change the live demo, or split deck work across subagents in slides-site — even if the request only says "fix slide 14", "the graphic is bad", "make it simpler" or "next paper".
---

# Paper-reading decks for slides-site

The decks are teaching material for a professor's reading group (Prof. Jiebo Luo's group). A listener who has not read the paper must follow every slide while Ertugrul talks. Each review he does comes back to one verdict: **too complicated, too wordy, or too AI-looking.** Everything below exists to avoid those three failures.

The VPT deck (`public/decks/2026-10-vpt-visual-prompt-tuning.html`) is the reference implementation. When unsure how something should look or be coded, screenshot the matching VPT slide and copy its approach. `references/feedback-log.md` has his exact verdicts, rejected and accepted, which are the best guide to his taste. Read it before any substantial design work.

## 1. The deck contract (don't break it)

- **One self-contained HTML file per deck** in `public/decks/`, with assets in `public/decks/<deck>/` (paper crops in `paper/`). `lib/decks.ts` lists the decks; the viewer is `app/view/[file]/`.
- **1280×720 canvas.** Use `#stage`, and one `<section class="slide" id="s-…" data-title="…">` per slide. `.slide.active` is the visible slide. Slide ids are stable names; never refer to slides by number in code.
- **Per-slide sections.** CSS and JS are both split into per-slide sections, headed `/* ---------- name ---------- */`. Add rules scoped to the slide (`#s-main …`); don't edit the shared type, layout and controls sections for one slide's needs.
- **Speaker notes:** `<aside class="notes">` on every slide. Rewrite them whenever the slide changes. Pressing N opens the presenter window.
- **Source line:** `<p class="src">` at the bottom (15px, `--ink-3`). He likes these, as short factual citations: "Dataset examples from Jia et al. 2022, Fig. 10. ViT-B/16: Dosovitskiy et al. 2021."
- **Page numbers are automatic.** The navigation JS injects `.pgnum` into every slide except the title, computed from slide order. Never hard-code numbers. Keep the bottom-right corner (right 40px, bottom ~38px) free, and keep `.src` stopping at right:100px.
- **Stepped slides:** `data-steps="N"` on the section, plus `hooks['<id>'] = step => draw(step)`. ArrowRight goes through the steps, then to the next slide; ArrowLeft goes back and arrives on the last step. A tab click must also set the global `step` (`step = +b.dataset.v`), so the arrow keys continue from there. Every tabbed explanatory slide works this way, because he presents every state.
- **Measured numbers come from data, never typed in.** His own results live in `const RESULTS = …;`, written by `demo/vpt/fill_deck.py` from `demo/vpt/weights/*.json`. Render from it, and keep that marker line intact.
- **PowerPoint export:**
  - `npm run pptx` (`scripts/build-pptx.mjs`) steps through every slide in headless Chrome.
  - `scripts/pptx-prep/<deck>.js` runs first, to freeze interactives. Update it when you change elements it queries.
  - Exported text boxes include the page numbers.
- **Live demo:**
  - The models run in the browser with onnxruntime-web. Files come from the free Hugging Face model repo `valinor61/vpt-medmnist`; override with `?models=<url>/`, or use a Gradio fallback with `?demo=<url>`.
  - `next.config.mjs` sets COOP/COEP (credentialless) so threads work.
  - Exports come from `demo/vpt/export_onnx.py` (`--regimes full,1k`) using `demo/vpt/.venv`; uploads go through `upload_models.py`.
  - Uploading to Hugging Face publishes files, so ask first.

## 2. Look: academic print

- **Fonts and colour.** Source Serif 4 for titles and big numbers; Source Sans 3 for body text; paper background `--paper` (#FBFAF7).
  - Colour carries meaning, taken from the paper's own figures:
    - **red `--tuned`** = trained / learned (prompts, head, the method being presented);
    - **blue `--frozen`** = frozen backbone;
    - **grey `--token`** = image or text tokens.
  - In charts: **red = the presented method, dark ink = full fine-tuning (the reference), grey = other baselines.** Keep this fixed across the whole deck.
- **No AI-looking chrome:**
  - eyebrow or kicker lines ("Parameter-efficient fine-tuning, reading 1 of 2 · next: LoRA");
  - tag labels ("Meaning 1 · still text");
  - numbered mini-headers ("1 · Cut and embed");
  - middle-dot label soup;
  - emoji or decorative icons;
  - progress dots;
  - teal pill buttons;
  - visible config fields ("browser" input + Load button).
- **No per-section restyling.** He called the tinted-background / top-rule / running-foot "My experiments" treatment cheap. Mark a new part with a plain divider slide in the normal style.
- **Charts are drawn in code (HTML or SVG), large and clean:**
  - horizontal bars start at 0;
  - values sit on or at the end of the bar;
  - names sit at the line ends instead of a legend;
  - a dashed reference line marks full fine-tuning;
  - no chartjunk;
  - axis text ≥15px.
  - Study `s-protocols`, `s-main`, `s-data`, `s-cost` and `s-slopes` for the pattern.
- **Images must be sharp and uncropped.**
  - Never `object-fit:cover` slivers of a composite paper figure.
  - Extract single example images from the paper PDF with PyMuPDF, either the embedded images directly (sharper than a render) or a 300-dpi render with a clip rect.
  - Save them as new files and look at every crop before using it.

## 3. Words: like he wrote it

- **Titles (h2):**
  - one line, ideally ≤9–10 words, plain and natural;
  - "How a Vision Transformer reads an image", "Where VPT falls short", "Summary".
  - Not claim-essays ("Main result: VPT-deep beats … while storing 1.18 models instead of 24"), not colon prefixes ("Before the method:", "Design choice 1:", "Live:", "Backup:"), and no hooks ("Try the numbers", "So what is a *visual* prompt?").
  - A title that is a slogan nobody would say ("A few learned inputs can replace fine-tuning, within limits") also fails.
- **Body text:**
  - **Short full sentences, never bullet points:** two or three per slide or per step. When asked to shorten, cut words, not into fragments.
  - **No bold run-in labels** such as "How to read it.", "Why this matters." or "What it costs.".
  - **Body ≥18px.**
- **Explain for humans, not machines.** Say what a thing is for ("[CLS] is one extra vector that collects a summary of the image; the head reads only it"). Don't list indices, tensor shapes or numbers that mean nothing alone. A big highlighted "38,400" means nothing to him; "0.14% of the model" or "99.4% less storage per task" does.
- **Remove redundant phrases.** "Nothing is selected:" next to "a fixed grid cuts every patch" is noise.
- **His experiments.** Use natural wording ("I tested VPT on three medical datasets", "Medical images, my own experiments"), never "Your own task".

## 4. Content rules

- **One idea per visual.** If a slide needs a legend plus a paragraph to decode, split it into arrow-key steps (one comparison or group per step) or simplify.
- **Interactives must show the audience a real effect.**
  - Good: drag the number of tasks and watch storage grow; switch shallow or deep; with prompts or without; 1,000 images or all.
  - Bad: hovering a patch to show its index; schematic 768-number strips; attention-matrix hovers.
- **Every number from the paper or the measured data.**
  - Read tables with PyMuPDF text extraction.
  - If a value exists only in a figure, and the figure is a vector drawing, read the plotted points from the PDF's path coordinates calibrated on the gridlines. Check them against any overlapping table value. Say so in the source line ("values read from its vector drawing in the PDF; the 100% points match Table 1").
  - Never invent numbers.
- **Don't draw what wasn't measured.** Two training sizes are two separate groups of bars, not a line between them. He called the slope chart "misleading".
- **Keep numbers consistent across slides.** Say what a percentage includes: "0.6% of the weights" includes prompts plus head, and the same base everywhere (343 MB copy, 2.2 MB VPT file).
- **Don't repeat a topic.** The "where the word prompt comes from" history needed at most two slides, not five.
- **Order matters.** Explain a concept before any slide that uses it: how a ViT reads an image comes before what a visual prompt is.
- **Cut slides that are already covered.** A backup table that repeats the result slides was "in the wrong place and wrong style"; move its one unique fact (the definition of balanced accuracy) into a source line.
- **Medical predictions** keep a quiet "Research demo, not a diagnostic tool".

## 5. Workflow

1. **Look before and after every change.** Screenshot with the bundled script, from the repo root:
   ```
   node .claude/skills/paper-deck/scripts/shot.mjs <deck.html> <scratch dir> all
   node .claude/skills/paper-deck/scripts/shot.mjs <deck.html> <dir> s-main --eval "dispatchEvent(new KeyboardEvent('keydown',{key:'ArrowRight'}))"
   ```
   - Read every PNG. Check for overflow, collisions with `.src` or the page number, ragged columns, dead space, and text under 18px.
   - Check every step, reached both by click and by arrow key.
   - Use `--base http://localhost:3000/decks/` for slides that fetch models.
   - In the browser pane, a hash-only navigation doesn't reload the page; call `location.reload()`.
2. **Show him the deck and let him point at problems.** Start the dev server with `preview_start` (`slides`). The deck is at `/decks/<file>`, the viewer at `/view/<file>`. He gives slide-numbered feedback; group it and act on all of it.
3. **Parallel agents for big rounds.** Split the work by slide ownership (one agent per slide or per closely related slide group, plus one for global pieces). Give every agent the brief in `references/agent-brief.md`, adapted to the round.
   - Agents edit the same HTML file at once. Small exact Edit replacements only, never whole-file rewrites, and re-read the region when Edit says the file changed.
   - Agents still break this sometimes. Afterwards, grep for each agent's key markers and run a full `all` screenshot pass to confirm nothing was lost.
   - Relay ownership changes with SendMessage when a new agent takes over slides another one might touch.
4. **Explain before changing when he asks a question.** "Explain it to me before changing anything" means answer only. Diagnose what on the slide caused the confusion, propose the fix, then do it when he says go.
5. **Save durable lessons.** New taste lessons go into `references/feedback-log.md`, so the next deck starts where this one ended.

## 6. Skeleton of a talk (what worked for VPT)

1. **Title:** one-line description, one clean visual of the idea, authors and venue, presenter and date.
2. **The problem:** why the paper exists. For VPT: pretrain once and adapt per task; full fine-tuning stores a copy per task; the cheaper methods before it.
3. **Background, in at most two slides, in dependency order.**
4. **The method:** paper figure plus equations, with a toggle for the variants; why it works, in one simple picture; parameter and cost counts; code.
5. **Results:** benchmark overview; main result; ablations; limits; costs. One chart per step.
6. **His own experiments:** plain divider slide; setup; results; costs; live demo on one slide.
7. **Summary:** the key numbers, one plain sentence each. Then references, grouped and consistently formatted, containing exactly what the deck cites.
