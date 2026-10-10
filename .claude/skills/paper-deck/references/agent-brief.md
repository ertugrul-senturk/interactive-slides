# Brief for parallel slide agents (copy, fill the <…> parts, save to the scratchpad, point every agent at it)

## The deck and the audience

**The deck.** `public/decks/<deck>.html` in C:/Users/ASUS/Desktop/slides-site is a single self-contained HTML file:
- CSS sits in one `<style>` block, split into per-slide sections such as `/* ---------- 4 · storage ---------- */`.
- Each slide is a `<section class="slide" id="s-…">` inside `#stage`, on a 1280×720 canvas.
- JS sits in one `<script>` block, also split into per-slide sections.

Each slide has an `<aside class="notes">` with speaker notes; update the notes when you change what a slide shows. Paper figures live in `public/decks/<deck>/paper/`. The paper PDF is at `<path>`. Read its text and tables with PyMuPDF, and use it as the source of truth: never invent numbers.

**The audience.** The presenter, Ertugrul Senturk, a grad student, gives this ~30-minute talk to his professor's lab group. Read `.claude/skills/paper-deck/SKILL.md` and `references/feedback-log.md` first; they describe his style and his past verdicts. The short version: simpler, clearer, fewer words in full sentences, plain one-line titles, no AI-looking chrome, interactives only when they show a real effect, charts drawn in code from real numbers, and sharp uncropped images.

**The reference look.** Before you design anything, screenshot `<reference slide ids>` and match their visual language.

## Your slides

<slide ids, the CSS/JS sections they use, the presenter's verbatim feedback, and the concrete direction for this round>

## Rules for working in parallel

Several agents edit this same file at the same time.

- **Scope:** edit only your own slides' `<section>`s and their CSS and JS sections. Don't touch other slides, the shared CSS (type, layout helpers, controls) or the navigation JS unless your task says so. For a variant of a shared class, add a slide-scoped rule.
- **Editing:** use the Edit tool with small, exact, unique replacements. Never rewrite the whole file with Write or a script. If Edit says the file changed, re-read just your region and retry.
- **Stable names:** keep slide ids stable unless your task is to delete or merge slides, and don't rename shared JS helpers.
- **Tabs:** a tabbed slide gets `data-steps="N"` plus `hooks['<id>'] = step => draw(step)`, and a click sets the global `step`. See `s-method` / `s-main` in the VPT deck.
- **Measured numbers:** render them from `const RESULTS = …;` (written by `demo/<…>/fill_deck.py`). Never hard-code them, and keep that marker line intact.
- **PowerPoint prep:** `scripts/pptx-prep/<deck>.js` queries some elements. Update it if you change what it relies on.
- **Page numbers:** they are automatic. Keep the bottom-right corner free, and keep content clear of the `.src` line.

## Verify

Screenshot your slides from the repo root:

```
node .claude/skills/paper-deck/scripts/shot.mjs <deck>.html "<your own scratch dir>" <your slide ids>
```

- **Every state:** use `--eval "…"` to click tabs or press ArrowRight. Check every step, reached both by click and by arrow key.
- **Look at the result:** read every PNG and iterate until each slide is clean and balanced, with no overflow or collisions.
- **Page errors:** a page error from someone else's slide may be another agent mid-edit, so re-run before worrying.
- **Don't** start a dev server, commit, or upload anything.

## Report back

- Your final slide ids and their h2 titles.
- What changed on each slide, in 1–2 sentences.
- Where each number comes from.
- Any files you created.
- What you were unsure about.
- Screenshot paths.
