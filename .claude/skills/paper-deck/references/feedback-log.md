# Feedback log: what Ertugrul rejected and what he accepted

These are concrete verdicts from reviewing the VPT deck (October 2026), grouped by theme. Each entry gives what was there, what he said, and what replaced it and was accepted. Use it to predict his reaction before he sees a slide.

## Chrome and "AI-looking" UI
- **Eyebrow line.** "Parameter-efficient fine-tuning, reading 1 of 2 · next: LoRA" over the title was "too AI generated". Removed from every slide.
- **Viewer bar.** The Next viewer's "← all slides | title | open file | [teal fullscreen pill]" read as AI. It became a quiet print-like bar: serif title, plain text links, thin rule, no pill.
- **Tags and mini-headers.** "Meaning 1 · still text", "1 · Cut and embed" and emoji in an SVG were removed. Plain typography only.
- **Section restyling.** A cool-tinted background, a dark top rule and an italic "My experiments" running foot for his part of the talk were "hell nah, it looks cheap". Replaced by a plain divider slide in the normal style.
- **Demo config.** A visible "browser" input plus a Load button on the live demo was "too AI". Removed; the fallback server is chosen with `?demo=<url>` instead.
- **Page numbers.** He wants them, but small, plain and automatic, bottom-right, with none on the title slide. Never "7 / 32" and never progress bars.

## Titles
- **Too long.** "Main result: VPT-deep beats full fine-tuning on 20 of the 24 tasks while storing 1.18 models instead of 24" became "VPT-deep beats full fine-tuning on 20 of 24 tasks".
- **Prefixes.** "Design choice 1: the prompt must be …" became "Where to put the prompt". "Before the method: a Vision Transformer turns an image into 197 vectors …" became "How a Vision Transformer reads an image".
- **Hooks.** "How much is actually trained and stored? Try the numbers" ("try the numbers meeh") became "How much VPT trains and stores per task".
- **Generic or slogan titles.** "What to remember" ("bad title") and then "A few learned inputs can replace fine-tuning, within limits" ("doesn't make sense") both failed. He chose "Summary" himself. For a summary or recap, plain beats clever.
- **Generated-sounding labels.** "Four things worth discussing" "says I am the AI". The questions slide was dropped altogether.

## Text
- **Too much text.** Long right-hand columns ("What full fine-tuning does." / "What it costs." …) were "good but too long and doesn't align with the page design". They were cut to two or three sentences placed where the layout needs them.
- **Bullets.** "When I say shorten I don't mean bullet points. I need full sentences like this but clearer."
- **Bold run-ins.** "How to read it.", "Why this matters." and "What the chart shows." read as templates. Write plain sentences instead.
- **Redundancy.** "Nothing is selected: every patch, read row by row, …" was "kind of redundant". He also asked what [CLS] does, so the sentence now says what it is for.
- **Wording of his own work.** "Your own task" was "looking so odd". Use "Medical images, my own experiments, shown later" and "I tested VPT on three medical datasets".

## Visuals
- **Cropped images.** Slices of the composite Fig. 10 strips with `object-position` were "cropped and not clear at all". The fix was single sharp examples taken from the PDF's embedded images, one per dataset and all the same size, labelled with the dataset name.
- **Too complicated for the audience.** These were all replaced by one simple bar chart or a one-idea picture:
  - seven columns of striped layer stacks for the cheaper methods;
  - the attention-matrix hover;
  - the 768-number vector strip;
  - the patch-index hover ("they are not machines, they can't identify this information");
  - the self-supervised dot plot with overlapping labels.
- **Paper figures.** He liked the figures themselves ("images are great from the paper"), but wanted them redrawn in code "similar but better UI", one graph per arrow-key step, each explained in plain words. Done for Fig. 3 to Fig. 7 from the PDF's vector point coordinates.
- **Misleading lines.** A slope chart joining the 1,000-image result to the full-data result implied linear behaviour between them. It is now two separate bar groups, a 0–100 axis, and "there are no points in between" in the source line.
- **Meaningless numbers.** A big red "38,400" ("no value or meaning for me") was replaced by a percentage of the model.
- **Summary row for storage.** It should show "space efficiency as a percentage per task" (99.4% less storage than a fine-tuned copy), not MB.
- **Summary row for his results.** It should give the exact average for VPT and full fine-tuning (87.6% vs 88.8%, with the linear probe at 83.3%), worded so both numbers are named.

## Structure
- **Too many slides on one topic.** Five slides on where "prompt" comes from became at most two: (1) text prompt to learned vectors in language; (2) the same idea on image patches, saying what the red vectors are (random at first, trained for the task, the same for every input, not taken from the image).
- **Wrong order.** The visual-prompt slide used patches before the ViT slide explained them. The two were swapped.
- **Redundant slides.** Two live demo slides ("second slide is redundant") were merged into one. The questions slide and the backup "all my numbers" table were removed.
- **Live demo.** He wanted the 1,000-image models next to the full-data models. A toggle switches all three methods. Switching swaps a 1.8 MB prompt file for VPT but needs another 172 MB model for full fine-tuning, which is a talking point.

## Accepted and praised
- The interactive storage slider ("tasks" and "model size" changing copies against one shared model).
- The source-line style.
- The VPT method slide with the paper figure, a shallow/deep toggle and equations. It was very good, though the wording needed simplifying.
- The round-2 charts on slides 12–18 ("it's going great, you have learned my style kinda"). Copy those patterns for new charts.
