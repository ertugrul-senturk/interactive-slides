// Runs inside the combined VPT + LoRA deck before it is converted to PowerPoint, where the controls cannot work.
// Assembled by scripts/merge-peft-deck.py: the VPT prep script without its demo part, then the LoRA prep script's demo part.
// The live demo shows a recorded run (see below).
(() => {
  // Storage slide: the slider and the model-size buttons can't be used in a static file, so the slide becomes
  // three pages that follow the talk (one task, then 24 tasks, then 24 tasks on ViT-H), and the controls
  // become a plain line saying which case the page shows.
  const tasks = document.getElementById("st-n"), sizes = document.getElementById("st-bb");
  if (tasks && sizes) {
    const ctls = tasks.closest(".st-top > div");
    ctls.querySelectorAll(".ctl").forEach(e => { e.style.display = "none"; });
    const line = document.createElement("p");
    line.style.cssText = "margin:0;font-size:20px;line-height:1.5;color:var(--ink-2)";
    ctls.prepend(line);
    const NAME = { B: "ViT-B, 86 million weights", L: "ViT-L, 303 million weights", H: "ViT-H, 632 million weights" };
    const show = (n, b) => {
      tasks.value = n; tasks.dispatchEvent(new Event("input"));
      sizes.querySelector(`button[data-b="${b}"]`).click();
      line.innerHTML = `Shown for <b>${n} task${n > 1 ? "s" : ""}</b> on <b>${NAME[b]}</b>.` +
        (n > 1 ? "" : " The next pages show 24 tasks, then the largest model.");
    };
    window.pptxSteps = Object.assign(window.pptxSteps || {}, { "s-storage": [() => show(1, "B"), () => show(24, "B"), () => show(24, "H")] });
    show(24, "B");
  }

  // Parameter slide: the prompt-length buttons go; the sentence beside them says which state is shown.
  const pctl = document.querySelector("#s-calc .pctl");
  if (pctl) {
    pctl.querySelectorAll(".lab, .seg").forEach(e => e.remove());
    const p = pctl.querySelector("p");
    p.style.cssText = "margin:0;max-width:none";
    p.textContent = "Shown for p = 50; at p = 200 the file is 7.7 MB and compute is about 2.1×. " + p.textContent;
  }

  // Live demo slide: the static file opens from disk, where the models cannot be downloaded. It shows one
  // real, recorded run instead (`node scripts/pptx-prep/2026-10-vpt-snapshot-demo.mjs 2026-10-peft-vpt-lora.html`
  // writes the block below), and the live code is kept from redrawing the slide when it is entered.
  // Two pages: the models trained on all training images, then the 1,000-image models. The task tabs, the
  // training-set toggle and the thumbnails can't be clicked in a static file, so they become one plain line.
  const DEMO = /* DEMO_SNAPSHOT */ null /* END_DEMO_SNAPSHOT */;
  const demoSlide = document.getElementById("s-demo");
  if (DEMO && demoSlide) {
    document.getElementById("stage").addEventListener("enter", e => { if (e.target.id === "s-demo") e.stopPropagation(); }, true);
    const RN = { full: "all training images", "1k": "1,000 training images" };
    const show = ({ regime, html }) => {
      demoSlide.querySelector(".demo").outerHTML = html;
      const d = demoSlide.querySelector(".demo");
      const taskName = d.querySelector('#task button[aria-pressed="true"]')?.textContent || "";
      d.querySelector("#task").outerHTML = `<p class="static-line" style="margin:0;font-size:20px;line-height:1.4;color:var(--ink-2);align-self:center">` +
        `${taskName}, one test image, models trained on <b>${RN[regime]}</b>.</p>`;
      d.querySelector(".regime").style.visibility = "hidden";
      d.querySelector("#thumbs")?.remove();
    };
    window.pptxSteps = Object.assign(window.pptxSteps || {}, { "s-demo": DEMO.map(snap => () => show(snap)) });
    show(DEMO[0]);
  }
  for (const st of document.querySelectorAll(".status")) {
    st.innerHTML = "<span>A recorded run. In the HTML version of this deck the four models run live in the browser.</span>" +
      "<span style='flex:1'></span><span>Research demo, not a diagnostic tool</span>";
  }
})();
