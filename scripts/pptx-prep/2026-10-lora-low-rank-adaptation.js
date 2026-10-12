// Runs inside the deck before it is converted to PowerPoint, where the controls cannot work.
// The live demo shows a recorded run (see below).
(() => {
  // Live demo slide: the static file opens from disk, where the models cannot be downloaded. It shows one
  // real, recorded run instead (`node scripts/pptx-prep/2026-10-vpt-snapshot-demo.mjs 2026-10-lora-low-rank-adaptation.html`
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
    st.innerHTML = "<span>A recorded run. In the HTML version of this deck the three models run live in the browser.</span>" +
      "<span style='flex:1'></span><span>Research demo, not a diagnostic tool</span>";
  }
})();
