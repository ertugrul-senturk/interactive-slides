// Runs inside the Smart Teeth deck before it is converted to PowerPoint and PDF, where the controls cannot work.
// Slides stepped by the arrow key (data-steps) become one page per step on their own and need nothing here.
(() => {
  const line = (txt) => { const p = document.createElement("p"); p.style.cssText = "margin:0 0 14px;font-size:19px;line-height:1.45;color:var(--ink-2)"; p.innerHTML = txt; return p; };

  // Numbering: the "Another view" button draws a new random mouth; the static file keeps the first one.
  const nb = document.getElementById("nb-new");
  if (nb) nb.remove();

  // AI cost: three sliders become one line, and the slide becomes two pages (a small and a large rollout).
  const cctl = document.getElementById("cost-ctl");
  if (cctl && window.stCost) {
    const l = line("");
    cctl.replaceWith(l);
    const show = (f, c, s) => { window.stCost.set(f, c, s); l.innerHTML = `Shown for <b>${f.toLocaleString("en-US")} families</b>, each asking <b>${c} questions</b> and making <b>${s} scans</b> a week.`; };
    window.pptxSteps = Object.assign(window.pptxSteps || {}, {
      "s-cost": [() => show(1000, 5, 2), () => show(20000, 10, 7)],
    });
    show(1000, 5, 2);
  }

  // Tooth words: the slider becomes a line, and the star rule is shown for one help.
  const wctl = document.getElementById("wd-ctl");
  if (wctl && window.stWords) { window.stWords(1); wctl.replaceWith(line("Shown for <b>1 help</b> in a round of six words.")); }

  // Risk: the checkboxes become plain marks, so nothing looks clickable.
  document.querySelectorAll("#rk-ans input[type=checkbox]").forEach((c) => {
    const m = document.createElement("span");
    m.textContent = c.checked ? "✓" : "";
    m.style.cssText = "font-size:18px;font-weight:600;color:var(--ink)";
    c.replaceWith(m);
  });
})();
