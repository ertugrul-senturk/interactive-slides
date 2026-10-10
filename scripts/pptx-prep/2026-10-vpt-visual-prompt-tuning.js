// Runs inside the deck before it is converted to PowerPoint, where the controls
// cannot work. The VPT figure (shallow / deep) and the attention slide (without / with prompts)
// are stepped by the arrow key like a build, so they need nothing here. The other interactive slides (result tabs) render a
// sensible default state. The storage slider is set to 24 tasks. The parameter slide shows p = 50.
(() => {
  // Storage slide: 24 tasks shows the cost much better than the live default of one task.
  const tasks = document.getElementById("st-n");
  if (tasks) { tasks.value = 24; tasks.dispatchEvent(new Event("input")); }

  // Parameter slide: the prompt-length buttons go; the sentence beside them says which state is shown.
  const pctl = document.querySelector("#s-calc .pctl");
  if (pctl) {
    pctl.querySelectorAll(".lab, .seg").forEach(e => e.remove());
    const p = pctl.querySelector("p");
    p.style.cssText = "margin:0;max-width:none";
    p.textContent = "Shown for p = 50; at p = 200 the file is 7.7 MB and compute is about 2.1×. " + p.textContent;
  }

  // Live demo slide: the models run in the browser, so the engine controls become a pointer to the live deck.
  for (const st of document.querySelectorAll(".status")) {
    st.innerHTML = "<span>Live demo: open the HTML version of this deck to run the models. Research demo, not a diagnostic tool.</span>";
  }
})();
