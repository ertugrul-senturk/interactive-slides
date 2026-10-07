// Runs inside the deck before it is converted to PowerPoint, where the controls
// cannot work. The VPT figure (slide 6) is stepped by the arrow key like a build,
// so it needs nothing here. The calculator keeps its default state (ViT-B, deep, p = 50).
(() => {
  // Calculator: the sliders and buttons become a one-line description of the state shown.
  const calc = document.querySelector("#s-calc .calc > div");
  const note = document.createElement("p");
  note.className = "sub";
  note.textContent = "Shown for ViT-B/16, VPT-deep, p = 50 prompts, 102 classes, 24 tasks. The live version has sliders.";
  calc.replaceChildren(note);

  // Live demo slides: they need the inference server, so the controls become a pointer to the live deck.
  for (const st of document.querySelectorAll(".status")) {
    st.innerHTML = "<span>Live demo: open the HTML version of this deck to run the model. Research demo, not a diagnostic tool.</span>";
  }
  document.getElementById("reshuffle")?.remove();
})();
