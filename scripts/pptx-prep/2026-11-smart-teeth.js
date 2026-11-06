// Runs inside the Smart Teeth deck before it is converted to PowerPoint and PDF, where the controls cannot work.
// Slides stepped by the arrow key (data-steps) become one page per step on their own and need nothing here.
(() => {
  // Numbering: the "Another view" button draws a new random mouth; the static file keeps the first one.
  const nb = document.getElementById("nb-new");
  if (nb) nb.remove();
})();
