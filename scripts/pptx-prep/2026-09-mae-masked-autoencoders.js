// Runs inside the deck before it is converted to PowerPoint, where the controls
// cannot work. Sets the interactive slides to a fixed state and removes the controls.
(() => {
  // Slide 3, masking demo: panda, random sampling, 75 percent; controls replaced by a caption.
  const panda = document.querySelector('#imgseg [data-img="panda"]');
  const random = document.querySelector('#stratseg [data-s="random"]');
  const ratio = document.getElementById("ratio");
  random.click();
  panda.click();
  ratio.value = 75;
  ratio.dispatchEvent(new Event("input", { bubbles: true }));
  const rows = ratio.closest(".col").querySelectorAll(".ctlrow");
  const note = document.createElement("p");
  note.className = "cap";
  note.style.marginTop = "18px";
  note.textContent = "Masking ratio 75 percent, patches chosen at random.";
  rows[0].replaceWith(note);
  rows[1].remove();

  // Slide 14, partial fine-tuning: the slider is replaced by the two cases side by side, 0 and 1 block.
  const BL = [
    { n: 0, label: "0 blocks tuned · linear probing, all 24 frozen", mae: 73.5, moco: 77.6 },
    { n: 1, label: "1 block tuned · the last block, in salmon", mae: 81.0, moco: 79.9 },
  ];
  const slider = document.getElementById("blocks");
  const col = slider.closest(".col");
  const wrap = document.createElement("div");
  for (const b of BL) {
    const row = document.createElement("div");
    row.style.marginTop = "14px";
    const diff = b.mae - b.moco;
    row.innerHTML =
      `<div class="ctlk">${b.label}</div>` +
      `<div class="stack" style="height:30px;margin:8px 0 6px 0">${Array.from({ length: 24 }, (_, i) => `<div${i >= 24 - b.n ? ' class="tuned"' : ""}></div>`).join("")}</div>` +
      `<div class="vs" style="margin-top:4px">` +
      `<div class="item mae"><div class="k">MAE</div><div class="v" style="font-size:30px">${b.mae.toFixed(1)}</div></div>` +
      `<div class="item"><div class="k">contrastive (MoCo v3)</div><div class="v" style="font-size:30px">${b.moco.toFixed(1)}</div></div>` +
      `<div class="item"><div class="k">difference</div><div class="v" style="font-size:30px;color:var(${diff >= 0 ? "--teal-deep" : "--salmon-deep"})">${(diff >= 0 ? "+" : "−") + Math.abs(diff).toFixed(1)}</div></div>` +
      `</div>`;
    wrap.appendChild(row);
  }
  col.querySelector(".ticks").remove();
  col.querySelector("#stack").remove();
  col.querySelector(".small").remove();
  col.querySelector(".vs").remove();
  slider.replaceWith(wrap);
})();
