// Runs inside the NeRF deck before it is converted to PowerPoint and PDF, where the controls cannot work.
// Slides stepped by the arrow key (tabs with data-steps) become one page per step on their own and need nothing here.
(() => {
  // Storage slide: the three drawings turn by themselves and the detail slider can't be dragged in a static file,
  // so the scene stops at a fixed angle, the slide becomes two pages (medium detail, then very high) and the slider
  // becomes a plain line.
  const sctl = document.getElementById("st-ctl");
  if (sctl && window.nerfStore) {
    const line = document.createElement("p");
    line.style.cssText = "margin:0;font-size:19px;line-height:1.45;color:var(--ink-2)";
    sctl.replaceWith(line);
    window.nerfStore.spin(false); window.nerfStore.turn(0.95, 0.42);
    const ssrc = document.querySelector("#s-store .src");
    if (ssrc) ssrc.innerHTML = ssrc.innerHTML.replace(", drawn live; drag a picture to turn it.", ".");
    const show = (i, txt) => { window.nerfStore.level(i); line.innerHTML = txt; };
    window.pptxSteps = Object.assign(window.pptxSteps || {}, {
      "s-store": [() => show(1, "Shown at <b>medium</b> detail; the next page shows very high."), () => show(3, "Shown at <b>very high</b> detail.")],
    });
    show(1, "Shown at <b>medium</b> detail; the next page shows very high.");
  }

  // Rendering slide: the pixel slider goes; the strip still marks the pixel whose ray is drawn.
  const ray = document.getElementById("vr-ray");
  if (ray) ray.style.visibility = "hidden";

  // Positional encoding slide: the network is trained live in the browser. Train both fits to the end now, so each
  // page shows the finished result (the same seed and the same 600 steps as in the talk).
  if (window.nerfPeFinish) window.nerfPeFinish();
})();

// My capture, live Gaussians: a file opened from disk cannot run the viewer (it fetches the Gaussians and needs the GPU),
// so the slide shows one frame of it, captured from the live view, and says so.
(() => {
  const fig = document.querySelector("#s-live .lv-fig");
  if (!fig) return;
  fig.innerHTML = '<img src="2026-11-nerf/globe/live-still.jpg" alt="One frame of the live view of the Gaussians" style="display:block;width:778px;height:498px">';
  const sp = document.getElementById("lv-speed");
  if (sp) sp.textContent = "Shown live in the talk; here one frame of it. A frame took about 4 ms to draw on my laptop, while NeRF needs 18 seconds for one image.";
})();

// My capture, new views: the video becomes its still frame (keeping the id, so the NeRF / Instant-NGP steps swap the still).
(() => {
  const v = document.getElementById("orb-video");
  if (!v) return;
  const img = document.createElement("img");
  img.src = v.getAttribute("poster");
  img.id = "orb-video";
  img.alt = "One frame of the orbit: color on the left, depth on the right";
  img.style.cssText = "display:block;width:1120px;height:auto;border:1px solid var(--rule);background:#fff";
  v.replaceWith(img);
})();
