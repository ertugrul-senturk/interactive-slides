// Runs inside the NeRF deck before it is converted to PowerPoint and PDF, where the controls cannot work.
// Slides stepped by the arrow key (tabs with data-steps) become one page per step on their own and need nothing here.
(() => {
  // Grid slide: the slider can't be dragged in a static file, so the slide becomes three pages that follow the notes
  // (32 cubes per side, then 128 as in Neural Volumes, then 1,024), and the slider becomes a plain line.
  const ctl = document.getElementById("grid-ctl");
  if (ctl && window.nerfGrid) {
    const NS = [8, 16, 32, 128, 512, 1024];
    const line = document.createElement("p");
    line.style.cssText = "margin:14px 0 0;font-size:19px;line-height:1.45;color:var(--ink-2)";
    ctl.replaceWith(line);
    const show = (i, extra) => { window.nerfGrid(i); line.innerHTML = `Shown for <b>${NS[i].toLocaleString("en-US")} cubes per side</b>.${extra}`; };
    window.pptxSteps = Object.assign(window.pptxSteps || {}, {
      "s-grid": [() => show(2, " The next pages show 128 and 1,024."), () => show(3, " Neural Volumes uses this grid."), () => show(5, "")],
    });
    show(2, " The next pages show 128 and 1,024.");
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

// My capture, new views: the video becomes its still frame.
(() => {
  const v = document.getElementById("orb-video");
  if (!v) return;
  const img = document.createElement("img");
  img.src = v.getAttribute("poster");
  img.alt = "One frame of the NeRF orbit: colour on the left, depth on the right";
  img.style.cssText = "display:block;width:1120px;height:auto;border:1px solid var(--rule);background:#fff";
  v.replaceWith(img);
})();
