"use client";

export default function Viewer() {
  function fullscreen() {
    const el = document.getElementById("deck") as HTMLIFrameElement | null;
    if (!el) return;
    el.requestFullscreen?.();
    el.focus();
  }
  return (
    <button type="button" className="fs" onClick={fullscreen}>fullscreen</button>
  );
}
