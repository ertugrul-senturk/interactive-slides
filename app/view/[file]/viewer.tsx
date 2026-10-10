"use client";

import { useEffect } from "react";

function deckFrame() {
  return document.getElementById("deck") as HTMLIFrameElement | null;
}

export default function Viewer() {
  // Give the deck keyboard focus right away, so the arrow keys work without a first click.
  useEffect(() => {
    const el = deckFrame();
    if (!el) return;
    const focus = () => el.contentWindow?.focus();
    focus();
    el.addEventListener("load", focus);
    return () => el.removeEventListener("load", focus);
  }, []);

  function fullscreen() {
    const el = deckFrame();
    if (!el) return;
    el.requestFullscreen?.();
    el.contentWindow?.focus();
  }
  return (
    <button type="button" className="fs" onClick={fullscreen}>Fullscreen</button>
  );
}
