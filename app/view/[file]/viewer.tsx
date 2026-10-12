"use client";

import { useEffect, useRef, useState, type ReactNode } from "react";
import type { Slide } from "@/lib/decks";

const OPEN_KEY = "viewer.slides";

// The page around the deck: the bar (links passed in as children), a PowerPoint-style slide list
// on the left that can be shown or hidden, and the deck itself. Fullscreen shows only the deck.
export default function Viewer({ href, title, slides, children }: { href: string; title: string; slides: Slide[]; children: ReactNode }) {
  const frame = useRef<HTMLIFrameElement>(null);
  const list = useRef<HTMLOListElement>(null);
  const [open, setOpen] = useState(true);
  const [cur, setCur] = useState(0);

  // remember whether the list was open; on a narrow screen it starts hidden
  useEffect(() => {
    let saved: string | null = null;
    try { saved = localStorage.getItem(OPEN_KEY); } catch {}
    setOpen(saved ? saved === "1" : innerWidth > 900);
  }, []);

  // Give the deck keyboard focus right away, so the arrow keys work without a first click,
  // and follow the deck's current slide by watching which slide has the class "active".
  useEffect(() => {
    const el = frame.current;
    if (!el) return;
    let watch: MutationObserver | null = null;
    const attach = () => {
      el.contentWindow?.focus();
      watch?.disconnect();
      const doc = el.contentDocument;
      if (!doc) return;
      const all = [...doc.querySelectorAll(".slide")];
      const sync = () => { const k = all.findIndex((s) => s.classList.contains("active")); if (k >= 0) setCur(k); };
      watch = new MutationObserver(sync);
      all.forEach((s) => watch!.observe(s, { attributes: true, attributeFilter: ["class"] }));
      sync();
    };
    if (el.contentDocument?.readyState === "complete") attach();
    el.addEventListener("load", attach);
    return () => { el.removeEventListener("load", attach); watch?.disconnect(); };
  }, []);

  // keep the current slide in view in the list, without moving the page
  useEffect(() => {
    const box = list.current;
    const item = box?.children[cur] as HTMLElement | undefined;
    if (!open || !box || !item) return;
    const top = item.offsetTop - box.offsetTop, bottom = top + item.offsetHeight;
    if (top < box.scrollTop + 8) box.scrollTop = top - 8;
    else if (bottom > box.scrollTop + box.clientHeight - 8) box.scrollTop = bottom - box.clientHeight + 8;
  }, [cur, open]);

  function toggle() {
    const next = !open;
    setOpen(next);
    try { localStorage.setItem(OPEN_KEY, next ? "1" : "0"); } catch {}
    frame.current?.contentWindow?.focus();
  }

  // the decks read the slide number from the address (#3) and follow it when it changes
  function go(k: number) {
    const win = frame.current?.contentWindow;
    if (!win) return;
    setCur(k);
    // an absolute address: a relative one would resolve against this page, not the deck
    const url = new URL(win.location.href);
    url.hash = String(k + 1);
    win.location.replace(url.href);
    win.focus();
  }

  function fullscreen() {
    const el = frame.current;
    if (!el) return;
    el.requestFullscreen?.();
    el.contentWindow?.focus();
  }

  return (
    <div className="viewer">
      <header className="bar">
        {children}
        <span className="barbtns">
          <button type="button" className="fs" aria-pressed={open} aria-controls="slidelist" onClick={toggle}
            title={open ? "Hide the slide list" : "Show the slide list"}>Slides</button>
          <button type="button" className="fs" onClick={fullscreen}>Fullscreen</button>
        </span>
      </header>
      <div className="viewbody">
        <ol id="slidelist" ref={list} className="slidelist" hidden={!open} aria-label="Slides">
          {slides.map((s, k) => (
            <li key={k}>
              <button type="button" className="thumb" aria-current={k === cur ? "true" : undefined} onClick={() => go(k)} title={s.title}>
                <span className="thumbnum">{k + 1}</span>
                <span className="thumbpic">
                  {s.thumb
                    ? <img src={s.thumb} alt={s.title} width={384} height={216} loading="lazy" decoding="async" />
                    : <span className="thumbtext">{s.title}</span>}
                </span>
              </button>
            </li>
          ))}
        </ol>
        <iframe id="deck" ref={frame} className="frame" src={href} title={title} allowFullScreen />
      </div>
    </div>
  );
}
