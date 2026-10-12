import fs from "node:fs";
import path from "node:path";

export type Deck = {
  file: string;      // file name inside public/decks
  href: string;      // URL to open the deck
  title: string;     // from <title> in the HTML, or the file name
  date: string | null; // YYYY-MM or YYYY-MM-DD prefix in the file name, if present
  sizeKB: number;
  pptx: { href: string; sizeKB: number } | null; // PowerPoint version with the same name, if present
  pdf: { href: string; sizeKB: number } | null;  // PDF version with the same name, if present
  paper: string | null;                          // the original paper, from <meta name="paper" content="…">
  citation: string | null;                       // short reference such as "Jia et al., ECCV 2022", from <meta name="citation">
  summary: string | null;                        // one or two sentences for the index, from <meta name="summary">
  thumb: string | null;                          // picture of the title slide, written by npm run thumbs
  slides: Slide[];                               // every slide in order, for the viewer's slide list
};

export type Slide = {
  title: string;         // data-title, or the slide's first heading
  thumb: string | null;  // small picture of the slide, written by npm run thumbs
};

const DECKS_DIR = path.join(process.cwd(), "public", "decks");

function titleFrom(html: string, fallback: string): string {
  const m = html.match(/<title>([^<]*)<\/title>/i);
  return m ? m[1].trim() : fallback;
}

function dateFrom(file: string): string | null {
  const m = file.match(/^(\d{4}-\d{2}(?:-\d{2})?)/);
  return m ? m[1] : null;
}

function metaFrom(html: string, name: string): string | null {
  const m = html.match(new RegExp(`<meta\\s+name=["']${name}["']\\s+content=["']([^"']+)["']`, "i"));
  return m ? m[1] : null;
}

function decode(text: string): string {
  return text
    .replace(/<[^>]+>/g, "")
    .replace(/&nbsp;/g, " ")
    .replace(/&#(\d+);/g, (_, n) => String.fromCodePoint(+n))
    .replace(/&#x([0-9a-f]+);/gi, (_, n) => String.fromCodePoint(parseInt(n, 16)))
    .replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&quot;/g, '"').replace(/&#39;|&apos;/g, "'")
    .replace(/&mdash;/g, "—").replace(/&ndash;/g, "–").replace(/&times;/g, "×")
    .replace(/&amp;/g, "&")
    .replace(/\s+/g, " ")
    .trim();
}

// The slides as the deck's own engine sees them: every element with class "slide", in order.
// Thumbnails come from public/decks/thumbs/<deck>/slides.json (npm run thumbs) and are matched
// by slide id, or by position when the deck has no ids and the slide count is unchanged.
function slidesFrom(html: string, file: string): Slide[] {
  const body = html.replace(/<script\b[\s\S]*?<\/script>/gi, "").replace(/<template\b[\s\S]*?<\/template>/gi, "");
  const tags = [...body.matchAll(/<[a-z]+\b[^>]*\bclass=["'](?:[^"']*\s)?slide(?:\s[^"']*)?["'][^>]*>/gi)];
  const base = file.replace(/\.html$/i, "");
  let shots: { id: string | null; file: string }[] = [];
  try {
    shots = JSON.parse(fs.readFileSync(path.join(DECKS_DIR, "thumbs", base, "slides.json"), "utf8"));
  } catch {}
  return tags.map((m, k) => {
    const tag = m[0];
    const id = tag.match(/\sid=["']([^"']+)["']/i)?.[1] ?? null;
    const named = tag.match(/\sdata-title=["']([^"']+)["']/i)?.[1];
    const rest = body.slice(m.index! + tag.length, tags[k + 1]?.index ?? body.length);
    const heading = rest.match(/<h[1-3]\b[^>]*>([\s\S]*?)<\/h[1-3]>/i)?.[1];
    const title = decode(named ?? heading ?? "") || `Slide ${k + 1}`;
    const shot = id ? shots.find((s) => s.id === id) : shots.length === tags.length ? shots[k] : undefined;
    const thumb = shot ? `/decks/thumbs/${encodeURIComponent(base)}/${encodeURIComponent(shot.file)}` : null;
    return { title, thumb };
  });
}

// a static version next to the deck with the same name: .pptx or .pdf (written by npm run export)
function sibling(file: string, ext: string): Deck["pptx"] {
  const name = file.replace(/\.html$/i, ext);
  const full = path.join(DECKS_DIR, name);
  if (!fs.existsSync(full)) return null;
  return { href: `/decks/${encodeURIComponent(name)}`, sizeKB: Math.round(fs.statSync(full).size / 1024) };
}

export function getDecks(): Deck[] {
  if (!fs.existsSync(DECKS_DIR)) return [];
  return fs
    .readdirSync(DECKS_DIR)
    .filter((f) => f.toLowerCase().endsWith(".html"))
    .map((file) => {
      const full = path.join(DECKS_DIR, file);
      const html = fs.readFileSync(full, "utf8");
      const stat = fs.statSync(full);
      const fallback = file.replace(/\.html$/i, "").replace(/^\d{4}-\d{2}(-\d{2})?[-_ ]*/, "").replace(/[-_]+/g, " ");
      return {
        file,
        href: `/decks/${encodeURIComponent(file)}`,
        title: titleFrom(html, fallback),
        date: dateFrom(file),
        sizeKB: Math.round(stat.size / 1024),
        pptx: sibling(file, ".pptx"),
        pdf: sibling(file, ".pdf"),
        paper: metaFrom(html, "paper"),
        citation: metaFrom(html, "citation"),
        summary: metaFrom(html, "summary"),
        thumb: sibling(file, ".jpg")?.href ?? null,
        slides: slidesFrom(html, file),
      };
    })
    .sort((a, b) => (b.date ?? "").localeCompare(a.date ?? "") || a.title.localeCompare(b.title));
}
