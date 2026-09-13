import fs from "node:fs";
import path from "node:path";

export type Deck = {
  file: string;      // file name inside public/decks
  href: string;      // URL to open the deck
  title: string;     // from <title> in the HTML, or the file name
  date: string | null; // YYYY-MM or YYYY-MM-DD prefix in the file name, if present
  sizeKB: number;
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
      };
    })
    .sort((a, b) => (b.date ?? "").localeCompare(a.date ?? "") || a.title.localeCompare(b.title));
}
