import Link from "next/link";
import { getDecks } from "@/lib/decks";

export const dynamic = "force-static";

export default function Home() {
  const decks = getDecks();
  return (
    <main>
      <h1>Paper slides</h1>
      <p className="lede">Presentations for the reading seminar. Select one to open it. Inside the deck, use the arrow keys; click the deck once first so it has focus.</p>
      {decks.length === 0 ? (
        <p className="empty">No decks yet. Drop an .html file into <code>public/decks</code>.</p>
      ) : (
        <ul className="list">
          {decks.map((d) => (
            <li key={d.file}>
              <Link className="row" href={`/view/${encodeURIComponent(d.file)}`}>
                <span className="date">{d.date ?? ""}</span>
                <span>
                  <div className="title">{d.title}</div>
                  <div className="meta">{d.file} · {d.sizeKB} KB</div>
                </span>
                <span className="open">open →</span>
              </Link>
            </li>
          ))}
        </ul>
      )}
      <p className="how">
        To add a deck, put its .html file in <code>public/decks/</code> and push. Prefix the file name with a date
        such as <code>2026-09-</code> to sort it, and the page title comes from the file&apos;s <code>&lt;title&gt;</code>.
      </p>
    </main>
  );
}
