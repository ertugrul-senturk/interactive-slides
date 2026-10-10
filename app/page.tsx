import Link from "next/link";
import { getDecks } from "@/lib/decks";

export const dynamic = "force-static";

export default function Home() {
  const decks = getDecks();
  return (
    <main>
      <h1>Paper slides</h1>
      <p className="lede">Presentations for the reading seminar. Select one to open it. Inside the deck, change slides with the arrow keys; clicking is for the interactive parts. Each deck can also be downloaded as a PowerPoint file.</p>
      {decks.length === 0 ? (
        <p className="empty">No decks yet. Drop an .html file into <code>public/decks</code>.</p>
      ) : (
        <ul className="list">
          {decks.map((d) => (
            <li key={d.file} className="row">
              <span className="date">{d.date ?? ""}</span>
              <Link href={`/view/${encodeURIComponent(d.file)}`}>
                <div className="title">{d.title}</div>
                <div className="meta">{d.file} · {d.sizeKB} KB</div>
              </Link>
              <span className="actions">
                {d.pptx && (
                  <a className="dl" href={d.pptx.href} download title={`PowerPoint, ${d.pptx.sizeKB} KB`}>
                    Download .pptx
                  </a>
                )}
                <Link className="open" href={`/view/${encodeURIComponent(d.file)}`}>Open</Link>
              </span>
            </li>
          ))}
        </ul>
      )}
      <p className="how">
        To add a deck, put its .html file in <code>public/decks/</code> and push. Prefix the file name with a date
        such as <code>2026-09-</code> to sort it, and the page title comes from the file&apos;s <code>&lt;title&gt;</code>. Run <code>npm run pptx</code> to
        create the PowerPoint version next to it.
      </p>
    </main>
  );
}
