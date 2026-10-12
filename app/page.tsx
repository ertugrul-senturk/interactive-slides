import Link from "next/link";
import { getDecks, type Deck } from "@/lib/decks";

export const dynamic = "force-static";

const mb = (kb: number) => `${(kb / 1024).toFixed(1)} MB`;

// "2026-10" → "October 2026"; decks without a date go under "Undated"
function monthOf(date: string | null): string {
  if (!date) return "Undated";
  const [y, m] = date.split("-").map(Number);
  return new Date(Date.UTC(y, m - 1, 1)).toLocaleString("en-US", { month: "long", year: "numeric", timeZone: "UTC" });
}

function byMonth(decks: Deck[]): [string, Deck[]][] {
  const groups = new Map<string, Deck[]>();
  for (const d of decks) {
    const key = monthOf(d.date);
    groups.set(key, [...(groups.get(key) ?? []), d]);
  }
  return [...groups];
}

export default function Home() {
  const decks = getDecks();
  return (
    <main className="index">
      <header className="masthead">
        <h1>Paper slides</h1>
        <p className="lede">
          Slides for the papers I present in our reading group at the University of Rochester.
        </p>
        <p className="byline">Ertugrul Senturk</p>
      </header>

      {decks.length === 0 ? (
        <p className="empty">No talks yet.</p>
      ) : (
        byMonth(decks).map(([month, group]) => (
          <section key={month} className="month">
            <h2>{month}</h2>
            {group.map((d) => {
              const view = `/view/${encodeURIComponent(d.file)}`;
              return (
                <article key={d.file} className="talk">
                  <Link href={view} className="cover" tabIndex={-1} aria-hidden="true">
                    {d.thumb ? <img src={d.thumb} alt="" loading="lazy" /> : <span className="cover-blank">{d.title}</span>}
                  </Link>
                  <div className="text">
                    <h3><Link href={view}>{d.title}</Link></h3>
                    {d.citation && <p className="cite">{d.citation}</p>}
                    {d.summary && <p className="summary">{d.summary}</p>}
                    <p className="links">
                      <Link className="open" href={view}>Open the slides</Link>
                      {d.paper && <a href={d.paper} target="_blank" rel="noopener" title="The original paper (PDF)">Paper</a>}
                      {d.pdf && <a href={d.pdf.href} download title={`Static version as PDF, ${mb(d.pdf.sizeKB)}`}>PDF</a>}
                      {d.pptx && <a href={d.pptx.href} download title={`Static version as PowerPoint, ${mb(d.pptx.sizeKB)}`}>PowerPoint</a>}
                    </p>
                  </div>
                </article>
              );
            })}
          </section>
        ))
      )}

      <footer className="colophon">
        Inside a talk, the arrow keys move between slides. The PDF and PowerPoint versions give every step of an
        interactive slide its own page.
      </footer>
    </main>
  );
}
