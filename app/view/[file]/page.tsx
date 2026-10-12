import Link from "next/link";
import { notFound } from "next/navigation";
import { getDecks } from "@/lib/decks";
import Viewer from "./viewer";

const mb = (kb: number) => `${(kb / 1024).toFixed(1)} MB`;

export const dynamic = "force-static";
export const dynamicParams = false;

export function generateStaticParams() {
  return getDecks().map((d) => ({ file: d.file }));
}

export async function generateMetadata({ params }: { params: Promise<{ file: string }> }) {
  const { file } = await params;
  const deck = getDecks().find((d) => d.file === decodeURIComponent(file));
  return { title: deck ? deck.title : "Deck" };
}

export default async function ViewPage({ params }: { params: Promise<{ file: string }> }) {
  const { file } = await params;
  const name = decodeURIComponent(file);
  const deck = getDecks().find((d) => d.file === name);
  if (!deck) notFound();
  return (
    <Viewer href={deck.href} title={deck.title} slides={deck.slides}>
      <Link href="/" className="home" title="All talks">Paper slides</Link>
      <span className="barsep" aria-hidden="true" />
      <span className="bartitle">
        {deck.title}
        {deck.citation && <span className="barcite">{deck.citation}</span>}
      </span>
      <nav className="baractions">
        {deck.paper && <a href={deck.paper} target="_blank" rel="noopener" title="The original paper (PDF)">Paper</a>}
        {deck.pdf && <a className="dl" href={deck.pdf.href} download title={`Static version as PDF, ${mb(deck.pdf.sizeKB)}`}>PDF</a>}
        {deck.pptx && <a className="dl" href={deck.pptx.href} download title={`Static version as PowerPoint, ${mb(deck.pptx.sizeKB)}`}>PowerPoint</a>}
      </nav>
    </Viewer>
  );
}
