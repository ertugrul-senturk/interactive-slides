import Link from "next/link";
import { notFound } from "next/navigation";
import { getDecks } from "@/lib/decks";
import Viewer from "./viewer";

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
    <div className="viewer">
      <header className="bar">
        <Link href="/" className="back">← all slides</Link>
        <span className="bartitle">{deck.title}</span>
        <span className="baractions">
          <a href={deck.href} target="_blank" rel="noopener">open file</a>
          <Viewer />
        </span>
      </header>
      <iframe id="deck" className="frame" src={deck.href} title={deck.title} allowFullScreen />
    </div>
  );
}
