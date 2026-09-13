import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Paper slides",
  description: "Slide decks for paper presentations",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
