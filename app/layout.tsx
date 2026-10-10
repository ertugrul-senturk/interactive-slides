import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Paper slides",
  description: "Slide decks for paper presentations",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="" />
        <link
          rel="stylesheet"
          href="https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&family=Source+Sans+3:wght@400;600&family=Source+Code+Pro:wght@400&display=swap"
        />
      </head>
      <body>{children}</body>
    </html>
  );
}
