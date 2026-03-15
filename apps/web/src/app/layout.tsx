import type { Metadata } from "next";
import { Inter } from "next/font/google";

import "../index.css";
import Header from "@/components/header";
import Providers from "@/components/providers";

const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "ProECG — Laudo de ECG com IA em Segundos",
  description:
    "Tire uma foto do ECG de papel e receba um laudo descritivo completo com hipóteses diagnósticas em segundos. Validado por cardiologistas.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="pt-BR" suppressHydrationWarning className="scroll-smooth">
      <body className={`${inter.variable} antialiased bg-gradient-to-br from-background to-secondary`}>
        <Providers>
          <div className="grid grid-rows-[auto_1fr] min-h-svh">
            <Header />
            {children}
          </div>
        </Providers>
      </body>
    </html>
  );
}
