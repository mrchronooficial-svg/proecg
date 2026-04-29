import type { ReactNode } from "react";
import { Inter } from "next/font/google";

const inter = Inter({
  subsets: ["latin"],
  weight: ["300", "400", "500", "600", "700", "800", "900"],
  variable: "--font-v2",
});

export default function V2Layout({ children }: { children: ReactNode }) {
  return <div className={`${inter.variable} font-[family-name:var(--font-v2)]`}>{children}</div>;
}
