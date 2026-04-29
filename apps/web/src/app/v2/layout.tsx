import type { ReactNode } from "react";
import { Inter } from "next/font/google";

const inter = Inter({
  subsets: ["latin"],
  weight: ["300", "400", "500", "600", "700", "800", "900"],
  variable: "--font-v2",
});

export default function V2Layout({ children }: { children: ReactNode }) {
  return (
    <>
      {/* Trava scroll horizontal globalmente enquanto a rota /v2 estiver ativa. */}
      <style>{`html,body{overflow-x:hidden!important;max-width:100vw;width:100%;margin:0}*,*::before,*::after{box-sizing:border-box}`}</style>
      <div className={`${inter.variable} w-full max-w-full overflow-x-hidden font-[family-name:var(--font-v2)]`}>
        {children}
      </div>
    </>
  );
}
