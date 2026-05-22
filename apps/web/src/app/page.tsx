import { Inter } from "next/font/google";

import { Navbar } from "@/components/marketing/navbar";
import { Hero } from "@/components/marketing/hero";
import { Logos } from "@/components/marketing/logos";
import { HowItWorks } from "@/components/marketing/how-it-works";
import { Benefits } from "@/components/marketing/benefits";
import { Diagnostics } from "@/components/marketing/diagnostics";
import { Impact } from "@/components/marketing/impact";
import { Security } from "@/components/marketing/security";
import { Pricing } from "@/components/marketing/pricing";
import { CTAFinal } from "@/components/marketing/cta-final";
import { FooterMarketing } from "@/components/marketing/footer";

const inter = Inter({
  subsets: ["latin"],
  weight: ["300", "400", "500", "600", "700", "800", "900"],
  variable: "--font-marketing",
});

export default function Home() {
  return (
    <>
      {/* Trava scroll horizontal globalmente na rota /. */}
      <style>{`html,body{overflow-x:hidden!important;max-width:100vw;width:100%;margin:0}*,*::before,*::after{box-sizing:border-box}`}</style>
      <div
        className={`${inter.variable} w-full max-w-full overflow-x-hidden font-[family-name:var(--font-marketing)]`}
      >
        <main className="overflow-x-hidden bg-white text-[#1d1d1f] antialiased">
          <Navbar />
          <Hero />
          <Logos />
          <HowItWorks />
          <Benefits />
          <Diagnostics />
          <Impact />
          <Security />
          <Pricing />
          <CTAFinal />
          <FooterMarketing />
        </main>
      </div>
    </>
  );
}
