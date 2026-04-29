import { Navbar } from "@/components/marketing/v2/navbar";
import { Hero } from "@/components/marketing/v2/hero";
import { Logos } from "@/components/marketing/v2/logos";
import { HowItWorks } from "@/components/marketing/v2/how-it-works";
import { Benefits } from "@/components/marketing/v2/benefits";
import { Diagnostics } from "@/components/marketing/v2/diagnostics";
import { Impact } from "@/components/marketing/v2/impact";
import { Security } from "@/components/marketing/v2/security";
import { Pricing } from "@/components/marketing/v2/pricing";
import { CTAFinal } from "@/components/marketing/v2/cta-final";
import { FooterMarketing } from "@/components/marketing/v2/footer";

export default function LandingV2() {
  return (
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
  );
}
