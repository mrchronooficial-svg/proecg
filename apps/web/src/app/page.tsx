import { Navbar } from "@/components/marketing/navbar";
import { Hero } from "@/components/marketing/hero";
import { SocialProof } from "@/components/marketing/social-proof";
import { HowItWorks } from "@/components/marketing/how-it-works";
import { Benefits } from "@/components/marketing/benefits";
import { LaudoExample } from "@/components/marketing/laudo-example";
import { PricingTable } from "@/components/marketing/pricing-table";
import { Compliance } from "@/components/marketing/compliance";
import { Testimonials } from "@/components/marketing/testimonials";
import { FAQ } from "@/components/marketing/faq";
import { CTAFinal } from "@/components/marketing/cta-final";
import { FooterMarketing } from "@/components/marketing/footer-marketing";

export default function Home() {
  return (
    <main className="flex flex-col scroll-smooth">
      <Navbar />
      <Hero />
      <SocialProof />
      <HowItWorks />
      <Benefits />
      <LaudoExample />
      <PricingTable />
      <Compliance />
      <Testimonials />
      <FAQ />
      <CTAFinal />
      <FooterMarketing />
    </main>
  );
}
