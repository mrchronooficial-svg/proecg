import { Hero } from "@/components/marketing/hero";
import { HowItWorks } from "@/components/marketing/how-it-works";
import { PricingTable } from "@/components/marketing/pricing-table";
import { FooterMarketing } from "@/components/marketing/footer-marketing";

export default function Home() {
  return (
    <main className="flex flex-col">
      <Hero />
      <HowItWorks />
      <PricingTable />
      <FooterMarketing />
    </main>
  );
}
