import type { ReactNode } from "react";

// Gradiente navy idêntico ao hero da landing (apps/web/src/components/marketing/hero.tsx)
const BG_GRADIENT =
  "radial-gradient(60% 80% at 70% 20%, rgba(26,26,62,0.9), transparent 60%), radial-gradient(50% 60% at 20% 80%, rgba(0,102,204,0.25), transparent 60%), linear-gradient(180deg, #050510 0%, #0d0d20 60%, #1a1a3e 100%)";

export function CheckoutShell({ children }: { children: ReactNode }) {
  return (
    <div
      className="relative min-h-svh w-full overflow-hidden bg-[#0a0a0a] text-white"
      style={{ backgroundImage: BG_GRADIENT }}
    >
      {children}
    </div>
  );
}
