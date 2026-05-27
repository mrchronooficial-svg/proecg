import type { ReactNode } from "react";

/** Campo com label flutuante dentro de um container arredondado escuro
 *  (padrão das telas de cadastro/checkout — referência ChatGPT). */
export function FloatingField({
  label,
  trailing,
  children,
}: {
  label: string;
  trailing?: ReactNode;
  children: ReactNode;
}) {
  return (
    <div className="rounded-2xl border border-white/12 bg-white/[0.04] px-5 py-2.5 transition focus-within:border-[#0a84ff] focus-within:ring-2 focus-within:ring-[#0a84ff]/25">
      <div className="flex items-center justify-between gap-2">
        <span className="text-[12px] font-medium text-white/45">{label}</span>
        {trailing}
      </div>
      {children}
    </div>
  );
}

export const bareInput =
  "w-full bg-transparent text-[15px] text-white placeholder-white/40 outline-none";
