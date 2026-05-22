import { AlertTriangle } from "lucide-react";

export function DisclaimerBanner() {
  return (
    <div
      role="note"
      className="flex items-center gap-2 rounded-xl bg-[#FFF8E1] px-4 py-3 text-[13px] font-medium leading-[1.3] text-[#92600A]"
    >
      <AlertTriangle size={16} strokeWidth={2} aria-hidden="true" />
      <span>
        Ferramenta de apoio à decisão clínica — não substitui avaliação médica.
      </span>
    </div>
  );
}
