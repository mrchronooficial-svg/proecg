import { AlertCircle } from "lucide-react";

export function DisclaimerFooter() {
  return (
    <footer className="mt-12 flex items-center justify-center gap-1.5 px-4 pb-4 text-center text-[13px] leading-[1.3] text-apple-text-tertiary">
      <AlertCircle size={14} aria-hidden="true" className="shrink-0" />
      <span>
        Ferramenta de apoio à decisão clínica — não substitui avaliação médica.
      </span>
    </footer>
  );
}
