interface ReportCardProps {
  text: string;
}

export function ReportCard({ text }: ReportCardProps) {
  return (
    <article className="rounded-2xl bg-apple-surface p-5 shadow-apple-sm">
      <p className="whitespace-pre-line text-[16px] leading-[1.6] text-apple-text">
        {text}
      </p>
      <hr className="my-4 border-apple-border-light" />
      <p className="text-[13px] italic leading-[1.4] text-apple-text-tertiary">
        Ferramenta de apoio à decisão clínica — não substitui avaliação médica.
      </p>
    </article>
  );
}
