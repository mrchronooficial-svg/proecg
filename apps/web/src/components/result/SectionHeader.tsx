interface SectionHeaderProps {
  title: string;
}

export function SectionHeader({ title }: SectionHeaderProps) {
  return (
    <h2 className="mb-2 text-[12px] font-semibold uppercase tracking-[0.06em] text-apple-accent">
      {title}
    </h2>
  );
}
