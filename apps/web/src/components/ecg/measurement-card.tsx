import { Card } from "@proecg/ui/components/card";

interface MeasurementCardProps {
  label: string;
  value: number | null;
  unit: string;
}

export function MeasurementCard({ label, value, unit }: MeasurementCardProps) {
  return (
    <Card className="relative glass flex flex-col items-center gap-1 p-4 overflow-hidden">
      <div className="absolute top-0 left-0 right-0 bg-primary h-1 rounded-full" />
      <span className="text-xs text-muted-foreground mt-1">{label}</span>
      <span className="text-2xl font-bold">
        {value != null ? value : "—"}
        <span className="ml-0.5 text-sm font-normal text-muted-foreground">
          {unit}
        </span>
      </span>
    </Card>
  );
}
