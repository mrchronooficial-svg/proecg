import { Card } from "@proecg/ui/components/card";

interface MeasurementCardProps {
  label: string;
  value: number;
  unit: string;
}

export function MeasurementCard({ label, value, unit }: MeasurementCardProps) {
  return (
    <Card className="glass flex flex-col items-center gap-1 p-4 overflow-hidden">
      <div className="absolute top-0 left-0 right-0 gradient-brand h-1 rounded-full" />
      <span className="text-xs text-muted-foreground mt-1">{label}</span>
      <span className="text-2xl font-bold">
        {value}
        <span className="ml-0.5 text-sm font-normal text-muted-foreground">
          {unit}
        </span>
      </span>
    </Card>
  );
}
