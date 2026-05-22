import { cn } from "@proecg/ui/lib/utils";

interface SkeletonProps extends React.HTMLAttributes<HTMLDivElement> {
  className?: string;
}

export function Skeleton({ className, ...props }: SkeletonProps) {
  return (
    <div
      className={cn(
        "rounded-xl bg-apple-border-light apple-animate-shimmer",
        className,
      )}
      {...props}
    />
  );
}
