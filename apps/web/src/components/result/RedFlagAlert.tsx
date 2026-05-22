import { AlertTriangle } from "lucide-react";
import type { RedFlag } from "@/types/result";

interface RedFlagAlertProps {
  flag: RedFlag;
}

export function RedFlagAlert({ flag }: RedFlagAlertProps) {
  const danger = flag.type === "danger";
  const colors = danger
    ? {
        bg: "bg-[#FEF2F2]",
        border: "border-[#FF3B30]",
        title: "text-[#DC2626]",
        icon: "text-[#FF3B30]",
      }
    : {
        bg: "bg-[#FFFBEB]",
        border: "border-[#F59E0B]",
        title: "text-[#B45309]",
        icon: "text-[#F59E0B]",
      };

  return (
    <div
      role="alert"
      className={`rounded-2xl ${colors.bg} border-l-4 ${colors.border} p-4`}
    >
      <div className="flex items-start gap-3">
        <AlertTriangle
          size={24}
          strokeWidth={2}
          className={`shrink-0 ${colors.icon}`}
          aria-hidden="true"
        />
        <div className="min-w-0 flex-1">
          <p
            className={`text-[13px] font-bold uppercase tracking-[0.06em] ${colors.title}`}
          >
            {flag.title}
          </p>
          <p className="mt-1 text-[17px] font-semibold leading-[1.3] text-apple-text">
            {flag.message}
          </p>
          {flag.suggestion && (
            <p className="mt-1.5 text-[14px] leading-[1.4] text-apple-text-secondary">
              {flag.suggestion}
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
