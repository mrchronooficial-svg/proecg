import { BottomNav } from "./BottomNav";
import { DashboardSidebar } from "./DashboardSidebar";
import { DisclaimerFooter } from "./DisclaimerFooter";
import { MobileHeader } from "./MobileHeader";

interface DashboardShellProps {
  children: React.ReactNode;
  userName: string;
  userEmail: string;
}

export function DashboardShell({
  children,
  userName,
  userEmail,
}: DashboardShellProps) {
  return (
    <div
      className="min-h-svh bg-apple-bg text-apple-text"
      style={{ fontFamily: "var(--font-geist-sans), -apple-system, BlinkMacSystemFont, sans-serif" }}
    >
      <MobileHeader userName={userName} userEmail={userEmail} />
      <DashboardSidebar />

      <main className="md:pl-[260px]">
        <div className="mx-auto flex min-h-[calc(100svh-3.5rem)] max-w-3xl flex-col px-4 pt-6 pb-28 md:min-h-svh md:px-8 md:pt-10 md:pb-10">
          <div className="flex-1">{children}</div>
          <DisclaimerFooter />
        </div>
      </main>

      <BottomNav />
    </div>
  );
}
