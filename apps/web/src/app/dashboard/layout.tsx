import { DashboardShell } from "@/components/dashboard/DashboardShell";
import { requireSubscription } from "@/lib/require-subscription";

export default async function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const { session } = await requireSubscription();

  return (
    <DashboardShell
      userName={session.user.name}
      userEmail={session.user.email}
    >
      {children}
    </DashboardShell>
  );
}
