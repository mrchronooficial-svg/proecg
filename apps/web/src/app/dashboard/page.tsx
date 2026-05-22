import { requireSubscription } from "@/lib/require-subscription";

import DashboardHome from "./dashboard";

export default async function DashboardPage() {
  const { session } = await requireSubscription();

  return (
    <DashboardHome
      userName={session.user.name}
    />
  );
}
