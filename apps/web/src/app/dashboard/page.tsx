import { requireSubscription } from "@/lib/require-subscription";

import Dashboard from "./dashboard";

export default async function DashboardPage() {
  const { session } = await requireSubscription();

  return <Dashboard session={session} />;
}
