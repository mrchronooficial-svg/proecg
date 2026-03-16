import { auth } from "@proecg/auth";
// import prisma from "@proecg/db"; // TODO: reativar com verificação de plano
import { headers } from "next/headers";
import { redirect } from "next/navigation";

export async function requireSubscription() {
  const session = await auth.api.getSession({
    headers: await headers(),
  });

  if (!session?.user) {
    redirect("/login");
  }

  // TODO: reativar verificação de plano quando pagamento estiver configurado
  // const subscription = await prisma.subscription.findFirst({
  //   where: {
  //     userId: session.user.id,
  //     status: "ACTIVE",
  //   },
  //   orderBy: { createdAt: "desc" },
  // });
  //
  // if (!subscription) {
  //   redirect("/assinar");
  // }

  return { session, subscription: null };
}
