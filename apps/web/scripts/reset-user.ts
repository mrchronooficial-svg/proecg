/**
 * Apaga o usuário rafaelmello97@gmail.com (e cascata: sessions, accounts,
 * verifications, ecgAnalysis, subscription) pra permitir signup limpo.
 * Roda com: tsx scripts/reset-user.ts
 */
import prisma from "@proecg/db";

const EMAIL = "rafaelmello97@gmail.com";

async function main() {
  const user = await prisma.user.findUnique({ where: { email: EMAIL } });
  if (!user) {
    console.log(`Usuário ${EMAIL} não existe. Nada a fazer.`);
    return;
  }
  console.log(`Apagando user ${user.id} (${user.email})...`);

  // Apaga registros dependentes manualmente caso onDelete:Cascade não esteja
  // configurado em todas as relações.
  await prisma.session.deleteMany({ where: { userId: user.id } });
  await prisma.account.deleteMany({ where: { userId: user.id } });
  // EcgAnalysis e Subscription são opcionais — limpa se houver
  try {
    await prisma.ecgAnalysis.deleteMany({ where: { userId: user.id } });
  } catch (e) {
    console.log("  (sem ecgAnalysis ou erro:", (e as Error).message, ")");
  }
  try {
    await prisma.subscription.deleteMany({ where: { userId: user.id } });
  } catch (e) {
    console.log("  (sem subscription ou erro:", (e as Error).message, ")");
  }
  await prisma.user.delete({ where: { id: user.id } });
  console.log("OK — usuário e dependências apagados.");
}

main()
  .catch((e) => {
    console.error(e);
    process.exitCode = 1;
  })
  .finally(async () => {
    await prisma.$disconnect();
  });
