import { auth } from "@proecg/auth";
import prisma from "@proecg/db";
import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";
import { z } from "zod";

import {
  markPasswordResetTokenUsed,
  validatePasswordResetToken,
} from "@/lib/password-token";

const bodySchema = z.object({
  token: z.string().min(1),
  newPassword: z.string().min(8, "A senha deve ter pelo menos 8 caracteres"),
});

export async function POST(request: NextRequest) {
  let parsed;
  try {
    parsed = bodySchema.parse(await request.json());
  } catch {
    return NextResponse.json(
      { error: "Dados inválidos. Verifique a senha e tente novamente." },
      { status: 400 },
    );
  }

  const { token, newPassword } = parsed;

  const { valid, userId } = await validatePasswordResetToken(token);
  if (!valid || !userId) {
    return NextResponse.json(
      { error: "Link inválido ou expirado. Solicite um novo." },
      { status: 400 },
    );
  }

  try {
    const ctx = await auth.$context;
    const hashed = await ctx.password.hash(newPassword);

    const credential = await prisma.account.findFirst({
      where: { userId, providerId: "credential" },
    });
    if (credential) {
      await prisma.account.update({
        where: { id: credential.id },
        data: { password: hashed },
      });
    } else {
      await prisma.account.create({
        data: {
          id: crypto.randomUUID(),
          userId,
          providerId: "credential",
          accountId: userId,
          password: hashed,
        },
      });
    }

    await markPasswordResetTokenUsed(token);
  } catch (err) {
    console.error("[reset-password] erro ao atualizar senha:", err);
    return NextResponse.json(
      { error: "Não foi possível atualizar a senha. Tente novamente." },
      { status: 500 },
    );
  }

  return NextResponse.json({ ok: true });
}
