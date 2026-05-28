import prisma from "@proecg/db";

const TOKEN_TTL_HOURS = 24;

export async function generatePasswordResetToken(
  userId: string,
): Promise<string> {
  const token = crypto.randomUUID();
  const expiresAt = new Date(Date.now() + TOKEN_TTL_HOURS * 60 * 60 * 1000);
  await prisma.passwordResetToken.create({
    data: { userId, token, expiresAt },
  });
  return token;
}

export async function validatePasswordResetToken(
  token: string,
): Promise<{ valid: boolean; userId: string | null }> {
  if (!token) return { valid: false, userId: null };
  const record = await prisma.passwordResetToken.findUnique({
    where: { token },
  });
  if (!record) return { valid: false, userId: null };
  if (record.used) return { valid: false, userId: null };
  if (record.expiresAt.getTime() < Date.now()) {
    return { valid: false, userId: null };
  }
  return { valid: true, userId: record.userId };
}

export async function markPasswordResetTokenUsed(token: string): Promise<void> {
  await prisma.passwordResetToken.updateMany({
    where: { token, used: false },
    data: { used: true },
  });
}
