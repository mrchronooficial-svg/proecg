import prisma from "@proecg/db";
import { env } from "@proecg/env/server";
import { betterAuth } from "better-auth";
import { prismaAdapter } from "better-auth/adapters/prisma";
import { nextCookies } from "better-auth/next-js";
import { Resend } from "resend";

const resend = env.RESEND_API_KEY ? new Resend(env.RESEND_API_KEY) : null;

async function sendEmail(to: string, subject: string, html: string) {
  if (!resend) {
    console.log(`[DEV] Email to ${to}: ${subject}`);
    return;
  }
  await resend.emails.send({
    from: "ProECG <noreply@proecg.com.br>",
    to,
    subject,
    html,
  });
}

export const auth = betterAuth({
  database: prismaAdapter(prisma, {
    provider: "postgresql",
  }),

  trustedOrigins: [env.CORS_ORIGIN],
  emailAndPassword: {
    enabled: true,
    sendResetPassword: async ({ user, url }) => {
      await sendEmail(
        user.email,
        "Redefinir senha — ProECG",
        `<p>Olá ${user.name},</p><p>Clique no link abaixo para redefinir sua senha:</p><p><a href="${url}">${url}</a></p><p>Se você não solicitou, ignore este email.</p>`,
      );
    },
  },
  emailVerification: {
    sendOnSignUp: true,
    autoSignInAfterVerification: true,
    sendVerificationEmail: async ({ user, url }) => {
      await sendEmail(
        user.email,
        "Verificar email — ProECG",
        `<p>Olá ${user.name},</p><p>Clique no link abaixo para verificar seu email:</p><p><a href="${url}">${url}</a></p>`,
      );
    },
  },
  secret: env.BETTER_AUTH_SECRET,
  baseURL: env.BETTER_AUTH_URL,
  plugins: [nextCookies()],
});
