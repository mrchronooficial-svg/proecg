import "dotenv/config";
import { createEnv } from "@t3-oss/env-core";
import { z } from "zod";

export const env = createEnv({
  server: {
    DATABASE_URL: z.string().min(1),
    BETTER_AUTH_SECRET: z.string().min(32),
    BETTER_AUTH_URL: z.url(),
    CORS_ORIGIN: z.url(),
    NODE_ENV: z.enum(["development", "production", "test"]).default("development"),

    // Resend (email)
    RESEND_API_KEY: z.string().min(1).optional(),

    // Asaas (payment)
    ASAAS_API_KEY: z.string().min(1).optional(),
    ASAAS_WEBHOOK_TOKEN: z.string().min(1).optional(),
    ASAAS_SANDBOX: z
      .string()
      .default("true")
      .transform((v) => v !== "false"),

    // Cloudflare R2 (storage)
    R2_ACCOUNT_ID: z.string().min(1).optional(),
    R2_ACCESS_KEY_ID: z.string().min(1).optional(),
    R2_SECRET_ACCESS_KEY: z.string().min(1).optional(),
    R2_BUCKET_NAME: z.string().min(1).optional(),
    R2_PUBLIC_URL: z.string().min(1).optional(),

    // Modal (AI backend)
    MODAL_ENDPOINT_URL: z.string().min(1).optional(),
    MODAL_TOKEN: z.string().min(1).optional(),
  },
  runtimeEnv: process.env,
  emptyStringAsUndefined: true,
});
