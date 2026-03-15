import { S3Client, PutObjectCommand } from "@aws-sdk/client-s3";
import { getSignedUrl } from "@aws-sdk/s3-request-presigner";
import { env } from "@proecg/env/server";

function getS3Client() {
  if (!env.R2_ACCOUNT_ID || !env.R2_ACCESS_KEY_ID || !env.R2_SECRET_ACCESS_KEY) {
    throw new Error("R2 credentials not configured");
  }

  return new S3Client({
    region: "auto",
    endpoint: `https://${env.R2_ACCOUNT_ID}.r2.cloudflarestorage.com`,
    credentials: {
      accessKeyId: env.R2_ACCESS_KEY_ID,
      secretAccessKey: env.R2_SECRET_ACCESS_KEY,
    },
  });
}

export async function getUploadUrl(key: string): Promise<string> {
  const client = getS3Client();
  const command = new PutObjectCommand({
    Bucket: env.R2_BUCKET_NAME,
    Key: key,
    ContentType: "image/jpeg",
  });

  return getSignedUrl(client, command, { expiresIn: 300 });
}

export function getPublicUrl(key: string): string {
  if (!env.R2_PUBLIC_URL) {
    throw new Error("R2_PUBLIC_URL not configured");
  }
  return `${env.R2_PUBLIC_URL}/${key}`;
}
