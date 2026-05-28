import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";

import { validatePasswordResetToken } from "@/lib/password-token";

export async function GET(request: NextRequest) {
  const token = request.nextUrl.searchParams.get("token") ?? "";
  const { valid } = await validatePasswordResetToken(token);
  return NextResponse.json({ valid });
}
