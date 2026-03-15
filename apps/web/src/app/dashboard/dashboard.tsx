"use client";

import Link from "next/link";
import { buttonVariants } from "@proecg/ui/components/button";
import { cn } from "@proecg/ui/lib/utils";

import { authClient } from "@/lib/auth-client";
import { RecentAnalyses } from "@/components/dashboard/recent-analyses";

export default function Dashboard({
  session,
}: {
  session: typeof authClient.$Infer.Session;
}) {
  return (
    <div className="mx-auto max-w-2xl px-4 py-8 pb-20 md:pb-8">
      <div className="mb-8 bg-primary text-primary-foreground rounded-2xl p-6 flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Olá, {session.user.name}</h1>
          <p className="text-sm text-white/80">
            O que gostaria de analisar hoje?
          </p>
        </div>
        <Link
          href="/dashboard/novo"
          className={cn(
            buttonVariants({ variant: "secondary" }),
            "bg-white text-primary hover:bg-white/90",
          )}
        >
          Novo ECG
        </Link>
      </div>

      <section>
        <h2 className="mb-4 text-lg font-semibold">Análises recentes</h2>
        <RecentAnalyses />
      </section>
    </div>
  );
}
