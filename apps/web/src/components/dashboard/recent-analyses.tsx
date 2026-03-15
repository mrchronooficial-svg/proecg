"use client";

import { useSuspenseQuery } from "@tanstack/react-query";
import Link from "next/link";
import { Card } from "@proecg/ui/components/card";

import { trpc } from "@/utils/trpc";

export function RecentAnalyses() {
  const { data } = useSuspenseQuery(
    trpc.ecg.listAnalyses.queryOptions({ limit: 5 }),
  );

  if (!data.items.length) {
    return (
      <p className="text-sm text-muted-foreground">
        Nenhuma análise ainda. Comece enviando um ECG.
      </p>
    );
  }

  return (
    <div className="space-y-3">
      {data.items.map((item) => (
        <Link key={item.id} href={`/dashboard/resultado/${item.id}`}>
          <Card className="flex items-center gap-4 p-4 transition-colors hover:bg-accent">
            {item.imageUrl && (
              <img
                src={item.imageUrl}
                alt="ECG"
                className="h-12 w-16 rounded object-cover"
              />
            )}
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium">
                {new Date(item.createdAt).toLocaleDateString("pt-BR")}
              </p>
              <p className="truncate text-xs text-muted-foreground">
                {item.status === "COMPLETED" ? "Concluído" : item.status}
              </p>
            </div>
          </Card>
        </Link>
      ))}
    </div>
  );
}
