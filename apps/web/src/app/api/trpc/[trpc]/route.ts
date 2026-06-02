import { createContext } from "@proecg/api/context";
import { appRouter } from "@proecg/api/routers/index";
import { fetchRequestHandler } from "@trpc/server/adapters/fetch";
import { NextRequest } from "next/server";

// O submitAnalysis aguarda síncronamente o Modal terminar (~60-240s
// dependendo do ECG). Vercel Pro default é 60s — precisa de 300s pra
// caber a chamada sincrona. (Hobby vai capar em 60s mesmo assim.)
export const maxDuration = 300;

function handler(req: NextRequest) {
  return fetchRequestHandler({
    endpoint: "/api/trpc",
    req,
    router: appRouter,
    createContext: () => createContext(req),
  });
}
export { handler as GET, handler as POST };
