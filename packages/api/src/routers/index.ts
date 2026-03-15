import { protectedProcedure, publicProcedure, router } from "../index";
import { subscriptionRouter } from "./subscription";
import { ecgRouter } from "./ecg";

export const appRouter = router({
  healthCheck: publicProcedure.query(() => {
    return "OK";
  }),
  privateData: protectedProcedure.query(({ ctx }) => {
    return {
      message: "This is private",
      user: ctx.session.user,
    };
  }),
  subscription: subscriptionRouter,
  ecg: ecgRouter,
});
export type AppRouter = typeof appRouter;
