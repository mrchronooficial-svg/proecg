import { TRPCError } from "@trpc/server";
import prisma from "@proecg/db";
import { z } from "zod";
import { createId } from "@paralleldrive/cuid2";

import { protectedProcedure, router } from "../index";
import { getUploadUrl, getPublicUrl } from "../lib/r2";
import { analyzeEcg } from "../lib/modal";

const EXPIRY_DAYS = 30;

export const ecgRouter = router({
  getUploadUrl: protectedProcedure.mutation(async ({ ctx }) => {
    const userId = ctx.session.user.id;
    const analysisId = createId();
    const key = `ecg/${userId}/${analysisId}.jpg`;

    const uploadUrl = await getUploadUrl(key);
    const imageUrl = getPublicUrl(key);

    const expiresAt = new Date();
    expiresAt.setDate(expiresAt.getDate() + EXPIRY_DAYS);

    await prisma.ecgAnalysis.create({
      data: {
        id: analysisId,
        userId,
        imageUrl,
        status: "UPLOADING",
        expiresAt,
      },
    });

    return { uploadUrl, analysisId, imageUrl };
  }),

  submitAnalysis: protectedProcedure
    .input(z.object({
      analysisId: z.string(),
      corners: z.object({
        top_left: z.tuple([z.number(), z.number()]),
        top_right: z.tuple([z.number(), z.number()]),
        bottom_right: z.tuple([z.number(), z.number()]),
        bottom_left: z.tuple([z.number(), z.number()]),
      }).optional(),
    }))
    .mutation(async ({ ctx, input }) => {
      const analysis = await prisma.ecgAnalysis.findUnique({
        where: { id: input.analysisId },
      });

      if (!analysis || analysis.userId !== ctx.session.user.id) {
        throw new TRPCError({
          code: "NOT_FOUND",
          message: "Análise não encontrada",
        });
      }

      await prisma.ecgAnalysis.update({
        where: { id: input.analysisId },
        data: { status: "PROCESSING" },
      });

      try {
        const report = await analyzeEcg(analysis.imageUrl, input.corners);

        await prisma.ecgAnalysis.update({
          where: { id: input.analysisId },
          data: {
            status: "COMPLETED",
            reportJson: JSON.parse(JSON.stringify(report)),
          },
        });

        return { status: "COMPLETED" as const, analysisId: input.analysisId };
      } catch (error) {
        await prisma.ecgAnalysis.update({
          where: { id: input.analysisId },
          data: { status: "FAILED" },
        });

        throw new TRPCError({
          code: "INTERNAL_SERVER_ERROR",
          message: "Erro ao processar ECG",
          cause: error,
        });
      }
    }),

  getAnalysis: protectedProcedure
    .input(z.object({ id: z.string() }))
    .query(async ({ ctx, input }) => {
      const analysis = await prisma.ecgAnalysis.findUnique({
        where: { id: input.id },
      });

      if (!analysis || analysis.userId !== ctx.session.user.id) {
        throw new TRPCError({
          code: "NOT_FOUND",
          message: "Análise não encontrada",
        });
      }

      if (analysis.expiresAt < new Date()) {
        throw new TRPCError({
          code: "NOT_FOUND",
          message: "Análise expirada",
        });
      }

      return {
        ...analysis,
        reportJson: analysis.reportJson as Record<string, unknown> | null,
      };
    }),

  listAnalyses: protectedProcedure
    .input(
      z.object({
        cursor: z.string().optional(),
        limit: z.number().min(1).max(50).default(10),
      }),
    )
    .query(async ({ ctx, input }) => {
      const items = await prisma.ecgAnalysis.findMany({
        where: {
          userId: ctx.session.user.id,
          expiresAt: { gt: new Date() },
        },
        orderBy: { createdAt: "desc" },
        take: input.limit + 1,
        select: {
          id: true,
          imageUrl: true,
          status: true,
          reportJson: true,
          createdAt: true,
          expiresAt: true,
        },
        ...(input.cursor
          ? { cursor: { id: input.cursor }, skip: 1 }
          : {}),
      });

      let nextCursor: string | undefined;
      if (items.length > input.limit) {
        const next = items.pop();
        nextCursor = next?.id;
      }

      return {
        items: items.map((item) => ({
          ...item,
          reportJson: item.reportJson as Record<string, unknown> | null,
        })),
        nextCursor,
      };
    }),
});
