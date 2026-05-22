/**
 * Cria EcgAnalysis simulada para IMG_1303 — ritmo sinusal com extrassístoles
 * supraventriculares isoladas (achado moderado, sem red flag).
 * Roda com: tsx scripts/seed-ecg-1303.ts
 */
import prisma from "@proecg/db";

const EMAIL = "rafaelmello97@gmail.com";

const REPORT = {
  measurements: {
    heartRate: 72,
    heartRateUnit: "bpm",
    pr: 168,
    prUnit: "ms",
    qrs: 96,
    qrsUnit: "ms",
    qt: 396,
    qtUnit: "ms",
    qtc: 432,
    qtcUnit: "ms",
    axis: 45,
    axisUnit: "°",
    rhythm: "Sinusal",
  },
  findings: [
    {
      code: "rhythm_sinus",
      description: "Ritmo sinusal regular",
      source: "rules",
    },
    {
      code: "premature_atrial",
      description: "Extrassístoles atriais isoladas (3 em 10s)",
      source: "rules",
      leads_affected: ["DII", "V1"],
    },
    {
      code: "axis_normal",
      description: "Eixo elétrico no quadrante normal (+45°)",
      source: "rules",
    },
    {
      code: "st_normal",
      description: "Sem alterações significativas do segmento ST",
      source: "rules",
    },
    {
      code: "t_normal",
      description: "Onda T sem alterações morfológicas",
      source: "rules",
    },
  ],
  diagnoses: [
    {
      code: "atrial_extrasystoles",
      description: "Extrassístoles atriais isoladas — sem repercussão hemodinâmica aparente",
      source: "rules",
    },
  ],
  redFlags: [],
  reportText:
    "Ritmo sinusal regular. Frequência cardíaca de 72 bpm. Eixo elétrico normal (+45°).\n\nIntervalo PR: 168ms (normal). Duração do QRS: 96ms (normal). Intervalo QT: 396ms. QTc corrigido (Bazett): 432ms (normal).\n\nObservadas 3 extrassístoles atriais isoladas em traçado de 10 segundos, sem repercussão hemodinâmica aparente.\n\nDemais segmentos sem alterações significativas. Onda T preservada em todas as derivações analisadas.",
  metadata: {
    layout: "standard_3x4_with_r1",
    pxPerMm: 13.18,
    leadsActive: 12,
    cnnAvailable: true,
    dotterAvailable: true,
  },
  processingTimeMs: 9842,
};

async function main() {
  const user = await prisma.user.findUnique({ where: { email: EMAIL } });
  if (!user) {
    console.error(`Usuário ${EMAIL} não existe.`);
    process.exitCode = 1;
    return;
  }

  const analysis = await prisma.ecgAnalysis.create({
    data: {
      userId: user.id,
      imageUrl: "/mock-ecgs/IMG_1303.png",
      reportJson: REPORT,
      status: "COMPLETED",
      expiresAt: new Date(Date.now() + 30 * 24 * 60 * 60 * 1000),
    },
  });
  console.log(`OK — analysis criada: ${analysis.id}`);
  console.log(`URL: http://localhost:3001/dashboard/resultado/${analysis.id}`);
}

main()
  .catch((e) => {
    console.error(e);
    process.exitCode = 1;
  })
  .finally(async () => {
    await prisma.$disconnect();
  });
