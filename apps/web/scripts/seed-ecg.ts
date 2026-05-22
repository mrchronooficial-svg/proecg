/**
 * Insere um EcgAnalysis mock para rafaelmello97@gmail.com.
 * Roda com: tsx scripts/seed-ecg.ts
 */
import prisma from "@proecg/db";

const EMAIL = "rafaelmello97@gmail.com";

const MOCK_REPORT = {
  measurements: {
    heartRate: 78,
    heartRateUnit: "bpm",
    pr: 180,
    prUnit: "ms",
    qrs: 88,
    qrsUnit: "ms",
    qt: 380,
    qtUnit: "ms",
    qtc: 420,
    qtcUnit: "ms",
    axis: 60,
    axisUnit: "°",
    rhythm: "sinusal",
  },
  findings: [
    {
      code: "normal_rhythm",
      description: "Ritmo sinusal regular",
      source: "rules",
    },
    {
      code: "normal_axis",
      description: "Eixo elétrico no quadrante normal (+60°)",
      source: "rules",
    },
  ],
  diagnoses: [
    {
      code: "normal",
      description: "Sem alterações significativas identificadas",
      source: "rules",
    },
  ],
  redFlags: [],
  reportText:
    "Ritmo: sinusal. FC: 78 bpm. Eixo: +60°.\nPR: 180ms. QRS: 88ms. QT: 380ms. QTc: 420ms (Bazett).\n\nSem alterações significativas ao eletrocardiograma.\n\nFerramenta de apoio à decisão clínica — não substitui avaliação médica. Correlacionar sempre com dados clínicos, exame físico e contexto do paciente.",
  metadata: {
    layout: "standard_3x4_with_r1",
    pxPerMm: 13.18,
    leadsActive: 12,
    cnnAvailable: true,
    dotterAvailable: true,
  },
  processingTimeMs: 8420,
};

async function main() {
  const user = await prisma.user.findUnique({ where: { email: EMAIL } });
  if (!user) {
    console.error(`Usuário ${EMAIL} não existe.`);
    process.exitCode = 1;
    return;
  }

  const expiresAt = new Date(Date.now() + 30 * 24 * 60 * 60 * 1000);
  const analysis = await prisma.ecgAnalysis.create({
    data: {
      userId: user.id,
      imageUrl:
        "https://pub-04f824ace48e445b8ca6011e9528c781.r2.dev/sample-ecg.jpg",
      reportJson: MOCK_REPORT,
      status: "COMPLETED",
      expiresAt,
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
