import { AnalysisList } from "@/components/ecg/analysis-list";

export default function HistoricoPage() {
  return (
    <div className="mx-auto max-w-2xl px-4 py-8 pb-20 md:pb-8">
      <h1 className="mb-6 text-2xl font-bold">Histórico</h1>
      <AnalysisList />
    </div>
  );
}
