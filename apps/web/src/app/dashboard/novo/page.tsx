import { UploadForm } from "@/components/ecg/upload-form";

export default function NovoEcgPage() {
  return (
    <div className="mx-auto max-w-2xl px-4 py-8 pb-20 md:pb-8">
      <h1 className="mb-6 text-2xl font-bold">Novo ECG</h1>
      <UploadForm />
    </div>
  );
}
