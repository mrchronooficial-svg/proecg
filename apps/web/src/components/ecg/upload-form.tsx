"use client";

import { Button } from "@proecg/ui/components/button";
import { Card } from "@proecg/ui/components/card";
import { useMutation } from "@tanstack/react-query";
import { useRouter } from "next/navigation";
import { useRef, useState } from "react";
import { toast } from "sonner";

import { trpc } from "@/utils/trpc";

type UploadState = "idle" | "preview" | "uploading" | "processing";

const PROCESSING_STEPS = [
  "Enviando imagem...",
  "Digitalizando ECG...",
  "Medindo intervalos...",
  "Aplicando regras clínicas...",
  "Classificando com IA...",
  "Montando laudo...",
];

export function UploadForm() {
  const router = useRouter();
  const inputRef = useRef<HTMLInputElement>(null);
  const [state, setState] = useState<UploadState>("idle");
  const [preview, setPreview] = useState<string | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [stepIndex, setStepIndex] = useState(0);

  const getUploadUrl = useMutation(trpc.ecg.getUploadUrl.mutationOptions());
  const submitAnalysis = useMutation(
    trpc.ecg.submitAnalysis.mutationOptions(),
  );

  function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const selected = e.target.files?.[0];
    if (!selected) return;
    setFile(selected);
    setPreview(URL.createObjectURL(selected));
    setState("preview");
  }

  async function handleSubmit() {
    if (!file) return;

    try {
      setState("uploading");
      setStepIndex(0);

      const { uploadUrl, analysisId } = await getUploadUrl.mutateAsync();

      await fetch(uploadUrl, {
        method: "PUT",
        body: file,
        headers: { "Content-Type": "image/jpeg" },
      });

      setState("processing");

      // Animate through processing steps
      const interval = setInterval(() => {
        setStepIndex((prev) => {
          if (prev < PROCESSING_STEPS.length - 1) return prev + 1;
          return prev;
        });
      }, 2000);

      await submitAnalysis.mutateAsync({ analysisId });

      clearInterval(interval);
      toast.success("ECG analisado com sucesso!");
      router.push(`/dashboard/resultado/${analysisId}`);
    } catch (error) {
      setState("preview");
      setStepIndex(0);
      toast.error(
        error instanceof Error ? error.message : "Erro ao processar ECG",
      );
    }
  }

  function handleReset() {
    setState("idle");
    setFile(null);
    setPreview(null);
    setStepIndex(0);
    if (inputRef.current) inputRef.current.value = "";
  }

  return (
    <Card className="glass p-6">
      {state === "idle" && (
        <div className="flex flex-col items-center gap-4">
          <div
            className="w-full border-2 border-dashed border-primary/30 rounded-xl p-12 hover:border-primary/50 transition-colors cursor-pointer flex flex-col items-center gap-4"
            onClick={() => inputRef.current?.click()}
          >
            <p className="text-center text-muted-foreground">
              Tire uma foto do ECG em papel ou selecione uma imagem da galeria.
            </p>
            <input
              ref={inputRef}
              type="file"
              accept="image/*"
              capture="environment"
              onChange={handleFileChange}
              className="hidden"
            />
            <Button size="lg">
              Selecionar foto
            </Button>
          </div>
        </div>
      )}

      {state === "preview" && preview && (
        <div className="flex flex-col items-center gap-4">
          <img
            src={preview}
            alt="Preview do ECG"
            className="max-h-80 w-full rounded-xl object-contain"
          />
          <div className="flex gap-3">
            <Button variant="outline" onClick={handleReset}>
              Trocar foto
            </Button>
            <Button onClick={handleSubmit}>Analisar ECG</Button>
          </div>
        </div>
      )}

      {(state === "uploading" || state === "processing") && (
        <div className="flex flex-col items-center gap-5 py-8">
          <div className="h-10 w-10 animate-spin rounded-full border-4 border-primary border-t-transparent" />
          <div className="text-center">
            <p className="font-medium text-foreground">
              {state === "uploading"
                ? PROCESSING_STEPS[0]
                : PROCESSING_STEPS[stepIndex]}
            </p>
            <p className="mt-1 text-sm text-muted-foreground">
              Isso pode levar até 30 segundos
            </p>
          </div>
          {/* Progress dots */}
          <div className="flex gap-1.5">
            {PROCESSING_STEPS.map((_, i) => (
              <div
                key={i}
                className={`h-1.5 w-1.5 rounded-full transition-colors ${
                  i <= stepIndex ? "bg-primary" : "bg-border"
                }`}
              />
            ))}
          </div>
        </div>
      )}
    </Card>
  );
}
