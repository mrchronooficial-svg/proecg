"use client";

import { Button } from "@proecg/ui/components/button";
import { Card } from "@proecg/ui/components/card";
import { useMutation } from "@tanstack/react-query";
import { useRouter } from "next/navigation";
import { useRef, useState } from "react";
import { toast } from "sonner";

import { trpc } from "@/utils/trpc";

type UploadState = "idle" | "preview" | "uploading" | "processing";

export function UploadForm() {
  const router = useRouter();
  const inputRef = useRef<HTMLInputElement>(null);
  const [state, setState] = useState<UploadState>("idle");
  const [preview, setPreview] = useState<string | null>(null);
  const [file, setFile] = useState<File | null>(null);

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

      const { uploadUrl, analysisId } = await getUploadUrl.mutateAsync();

      await fetch(uploadUrl, {
        method: "PUT",
        body: file,
        headers: { "Content-Type": "image/jpeg" },
      });

      setState("processing");

      await submitAnalysis.mutateAsync({ analysisId });

      toast.success("ECG analisado com sucesso!");
      router.push(`/dashboard/resultado/${analysisId}`);
    } catch (error) {
      setState("preview");
      toast.error(
        error instanceof Error ? error.message : "Erro ao processar ECG",
      );
    }
  }

  function handleReset() {
    setState("idle");
    setFile(null);
    setPreview(null);
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
            className="max-h-80 w-full rounded-md object-contain"
          />
          <div className="flex gap-3">
            <Button variant="outline" onClick={handleReset}>
              Trocar foto
            </Button>
            <Button onClick={handleSubmit}>Analisar ECG</Button>
          </div>
        </div>
      )}

      {state === "uploading" && (
        <div className="flex flex-col items-center gap-4 py-8">
          <div className="h-8 w-8 animate-spin rounded-full border-4 border-primary border-t-transparent" />
          <p className="text-muted-foreground">Enviando imagem...</p>
        </div>
      )}

      {state === "processing" && (
        <div className="flex flex-col items-center gap-4 py-8">
          <div className="h-8 w-8 animate-spin rounded-full border-4 border-primary border-t-transparent" />
          <p className="text-muted-foreground">
            Analisando ECG... Isso pode levar alguns segundos.
          </p>
        </div>
      )}
    </Card>
  );
}
