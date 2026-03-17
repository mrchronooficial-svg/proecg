"use client";

import { Button } from "@proecg/ui/components/button";
import { Card } from "@proecg/ui/components/card";
import { useMutation } from "@tanstack/react-query";
import { useRouter } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import { toast } from "sonner";

import { trpc } from "@/utils/trpc";
import { EcgCropTool, type Corners } from "./EcgCropTool";

type UploadState = "idle" | "preview" | "cropping" | "uploading" | "processing";

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
  const cameraRef = useRef<HTMLInputElement>(null);
  const galleryRef = useRef<HTMLInputElement>(null);
  const [state, setState] = useState<UploadState>("idle");
  const [preview, setPreview] = useState<string | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [stepIndex, setStepIndex] = useState(0);
  const [imageDims, setImageDims] = useState<{ w: number; h: number } | null>(null);
  const [corners, setCorners] = useState<Corners | null>(null);

  const getUploadUrl = useMutation(trpc.ecg.getUploadUrl.mutationOptions());
  const submitAnalysis = useMutation(
    trpc.ecg.submitAnalysis.mutationOptions(),
  );

  function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const selected = e.target.files?.[0];
    if (!selected) return;
    setFile(selected);
    const url = URL.createObjectURL(selected);
    setPreview(url);

    // Get image dimensions
    const img = new Image();
    img.onload = () => {
      setImageDims({ w: img.naturalWidth, h: img.naturalHeight });
      setState("cropping");
    };
    img.src = url;
  }

  function handleCropConfirm(cropCorners: Corners) {
    setCorners(cropCorners);
    setState("preview");
  }

  function handleRetake() {
    setState("idle");
    setFile(null);
    setPreview(null);
    setImageDims(null);
    setCorners(null);
    if (cameraRef.current) cameraRef.current.value = "";
    if (galleryRef.current) galleryRef.current.value = "";
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

      await submitAnalysis.mutateAsync({
        analysisId,
        corners: corners ?? undefined,
      });

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

  return (
    <>
      {/* Crop tool — full screen overlay */}
      {state === "cropping" && preview && imageDims && (
        <div className="fixed inset-0 z-50">
          <EcgCropTool
            imageUrl={preview}
            imageWidth={imageDims.w}
            imageHeight={imageDims.h}
            onConfirm={handleCropConfirm}
            onRetake={handleRetake}
          />
        </div>
      )}

      {/* Normal upload card */}
      <Card className="glass p-6">
        {state === "idle" && (
          <div className="flex flex-col items-center gap-4">
            <div className="w-full border-2 border-dashed border-primary/30 rounded-xl p-8 flex flex-col items-center gap-5">
              <p className="text-center text-muted-foreground">
                Tire uma foto do ECG em papel ou envie uma imagem da galeria.
              </p>

              {/* Input câmera (abre câmera no mobile) */}
              <input
                ref={cameraRef}
                type="file"
                accept="image/*"
                capture="environment"
                onChange={handleFileChange}
                className="hidden"
              />

              {/* Input galeria (abre seletor de arquivos/galeria) */}
              <input
                ref={galleryRef}
                type="file"
                accept="image/*"
                onChange={handleFileChange}
                className="hidden"
              />

              <div className="flex w-full flex-col gap-3 sm:flex-row sm:justify-center">
                <Button
                  size="lg"
                  onClick={() => cameraRef.current?.click()}
                  className="w-full sm:w-auto"
                >
                  Tirar foto
                </Button>
                <Button
                  size="lg"
                  variant="outline"
                  onClick={() => galleryRef.current?.click()}
                  className="w-full sm:w-auto"
                >
                  Enviar imagem
                </Button>
              </div>
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
            <p className="text-sm text-muted-foreground">
              Bordas do ECG selecionadas. Pronto para analisar.
            </p>
            <div className="flex gap-3">
              <Button variant="outline" onClick={() => setState("cropping")}>
                Ajustar bordas
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
    </>
  );
}
