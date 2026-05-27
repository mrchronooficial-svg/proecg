"use client";

import { useMutation, useQuery } from "@tanstack/react-query";
import { ArrowLeft, Check, CreditCard, QrCode } from "lucide-react";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { toast } from "sonner";

import { CheckoutShell } from "@/components/checkout/CheckoutShell";
import { bareInput, FloatingField } from "@/components/checkout/FloatingField";
import { authClient } from "@/lib/auth-client";
import {
  PLAN_BENEFITS,
  resolvePlan,
  type CheckoutPlan,
} from "@/lib/checkout-plans";
import { trpc } from "@/utils/trpc";

type Method = "card" | "pix";

interface PixData {
  qrCodeBase64: string;
  qrCodeText: string;
  paymentId: string;
}

function onlyDigits(v: string) {
  return v.replace(/\D/g, "");
}
function maskCard(v: string) {
  return onlyDigits(v).slice(0, 19).replace(/(\d{4})(?=\d)/g, "$1 ");
}
function maskExpiry(v: string) {
  const d = onlyDigits(v).slice(0, 4);
  return d.length > 2 ? `${d.slice(0, 2)}/${d.slice(2)}` : d;
}
function maskCpf(v: string) {
  const d = onlyDigits(v).slice(0, 11);
  return d
    .replace(/(\d{3})(\d)/, "$1.$2")
    .replace(/(\d{3})(\d)/, "$1.$2")
    .replace(/(\d{3})(\d{1,2})$/, "$1-$2");
}

export default function CheckoutPage() {
  const router = useRouter();
  const { data: session, isPending: sessionPending } = authClient.useSession();

  const [plan, setPlan] = useState<CheckoutPlan | null>(null);
  const [method, setMethod] = useState<Method>("card");

  const [cardNumber, setCardNumber] = useState("");
  const [expiry, setExpiry] = useState("");
  const [ccv, setCcv] = useState("");
  const [cpf, setCpf] = useState("");

  const [pix, setPix] = useState<PixData | null>(null);
  const [subscriptionId, setSubscriptionId] = useState<string | null>(null);

  useEffect(() => {
    const stored =
      typeof window !== "undefined"
        ? window.sessionStorage.getItem("checkout_plan")
        : null;
    setPlan(resolvePlan(stored));
  }, []);

  useEffect(() => {
    if (!sessionPending && !session) router.replace("/signup");
  }, [sessionPending, session, router]);

  const checkoutCard = useMutation(
    trpc.subscription.checkoutCard.mutationOptions({
      onSuccess: () => router.push("/checkout/success"),
      onError: (e) => toast.error(e.message || "Erro ao processar pagamento."),
    }),
  );

  const checkoutPix = useMutation(
    trpc.subscription.checkoutPix.mutationOptions({
      onSuccess: (data) => {
        setSubscriptionId(data.subscriptionId);
        if (data.pix) {
          setPix({
            qrCodeBase64: data.pix.qrCodeBase64,
            qrCodeText: data.pix.qrCodeText,
            paymentId: data.pix.paymentId,
          });
        } else if (data.invoiceUrl) {
          window.open(data.invoiceUrl, "_blank");
        }
      },
      onError: (e) => toast.error(e.message || "Erro ao gerar Pix."),
    }),
  );

  const pixStatus = useQuery({
    ...trpc.subscription.checkPixPayment.queryOptions({
      subscriptionId: subscriptionId ?? "",
    }),
    enabled: !!subscriptionId && !!pix,
    refetchInterval: 3000,
  });

  useEffect(() => {
    if (pixStatus.data?.isActive) router.push("/checkout/success");
  }, [pixStatus.data?.isActive, router]);

  if (!plan || sessionPending) {
    return (
      <CheckoutShell>
        <div className="flex min-h-svh items-center justify-center text-white/60">
          Carregando…
        </div>
      </CheckoutShell>
    );
  }

  function submitCard(e: React.FormEvent) {
    e.preventDefault();
    if (!plan) return;
    const [mm, aa] = expiry.split("/");
    const cpfDigits = onlyDigits(cpf);
    if (onlyDigits(cardNumber).length < 13) return toast.error("Número do cartão inválido.");
    if (!mm || !aa || aa.length !== 2) return toast.error("Validade inválida (MM/AA).");
    if (ccv.length < 3) return toast.error("CVV inválido.");
    if (cpfDigits.length !== 11) return toast.error("CPF inválido.");

    checkoutCard.mutate({
      plan: plan.id,
      creditCard: {
        number: onlyDigits(cardNumber),
        expiryMonth: mm,
        expiryYear: `20${aa}`,
        ccv,
      },
      cpf: cpfDigits,
    });
  }

  function submitPix(e: React.FormEvent) {
    e.preventDefault();
    if (!plan) return;
    const cpfDigits = onlyDigits(cpf);
    if (cpfDigits.length !== 11) return toast.error("CPF inválido.");
    checkoutPix.mutate({ plan: plan.id, cpf: cpfDigits });
  }

  const busy = checkoutCard.isPending || checkoutPix.isPending;

  return (
    <CheckoutShell>
      <div className="mx-auto max-w-5xl px-6 py-12">
        <button
          onClick={() => router.push("/signup/profile")}
          className="mb-6 inline-flex items-center gap-2 text-[22px] font-bold tracking-[-0.02em] transition hover:opacity-80"
        >
          <ArrowLeft size={22} className="text-white/70" /> Configure seu plano
        </button>

        <div className="grid grid-cols-1 gap-8 md:grid-cols-[1fr_380px]">
          {/* Formulário */}
          <div>
            {pix ? (
              <div className="rounded-3xl border border-white/10 bg-white/[0.05] p-6 backdrop-blur md:p-8">
                <PixView pix={pix} polling={pixStatus.isFetching} />
              </div>
            ) : (
              <>
                <p className="mb-3 text-[15px] font-semibold">Método de pagamento</p>
                <div className="mb-4 grid grid-cols-2 gap-3">
                  <MethodCard
                    active={method === "card"}
                    onClick={() => setMethod("card")}
                    icon={<CreditCard size={20} />}
                    label="Cartão"
                  />
                  <MethodCard
                    active={method === "pix"}
                    onClick={() => setMethod("pix")}
                    icon={<QrCode size={20} />}
                    label="Pix"
                  />
                </div>

                {method === "card" ? (
                  <form onSubmit={submitCard} className="space-y-3">
                    <FloatingField label="Número do cartão" trailing={<CardBrands />}>
                      <input
                        inputMode="numeric"
                        value={cardNumber}
                        onChange={(e) => setCardNumber(maskCard(e.target.value))}
                        placeholder="0000 0000 0000 0000"
                        className={bareInput}
                      />
                    </FloatingField>
                    <div className="grid grid-cols-2 gap-3">
                      <FloatingField label="Data de validade">
                        <input
                          inputMode="numeric"
                          value={expiry}
                          onChange={(e) => setExpiry(maskExpiry(e.target.value))}
                          placeholder="MM/AA"
                          className={bareInput}
                        />
                      </FloatingField>
                      <FloatingField label="Código de segurança">
                        <input
                          inputMode="numeric"
                          value={ccv}
                          onChange={(e) => setCcv(onlyDigits(e.target.value).slice(0, 4))}
                          placeholder="000"
                          className={bareInput}
                        />
                      </FloatingField>
                    </div>
                    <FloatingField label="CPF">
                      <input
                        inputMode="numeric"
                        value={cpf}
                        onChange={(e) => setCpf(maskCpf(e.target.value))}
                        placeholder="000.000.000-00"
                        className={bareInput}
                      />
                    </FloatingField>
                  </form>
                ) : (
                  <form onSubmit={submitPix} className="space-y-3">
                    <FloatingField label="CPF">
                      <input
                        inputMode="numeric"
                        value={cpf}
                        onChange={(e) => setCpf(maskCpf(e.target.value))}
                        placeholder="000.000.000-00"
                        className={bareInput}
                      />
                    </FloatingField>
                    <p className="px-1 text-[14px] text-white/60">
                      Ao clicar em Assinar, um QR Code Pix será gerado nesta tela.
                    </p>
                  </form>
                )}
              </>
            )}
          </div>

          {/* Resumo — card escuro elevado */}
          <aside className="h-fit rounded-3xl border border-white/10 bg-white/[0.05] p-7 backdrop-blur">
            <h2 className="text-[22px] font-bold tracking-[-0.01em]">Plano {plan.name}</h2>
            <p className="mt-1 text-[14px] text-white/55">Principais recursos</p>
            <ul className="mt-4 space-y-3.5">
              {PLAN_BENEFITS.map((b) => (
                <li key={b} className="flex items-start gap-3 text-[14px] text-white/85">
                  <Check size={18} strokeWidth={2.5} className="mt-0.5 shrink-0 text-[#0a84ff]" />
                  {b}
                </li>
              ))}
            </ul>
            <hr className="my-5 border-white/10" />
            <Row label={`Assinatura ${plan.cycleLabel}`} value={plan.totalLabel} />
            <Row label="Imposto estimado" value="R$ 0,00" />
            <div className="mt-3 flex items-baseline justify-between">
              <span className="text-[15px] font-bold">A pagar hoje</span>
              <span className="text-[18px] font-extrabold">{plan.totalLabel}</span>
            </div>

            {!pix && (
              <button
                onClick={method === "card" ? submitCard : submitPix}
                disabled={busy}
                className="mt-6 w-full rounded-full bg-[#0a84ff] px-6 py-4 text-[16px] font-semibold text-white transition-all hover:bg-[#0066cc] disabled:cursor-not-allowed disabled:opacity-60"
              >
                {busy ? "Processando…" : "Assinar"}
              </button>
            )}
            <p className="mt-3 text-center text-[13px] font-medium text-white/55">
              Compra segura. Cancele quando quiser.
            </p>
          </aside>
        </div>

        <p className="mt-8 max-w-3xl text-[12px] leading-relaxed text-white/40">
          Renovação {plan.cycleLabel} até cancelar. Cobraremos {plan.totalLabel} por
          ciclo. Cancele quando quiser em Configurações. Ao assinar, você concorda com
          os Termos de Uso e reconhece que leu a Política de Privacidade.
        </p>
      </div>
    </CheckoutShell>
  );
}

function MethodCard({
  active,
  onClick,
  icon,
  label,
}: {
  active: boolean;
  onClick: () => void;
  icon: React.ReactNode;
  label: string;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`flex flex-col items-start gap-2 rounded-2xl border p-4 text-left transition ${
        active
          ? "border-[#0a84ff] bg-[#0a84ff]/10"
          : "border-white/12 bg-white/[0.03] hover:bg-white/[0.06]"
      }`}
    >
      <span className={active ? "text-[#0a84ff]" : "text-white/70"}>{icon}</span>
      <span className="text-[14px] font-semibold">{label}</span>
    </button>
  );
}

function CardBrands() {
  return (
    <div className="flex items-center gap-1 text-[9px] font-bold">
      <span className="rounded bg-white/10 px-1.5 py-0.5 text-white/70">VISA</span>
      <span className="rounded bg-white/10 px-1.5 py-0.5 text-white/70">MASTER</span>
    </div>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between py-1 text-[14px]">
      <span className="text-white/55">{label}</span>
      <span className="font-medium">{value}</span>
    </div>
  );
}

function PixView({ pix, polling }: { pix: PixData; polling: boolean }) {
  const [copied, setCopied] = useState(false);
  return (
    <div className="flex flex-col items-center text-center">
      <h2 className="mb-1 text-[20px] font-bold tracking-[-0.01em]">Escaneie para pagar</h2>
      <p className="mb-4 text-[14px] text-white/60">
        Abra o app do seu banco e escaneie o QR Code.
      </p>
      <div className="rounded-2xl bg-white p-3">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={`data:image/png;base64,${pix.qrCodeBase64}`}
          alt="QR Code Pix"
          className="size-52"
        />
      </div>
      <button
        type="button"
        onClick={() => {
          navigator.clipboard.writeText(pix.qrCodeText);
          setCopied(true);
          setTimeout(() => setCopied(false), 2000);
        }}
        className="mt-4 w-full rounded-full border border-white/15 bg-white/[0.04] px-4 py-3 text-[14px] font-medium text-white transition hover:bg-white/10"
      >
        {copied ? "Código copiado!" : "Copiar código Pix"}
      </button>
      <p className="mt-4 flex items-center gap-2 text-[14px] text-white/60">
        <span className="size-2 animate-pulse rounded-full bg-amber-400" />
        {polling ? "Aguardando confirmação do pagamento…" : "Aguardando pagamento…"}
      </p>
    </div>
  );
}
