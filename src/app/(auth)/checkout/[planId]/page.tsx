// src/app/(auth)/checkout/[planId]/page.tsx
// Página de Checkout Transparente do ProECG
// Rota: /checkout/MONTHLY | /checkout/SEMI_ANNUAL | /checkout/ANNUAL
// Pré-requisito: usuário autenticado (redirect para /login se não estiver)

"use client";

import { useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { getPlan, PLAN_FEATURES, type PlanId } from "@/lib/plans";
import { trpc } from "@/lib/trpc";

type PaymentMethod = "CREDIT_CARD" | "PIX";

export default function CheckoutPage() {
  const params = useParams();
  const router = useRouter();
  const planId = params.planId as PlanId;

  let plan;
  try {
    plan = getPlan(planId);
  } catch {
    router.push("/planos");
    return null;
  }

  const [paymentMethod, setPaymentMethod] = useState<PaymentMethod>("CREDIT_CARD");
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [pixData, setPixData] = useState<{
    qrCodeBase64: string;
    qrCodeText: string;
  } | null>(null);
  const [success, setSuccess] = useState(false);

  // Dados do formulário
  const [form, setForm] = useState({
    name: "",
    cpf: "",
    phone: "",
    postalCode: "",
    addressNumber: "",
    // Cartão
    cardNumber: "",
    cardName: "",
    cardExpiry: "", // MM/AA
    cardCvv: "",
  });

  const checkoutCard = trpc.subscription.checkoutCard.useMutation();
  const checkoutPix = trpc.subscription.checkoutPix.useMutation();

  function updateField(field: string, value: string) {
    setForm((prev) => ({ ...prev, [field]: value }));
    setError(null);
  }

  // Formatar CPF: 123.456.789-00
  function formatCpf(value: string) {
    const digits = value.replace(/\D/g, "").slice(0, 11);
    if (digits.length <= 3) return digits;
    if (digits.length <= 6) return `${digits.slice(0, 3)}.${digits.slice(3)}`;
    if (digits.length <= 9)
      return `${digits.slice(0, 3)}.${digits.slice(3, 6)}.${digits.slice(6)}`;
    return `${digits.slice(0, 3)}.${digits.slice(3, 6)}.${digits.slice(6, 9)}-${digits.slice(9)}`;
  }

  // Formatar número do cartão: 1234 5678 9012 3456
  function formatCardNumber(value: string) {
    const digits = value.replace(/\D/g, "").slice(0, 16);
    return digits.replace(/(\d{4})(?=\d)/g, "$1 ");
  }

  // Formatar validade: MM/AA
  function formatExpiry(value: string) {
    const digits = value.replace(/\D/g, "").slice(0, 4);
    if (digits.length <= 2) return digits;
    return `${digits.slice(0, 2)}/${digits.slice(2)}`;
  }

  // Formatar CEP: 12345-678
  function formatCep(value: string) {
    const digits = value.replace(/\D/g, "").slice(0, 8);
    if (digits.length <= 5) return digits;
    return `${digits.slice(0, 5)}-${digits.slice(5)}`;
  }

  // Formatar telefone: (11) 99999-9999
  function formatPhone(value: string) {
    const digits = value.replace(/\D/g, "").slice(0, 11);
    if (digits.length <= 2) return digits;
    if (digits.length <= 7) return `(${digits.slice(0, 2)}) ${digits.slice(2)}`;
    return `(${digits.slice(0, 2)}) ${digits.slice(2, 7)}-${digits.slice(7)}`;
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setIsProcessing(true);
    setError(null);

    const cpfDigits = form.cpf.replace(/\D/g, "");

    try {
      if (paymentMethod === "CREDIT_CARD") {
        const [expiryMonth, expiryYear] = form.cardExpiry.split("/");
        const result = await checkoutCard.mutateAsync({
          planId,
          creditCard: {
            holderName: form.cardName || form.name,
            number: form.cardNumber.replace(/\s/g, ""),
            expiryMonth: expiryMonth,
            expiryYear: `20${expiryYear}`,
            ccv: form.cardCvv,
          },
          holderInfo: {
            name: form.name,
            cpfCnpj: cpfDigits,
            postalCode: form.postalCode.replace(/\D/g, ""),
            addressNumber: form.addressNumber,
            phone: form.phone.replace(/\D/g, ""),
          },
        });

        setSuccess(true);
        // Redirecionar para dashboard após 2 segundos
        setTimeout(() => router.push("/dashboard"), 2000);
      } else {
        // PIX
        const result = await checkoutPix.mutateAsync({
          planId,
          name: form.name,
          cpfCnpj: cpfDigits,
        });

        if (result.pix) {
          setPixData({
            qrCodeBase64: result.pix.qrCodeBase64,
            qrCodeText: result.pix.qrCodeText,
          });
        }
      }
    } catch (err: any) {
      setError(err.message || "Erro ao processar pagamento. Tente novamente.");
    } finally {
      setIsProcessing(false);
    }
  }

  async function copyPixCode() {
    if (pixData?.qrCodeText) {
      await navigator.clipboard.writeText(pixData.qrCodeText);
    }
  }

  // ============================================================
  // TELA DE SUCESSO (cartão aprovado)
  // ============================================================
  if (success) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50 px-4">
        <div className="bg-white rounded-2xl shadow-lg p-8 max-w-md w-full text-center">
          <div className="w-16 h-16 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-4">
            <svg className="w-8 h-8 text-green-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
            </svg>
          </div>
          <h1 className="text-2xl font-bold text-gray-900 mb-2">Pagamento confirmado!</h1>
          <p className="text-gray-600 mb-2">
            Plano <strong>{plan.name}</strong> ativado com sucesso.
          </p>
          <p className="text-gray-500 text-sm mb-6">
            Enviamos um email de confirmação. Você está sendo redirecionado...
          </p>
          <div className="animate-spin w-6 h-6 border-2 border-blue-600 border-t-transparent rounded-full mx-auto" />
        </div>
      </div>
    );
  }

  // ============================================================
  // TELA DO PIX (QR Code gerado)
  // ============================================================
  if (pixData) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50 px-4">
        <div className="bg-white rounded-2xl shadow-lg p-8 max-w-md w-full text-center">
          <h1 className="text-xl font-bold text-gray-900 mb-1">Pague com Pix</h1>
          <p className="text-gray-500 text-sm mb-6">
            Escaneie o QR Code abaixo com o app do seu banco
          </p>

          {/* QR Code */}
          <div className="bg-white border-2 border-gray-200 rounded-xl p-4 inline-block mb-4">
            <img
              src={`data:image/png;base64,${pixData.qrCodeBase64}`}
              alt="QR Code Pix"
              className="w-56 h-56"
            />
          </div>

          {/* Copia e cola */}
          <div className="mb-6">
            <p className="text-xs text-gray-400 mb-2">Ou copie o código Pix:</p>
            <div className="flex gap-2">
              <input
                type="text"
                readOnly
                value={pixData.qrCodeText}
                className="flex-1 bg-gray-100 rounded-lg px-3 py-2 text-xs text-gray-600 truncate"
              />
              <button
                onClick={copyPixCode}
                className="bg-blue-600 text-white px-4 py-2 rounded-lg text-sm font-medium hover:bg-blue-700 transition-colors"
              >
                Copiar
              </button>
            </div>
          </div>

          {/* Info */}
          <div className="bg-blue-50 rounded-lg p-4 text-left">
            <p className="text-sm text-blue-800">
              <strong>Plano {plan.name}</strong> — R$ {plan.totalPrice.toLocaleString("pt-BR")}
            </p>
            <p className="text-xs text-blue-600 mt-1">
              Após o pagamento, seu acesso será liberado automaticamente em até 1 minuto.
            </p>
          </div>

          <button
            onClick={() => router.push("/dashboard")}
            className="mt-4 text-sm text-gray-500 hover:text-gray-700 transition-colors"
          >
            Já paguei, ir para o dashboard →
          </button>
        </div>
      </div>
    );
  }

  // ============================================================
  // FORMULÁRIO DE CHECKOUT
  // ============================================================
  return (
    <div className="min-h-screen bg-gray-50 py-8 px-4">
      <div className="max-w-lg mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-2xl font-bold text-gray-900">Finalizar assinatura</h1>
          <p className="text-gray-500 mt-1">ProECG — Plano {plan.name}</p>
        </div>

        {/* Resumo do plano */}
        <div className="bg-white rounded-xl border border-gray-200 p-5 mb-6">
          <div className="flex justify-between items-center">
            <div>
              <p className="font-semibold text-gray-900">Plano {plan.name}</p>
              <p className="text-sm text-gray-500">{plan.description}</p>
            </div>
            <div className="text-right">
              <p className="text-2xl font-bold text-gray-900">
                R$ {plan.pricePerMonth}
              </p>
              <p className="text-xs text-gray-400">/mês</p>
            </div>
          </div>
          {plan.cycle !== "MONTHLY" && (
            <div className="mt-3 pt-3 border-t border-gray-100 flex justify-between text-sm">
              <span className="text-gray-500">
                Cobrança {plan.cycle === "ANNUAL" ? "anual" : "semestral"}
              </span>
              <span className="font-medium text-gray-700">
                R$ {plan.totalPrice.toLocaleString("pt-BR")}
              </span>
            </div>
          )}
        </div>

        <form onSubmit={handleSubmit}>
          {/* Dados pessoais */}
          <div className="bg-white rounded-xl border border-gray-200 p-5 mb-4">
            <h2 className="text-sm font-semibold text-gray-700 mb-4">Seus dados</h2>
            <div className="space-y-3">
              <input
                type="text"
                placeholder="Nome completo"
                value={form.name}
                onChange={(e) => updateField("name", e.target.value)}
                required
                className="w-full border border-gray-300 rounded-lg px-4 py-3 text-sm focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none"
              />
              <div className="grid grid-cols-2 gap-3">
                <input
                  type="text"
                  placeholder="CPF"
                  value={formatCpf(form.cpf)}
                  onChange={(e) => updateField("cpf", e.target.value)}
                  required
                  inputMode="numeric"
                  className="border border-gray-300 rounded-lg px-4 py-3 text-sm focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none"
                />
                <input
                  type="text"
                  placeholder="Telefone"
                  value={formatPhone(form.phone)}
                  onChange={(e) => updateField("phone", e.target.value)}
                  required
                  inputMode="tel"
                  className="border border-gray-300 rounded-lg px-4 py-3 text-sm focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none"
                />
              </div>
              <div className="grid grid-cols-3 gap-3">
                <input
                  type="text"
                  placeholder="CEP"
                  value={formatCep(form.postalCode)}
                  onChange={(e) => updateField("postalCode", e.target.value)}
                  required
                  inputMode="numeric"
                  className="col-span-2 border border-gray-300 rounded-lg px-4 py-3 text-sm focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none"
                />
                <input
                  type="text"
                  placeholder="Nº"
                  value={form.addressNumber}
                  onChange={(e) => updateField("addressNumber", e.target.value)}
                  required
                  className="border border-gray-300 rounded-lg px-4 py-3 text-sm focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none"
                />
              </div>
            </div>
          </div>

          {/* Forma de pagamento */}
          <div className="bg-white rounded-xl border border-gray-200 p-5 mb-4">
            <h2 className="text-sm font-semibold text-gray-700 mb-4">Forma de pagamento</h2>

            {/* Toggle cartão/pix */}
            <div className="flex bg-gray-100 rounded-lg p-1 mb-4">
              <button
                type="button"
                onClick={() => setPaymentMethod("CREDIT_CARD")}
                className={`flex-1 py-2.5 rounded-md text-sm font-medium transition-all ${
                  paymentMethod === "CREDIT_CARD"
                    ? "bg-white text-gray-900 shadow-sm"
                    : "text-gray-500 hover:text-gray-700"
                }`}
              >
                💳 Cartão de Crédito
              </button>
              <button
                type="button"
                onClick={() => setPaymentMethod("PIX")}
                className={`flex-1 py-2.5 rounded-md text-sm font-medium transition-all ${
                  paymentMethod === "PIX"
                    ? "bg-white text-gray-900 shadow-sm"
                    : "text-gray-500 hover:text-gray-700"
                }`}
              >
                ⚡ Pix
              </button>
            </div>

            {/* Campos do cartão */}
            {paymentMethod === "CREDIT_CARD" && (
              <div className="space-y-3">
                <input
                  type="text"
                  placeholder="Número do cartão"
                  value={formatCardNumber(form.cardNumber)}
                  onChange={(e) => updateField("cardNumber", e.target.value)}
                  required
                  inputMode="numeric"
                  autoComplete="cc-number"
                  className="w-full border border-gray-300 rounded-lg px-4 py-3 text-sm focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none"
                />
                <input
                  type="text"
                  placeholder="Nome impresso no cartão"
                  value={form.cardName}
                  onChange={(e) => updateField("cardName", e.target.value)}
                  required
                  autoComplete="cc-name"
                  className="w-full border border-gray-300 rounded-lg px-4 py-3 text-sm focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none"
                />
                <div className="grid grid-cols-2 gap-3">
                  <input
                    type="text"
                    placeholder="MM/AA"
                    value={formatExpiry(form.cardExpiry)}
                    onChange={(e) => updateField("cardExpiry", e.target.value)}
                    required
                    inputMode="numeric"
                    autoComplete="cc-exp"
                    maxLength={5}
                    className="border border-gray-300 rounded-lg px-4 py-3 text-sm focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none"
                  />
                  <input
                    type="text"
                    placeholder="CVV"
                    value={form.cardCvv}
                    onChange={(e) =>
                      updateField("cardCvv", e.target.value.replace(/\D/g, "").slice(0, 4))
                    }
                    required
                    inputMode="numeric"
                    autoComplete="cc-csc"
                    maxLength={4}
                    className="border border-gray-300 rounded-lg px-4 py-3 text-sm focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none"
                  />
                </div>
              </div>
            )}

            {/* Info do Pix */}
            {paymentMethod === "PIX" && (
              <div className="bg-green-50 rounded-lg p-4">
                <p className="text-sm text-green-800">
                  Ao clicar em "Pagar", um QR Code Pix será gerado para você escanear com o app do seu banco.
                  O acesso é liberado automaticamente após o pagamento.
                </p>
              </div>
            )}
          </div>

          {/* Erro */}
          {error && (
            <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-4">
              <p className="text-sm text-red-700">{error}</p>
            </div>
          )}

          {/* Botão de pagamento */}
          <button
            type="submit"
            disabled={isProcessing}
            className={`w-full py-4 rounded-xl text-white font-semibold text-base transition-all ${
              isProcessing
                ? "bg-gray-400 cursor-not-allowed"
                : "bg-blue-600 hover:bg-blue-700 active:scale-[0.98] shadow-lg shadow-blue-600/25"
            }`}
          >
            {isProcessing ? (
              <span className="flex items-center justify-center gap-2">
                <span className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                Processando...
              </span>
            ) : paymentMethod === "CREDIT_CARD" ? (
              `Pagar R$ ${plan.totalPrice.toLocaleString("pt-BR")}`
            ) : (
              `Gerar Pix — R$ ${plan.totalPrice.toLocaleString("pt-BR")}`
            )}
          </button>

          {/* Segurança */}
          <p className="text-center text-xs text-gray-400 mt-4">
            🔒 Pagamento processado com segurança pelo Asaas.
            {paymentMethod === "CREDIT_CARD" &&
              " Seus dados do cartão não são armazenados."}
          </p>
          <p className="text-center text-xs text-gray-400 mt-1">
            Cancele quando quiser · Sem multa · Sem burocracia
          </p>
        </form>
      </div>
    </div>
  );
}
