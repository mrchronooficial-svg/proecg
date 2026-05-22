// src/lib/asaas.ts
// Client da API do Asaas — Checkout Transparente
// Docs: https://docs.asaas.com

const ASAAS_API_KEY = process.env.ASAAS_API_KEY!;
const ASAAS_BASE_URL =
  process.env.ASAAS_SANDBOX === "true"
    ? "https://sandbox.asaas.com/api/v3"
    : "https://api.asaas.com/api/v3";

// ============================================================
// HTTP helper
// ============================================================

async function asaasRequest<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  const url = `${ASAAS_BASE_URL}${endpoint}`;
  const response = await fetch(url, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      access_token: ASAAS_API_KEY,
      ...options.headers,
    },
  });

  if (!response.ok) {
    const errorBody = await response.text();
    console.error(`[Asaas] Erro ${response.status} em ${endpoint}:`, errorBody);

    let parsed: any = {};
    try {
      parsed = JSON.parse(errorBody);
    } catch {}

    throw new AsaasError(
      parsed?.errors?.[0]?.description || `Asaas API error ${response.status}`,
      response.status,
      parsed?.errors
    );
  }

  return response.json() as Promise<T>;
}

export class AsaasError extends Error {
  constructor(
    message: string,
    public statusCode: number,
    public errors?: Array<{ code: string; description: string }>
  ) {
    super(message);
    this.name = "AsaasError";
  }
}

// ============================================================
// CLIENTES
// ============================================================

export interface AsaasCustomer {
  id: string;
  name: string;
  email: string;
  cpfCnpj?: string;
  mobilePhone?: string;
}

export async function getOrCreateCustomer(input: {
  name: string;
  email: string;
  cpfCnpj: string;
  mobilePhone?: string;
  externalReference?: string;
}): Promise<AsaasCustomer> {
  // Buscar existente pelo CPF
  const byCpf = await asaasRequest<{ data: AsaasCustomer[] }>(
    `/customers?cpfCnpj=${input.cpfCnpj}`
  );
  if (byCpf.data.length > 0) return byCpf.data[0];

  // Buscar existente pelo email
  const byEmail = await asaasRequest<{ data: AsaasCustomer[] }>(
    `/customers?email=${encodeURIComponent(input.email)}`
  );
  if (byEmail.data.length > 0) return byEmail.data[0];

  // Criar novo
  return asaasRequest<AsaasCustomer>("/customers", {
    method: "POST",
    body: JSON.stringify(input),
  });
}

// ============================================================
// TOKENIZAÇÃO DE CARTÃO (Checkout Transparente)
// ============================================================

export interface TokenizeCardInput {
  customer: string; // Asaas customer ID
  creditCard: {
    holderName: string;
    number: string;
    expiryMonth: string;
    expiryYear: string;
    ccv: string;
  };
  creditCardHolderInfo: {
    name: string;
    email: string;
    cpfCnpj: string;
    postalCode: string;
    addressNumber: string;
    phone: string;
  };
}

export interface CardToken {
  creditCardNumber: string; // últimos 4 dígitos mascarados
  creditCardBrand: string;
  creditCardToken: string;
}

export async function tokenizeCard(
  input: TokenizeCardInput
): Promise<CardToken> {
  return asaasRequest<CardToken>("/creditCard/tokenize", {
    method: "POST",
    body: JSON.stringify(input),
  });
}

// ============================================================
// ASSINATURAS (Checkout Transparente)
// ============================================================

export interface CreateSubscriptionInput {
  customer: string;
  billingType: "CREDIT_CARD" | "PIX" | "BOLETO";
  value: number;
  cycle: "MONTHLY" | "SEMIANNUALLY" | "YEARLY";
  nextDueDate: string; // YYYY-MM-DD
  description: string;
  externalReference?: string;
  // Dados do cartão para checkout transparente
  creditCard?: {
    holderName: string;
    number: string;
    expiryMonth: string;
    expiryYear: string;
    ccv: string;
  };
  creditCardHolderInfo?: {
    name: string;
    email: string;
    cpfCnpj: string;
    postalCode: string;
    addressNumber: string;
    phone: string;
  };
  // OU usar token previamente gerado
  creditCardToken?: string;
  // IP do cliente (obrigatório para cartão no checkout transparente)
  remoteIp?: string;
}

export interface AsaasSubscription {
  id: string;
  customer: string;
  billingType: string;
  value: number;
  cycle: string;
  nextDueDate: string;
  status: "ACTIVE" | "INACTIVE" | "EXPIRED";
  creditCard?: {
    creditCardNumber: string;
    creditCardBrand: string;
  };
}

export async function createSubscription(
  input: CreateSubscriptionInput
): Promise<AsaasSubscription> {
  return asaasRequest<AsaasSubscription>("/subscriptions", {
    method: "POST",
    body: JSON.stringify(input),
  });
}

// ============================================================
// PAGAMENTO AVULSO via PIX (para planos sem cartão)
// ============================================================

export interface CreatePixPaymentInput {
  customer: string;
  billingType: "PIX";
  value: number;
  dueDate: string;
  description: string;
  externalReference?: string;
}

export interface AsaasPayment {
  id: string;
  customer: string;
  subscription?: string;
  value: number;
  status: string;
  billingType: string;
  invoiceUrl?: string;
  bankSlipUrl?: string;
  dueDate: string;
  paymentDate?: string;
  externalReference?: string;
}

export interface PixQrCode {
  encodedImage: string; // Base64 da imagem QR
  payload: string; // Código copia-e-cola
  expirationDate: string;
}

export async function createPixPayment(
  input: CreatePixPaymentInput
): Promise<AsaasPayment> {
  return asaasRequest<AsaasPayment>("/payments", {
    method: "POST",
    body: JSON.stringify(input),
  });
}

export async function getPixQrCode(paymentId: string): Promise<PixQrCode> {
  return asaasRequest<PixQrCode>(`/payments/${paymentId}/pixQrCode`);
}

// ============================================================
// ASSINATURA PIX (recorrente)
// ============================================================

export async function createPixSubscription(input: {
  customer: string;
  value: number;
  cycle: "MONTHLY" | "SEMIANNUALLY" | "YEARLY";
  nextDueDate: string;
  description: string;
  externalReference?: string;
}): Promise<AsaasSubscription> {
  return asaasRequest<AsaasSubscription>("/subscriptions", {
    method: "POST",
    body: JSON.stringify({
      ...input,
      billingType: "UNDEFINED", // Gera boleto+pix combo
    }),
  });
}

// ============================================================
// CONSULTAS
// ============================================================

export async function getSubscription(
  subscriptionId: string
): Promise<AsaasSubscription> {
  return asaasRequest<AsaasSubscription>(
    `/subscriptions/${subscriptionId}`
  );
}

export async function getSubscriptionPayments(
  subscriptionId: string
): Promise<{ data: AsaasPayment[] }> {
  return asaasRequest<{ data: AsaasPayment[] }>(
    `/subscriptions/${subscriptionId}/payments`
  );
}

export async function cancelSubscription(
  subscriptionId: string
): Promise<{ deleted: boolean }> {
  return asaasRequest<{ deleted: boolean }>(
    `/subscriptions/${subscriptionId}`,
    { method: "DELETE" }
  );
}

export async function getPayment(paymentId: string): Promise<AsaasPayment> {
  return asaasRequest<AsaasPayment>(`/payments/${paymentId}`);
}

export async function getPaymentPixQrCode(
  paymentId: string
): Promise<PixQrCode> {
  return asaasRequest<PixQrCode>(`/payments/${paymentId}/pixQrCode`);
}
