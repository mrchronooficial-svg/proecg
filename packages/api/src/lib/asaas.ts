import { env } from "@proecg/env/server";

const BASE_URL = env.ASAAS_SANDBOX
  ? "https://sandbox.asaas.com/api"
  : "https://api.asaas.com";

export class AsaasError extends Error {
  constructor(
    message: string,
    public statusCode: number,
    public errors?: Array<{ code: string; description: string }>,
  ) {
    super(message);
    this.name = "AsaasError";
  }
}

async function asaasFetch<T>(
  path: string,
  options: { method?: string; body?: unknown } = {},
): Promise<T> {
  const apiKey = env.ASAAS_API_KEY;
  if (!apiKey) {
    throw new Error("ASAAS_API_KEY not configured");
  }

  const res = await fetch(`${BASE_URL}${path}`, {
    method: options.method ?? "GET",
    headers: {
      "Content-Type": "application/json",
      access_token: apiKey,
    },
    body: options.body ? JSON.stringify(options.body) : undefined,
  });

  if (!res.ok) {
    const text = await res.text();
    let parsed: { errors?: Array<{ code: string; description: string }> } = {};
    try {
      parsed = JSON.parse(text);
    } catch {
      /* corpo não-JSON */
    }
    throw new AsaasError(
      parsed.errors?.[0]?.description ?? `Asaas API error ${res.status}`,
      res.status,
      parsed.errors,
    );
  }

  return res.json() as Promise<T>;
}

// ============================================================
// Clientes
// ============================================================

interface AsaasCustomerRequest {
  name: string;
  email: string;
}

interface AsaasCustomerResponse {
  id: string;
  name: string;
  email: string;
  cpfCnpj?: string;
}

export async function createCustomer(data: AsaasCustomerRequest) {
  return asaasFetch<AsaasCustomerResponse>("/v3/customers", {
    method: "POST",
    body: data,
  });
}

/** Busca cliente por CPF, depois por email; cria se não existir. */
export async function getOrCreateCustomer(input: {
  name: string;
  email: string;
  cpfCnpj: string;
  mobilePhone?: string;
  externalReference?: string;
}): Promise<AsaasCustomerResponse> {
  const byCpf = await asaasFetch<{ data: AsaasCustomerResponse[] }>(
    `/v3/customers?cpfCnpj=${input.cpfCnpj}`,
  );
  if (byCpf.data.length > 0) return byCpf.data[0]!;

  const byEmail = await asaasFetch<{ data: AsaasCustomerResponse[] }>(
    `/v3/customers?email=${encodeURIComponent(input.email)}`,
  );
  if (byEmail.data.length > 0) return byEmail.data[0]!;

  return asaasFetch<AsaasCustomerResponse>("/v3/customers", {
    method: "POST",
    body: input,
  });
}

// ============================================================
// Assinaturas
// ============================================================

interface CreditCard {
  holderName: string;
  number: string;
  expiryMonth: string;
  expiryYear: string;
  ccv: string;
}

interface CreditCardHolderInfo {
  name: string;
  email: string;
  cpfCnpj: string;
  postalCode: string;
  addressNumber: string;
  phone: string;
}

interface AsaasSubscriptionRequest {
  customer: string;
  billingType: "PIX" | "CREDIT_CARD" | "BOLETO" | "UNDEFINED";
  value: number;
  cycle: "MONTHLY" | "SEMIANNUALLY" | "YEARLY";
  nextDueDate: string;
  description?: string;
  externalReference?: string;
  creditCard?: CreditCard;
  creditCardHolderInfo?: CreditCardHolderInfo;
  creditCardToken?: string;
  remoteIp?: string;
}

interface AsaasSubscriptionResponse {
  id: string;
  status: string;
  customer: string;
  value: number;
  cycle: string;
  nextDueDate: string;
  paymentLink?: string;
  creditCard?: {
    creditCardNumber: string;
    creditCardBrand: string;
  };
}

export async function createSubscription(data: AsaasSubscriptionRequest) {
  return asaasFetch<AsaasSubscriptionResponse>("/v3/subscriptions", {
    method: "POST",
    body: data,
  });
}

/** Assinatura recorrente cobrada via Pix (billingType PIX). */
export async function createPixSubscription(input: {
  customer: string;
  value: number;
  cycle: "MONTHLY" | "SEMIANNUALLY" | "YEARLY";
  nextDueDate: string;
  description: string;
  externalReference?: string;
}) {
  return asaasFetch<AsaasSubscriptionResponse>("/v3/subscriptions", {
    method: "POST",
    body: { ...input, billingType: "PIX" },
  });
}

export async function getSubscription(asaasId: string) {
  return asaasFetch<AsaasSubscriptionResponse>(`/v3/subscriptions/${asaasId}`);
}

export async function cancelSubscription(asaasId: string) {
  return asaasFetch<{ deleted: boolean }>(`/v3/subscriptions/${asaasId}`, {
    method: "DELETE",
  });
}

// ============================================================
// Pagamentos / Pix
// ============================================================

interface AsaasPayment {
  id: string;
  customer: string;
  subscription?: string;
  value: number;
  status: string;
  billingType: string;
  invoiceUrl?: string;
  dueDate: string;
}

interface PixQrCode {
  encodedImage: string; // Base64 do QR
  payload: string; // copia-e-cola
  expirationDate: string;
}

export async function getSubscriptionPayments(subscriptionId: string) {
  return asaasFetch<{ data: AsaasPayment[] }>(
    `/v3/subscriptions/${subscriptionId}/payments`,
  );
}

export async function getPaymentPixQrCode(paymentId: string) {
  return asaasFetch<PixQrCode>(`/v3/payments/${paymentId}/pixQrCode`);
}
