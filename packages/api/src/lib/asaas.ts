import { env } from "@proecg/env/server";

const BASE_URL = env.ASAAS_SANDBOX
  ? "https://sandbox.asaas.com/api"
  : "https://api.asaas.com";

interface AsaasCustomerRequest {
  name: string;
  email: string;
}

interface AsaasCustomerResponse {
  id: string;
  name: string;
  email: string;
}

interface AsaasSubscriptionRequest {
  customer: string;
  billingType: "PIX" | "CREDIT_CARD" | "BOLETO";
  value: number;
  cycle: "MONTHLY" | "SEMIANNUALLY" | "YEARLY";
  nextDueDate: string;
  description?: string;
}

interface AsaasSubscriptionResponse {
  id: string;
  status: string;
  customer: string;
  value: number;
  cycle: string;
  nextDueDate: string;
  paymentLink?: string;
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
    throw new Error(`Asaas API error ${res.status}: ${text}`);
  }

  return res.json() as Promise<T>;
}

export async function createCustomer(data: AsaasCustomerRequest) {
  return asaasFetch<AsaasCustomerResponse>("/v3/customers", {
    method: "POST",
    body: data,
  });
}

export async function createSubscription(data: AsaasSubscriptionRequest) {
  return asaasFetch<AsaasSubscriptionResponse>("/v3/subscriptions", {
    method: "POST",
    body: data,
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
