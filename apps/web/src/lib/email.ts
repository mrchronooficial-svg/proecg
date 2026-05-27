import { env } from "@proecg/env/server";
import { Resend } from "resend";

const resend = env.RESEND_API_KEY ? new Resend(env.RESEND_API_KEY) : null;

const ACCENT = "#5B65DC";
const TEXT = "#122056";

interface ConfirmationInput {
  to: string;
  name: string;
  planName: string;
  totalLabel: string;
  nextBilling: string;
  setPasswordUrl: string;
}

/** Email de confirmação de pagamento + link para o médico definir a senha. */
export async function sendPaymentConfirmationEmail(input: ConfirmationInput) {
  const html = confirmationHtml(input);
  const subject = "Bem-vindo ao ProECG! Seu acesso está ativo 🎉";

  if (!resend) {
    console.log(`[DEV] Email de confirmação para ${input.to}: ${subject}`);
    return;
  }
  await resend.emails.send({
    from: "ProECG <noreply@proecg.com.br>",
    to: input.to,
    subject,
    html,
  });
}

function confirmationHtml({
  name,
  planName,
  totalLabel,
  nextBilling,
  setPasswordUrl,
}: ConfirmationInput) {
  return `
  <div style="margin:0;padding:0;background:#FAFAFD;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Inter,sans-serif;color:${TEXT}">
    <div style="max-width:480px;margin:0 auto;padding:32px 20px">
      <div style="font-size:22px;font-weight:800;text-align:center;margin-bottom:24px">
        Pro<span style="color:${ACCENT}">ECG</span>
      </div>
      <div style="background:#fff;border:1px solid #E2E4F0;border-radius:16px;padding:32px">
        <h1 style="font-size:22px;font-weight:800;margin:0 0 8px">Olá, ${escapeHtml(name)}!</h1>
        <p style="font-size:15px;line-height:1.6;color:#4A5078;margin:0 0 20px">
          Seu plano <strong>${escapeHtml(planName)}</strong> foi ativado com sucesso.
        </p>

        <div style="background:#EEEFFD;border-radius:12px;padding:16px 20px;margin-bottom:24px">
          <table style="width:100%;font-size:14px;color:${TEXT}">
            <tr><td style="padding:4px 0;color:#4A5078">Plano</td><td style="text-align:right;font-weight:600">${escapeHtml(planName)}</td></tr>
            <tr><td style="padding:4px 0;color:#4A5078">Valor</td><td style="text-align:right;font-weight:600">${escapeHtml(totalLabel)}</td></tr>
            <tr><td style="padding:4px 0;color:#4A5078">Próxima cobrança</td><td style="text-align:right;font-weight:600">${escapeHtml(nextBilling)}</td></tr>
          </table>
        </div>

        <p style="font-size:15px;line-height:1.6;color:#4A5078;margin:0 0 16px">
          Para acessar o ProECG, defina sua senha de acesso:
        </p>
        <a href="${setPasswordUrl}" style="display:block;background:${ACCENT};color:#fff;text-decoration:none;text-align:center;font-weight:600;padding:14px 24px;border-radius:12px;font-size:15px">
          Definir minha senha
        </a>
        <p style="font-size:12px;line-height:1.5;color:#AEAEB2;margin:16px 0 0">
          Se o botão não funcionar, copie e cole este link no navegador:<br/>${setPasswordUrl}
        </p>
      </div>

      <p style="font-size:12px;line-height:1.5;color:#AEAEB2;text-align:center;margin:24px 0 0">
        Dúvidas? suporte@proecg.com.br · © 2026 ProECG<br/>
        Ferramenta de apoio à decisão clínica — não substitui avaliação médica.
      </p>
    </div>
  </div>`;
}

function escapeHtml(s: string) {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}
