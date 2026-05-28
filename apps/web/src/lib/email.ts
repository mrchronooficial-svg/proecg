import { env } from "@proecg/env/server";
import { Resend } from "resend";

const resend = env.RESEND_API_KEY ? new Resend(env.RESEND_API_KEY) : null;

const ACCENT = "#5B65DC";
const TEXT = "#122056";
const MUTED = "#4A5078";
const BG = "#FAFAFD";
const CARD_BG = "#EEEFFD";
const BORDER = "#E2E4F0";

const FROM = "ProECG <noreply@proecg.com.br>";
const APP_URL = "https://proecg-web.vercel.app";

export const PROVISIONAL_PASSWORD = "Abcd@1234";

interface ConfirmationInput {
  to: string;
  name: string;
  planName: string;
  totalLabel: string;
  nextBilling: string;
  authProvider: "email" | "google";
  /** Provisória — só usada quando authProvider === "email". */
  provisionalPassword?: string;
  /** Link com token único pra troca de senha (somente authProvider === "email"). */
  resetPasswordUrl?: string;
}

export async function sendPaymentConfirmationEmail(input: ConfirmationInput) {
  const html = confirmationHtml(input);
  const subject = "Bem-vindo ao ProECG! Seu acesso está ativo";

  if (!resend) {
    console.log(`[DEV] Email de confirmação para ${input.to}: ${subject}`);
    return;
  }
  await resend.emails.send({
    from: FROM,
    to: input.to,
    subject,
    html,
  });
}

function confirmationHtml(input: ConfirmationInput) {
  const {
    to,
    name,
    planName,
    totalLabel,
    nextBilling,
    authProvider,
    provisionalPassword,
    resetPasswordUrl,
  } = input;

  const dashboardUrl = `${APP_URL}/dashboard`;
  const accessBlock =
    authProvider === "email"
      ? emailAccessBlock(to, provisionalPassword ?? "", resetPasswordUrl ?? "")
      : googleAccessBlock(to);

  return `<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>Bem-vindo ao ProECG</title>
</head>
<body style="margin:0;padding:0;background:${BG};font-family:'Inter','DM Sans',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;color:${TEXT}">
  <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background:${BG};padding:32px 16px">
    <tr><td align="center">
      <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="max-width:520px">
        <tr><td align="center" style="padding-bottom:24px">
          <div style="font-size:24px;font-weight:800;letter-spacing:-0.02em;color:${TEXT}">
            Pro<span style="color:${ACCENT}">ECG</span>
          </div>
        </td></tr>

        <tr><td style="background:#FFFFFF;border:1px solid ${BORDER};border-radius:16px;padding:32px 28px">
          <h1 style="font-size:22px;font-weight:800;margin:0 0 8px;color:${TEXT};letter-spacing:-0.01em">
            Olá, ${escapeHtml(name)}! 🎉
          </h1>
          <p style="font-size:15px;line-height:1.6;color:${MUTED};margin:0 0 20px">
            Seu plano <strong style="color:${TEXT}">${escapeHtml(planName)}</strong> foi ativado com sucesso.
          </p>

          <div style="background:${CARD_BG};border-radius:12px;padding:18px 20px;margin-bottom:24px">
            <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="font-size:14px;color:${TEXT}">
              <tr>
                <td style="padding:4px 0;color:${MUTED}">Plano</td>
                <td style="text-align:right;font-weight:600">${escapeHtml(planName)}</td>
              </tr>
              <tr>
                <td style="padding:4px 0;color:${MUTED}">Valor cobrado</td>
                <td style="text-align:right;font-weight:600">${escapeHtml(totalLabel)}</td>
              </tr>
              <tr>
                <td style="padding:4px 0;color:${MUTED}">Próxima cobrança</td>
                <td style="text-align:right;font-weight:600">${escapeHtml(nextBilling)}</td>
              </tr>
            </table>
          </div>

          ${accessBlock}

          <a href="${escapeAttr(dashboardUrl)}" style="display:block;background:${ACCENT};color:#FFFFFF;text-decoration:none;text-align:center;font-weight:600;padding:14px 24px;border-radius:12px;font-size:15px;margin-top:24px">
            Acessar o ProECG
          </a>
        </td></tr>

        <tr><td align="center" style="padding-top:24px">
          <p style="font-size:13px;line-height:1.6;color:#7A7F99;margin:0 0 8px">
            Dúvidas? <a href="mailto:suporte@proecg.com.br" style="color:${ACCENT};text-decoration:none">suporte@proecg.com.br</a>
          </p>
          <p style="font-size:12px;line-height:1.6;color:#AEAEB2;margin:0 0 8px">
            Ferramenta de apoio à decisão clínica — não substitui avaliação médica.
          </p>
          <p style="font-size:12px;line-height:1.6;color:#AEAEB2;margin:0">
            © 2026 ProECG. Todos os direitos reservados.
          </p>
        </td></tr>
      </table>
    </td></tr>
  </table>
</body>
</html>`;
}

function emailAccessBlock(
  email: string,
  provisional: string,
  resetUrl: string,
) {
  return `
    <div style="border-top:1px solid ${BORDER};padding-top:20px;margin-top:4px">
      <p style="font-size:15px;font-weight:600;color:${TEXT};margin:0 0 12px">
        Seus dados de acesso:
      </p>
      <div style="background:#F4F5FB;border:1px solid ${BORDER};border-radius:12px;padding:14px 18px;font-size:14px;color:${TEXT};margin-bottom:14px">
        <div style="padding:4px 0"><span style="color:${MUTED}">Email:</span> <strong>${escapeHtml(email)}</strong></div>
        <div style="padding:4px 0"><span style="color:${MUTED}">Senha provisória:</span> <strong style="font-family:'SF Mono',Menlo,Consolas,monospace">${escapeHtml(provisional)}</strong></div>
      </div>
      <p style="font-size:13px;line-height:1.5;color:#B7791F;background:#FFF8E6;border-radius:10px;padding:10px 14px;margin:0 0 14px">
        ⚠️ Por segurança, recomendamos alterar sua senha no primeiro acesso.
      </p>
      <a href="${escapeAttr(resetUrl)}" style="display:block;background:#FFFFFF;color:${ACCENT};border:1px solid ${ACCENT};text-decoration:none;text-align:center;font-weight:600;padding:12px 24px;border-radius:12px;font-size:14px">
        Alterar minha senha
      </a>
    </div>`;
}

function googleAccessBlock(email: string) {
  return `
    <div style="border-top:1px solid ${BORDER};padding-top:20px;margin-top:4px">
      <p style="font-size:15px;font-weight:600;color:${TEXT};margin:0 0 8px">
        Como acessar:
      </p>
      <p style="font-size:14px;line-height:1.6;color:${MUTED};margin:0">
        Você pode acessar usando sua conta Google
        (<strong style="color:${TEXT}">${escapeHtml(email)}</strong>).
      </p>
    </div>`;
}

function escapeHtml(s: string) {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function escapeAttr(s: string) {
  return escapeHtml(s);
}
