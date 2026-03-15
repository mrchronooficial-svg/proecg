# Arquitetura — ProECG

## Visão Geral

```
┌──────────────┐     ┌────────────────────┐     ┌──────────────┐
│   Médico     │────▶│  Next.js (Vercel)   │────▶│ Neon Postgres│
│  (celular)   │     │  Frontend + API     │     │  (users, sub,│
└──────────────┘     │  tRPC, Auth, CRUD   │     │   analyses)  │
                     └────────┬───────────┘     └──────────────┘
                              │
                     Upload foto ECG
                              │
                     ┌────────▼───────────┐     ┌──────────────┐
                     │  Cloudflare R2     │     │ Modal        │
                     │  (storage fotos)   │     │ (serverless) │
                     └────────────────────┘     │              │
                              │                 │ Digitalizar  │
                     URL da foto ───────────────▶ Medir        │
                                                │ Classificar  │
                              ┌─────────────────│ Montar laudo │
                              │                 └──────────────┘
                     JSON com laudo
                              │
                     ┌────────▼───────────┐
                     │  Tela de resultado  │
                     │  (médico vê laudo)  │
                     └────────────────────┘
```

## Decisões Arquiteturais

### ADR-001: Python como serverless function (Modal)
- **Contexto:** IA de ECG requer PyTorch, OpenCV, scipy — bibliotecas Python pesadas que não rodam em Vercel. Mas manter um servidor Python 24h é complexidade desnecessária no MVP.
- **Decisão:** Frontend e CRUD em Next.js (Vercel). Pipeline de IA em Python via Modal (serverless, roda sob demanda).
- **Consequência:** Um projeto só para gerenciar (Next.js). Python deploya com 1 comando (`modal deploy`). Paga só quando usa. Cold start de ~5-10 seg na primeira chamada. Se cold start incomodar no futuro, migrar para servidor dedicado (Railway/Render) é simples — o código Python é o mesmo.

### ADR-002: Cloudflare R2 para storage
- **Contexto:** Fotos de ECG (~1-5 MB cada) precisam de object storage. S3 cobra egress, R2 não.
- **Decisão:** Cloudflare R2 com presigned URLs para upload direto do browser.
- **Consequência:** Zero custo de egress. Upload rápido via CDN global.

### ADR-003: Asaas para pagamentos
- **Contexto:** Precisa de Pix + cartão com recorrência mensal. Público brasileiro.
- **Decisão:** Asaas (API brasileira, suporte a Pix e cartão, webhook de confirmação).
- **Consequência:** Webhook POST para /api/webhooks/asaas confirma pagamento → libera acesso.

### ADR-004: Neon (Postgres serverless) para banco
- **Contexto:** Rafael já usa Neon em outros projetos. Dados são leves (users, subscriptions, analysis metadata).
- **Decisão:** Neon com Prisma ORM.
- **Consequência:** Serverless, escala automaticamente. Branch de dev/prod disponível.

### ADR-005: Modelo híbrido (CNN + regras) para classificação
- **Contexto:** ~30 diagnósticos com diferentes perfis de dados. Alguns têm critérios numéricos claros, outros são padrões visuais.
- **Decisão:** CNN para padrões visuais + regras clínicas para critérios objetivos. Resultados combinados.
- **Consequência:** Mais robusto que CNN pura. Regras nunca "alucinam". CNN complementa onde regras não alcançam.

### ADR-006: Templates fixos para laudo no MVP
- **Contexto:** LLM gera texto mais fluente, mas adiciona custo e risco de alucinação.
- **Decisão:** MVP usa templates fixos. Fase 2 migra para LLM (Claude API).
- **Consequência:** Laudo mais "robótico" mas previsível e gratuito. Migração para LLM é simples (trocar 1 função).

## Modal Function — Contrato

### analyze(image_url) → JSON

**Request:**
```json
{
  "image_url": "https://r2.proecg.com/ecgs/abc123.jpg"
}
```

**Response:**
```json
{
  "success": true,
  "measurements": {
    "heart_rate": 78,
    "axis": 60,
    "pr_interval": 180,
    "qrs_duration": 88,
    "qt_interval": 380,
    "qtc_bazett": 420,
    "rhythm": "sinusal"
  },
  "findings": [
    {
      "code": "st_elevation_anterior",
      "description": "Supradesnivelamento do segmento ST em V1-V4",
      "source": "cnn",
      "leads_affected": ["V1", "V2", "V3", "V4"]
    }
  ],
  "diagnoses": [
    {
      "code": "sca_com_supra",
      "description": "Sugestivo de síndrome coronariana aguda com supra de ST em parede anterior",
      "source": "cnn+rules"
    }
  ],
  "report_text": "Ritmo: sinusal. FC: 78 bpm. Eixo: +60°.\nPR: 180ms. QRS: 88ms. QT: 380ms. QTc: 420ms (Bazett).\n\nAchados: Supradesnivelamento do segmento ST em V1-V4.\nSugestivo de: Síndrome coronariana aguda com supra de ST em parede anterior.\n\n⚠️ Ferramenta de apoio — correlacionar com dados clínicos.",
  "processing_time_ms": 1850
}
```

## Fluxo de Upload

1. Médico tira foto → browser obtém presigned URL do R2 via tRPC
2. Browser faz PUT direto no R2 (upload direto, não passa pelo Next.js)
3. Next.js recebe confirmação → chama Modal function com a URL da imagem
4. Modal baixa imagem do R2, processa, retorna JSON
5. Next.js salva resultado no Neon (EcgAnalysis) e mostra na tela

## Segurança

- Modal function autenticada via token (apenas Next.js pode chamar)
- Presigned URLs do R2 expiram em 5 minutos
- Fotos expiram em 30 dias (lifecycle rule no R2)
- Sem dados identificáveis de pacientes em nenhum ponto
