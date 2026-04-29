# ProECG

Plataforma SaaS mobile-first para médicos de emergência, UTI e UBS fazerem upload de fotos de ECG em papel e receberem laudo descritivo automático com hipóteses diagnósticas via IA. O sistema digitaliza a foto, mede intervalos, classifica padrões e monta o laudo — tudo em segundos.

## Stack

- **Frontend:** Next.js 14+ (App Router) + Tailwind CSS
- **API:** tRPC (rotas internas Next.js)
- **Auth:** Better Auth (email+senha + Google OAuth)
- **Database:** PostgreSQL via Neon
- **ORM:** Prisma
- **Pagamento:** Asaas (Pix + cartão, recorrência mensal)
- **Storage:** Cloudflare R2 (fotos de ECG)
- **Backend IA:** Python via Modal (serverless — roda sob demanda, sem servidor)
- **Deploy:** Vercel (frontend) + Modal (Python serverless)
- **Package Manager:** npm

## Comandos

```bash
# Frontend (Next.js)
npm run dev          # Dev server (localhost:3000)
npm run build        # Build produção
npm run lint         # ESLint
npm run typecheck    # Verificar tipos
npm run db:push      # Sync schema Prisma → Neon
npm run db:studio    # Prisma Studio

# Backend IA (Modal — serverless)
cd modal_functions
pip install modal
modal serve analyze.py    # Dev local (testa a function)
modal deploy analyze.py   # Deploy para produção (1 comando)
```

## Estrutura

```
proecg/
├── CLAUDE.md
├── package.json
├── .env                        # Variáveis de ambiente (gitignore)
│
├── src/                        # Frontend + API Next.js
│   ├── app/
│   │   ├── (marketing)/        # Landing page, pricing (público)
│   │   ├── (auth)/             # Login, cadastro, recuperar senha
│   │   ├── (dashboard)/        # Rotas autenticadas (upload, resultado, histórico)
│   │   └── api/
│   │       ├── trpc/           # Endpoint tRPC
│   │       └── webhooks/       # Webhook Asaas (confirmação pagamento)
│   ├── components/
│   │   ├── ui/                 # Componentes base (shadcn)
│   │   └── ecg/               # Componentes específicos ECG (upload, laudo, histórico)
│   ├── server/
│   │   ├── trpc/              # Routers tRPC (user, subscription, ecg, report)
│   │   └── auth.ts            # Config Better Auth
│   ├── lib/
│   │   ├── trpc.ts            # Cliente tRPC
│   │   ├── auth-client.ts     # Cliente Better Auth
│   │   ├── asaas.ts           # Client Asaas API
│   │   └── r2.ts              # Client Cloudflare R2
│   └── prisma/
│       └── schema.prisma
│
├── modal_functions/             # Backend Python (serverless via Modal)
│   ├── analyze.py              # Modal function: pipeline completo
│   ├── requirements.txt
│   ├── config.py               # Constantes, paths, limiares
│   ├── models/                 # Pesos dos modelos treinados
│   │   ├── digitizer/          # Pesos Dotter + Leader (UNet treinados por nós)
│   │   └── classifier/         # CNN classificação (.pth)
│   ├── pipeline/
│   │   ├── digitize.py         # Foto → sinal digital (12 derivações)
│   │   ├── measure.py          # Sinal → medições (FC, PR, QRS, QT, QTc, eixo)
│   │   ├── rules.py            # Regras clínicas (critérios numéricos)
│   │   ├── classify.py         # CNN classificação de padrões
│   │   ├── report.py           # Combina tudo → laudo descritivo (template)
│   │   └── orchestrator.py     # Pipeline completo (chama tudo em ordem)
│   └── tests/
│       ├── test_digitize.py
│       ├── test_measure.py
│       ├── test_rules.py
│       └── test_pipeline.py
│
├── training/                   # Scripts de treino (rodam no Colab, NÃO em produção)
│   ├── generate_synthetic.py   # Gerar imagens sintéticas de ECG em papel virtual
│   ├── train_dotter.py         # Treinar UNet Dotter (detecção de grid)
│   ├── train_leader.py         # Treinar UNet Leader (segmentação de leads)
│   ├── train_classifier.py     # Treinar CNN de classificação
│   ├── evaluate.py             # Avaliar acurácia por diagnóstico
│   └── notebooks/
│       ├── 01_explorar_dados.ipynb
│       ├── 02_gerar_sinteticos.ipynb
│       └── 03_treinar_modelos.ipynb
│
└── docs/
    ├── REGRAS_CLINICAS.md      # Critérios diagnósticos escritos pelo cardiologista
    ├── DIAGNOSTICOS.md         # Lista dos ~30 diagnósticos e status (ativo/validação)
    ├── PROECG_MAPA_COMPLETO_DO_PROJETO.md  # Mapa detalhado de todas as etapas
    ├── ANALISE_PAPER_PMCARDIO.md           # Análise técnica do paper de referência
    ├── ARQUITETURA.md          # Decisões técnicas
    └── API_IA.md               # Contrato da Modal function (request/response)
```

## Arquitetura: Next.js + Modal (serverless)

```
Médico (celular) → Next.js (Vercel)
                      ├── Auth, pagamento, CRUD → Neon (Postgres)
                      ├── Upload foto → Cloudflare R2
                      └── Chama Modal function (serverless, sob demanda)
                            ├── Digitaliza (pipeline próprio baseado em PMcardio)
                            ├── Mede (FC, PR, QRS, QT, QTc, eixo)
                            ├── Regras clínicas checam critérios
                            ├── CNN classifica padrões visuais
                            └── Template monta laudo → retorna JSON
```

O Next.js NUNCA roda código Python. Chama a Modal function via HTTP (POST com URL da imagem, resposta JSON com laudo). Modal roda sob demanda — sem servidor 24h. Cold start de ~5-10 seg na primeira chamada; depois fica rápido. Se no futuro o cold start incomodar, migrar para servidor dedicado (Railway/Render) é simples — o código Python é o mesmo.

## Entidades do Banco (Prisma)

- **User** — id, email, name, authProvider, createdAt
- **Subscription** — id, userId, plan (MONTHLY|SEMI|ANNUAL), status, asaasId, startsAt, endsAt
- **EcgAnalysis** — id, userId, imageUrl (R2), reportJson, createdAt, expiresAt (30 dias)

Relações: User 1:N Subscription, User 1:N EcgAnalysis.

## Planos e Preços

| Plano | Preço/mês | Duração |
|-------|-----------|---------|
| Mensal | R$ 267 | 1 mês |
| Semestral | R$ 227/mês | 6 meses |
| Anual | R$ 197/mês | 12 meses |

Pagamento via Asaas (Pix + cartão). Sem plano gratuito ou trial no MVP.

## Fluxo Principal

1. Médico acessa o site no celular → cadastro (email+senha ou Google)
2. Escolhe plano e paga (Asaas) → webhook confirma → acesso liberado
3. Dashboard: botão "Novo ECG" → tira foto com câmera do celular
4. Foto sobe para R2 → Next.js chama Modal function → laudo retorna em segundos
5. Resultado aparece na tela com: medições + achados + hipóteses diagnósticas
6. Pode: exportar PDF, enviar por WhatsApp/email, ver no histórico (30 dias)
7. Após 30 dias, análise expira (médico pode exportar antes)

## Motor de IA — Como Funciona

### Camada 1: Digitalização (pipeline próprio inspirado no PMcardio)
Pipeline de 6 módulos construído do zero, baseado na arquitetura documentada no paper "High Precision ECG Digitization Using AI" (Demolder et al., PMcardio/Powerful Medical):

1. **Pré-processamento:** Crop automático do papel + correção de perspectiva (OpenCV)
2. **Dotter (UNet):** Detecta interseções do grid milimetrado → máscara de keypoints
3. **Gridder:** Organiza keypoints em matriz, interpola gaps
4. **Undistortion:** Corrige distorção do papel quadrado a quadrado (homografia)
5. **Leader (UNet):** Segmenta traçados dos leads na imagem normalizada → máscara binária
6. **Extração de sinal:** Converte máscara em 12 arrays de µV × tempo

Ambos os UNets (Dotter e Leader) usam arquitetura UNet + ResBlocks + SiLU, treinados com BCEWithLogitsLoss + ADAM, LR 0.005, 300 epochs, patches 256×256px. Treinados primeiro com dataset sintético (~5.000 imagens) e depois refinados com fotos reais de ECGs brasileiros (~100-200 fotos anotadas).

Padrão brasileiro default: 25mm/s, 10mm/mV, layout 3×4+1 (DII longo), grid 1mm/5mm.

Paper de referência completo analisado em `docs/ANALISE_PAPER_PMCARDIO.md`.

### Camada 2: Medições (código matemático, sem IA)
Algoritmos calculam: FC, eixo elétrico, intervalo PR, duração QRS, QT, QTc (Bazett), análise do segmento ST. Código determinístico usando scipy/neurokit2.

### Camada 3: Classificação (híbrida: CNN + regras)

**Regras clínicas** (critérios numéricos definidos pelo cardiologista):
- Implementadas em `modal_functions/pipeline/rules.py`
- Critérios detalhados em `docs/REGRAS_CLINICAS.md`
- Exemplos: BAV 1º (PR>200ms), QT longo (QTc>470ms mulher), WPW (PR<120ms+delta)

**CNN** (padrões visuais treinados com PTB-XL + CODE-15%):
- Modelo ResNet-1D treinado offline no Colab
- Arquivo .pth salvo em `modal_functions/models/classifier/`
- Cobre: SCA supra, FA, flutter, TV, hipercalemia, pericardite, etc.

**Grad-CAM (heatmap de explicabilidade):**
- Para cada diagnóstico positivo: gera mapa de calor mostrando quais trechos do ECG influenciaram a decisão da IA
- Semelhante ao ECGxplain™ do PMcardio
- Exibe score de confiança por derivação

**Diagnósticos que não atingirem precisão aceitável ficam ocultos no MVP** (status "em validação" no `docs/DIAGNOSTICOS.md`). O cardiologista define o limiar.

### Camada 4: Laudo (template)
Template fixo combina medições + regras + CNN → texto descritivo. Sem LLM no MVP.

Exemplo de output:
```
Ritmo: sinusal. FC: 78 bpm. Eixo: +60°.
PR: 180ms. QRS: 88ms. QT: 380ms. QTc: 420ms (Bazett).

Achados: Supradesnivelamento do segmento ST em V1-V4.
Sugestivo de: Síndrome coronariana aguda com supra de ST em parede anterior.

⚠️ Ferramenta de apoio — correlacionar com dados clínicos.
```

## Diagnósticos Previstos (~30)

Lista completa em `docs/DIAGNOSTICOS.md`. Inclui:
SCA com/sem supra, BRE, BRD, FA, flutter, TSV, TV mono/poli, torsades, BAV (1º, 2º Mobitz 1/2, total), hipercalemia, hipocalemia, padrões equivalentes de infarto (Winter, supra aVR, infarto posterior, Sgarbossa, Wellens), taquicardia atrial, WPW, FA pré-excitada, bradicardia sinusal grave, Brugada, QT longo, QT curto, embolia pulmonar, pericardite, tamponamento.

Se nenhum diagnóstico for detectado → laudo puramente descritivo (medições + "sem alterações significativas").

## Dados do Paciente (MVP)

Anônimo. Sem nome, CPF ou dados identificáveis. Apenas a foto do ECG é enviada.
Dados clínicos (idade, sexo, sintomas) serão adicionados na Fase 2 (Motor 2), não no MVP.

## Segurança e Legal

- Disclaimer obrigatório em todo laudo: "Ferramenta de apoio à decisão clínica — não substitui avaliação médica"
- Termos de uso + política de privacidade com checkbox no cadastro
- Dados anônimos, sem PII de pacientes
- LGPD: dados em Neon (Postgres serverless), fotos em R2

## ⛔ NÃO Fazer

- NÃO modificar `schema.prisma` sem aprovar migração
- NÃO instalar dependências sem confirmar
- NÃO usar `any` no TypeScript
- NÃO criar API routes fora do tRPC (exceto webhook Asaas)
- NÃO colocar lógica de IA no Next.js — toda IA fica nas Modal functions (Python)
- NÃO armazenar dados identificáveis de pacientes (nome, CPF)
- NÃO apresentar diagnóstico como definitivo — sempre "sugestivo de", "compatível com"
- NÃO mostrar nível de confiança/probabilidade ao médico
- NÃO rodar treinamento de modelo em produção — treino é offline no Colab

## Workflow

1. Pergunte se algo não estiver claro
2. Proponha plano ANTES de implementar
3. Espere aprovação antes de executar
4. Rode `npm run typecheck` após mudanças no Next.js
5. Teste cada feature antes de prosseguir
6. Commits frequentes e descritivos

## Verificação

```bash
npm run build && npm run typecheck && npm run lint
```

## Links e Referências

- Modal: https://modal.com/docs
- Better Auth: https://www.better-auth.com/docs
- tRPC: https://trpc.io/docs
- Prisma: https://www.prisma.io/docs
- Asaas API: https://docs.asaas.com
- Cloudflare R2: https://developers.cloudflare.com/r2
- PTB-XL: https://physionet.org/content/ptb-xl/1.0.3/
- CODE-15%: https://zenodo.org/records/4916206
- PM-ECG-ID (benchmark): https://doi.org/10.5281/zenodo.13617673
- Paper PMcardio digitização: https://doi.org/10.1101/2024.08.31.24312876
- ECG-Image-Kit (geração sintética): https://github.com/alphanumericslab/ecg-image-kit

## Roadmap

- **MVP (atual):** Motor 1 — foto → laudo descritivo com medições e hipóteses (template)
- **Fase 2:** Motor 2 — dados clínicos do paciente + conduta terapêutica via LLM (Claude API)
- **Fase 3:** Laudo descritivo via LLM (substituir templates por texto fluente)
- **Futuro:** App mobile nativo, retreino com dados reais, registro Anvisa
