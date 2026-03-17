# CLAUDE_MOTOR2.md — Motor 2: Interpretação de ECG via Claude Vision

> **Este documento é a especificação completa para implementar o Motor 2 do ProECG.**
> O Motor 2 usa a API Claude Vision para interpretar a imagem do ECG DIRETAMENTE, sem digitalização.
> É o caminho mais rápido para um produto funcional em produção.
>
> Leia TUDO antes de implementar. Este é o motor PRINCIPAL do MVP.

---

## ÍNDICE

1. [Contexto e Decisão Estratégica](#1-contexto)
2. [Arquitetura do Motor 2](#2-arquitetura)
3. [Fluxo Completo](#3-fluxo-completo)
4. [O Prompt — Engenharia de Prompt para ECG](#4-o-prompt)
5. [Parsing da Resposta — Claude → JSON Estruturado](#5-parsing-da-resposta)
6. [Integração com o Pipeline Existente](#6-integração)
7. [Contrato JSON de Saída](#7-contrato-json)
8. [Tratamento de Erros e Edge Cases](#8-erros-e-edge-cases)
9. [Custo e Otimização](#9-custo)
10. [Fases de Implementação](#10-fases)
11. [Regras Clínicas do Prompt](#11-regras-clínicas)
12. [Regras Invioláveis](#12-regras-invioláveis)

---

## 1. CONTEXTO

### Por que Motor 2

O Motor 1 (digitalização + pipeline numérico) está implementado mas a qualidade da digitalização ainda não é clínica — o sinal extraído tem artefatos de escada/plateau, e as medições falham (ritmo indeterminado, sem detecção de achados).

O Motor 2 contorna o problema inteiro: envia a foto do ECG diretamente para a API Claude Vision, que interpreta a imagem como um cardiologista faria — reconhecendo padrões visuais, estimando medições, e gerando o laudo.

### Relação Motor 1 × Motor 2

```
MOTOR 1 (Pipeline numérico) — em desenvolvimento, melhora em paralelo
  Foto → digitalizar → sinal → medir → regras/CNN → laudo
  Custo: R$ 0 por ECG
  Status: NÃO funciona para produção ainda

MOTOR 2 (Claude Vision) — implementar AGORA como motor principal do MVP
  Foto → Claude Vision API → laudo
  Custo: ~R$ 0,15-0,50 por ECG
  Status: funciona HOJE

ESTRATÉGIA:
  MVP lança com Motor 2
  Motor 1 melhora em paralelo
  Quando Motor 1 estiver bom: usa Motor 1 como primário, Motor 2 como fallback
```

---

## 2. ARQUITETURA

### Visão geral

```
Médico (celular)
    │
    ▼
  Tira foto do ECG
    │
    ▼
  [Crop Tool] ── médico ajusta cantos (CLAUDE_CROP_TOOL.md)
    │
    ▼
  Upload foto → Cloudflare R2
    │
    ▼
  Next.js chama Modal function
    │
    ▼
  Modal function:
    │
    ├──→ [Motor 2] Envia foto para Claude Vision API
    │         │
    │         ▼
    │    Claude analisa imagem, retorna JSON estruturado
    │         │
    │         ▼
    │    Parse JSON → measurements, findings, diagnoses, report_text
    │
    ▼
  Retorna JSON para frontend → médico vê laudo
```

### O que MUDA vs arquitetura atual

| Componente | Antes | Depois |
|-----------|-------|--------|
| `orchestrator.py` | Chama digitize → measure → rules → classify → report | Chama `motor2_analyze()` → retorna laudo direto |
| `digitize/` | Obrigatório | NÃO USADO pelo Motor 2 |
| `measure.py` | Obrigatório | NÃO USADO pelo Motor 2 |
| `rules.py` | Obrigatório | Regras estão DENTRO do prompt do Claude |
| `classify.py` | Obrigatório | NÃO USADO pelo Motor 2 |
| `report.py` | Obrigatório | NÃO USADO (Claude gera o texto direto) |
| **Novo: `motor2.py`** | Não existia | Envia foto + prompt → recebe JSON |

### Dependência

- **Anthropic Python SDK:** `pip install anthropic`
- **API Key:** variável de ambiente `ANTHROPIC_API_KEY`
- **Modelo:** `claude-sonnet-4-20250514` (melhor custo/benefício para visão)

---

## 3. FLUXO COMPLETO

### Passo a passo técnico

```python
# Pseudocódigo do Motor 2

def motor2_analyze(image: PIL.Image) -> dict:
    """Envia foto de ECG para Claude Vision e retorna laudo estruturado."""

    # 1. Converter imagem para base64
    image_b64 = pil_to_base64(image)

    # 2. Montar mensagem com imagem + prompt de interpretação
    message = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 4096,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": ECG_INTERPRETATION_PROMPT,  # Ver seção 4
                    },
                ],
            }
        ],
    }

    # 3. Chamar API
    response = anthropic_client.messages.create(**message)

    # 4. Parsear resposta JSON
    result = parse_ecg_response(response.content[0].text)

    # 5. Retornar no formato do contrato
    return result
```

---

## 4. O PROMPT — Engenharia de Prompt para ECG

### Este é o componente mais crítico do Motor 2.

O prompt deve fazer Claude:
1. Analisar a imagem como cardiologista experiente
2. Retornar dados ESTRUTURADOS (JSON), não texto livre
3. Seguir terminologia e padrões da SBC
4. Usar linguagem sugestiva (nunca afirmativa)
5. Incluir disclaimer

### Prompt completo

```python
ECG_INTERPRETATION_PROMPT = """Você é um cardiologista especialista em eletrocardiografia com 20 anos de experiência em emergência e UTI no Brasil. Analise esta imagem de ECG e retorne EXCLUSIVAMENTE um JSON válido (sem markdown, sem ```json, sem texto antes ou depois).

## Instruções de Análise

0. ANTES DE TUDO: verifique se a imagem está na orientação correta. ECGs podem ser fotografados de cabeça para baixo ou rotacionados 90°/180°. Sinais de orientação incorreta: rótulos das derivações (I, II, III, aVR, V1...) invertidos ou de ponta-cabeça, texto manuscrito de cabeça para baixo, layout que não faz sentido anatômico (ex: aVR positivo quando deveria ser negativo). Se a imagem estiver invertida, CORRIJA MENTALMENTE a orientação antes de interpretar. Indique no campo technical_params se a imagem estava rotacionada.
1. Identifique os parâmetros técnicos do ECG (velocidade do papel, ganho, layout)
2. Analise sistematicamente: ritmo, frequência, eixo, intervalos, morfologia de cada derivação
3. Identifique TODOS os achados anormais
4. Formule hipóteses diagnósticas baseadas nos achados
5. Gere um laudo descritivo seguindo padrão SBC

## Regras Clínicas Obrigatórias

### Ritmo
- Sinusal: onda P positiva em DII, negativa em aVR, antes de cada QRS, com intervalo PR constante
- Fibrilação atrial: ausência de ondas P, intervalo RR irregular, linha de base irregular
- Flutter atrial: ondas F (dentes de serra) em DII/DIII/aVF, frequência atrial ~300 bpm
- Se não conseguir determinar o ritmo com certeza: "ritmo indeterminado"

### Frequência Cardíaca
- Contar quadrados grandes entre dois R-R: FC = 300 / nº quadrados grandes
- Se ritmo irregular: estimar FC média no DII longo

### Eixo Elétrico
- Normal: -30° a +90° (QRS positivo em DI e aVF)
- Desvio esquerdo: < -30° (QRS negativo em DII e aVF)
- Desvio direito: > +90° (QRS negativo em DI)

### Intervalos (a 25 mm/s: 1mm = 40ms, 1 quadrado grande = 200ms)
- PR normal: 120-200 ms. >200ms = BAV 1º grau. <120ms + delta = WPW
- QRS normal: <120 ms. ≥120ms = bloqueio de ramo
- QT: medir do início do QRS ao fim da T. Corrigir: QTc = QT / √(RR em segundos)
- QTc normal: <450ms (homem), <470ms (mulher)

### Supradesnivelamento de ST (CRÍTICO — não perder)
- Elevação ≥1mm (0.1mV) em derivações dos membros
- Elevação ≥2mm (0.2mV) em V1-V3, ≥1mm em V4-V6
- Se presente: identificar parede (anterior V1-V4, lateral V5-V6/DI/aVL, inferior DII/DIII/aVF)
- Procurar imagem em espelho (infra recíproco)

### Supra de ST na presença de BRD ou BRE (MUITO CRÍTICO — errar para o lado da cautela)
- BRD NÃO mascara supra de ST. Se houver BRD + supra de ST concordante com a deflexão principal do QRS → IAMCSST até provar o contrário. NÃO atribuir o supra apenas a "alterações secundárias ao BRD".
- BRE pode mascarar supra. Aplicar critérios de Sgarbossa/Smith modificados:
  - Supra ≥1mm concordante com QRS = altamente sugestivo de IAM
  - Supra ≥5mm discordante com QRS = sugestivo
  - Razão ST/S > -0.25 em qualquer derivação = sugestivo (critério de Smith)
- NA DÚVIDA entre "alteração secundária" e "supra real": SEMPRE relatar o supra e sugerir correlação clínica. É preferível um falso positivo (médico investiga e descarta) do que um falso negativo (infarto não detectado).

### REGRA DE OURO DE SENSIBILIDADE
- Para achados de EMERGÊNCIA (supra de ST, TV, FV, BAV total, hipercalemia grave): ERRAR PARA O LADO DA SENSIBILIDADE. É melhor reportar um achado duvidoso com linguagem de cautela ("Não é possível excluir supra de ST em V2-V4 — correlacionar com clínica") do que omitir um infarto.
- Para achados de ROTINA (eixo borderline, PR limítrofe): pode ser mais específico e não reportar se duvidoso.

### Infradesnivelamento de ST
- Depressão ≥0.5mm (0.05mV) é significativa
- Pode indicar isquemia, efeito digitálico, ou alteração recíproca

### Bloqueios de Ramo
- BRD: QRS ≥120ms + rSR' em V1 + S empastada em DI/V6
- BRE: QRS ≥120ms + ausência de Q em DI/V5/V6 + R monofásica alargada

### Sobrecarga
- SVE: Sokolow (S em V1 + R em V5/V6 ≥35mm), Cornell, ou padrão de strain
- SVD: R>S em V1, desvio do eixo para direita, onda P pulmonale

### Ondas Q Patológicas
- Duração ≥40ms ou profundidade ≥25% da onda R na mesma derivação
- Localizar parede afetada

## Formato de Resposta — JSON OBRIGATÓRIO

Retorne EXATAMENTE este JSON (sem texto adicional):

{
  "technical_params": {
    "paper_speed_mm_s": 25,
    "gain_mm_mv": 10,
    "layout": "3x4_dii_longo",
    "quality": "boa|regular|ruim",
    "orientation": "normal|rotacionado_180|rotacionado_90|incerto"
  },
  "measurements": {
    "heart_rate": <número inteiro em bpm | null se indeterminado>,
    "heart_rate_unit": "bpm",
    "rhythm": "<sinusal|fibrilação atrial|flutter atrial|taquicardia ventricular|ritmo indeterminado|outro>",
    "rhythm_detail": "<descrição breve do ritmo>",
    "axis": <número inteiro em graus | null>,
    "axis_classification": "<normal|desvio esquerdo|desvio direito|indeterminado>",
    "pr_interval": <número inteiro em ms | null>,
    "pr_unit": "ms",
    "qrs_duration": <número inteiro em ms | null>,
    "qrs_unit": "ms",
    "qt_interval": <número inteiro em ms | null>,
    "qt_unit": "ms",
    "qtc_bazett": <número inteiro em ms | null>,
    "qtc_unit": "ms"
  },
  "findings": [
    {
      "code": "<código_snake_case>",
      "description": "<descrição em português>",
      "leads_affected": ["<derivações>"],
      "severity": "<normal|atenção|urgente|emergência>"
    }
  ],
  "diagnoses": [
    {
      "code": "<código_snake_case>",
      "description": "<Sugestivo de ... | Compatível com ...>",
      "category": "<isquemia|arritmia|bloqueio|sobrecarga|distúrbio eletrolítico|outro>",
      "urgency": "<rotina|urgente|emergência>"
    }
  ],
  "report_text": "<Laudo descritivo completo em português, padrão SBC. Começar com ritmo e FC. Depois eixo e intervalos. Depois achados por derivação. Depois conclusão com hipóteses diagnósticas. Terminar SEMPRE com: ⚠️ Ferramenta de apoio à decisão clínica — não substitui avaliação médica.>"
}

## Regras de Linguagem do Laudo (report_text)

- NUNCA usar linguagem afirmativa: "O paciente TEM infarto" ❌
- SEMPRE usar linguagem sugestiva: "Achados sugestivos de...", "Compatível com..." ✅
- NUNCA mostrar probabilidade ou nível de confiança
- Se nenhum achado anormal: "Eletrocardiograma sem alterações significativas"
- Se não conseguir avaliar algo: dizer explicitamente (ex: "Derivação V4 parcialmente obstruída, avaliação limitada")
- SEMPRE terminar com o disclaimer: ⚠️ Ferramenta de apoio à decisão clínica — não substitui avaliação médica.

## IMPORTANTE

- Analise TODAS as 12 derivações sistematicamente
- NÃO invente dados — se não conseguir medir algo, retorne null
- Se a imagem estiver ruim (borrada, cortada, escura), indique no campo quality e nos achados
- Se a imagem estiver de cabeça para baixo ou rotacionada, corrija mentalmente ANTES de interpretar e indique no campo orientation
- Quando a qualidade da imagem limitar a análise de uma derivação específica, DIGA EXPLICITAMENTE (ex: "V4 com resolução insuficiente para avaliar ST com segurança — não é possível excluir supra")
- Em caso de dúvida sobre achado de emergência: REPORTE com linguagem de cautela, nunca omita
- O campo report_text deve ser o laudo COMPLETO que o médico vai ler
- Retorne APENAS o JSON, sem texto antes ou depois, sem blocos de código markdown
"""
```

---

## 5. PARSING DA RESPOSTA

### De texto Claude → JSON estruturado

```python
import json
import re

def parse_ecg_response(raw_text: str) -> dict:
    """Parseia a resposta do Claude Vision em JSON estruturado.

    Trata:
    - Resposta pode vir com ```json ... ``` (apesar de pedirmos que não)
    - Resposta pode ter texto antes/depois do JSON
    - JSON pode ter trailing commas ou outros problemas menores
    """
    text = raw_text.strip()

    # Remover blocos de código markdown se presentes
    text = re.sub(r'^```json\s*', '', text)
    text = re.sub(r'^```\s*', '', text)
    text = re.sub(r'\s*```$', '', text)

    # Tentar encontrar o JSON na resposta
    # Procurar de { até o último }
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        text = text[start:end + 1]

    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        # Tentar remover trailing commas
        text = re.sub(r',\s*}', '}', text)
        text = re.sub(r',\s*]', ']', text)
        try:
            result = json.loads(text)
        except json.JSONDecodeError:
            # Falha total no parse — retornar erro estruturado
            return {
                "success": False,
                "error": "Falha ao parsear resposta da IA",
                "raw_response": raw_text[:500],
                "measurements": {},
                "findings": [],
                "diagnoses": [],
                "report_text": "",
            }

    # Validar campos obrigatórios
    result.setdefault("measurements", {})
    result.setdefault("findings", [])
    result.setdefault("diagnoses", [])
    result.setdefault("report_text", "")

    return result
```

### Validação do JSON

Após parse, validar:

```python
def validate_ecg_result(result: dict) -> dict:
    """Valida e sanitiza o resultado do Claude Vision."""

    measurements = result.get("measurements", {})

    # FC: deve ser entre 20-300 ou null
    hr = measurements.get("heart_rate")
    if hr is not None and (hr < 20 or hr > 300):
        measurements["heart_rate"] = None

    # Intervalos: devem ser positivos e plausíveis
    for key, (min_val, max_val) in {
        "pr_interval": (50, 500),
        "qrs_duration": (40, 250),
        "qt_interval": (200, 700),
        "qtc_bazett": (200, 700),
    }.items():
        val = measurements.get(key)
        if val is not None and (val < min_val or val > max_val):
            measurements[key] = None

    # Eixo: -180 a 180
    axis = measurements.get("axis")
    if axis is not None and (axis < -180 or axis > 180):
        measurements["axis"] = None

    # Garantir que report_text termina com disclaimer
    report = result.get("report_text", "")
    disclaimer = "⚠️ Ferramenta de apoio à decisão clínica — não substitui avaliação médica."
    if disclaimer not in report:
        report = report.rstrip() + "\n\n" + disclaimer
        result["report_text"] = report

    # Remover scores/confiança dos findings (não mostrar ao médico)
    for finding in result.get("findings", []):
        finding.pop("score", None)
        finding.pop("confidence", None)

    result["measurements"] = measurements
    return result
```

---

## 6. INTEGRAÇÃO COM PIPELINE EXISTENTE

### Novo arquivo: `modal_functions/pipeline/motor2.py`

Este é o arquivo principal do Motor 2. Contém:
- `motor2_analyze(image: PIL.Image) → dict` — função principal
- O prompt de interpretação
- Parsing e validação
- Tratamento de erros

### Modificar `orchestrator.py`

```python
# orchestrator.py — MODIFICAR a função analyze()

def analyze(
    image_url: str,
    corners: dict | None = None,
    use_placeholder: bool = False,
    engine: str = "motor2",           # ← NOVO PARÂMETRO
) -> dict:
    """Pipeline completo de análise de ECG.

    Args:
        image_url: URL da imagem no R2.
        corners: 4 cantos para crop (opcional).
        use_placeholder: sinal sintético para teste.
        engine: "motor2" (Claude Vision) ou "motor1" (pipeline numérico).
    """
    start_time = time.perf_counter()

    try:
        # 1. Baixar imagem
        image = _download_image(image_url)

        # 1.5 Aplicar crop se corners fornecidos
        if corners is not None:
            image = _apply_perspective_crop(image, corners)

        # 2. Escolher motor
        if engine == "motor2":
            from .motor2 import motor2_analyze
            result = motor2_analyze(image)
        else:
            # Motor 1 — pipeline numérico (atual)
            signal_12lead = _digitize_ecg(image, use_placeholder=use_placeholder)
            measurements = measure_ecg(signal_12lead, fs=500)
            rule_findings = apply_clinical_rules(measurements)
            cnn_findings = classify_ecg(signal_12lead)
            report = generate_report(measurements, rule_findings, cnn_findings)
            result = {
                "measurements": _build_measurements_response(measurements),
                "findings": _strip_scores(report["findings"]),
                "diagnoses": report["diagnoses"],
                "report_text": report["report_text"],
            }

        elapsed_ms = int((time.perf_counter() - start_time) * 1000)

        response = _sanitize_for_json({
            "success": True,
            "engine": engine,
            **result,
            "processing_time_ms": elapsed_ms,
        })

        return response

    except Exception as e:
        elapsed_ms = int((time.perf_counter() - start_time) * 1000)
        return {
            "success": False,
            "error": f"Erro ao processar ECG: {type(e).__name__}: {str(e)}",
            "engine": engine,
            "measurements": {},
            "findings": [],
            "diagnoses": [],
            "report_text": "",
            "processing_time_ms": elapsed_ms,
        }
```

### Modificar `analyze.py` (Modal function)

```python
# analyze.py — aceitar engine como parâmetro

@app.function(...)
def analyze_ecg(
    image_url: str,
    corners: dict | None = None,
    engine: str = "motor2",
) -> dict:
    from pipeline.orchestrator import analyze
    return analyze(image_url, corners=corners, engine=engine)
```

### Variável de ambiente necessária

```bash
# .env
ANTHROPIC_API_KEY=sk-ant-...   # Chave da API Anthropic
```

No Modal:
```python
# analyze.py
import modal

app = modal.App("proecg")

@app.function(
    secrets=[modal.Secret.from_name("anthropic-key")],  # ou from_dotenv()
    ...
)
```

---

## 7. CONTRATO JSON DE SAÍDA

O Motor 2 DEVE retornar o MESMO formato que o pipeline atual. O frontend não precisa saber qual motor foi usado.

```json
{
  "success": true,
  "engine": "motor2",
  "measurements": {
    "heart_rate": 75,
    "heart_rate_unit": "bpm",
    "rhythm": "fibrilação atrial",
    "axis": 45,
    "axis_unit": "°",
    "pr_interval": null,
    "pr_unit": "ms",
    "qrs_duration": 88,
    "qrs_unit": "ms",
    "qt_interval": 380,
    "qt_unit": "ms",
    "qtc_bazett": 430,
    "qtc_unit": "ms"
  },
  "findings": [
    {
      "code": "atrial_fibrillation",
      "description": "Fibrilação atrial com resposta ventricular controlada",
      "leads_affected": ["II", "DII_longo"],
      "severity": "atenção"
    },
    {
      "code": "st_elevation_anterior",
      "description": "Supradesnivelamento do segmento ST em V2-V4",
      "leads_affected": ["V2", "V3", "V4"],
      "severity": "emergência"
    }
  ],
  "diagnoses": [
    {
      "code": "sca_com_supra",
      "description": "Sugestivo de Síndrome Coronariana Aguda com supradesnivelamento de ST em parede anterior",
      "category": "isquemia",
      "urgency": "emergência"
    },
    {
      "code": "fibrilacao_atrial",
      "description": "Compatível com fibrilação atrial",
      "category": "arritmia",
      "urgency": "urgente"
    }
  ],
  "report_text": "Ritmo de fibrilação atrial com resposta ventricular controlada. Frequência cardíaca média de aproximadamente 75 bpm.\n\nEixo elétrico no plano frontal: normal (~+45°).\n\nIntervalos: PR não mensurável (fibrilação atrial). QRS: ~88ms (complexos estreitos). QT: ~380ms. QTc (Bazett): ~430ms.\n\nAchados: Supradesnivelamento do segmento ST em V2-V4, mais pronunciado em V3, com morfologia de lesão aguda. Possível infradesnivelamento recíproco de ST em DII, DIII e aVF.\n\nHipótese diagnóstica: Achados sugestivos de Síndrome Coronariana Aguda com supradesnivelamento de ST (IAMCSST) em parede anterior, em paciente com fibrilação atrial.\n\n⚠️ Ferramenta de apoio à decisão clínica — não substitui avaliação médica.",
  "processing_time_ms": 3200
}
```

---

## 8. TRATAMENTO DE ERROS E EDGE CASES

### Erros da API

| Erro | Tratamento |
|------|-----------|
| API timeout (>30s) | Retry 1 vez. Se falhar: tentar Motor 1 como fallback |
| Rate limit (429) | Esperar e retry com exponential backoff |
| API key inválida | Erro fatal, logar, retornar erro ao frontend |
| Resposta não-JSON | Tentar extrair JSON da resposta. Se falhar: retry com prompt mais enfático |
| Imagem muito grande | Reduzir para max 2048px antes de enviar |

### Edge cases clínicos

| Caso | Tratamento no prompt |
|------|---------------------|
| ECG parcialmente obstruído (carimbo, dedo) | Claude deve indicar no campo quality e nos findings quais leads estão comprometidos |
| ECG muito escuro/borrado | quality = "ruim", achados limitados |
| ECG de 6 derivações (não 12) | Claude identifica e analisa o que tem |
| Foto de tela (ECG digital no monitor) | Claude lida normalmente |
| Não é um ECG (foto aleatória) | Claude deve retornar erro: "A imagem não parece ser um eletrocardiograma" |

### Fallback Motor 2 → Motor 1

```python
def analyze_with_fallback(image, corners=None):
    """Tenta Motor 2 primeiro. Se falhar, tenta Motor 1."""
    try:
        result = motor2_analyze(image)
        if result.get("success") is not False:
            return result
    except Exception as e:
        logger.warning("Motor 2 falhou: %s. Tentando Motor 1.", e)

    # Fallback para Motor 1
    try:
        signal = digitize_ecg(image)
        measurements = measure_ecg(signal, fs=500)
        # ... resto do pipeline
    except Exception as e:
        logger.error("Motor 1 também falhou: %s", e)
        return {"success": False, "error": str(e)}
```

---

## 9. CUSTO E OTIMIZAÇÃO

### Custo estimado por ECG

| Modelo Claude | Input (imagem ~1MB) | Output (~2000 tokens) | Custo total |
|--------------|--------------------|-----------------------|-------------|
| Sonnet 4 | ~$0.003 (input) | ~$0.03 (output) | **~$0.03-0.05** (~R$ 0.15-0.25) |
| Opus 4 | ~$0.015 (input) | ~$0.075 (output) | **~$0.09** (~R$ 0.45) |

**Recomendação:** Usar **Sonnet 4** (`claude-sonnet-4-20250514`). Custo 3× menor que Opus, qualidade suficiente para ECG.

### Conta de padaria

| Cenário | ECGs/dia | Custo/mês API | Receita/mês | Margem |
|---------|---------|---------------|-------------|--------|
| 1 médico, 10 ECGs/dia | 300 | ~R$ 50 | R$ 197 | R$ 147 (75%) |
| 1 médico, 30 ECGs/dia | 900 | ~R$ 150 | R$ 197 | R$ 47 (24%) |
| 10 médicos, 15 ECGs/dia | 4.500 | ~R$ 750 | R$ 1.970 | R$ 1.220 (62%) |

**A conta fecha em todos os cenários.** Mesmo no pior caso (30 ECGs/dia), ainda tem 24% de margem.

### Otimizações para reduzir custo

1. **Reduzir tamanho da imagem:** Enviar max 1500×1500px (suficiente para Claude ler ECG). Imagem de 6000px é desperdício de tokens de input.
2. **Cache de resultados:** Se o mesmo image_url for analisado 2× (ex: refresh da página), retornar resultado cacheado do banco.
3. **Compressão JPEG:** Antes de base64, comprimir para quality=85. ECG não precisa de qualidade fotográfica.

---

## 10. FASES DE IMPLEMENTAÇÃO

### Fase 1 — Motor 2 básico (funcional)

```
1.1 Criar modal_functions/pipeline/motor2.py com:
    - motor2_analyze(image) → dict
    - ECG_INTERPRETATION_PROMPT (copiar da seção 4)
    - parse_ecg_response()
    - validate_ecg_result()

1.2 Instalar anthropic SDK: adicionar "anthropic" ao requirements.txt

1.3 Configurar ANTHROPIC_API_KEY como secret no Modal

1.4 TESTAR isoladamente:
    - Carregar a foto teste_2.jpeg
    - Rodar motor2_analyze(image)
    - Verificar que retorna JSON válido com medições e laudo
    - Comparar com laudo do cardiologista
```

**Critério de sucesso:** Motor 2 retorna laudo clinicamente correto para a foto de teste.

### Fase 2 — Integrar no orchestrator

```
2.1 Modificar orchestrator.py para aceitar engine="motor2" (default)
2.2 Modificar analyze.py (Modal) para passar engine
2.3 TESTAR end-to-end:
    - Chamar analyze(image_url, engine="motor2")
    - Verificar JSON de resposta completo
    - Verificar que frontend exibe laudo corretamente
```

**Critério de sucesso:** Fluxo completo funciona: foto → API → laudo na tela.

### Fase 3 — Otimizações

```
3.1 Reduzir imagem antes de enviar (max 1500px)
3.2 Compressão JPEG (quality=85)
3.3 Cache de resultados no banco (EcgAnalysis)
3.4 Timeout e retry com backoff
3.5 Logging de custo por chamada
```

### Fase 4 — Fallback Motor 2 → Motor 1

```
4.1 Implementar analyze_with_fallback()
4.2 Se API Claude falhar (timeout, rate limit): tentar Motor 1
4.3 Adicionar campo "engine" na resposta para saber qual foi usado
```

---

## 11. REGRAS CLÍNICAS DO PROMPT

As regras clínicas que estão no `rules.py` (21 KB) devem estar refletidas no prompt. Claude já conhece todas essas regras (é conhecimento médico padrão), mas o prompt reforça as mais importantes para garantir que não perca nada.

### Diagnósticos que o prompt DEVE cobrir (mesma lista do DIAGNOSTICOS.md)

**Grupo A (critérios objetivos — Claude deve medir):**
- BAV 1º grau (PR > 200ms)
- BRD completo (QRS ≥ 120ms + rSR' V1)
- BRE completo (QRS ≥ 120ms + R monofásica em V5/V6)
- QT longo (QTc > 470ms mulher / > 450ms homem)
- QT curto (QTc < 340ms)
- Bradicardia sinusal grave (FC < 50 + sinusal)
- WPW (PR < 120ms + delta + QRS > 100ms)
- Sgarbossa em BRE (score ≥ 3)

**Grupo B (padrões visuais — Claude é excelente nisso):**
- SCA com supra de ST
- Fibrilação atrial
- Flutter atrial
- TV monomórfica
- BAV total
- Hipercalemia
- Pericardite aguda

**Grupo C (sutis — Claude deve tentar mas pode errar):**
- SCA sem supra
- Wellens
- Winter
- Brugada tipo 1
- Embolia pulmonar (S1Q3T3)

### Severidade e urgência

O prompt deve classificar achados em:
- **normal** — sem alteração
- **atenção** — alteração que requer acompanhamento (ex: BAV 1º)
- **urgente** — alteração que requer avaliação médica breve (ex: FA)
- **emergência** — alteração que pode necessitar intervenção imediata (ex: IAMCSST, TV)

---

## 12. REGRAS INVIOLÁVEIS

### NÃO FAZER

- **NÃO usar Opus** no MVP — Sonnet é suficiente e 3× mais barato
- **NÃO mostrar** ao médico que é IA/Claude gerando o laudo — a experiência é "ProECG"
- **NÃO remover** o Motor 1 — ele continua existindo para ser melhorado em paralelo
- **NÃO enviar** dados de paciente na chamada API (não tem mesmo, ECG é anônimo)
- **NÃO cachear** a API key no código — usar variável de ambiente / secret
- **NÃO fazer** mais de 1 retry em caso de falha (custo)
- **NÃO mostrar** nível de confiança ao médico
- **NÃO usar** linguagem afirmativa nos diagnósticos

### SEMPRE FAZER

- **SEMPRE** incluir disclaimer no laudo
- **SEMPRE** usar linguagem sugestiva ("Sugestivo de...", "Compatível com...")
- **SEMPRE** retornar JSON no formato do contrato (seção 7)
- **SEMPRE** reduzir imagem antes de enviar (economia de tokens)
- **SEMPRE** logar tempo e custo de cada chamada
- **SEMPRE** ter fallback se API falhar
- **SEMPRE** validar JSON de resposta (medições plausíveis, campos obrigatórios)
- **SEMPRE** sanitizar para JSON-safe (sem NaN, Inf, tipos numpy)

---

## APÊNDICE A — Exemplo de Chamada Completa

```python
import anthropic
import base64
import io
from PIL import Image

def motor2_analyze(image: Image.Image) -> dict:
    """Motor 2: interpreta ECG via Claude Vision API."""

    # 1. Reduzir imagem
    max_size = 1500
    w, h = image.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    # 2. Converter para base64 JPEG
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    image_b64 = base64.standard_b64encode(buffer.getvalue()).decode("utf-8")

    # 3. Chamar API
    client = anthropic.Anthropic()  # usa ANTHROPIC_API_KEY do env

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": ECG_INTERPRETATION_PROMPT,
                    },
                ],
            }
        ],
    )

    # 4. Parsear resposta
    raw_text = response.content[0].text
    result = parse_ecg_response(raw_text)
    result = validate_ecg_result(result)

    # 5. Adicionar metadata
    result["success"] = True
    result["engine"] = "motor2"
    result["api_tokens_used"] = {
        "input": response.usage.input_tokens,
        "output": response.usage.output_tokens,
    }

    return result
```

---

## APÊNDICE B — Dependências adicionais

Adicionar ao `modal_functions/requirements.txt`:

```
anthropic>=0.40.0
```

---

## APÊNDICE C — Secret no Modal

```python
# Configurar via Modal CLI:
# modal secret create anthropic-key ANTHROPIC_API_KEY=sk-ant-...

# Ou via dashboard Modal:
# https://modal.com/secrets → New Secret → anthropic-key

# No analyze.py:
@app.function(
    secrets=[modal.Secret.from_name("anthropic-key")],
    # ...
)
```

---

*Documento gerado em 17/03/2026. Versão 1.0.*
*Motor 2 = caminho mais rápido para produto em produção.*
*Motor 1 continua sendo melhorado em paralelo.*
