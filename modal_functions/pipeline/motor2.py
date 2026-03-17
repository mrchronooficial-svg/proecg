"""
Motor 2 — Interpretação de ECG via Claude Vision API.

Envia foto do ECG diretamente para Claude, que interpreta como cardiologista
e retorna JSON estruturado com medições, achados, diagnósticos e laudo.

Uso: result = motor2_analyze(image)
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
import time
from typing import Any

from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 4096
MAX_IMAGE_SIZE = 1500      # px (lado maior)
JPEG_QUALITY = 85
API_TIMEOUT = 30            # segundos
MAX_RETRIES = 1

# ---------------------------------------------------------------------------
# Prompt de interpretação (EXATAMENTE da seção 4 do CLAUDE_MOTOR2.md)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Parsing e validação
# ---------------------------------------------------------------------------

def parse_ecg_response(raw_text: str) -> dict:
    """Parseia a resposta do Claude Vision em JSON estruturado."""
    text = raw_text.strip()

    # Remover blocos de código markdown se presentes
    text = re.sub(r'^```json\s*', '', text)
    text = re.sub(r'^```\s*', '', text)
    text = re.sub(r'\s*```$', '', text)

    # Encontrar o JSON na resposta
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
            return {
                "success": False,
                "error": "Falha ao parsear resposta da IA",
                "raw_response": raw_text[:500],
                "measurements": {},
                "findings": [],
                "diagnoses": [],
                "report_text": "",
            }

    result.setdefault("measurements", {})
    result.setdefault("findings", [])
    result.setdefault("diagnoses", [])
    result.setdefault("report_text", "")

    return result


def validate_ecg_result(result: dict) -> dict:
    """Valida e sanitiza o resultado do Claude Vision."""
    measurements = result.get("measurements", {})

    # FC: 20-300 ou null
    hr = measurements.get("heart_rate")
    if hr is not None and (not isinstance(hr, (int, float)) or hr < 20 or hr > 300):
        measurements["heart_rate"] = None

    # Intervalos: positivos e plausíveis
    for key, (min_val, max_val) in {
        "pr_interval": (50, 500),
        "qrs_duration": (40, 250),
        "qt_interval": (200, 700),
        "qtc_bazett": (200, 700),
    }.items():
        val = measurements.get(key)
        if val is not None and (not isinstance(val, (int, float)) or val < min_val or val > max_val):
            measurements[key] = None

    # Eixo: -180 a 180
    axis = measurements.get("axis")
    if axis is not None and (not isinstance(axis, (int, float)) or axis < -180 or axis > 180):
        measurements["axis"] = None

    # Garantir units
    measurements.setdefault("heart_rate_unit", "bpm")
    measurements.setdefault("pr_unit", "ms")
    measurements.setdefault("qrs_unit", "ms")
    measurements.setdefault("qt_unit", "ms")
    measurements.setdefault("qtc_unit", "ms")
    measurements.setdefault("axis_unit", "°")

    # Disclaimer
    report = result.get("report_text", "")
    disclaimer = "⚠️ Ferramenta de apoio à decisão clínica — não substitui avaliação médica."
    if disclaimer not in report:
        report = report.rstrip() + "\n\n" + disclaimer
        result["report_text"] = report

    # Remover scores dos findings
    for finding in result.get("findings", []):
        finding.pop("score", None)
        finding.pop("confidence", None)

    result["measurements"] = measurements
    return result


# ---------------------------------------------------------------------------
# Preparação da imagem
# ---------------------------------------------------------------------------

def _prepare_image(image: Image.Image) -> str:
    """Reduz, comprime e converte imagem para base64 JPEG."""
    # Converter para RGB se necessário
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Reduzir para max MAX_IMAGE_SIZE px
    w, h = image.size
    if max(w, h) > MAX_IMAGE_SIZE:
        scale = MAX_IMAGE_SIZE / max(w, h)
        image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        logger.info("Imagem reduzida: %dx%d → %dx%d", w, h, image.size[0], image.size[1])

    # Comprimir para JPEG
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=JPEG_QUALITY)
    jpeg_bytes = buffer.getvalue()

    logger.info("Imagem preparada: %dx%d, %d KB JPEG",
                image.size[0], image.size[1], len(jpeg_bytes) // 1024)

    return base64.standard_b64encode(jpeg_bytes).decode("utf-8")


# ---------------------------------------------------------------------------
# Função principal
# ---------------------------------------------------------------------------

def motor2_analyze(image: Image.Image) -> dict:
    """Motor 2: interpreta ECG via Claude Vision API.

    Args:
        image: PIL Image do ECG (qualquer tamanho/orientação).

    Returns:
        dict com measurements, findings, diagnoses, report_text.

    Raises:
        RuntimeError se API falhar após retries.
    """
    import anthropic

    # 1. Preparar imagem
    image_b64 = _prepare_image(image)

    # 2. Criar client
    client = anthropic.Anthropic()  # usa ANTHROPIC_API_KEY do env

    # 3. Chamar API com retry
    response = None
    last_error = None
    
    for attempt in range(1 + MAX_RETRIES):
        try:
            t0 = time.perf_counter()

            response = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                timeout=API_TIMEOUT,
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

            elapsed = time.perf_counter() - t0
            logger.info("Claude API: %.1fs, input=%d tokens, output=%d tokens",
                        elapsed, response.usage.input_tokens, response.usage.output_tokens)

            break  # Sucesso

        except anthropic.APITimeoutError as e:
            last_error = e
            logger.warning("Claude API timeout (tentativa %d/%d): %s",
                           attempt + 1, 1 + MAX_RETRIES, e)
            if attempt < MAX_RETRIES:
                time.sleep(2)
                continue
        except anthropic.RateLimitError as e:
            last_error = e
            logger.warning("Claude API rate limit (tentativa %d/%d): %s",
                           attempt + 1, 1 + MAX_RETRIES, e)
            if attempt < MAX_RETRIES:
                time.sleep(5)
                continue
        except anthropic.APIError as e:
            last_error = e
            logger.error("Claude API error: %s", e)
            # Não faz retry para outros erros, mas também não dá break
            # Deixa o loop terminar naturalmente

    # Verificar se a chamada foi bem sucedida
    if response is None:
        error_msg = f"Claude API falhou após {1 + MAX_RETRIES} tentativas: {last_error}"
        logger.error(error_msg)
        return {
            "success": False,
            "error": error_msg,
            "measurements": {},
            "findings": [],
            "diagnoses": [],
            "report_text": "",
        }

    # 4. Parsear resposta
    raw_text = response.content[0].text
    result = parse_ecg_response(raw_text)

    if result.get("success") is False:
        logger.error("Falha no parse: %s", result.get("error"))
        return result

    # 5. Validar
    result = validate_ecg_result(result)

    # 6. Metadata
    result["success"] = True
    result["engine"] = "motor2"
    result["api_tokens"] = {
        "input": response.usage.input_tokens,
        "output": response.usage.output_tokens,
    }

    return result
