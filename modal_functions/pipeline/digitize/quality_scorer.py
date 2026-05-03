"""
Módulo 0 — Quality Scorer
=========================

Roda ANTES de qualquer modelo (Dotter/Leader). Decide se a foto tem
qualidade suficiente pra processamento. Se não, rejeita imediatamente
com mensagem amigável pro médico — economiza GPU e dá feedback rápido.

Checks (em ordem de prioridade):
  1. Resolução mínima (lado maior ≥ 1000 px)
  2. ECG detectado (grid periódico via autocorrelação de bordas)
  3. Foco (Laplacian variance ≥ threshold)
  4. Exposição (60 ≤ brilho médio ≤ 230)
  5. Proporção (1.5–3.0) — apenas warning, não rejeita

Score (peso → componente):
  0.2 — resolução      0.3 — ecg detectado     0.2 — foco
  0.2 — exposição       0.1 — proporção

Decisão final:
  • Qualquer check estrito (1–4) reprovado     → reject (score informativo)
  • Tudo passa, score > 0.7                    → accept clean
  • Tudo passa, score 0.5–0.7                  → accept com warning
  • Score < 0.5                                → reject (defesa em profundidade)
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Thresholds (calibrados pelos testes em 10 imagens; ver test_quality_scorer.py)
# ---------------------------------------------------------------------------

MIN_LONGER_SIDE_PX = 1000
MIN_PERIODIC_SCORE = 0.20      # autocorrelação min(h, v) — calibrado
MIN_LAPLACIAN_VARIANCE = 50.0  # variance — calibrado
MIN_BRIGHTNESS = 60.0
MAX_BRIGHTNESS = 230.0
# Aspect ratio: a foto crua do celular geralmente é 4:3 (1.33) ou 16:9 (1.78).
# O ECG em si é ~3:1, mas isso só vale APÓS o crop do papel. Aqui, aceitamos
# qualquer proporção razoável (quase-quadrado até muito comprido). Só fora
# desses limites = warning (provavelmente foto extremamente distorcida).
MIN_ASPECT_RATIO = 1.05
MAX_ASPECT_RATIO = 5.0

# Pesos do score
W_RESOLUTION = 0.20
W_ECG = 0.30
W_FOCUS = 0.20
W_EXPOSURE = 0.20
W_ASPECT = 0.10


# ---------------------------------------------------------------------------
# Detecção de grid periódico (FFT/autocorrelação)
# ---------------------------------------------------------------------------

def _periodicity_score(gray: np.ndarray, max_dim: int = 768) -> tuple[float, float]:
    """Quantifica periodicidade horizontal e vertical no mapa de bordas.

    Um grid de ECG tem linhas verticais e horizontais regulares. Projetar
    as bordas em cada eixo e autocorrelacionar revela picos no espaçamento
    do grid (1mm/5mm). Imagens não-ECG (fotos, texto, ruído) têm
    autocorrelação plana.

    Returns:
        (h_score, v_score) — pico de autocorrelação em cada eixo (0..1).
    """
    h, w = gray.shape
    # Resize pra acelerar (mantendo proporção)
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        gray = cv2.resize(
            gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
        )
        h, w = gray.shape

    # Mapa de bordas (Canny realça linhas do grid)
    edges = cv2.Canny(gray, 40, 120)

    # Projeções ortogonais
    col_proj = edges.sum(axis=0).astype(np.float32)  # detecta linhas verticais
    row_proj = edges.sum(axis=1).astype(np.float32)  # detecta linhas horizontais

    def _ac_peak(sig: np.ndarray) -> float:
        sig = sig - sig.mean()
        if sig.std() < 1e-6:
            return 0.0
        n = len(sig)
        # Autocorrelação normalizada
        ac = np.correlate(sig, sig, mode="full")[n - 1:]
        ac = ac / ac[0]
        # Ignora lag 0 e vizinhança imediata (DC residual)
        # Procura pico no range correspondente a 1mm–20mm — sem saber px/mm,
        # usa janela genérica: 5 px até n/4
        skip = max(5, n // 100)
        end = n // 3
        if end <= skip + 1:
            return 0.0
        peaks = ac[skip:end]
        return float(max(0.0, peaks.max()))

    return _ac_peak(col_proj), _ac_peak(row_proj)


# ---------------------------------------------------------------------------
# Função pública
# ---------------------------------------------------------------------------

def score_quality(image_bgr: np.ndarray) -> dict:
    """Avalia qualidade da foto e decide se processa ou rejeita.

    Args:
        image_bgr: imagem BGR (numpy array) carregada pelo cv2.

    Returns:
        dict com:
          accept: bool
          score: float (0..1)
          rejection_reason: str | None — mensagem pro médico
          warning: str | None — alerta sem bloquear
          details: dict por check com métrica e pass/fail

    A função NUNCA levanta exceção em caso de imagem inválida; sempre retorna
    um dict com `accept=False` e razão.
    """
    if image_bgr is None or image_bgr.size == 0:
        return {
            "accept": False,
            "score": 0.0,
            "rejection_reason": "Imagem inválida ou vazia.",
            "warning": None,
            "details": {},
        }

    h, w = image_bgr.shape[:2]
    if image_bgr.ndim == 2:
        gray = image_bgr
    else:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # ---------- 1. Resolução ----------
    longer = int(max(h, w))
    res_pass = longer >= MIN_LONGER_SIDE_PX
    resolution = {
        "width": int(w),
        "height": int(h),
        "longer_side": longer,
        "pass": bool(res_pass),
    }

    # ---------- 2. ECG detectado ----------
    # Pula FFT se resolução é absurdamente baixa (economia)
    if longer < 200:
        h_score, v_score = 0.0, 0.0
    else:
        h_score, v_score = _periodicity_score(gray)
    periodic_score = float(min(h_score, v_score))
    ecg_pass = periodic_score >= MIN_PERIODIC_SCORE
    ecg_detected = {
        "periodic_score": periodic_score,
        "h_score": float(h_score),
        "v_score": float(v_score),
        "pass": bool(ecg_pass),
    }

    # ---------- 3. Foco (Laplacian variance) ----------
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    focus_pass = lap_var >= MIN_LAPLACIAN_VARIANCE
    focus = {"laplacian_var": lap_var, "pass": bool(focus_pass)}

    # ---------- 4. Exposição ----------
    mean_b = float(gray.mean())
    bright_pass = MIN_BRIGHTNESS <= mean_b <= MAX_BRIGHTNESS
    exposure = {"mean_brightness": mean_b, "pass": bool(bright_pass)}

    # ---------- 5. Aspect ratio ----------
    aspect = float(max(h, w)) / float(max(min(h, w), 1))
    aspect_pass = MIN_ASPECT_RATIO <= aspect <= MAX_ASPECT_RATIO
    aspect_ratio = {"ratio": aspect, "pass": bool(aspect_pass)}

    # ---------- Score ----------
    score = (
        W_RESOLUTION * float(res_pass)
        + W_ECG * float(ecg_pass)
        + W_FOCUS * float(focus_pass)
        + W_EXPOSURE * float(bright_pass)
        + W_ASPECT * float(aspect_pass)
    )

    # ---------- Decisão & mensagens ----------
    rejection_reason: Optional[str] = None
    warning: Optional[str] = None

    # Mensagens individuais (ordem de prioridade)
    if not res_pass:
        rejection_reason = (
            "Foto com resolução muito baixa. Tire a foto mais perto do ECG "
            "ou use a câmera traseira."
        )
    elif not ecg_pass:
        rejection_reason = (
            "Não foi possível identificar um ECG na imagem. Certifique-se de "
            "que o ECG está visível e centralizado na foto."
        )
    elif not focus_pass:
        rejection_reason = (
            "Foto desfocada. Segure o celular firme e toque na tela para "
            "focar antes de tirar a foto."
        )
    elif not bright_pass:
        if mean_b < MIN_BRIGHTNESS:
            rejection_reason = (
                "Foto muito escura. Tire a foto em ambiente mais iluminado."
            )
        else:
            rejection_reason = (
                "Foto estourada. Evite flash direto e reflexos no papel."
            )

    # Defesa em profundidade — score < 0.5 sem nenhum check estrito ter falhado
    # é teoricamente impossível, mas mantém a regra do spec
    if rejection_reason is None and score < 0.5:
        rejection_reason = "Qualidade da foto insuficiente."

    accept = rejection_reason is None

    if accept:
        # Warning de aspect ratio (ECG vertical/quadrado pode ser foto rotacionada)
        if not aspect_pass:
            warning = (
                "Proporção da foto incomum — pode estar girada. "
                "O sistema tentará corrigir automaticamente."
            )
        elif score < 0.7:
            warning = (
                "Qualidade da foto é mediana — o resultado pode ter "
                "menor precisão."
            )

    result = {
        "accept": bool(accept),
        "score": float(score),
        "rejection_reason": rejection_reason,
        "warning": warning,
        "details": {
            "resolution": resolution,
            "ecg_detected": ecg_detected,
            "focus": focus,
            "exposure": exposure,
            "aspect_ratio": aspect_ratio,
        },
    }

    if accept:
        logger.info(
            "Quality OK: score=%.2f (res=%d, ecg=%.2f, foc=%.0f, exp=%.0f, asp=%.2f)",
            score, longer, periodic_score, lap_var, mean_b, aspect,
        )
    else:
        logger.warning("Quality REJECT: %s (score=%.2f)", rejection_reason, score)

    return result
