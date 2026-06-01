"""
Pipeline de Digitalização de ECG baseado no PMcardio — ProECG

Pipeline de 6 módulos sequenciais (Demolder et al., 2025):
  1. Pré-processamento: crop do papel + correção de perspectiva
  2. Dotter: detecta pontos de interseção do grid milimetrado
  3. Gridder: organiza pontos em matriz + interpola gaps
  4. Undistortion: corrige distorção do papel quadrado a quadrado
  5. Leader: segmenta traçados dos leads
  6. Extração de sinal: máscara → 12 arrays µV

Padrão brasileiro: 25mm/s, 10mm/mV.
Layouts suportados:
  - 3×4+1: 3 linhas × 4 derivações + DII longo (detectado se 3-4 faixas)
  - 6×2+1: 6 linhas × 2 derivações + DII longo (detectado se 6-7 faixas)

Os módulos 2 (Dotter) e 5 (Leader) usam UNet de training/models/unet.py.
  - Dotter: UNet treinado (dotter_best.pth) com fallback mock (HSV) se pesos ausentes
  - Leader: UNet treinado (leader_best.pt) com fallback mock (threshold) se pesos ausentes
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import label as ndlabel
from scipy.signal import medfilt

# from .format_detector import detect_layout  # DESATIVADO — layout vem do LeadIdentifier do Stenhede
from .constants import (
    GAIN_DEFAULT,
    GRID_MAJOR_MM,
    GRID_MINOR_MM,
    LEAD_ORDER,
    OUTPUT_LENGTH,
    PAPER_SPEED_DEFAULT,
    SAMPLING_RATE,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes do pipeline PMcardio
# ---------------------------------------------------------------------------

PATCH_SIZE = 256             # tamanho de patch para UNet (Dotter/Leader)
PATCH_OVERLAP = 32           # sobreposição entre patches

# Margem de seguranca do crop por perspectiva: expande os 4 cantos detectados
# pra fora do centroide por essa fracao, evitando perder bordas do papel.
# (approxPolyDP tende a encolher o quadrilatero pra dentro do papel real)
CROP_SAFETY_MARGIN_FRAC = 0.06

# Layout 3×4+1 (padrão brasileiro mais comum)
LEAD_LAYOUT_3x4 = [
    ["I",   "aVR", "V1", "V4"],
    ["II",  "aVL", "V2", "V5"],
    ["III", "aVF", "V3", "V6"],
]

# Layout 6×2+1 (alternativo brasileiro)
LEAD_LAYOUT_6x2 = [
    ["I",   "V1"],
    ["II",  "V2"],
    ["III", "V3"],
    ["aVR", "V4"],
    ["aVL", "V5"],
    ["aVF", "V6"],
]

RHYTHM_LEAD = "II"  # DII longo (rhythm strip)

SHORT_LEAD_SEC_3x4 = 2.5   # 10s / 4 colunas
SHORT_LEAD_SEC_6x2 = 5.0   # 10s / 2 colunas
LONG_LEAD_SEC = 10.0        # duração do rhythm lead


class ECGDigitizer:
    """Orquestra os 6 módulos de digitalização de ECG (arquitetura PMcardio).

    Uso:
        digitizer = ECGDigitizer()
        result = digitizer.run("/path/to/ecg_photo.jpg")
        # result["signals"]["I"] → np.array de µV
        # result["sampling_rate"] → int (Hz)
    """

    # Caminho default dos pesos — relativo ao diretório do módulo
    _DEFAULT_DOTTER_WEIGHTS = str(
        Path(__file__).resolve().parents[2] / "models" / "digitizer" / "dotter_best.pth"
    )
    _DEFAULT_LEADER_WEIGHTS = str(
        Path(__file__).resolve().parents[2] / "models" / "digitizer" / "leader_best.pt"
    )

    def __init__(
        self,
        use_mock: bool = False,
        dotter_weights: Optional[str] = None,
        leader_weights: Optional[str] = None,
    ):
        """
        Args:
            use_mock: Se True, usa detecção mock (HSV/threshold) em vez dos UNets.
                      Se False, usa UNet real para Dotter (com fallback mock se
                      pesos não disponíveis). Leader continua mock até ter pesos.
            dotter_weights: Caminho para pesos do Dotter (.pt).
                            Default: modal_functions/models/digitizer/dotter_best.pth
            leader_weights: Caminho para pesos do Leader (.pt). Ignorado se use_mock=True.
        """
        self.use_mock = use_mock
        self.dotter_weights = dotter_weights or self._DEFAULT_DOTTER_WEIGHTS
        self.leader_weights = leader_weights or self._DEFAULT_LEADER_WEIGHTS
        self._dotter_model = None
        self._leader_model = None
        self._ocr_reader = None
        self.px_per_mm: float = 0.0

    # Tokens normalizados (lowercase, sem espaços) que indicam ECG na orientação correta.
    # Cada token detectado conta 1 ponto único por orientação (dedup).
    # Inclui labels de derivações (EN), calibração e palavras típicas do
    # cabeçalho de ECGs em português brasileiro.
    _ECG_ORIENTATION_TOKENS = (
        "avr", "avl", "avf",
        "v1", "v2", "v3", "v4", "v5", "v6",
        "mm/s", "mm/mv", "25mm/s", "10mm/mv",
        "25mm", "10mm",
        "speed", "limb", "gain",
        "ii", "iii",
        # PT-BR: cabeçalho típico de ECGs brasileiros
        "ritmo", "sinusal", "freq", "frequencia", "frequência",
        "padrao", "padrão", "deriv", "posicion",
        "eixo", "intervalo", "normal", "bpm",
        "paciente", "idade", "sexo", "desconh",
    )

    # =====================================================================
    # Ponto de entrada
    # =====================================================================

    def run(self, image_path: str) -> dict:
        """Executa o pipeline completo: foto → 12 sinais digitais.

        Args:
            image_path: Caminho da imagem (JPEG/PNG).

        Returns:
            dict com:
                signals: dict[str, np.ndarray] — 12 arrays em µV
                sampling_rate: int — Hz (derivado do px_per_mm)
                px_per_mm: float — resolução do grid
                grid_shape: tuple — (linhas, colunas) do grid detectado
                quality_flags: dict — flags de qualidade por etapa
        """
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Não foi possível carregar: {image_path}")

        quality_flags = {}

        # 0. Quality scoring — rejeita rápido se foto não tem qualidade,
        # antes de gastar GPU/CPU com Dotter/Leader.
        from .quality_scorer import score_quality
        qa = score_quality(img)
        quality_flags["quality_check"] = qa
        if not qa["accept"]:
            logger.warning(
                "Pipeline rejeitado pela checagem de qualidade: %s",
                qa["rejection_reason"],
            )
            raise ValueError(
                f"Imagem rejeitada na checagem de qualidade: {qa['rejection_reason']}"
            )

        # 1. Pré-processamento
        cropped, crop_info = self.preprocess(img)
        quality_flags["crop"] = crop_info

        # 2. Dotter — detectar pontos de grid
        #    UNet real por default; dotter() faz fallback p/ mock se pesos ausentes
        if self.use_mock:
            grid_mask, keypoints = self.dotter_mock(cropped)
        else:
            grid_mask, keypoints = self.dotter(cropped)
            if len(keypoints) == 0:
                logger.warning("Dotter UNet não detectou pontos, tentando mock")
                grid_mask, keypoints = self.dotter_mock(cropped)
        quality_flags["grid_points_detected"] = len(keypoints)

        # 3. Gridder — organizar pontos em matriz + calcular px_per_mm
        grid_matrix, grid_info = self.gridder(keypoints, cropped.shape[:2])
        quality_flags["grid_shape"] = grid_info["shape"]
        quality_flags["px_per_mm"] = grid_info["px_per_mm"]
        quality_flags["interpolated_points"] = grid_info["interpolated"]

        # 4. Undistortion — corrigir distorção do papel
        # Só aplica se grid denso o suficiente (≥100 keypoints originais);
        # com grid esparso a homografia por quadrado produz artefatos.
        min_kp_for_undistort = 100
        if len(keypoints) >= min_kp_for_undistort:
            normalized = self.undistort(
                cropped, grid_matrix, grid_info["px_per_mm"]
            )
            quality_flags["undistortion"] = "applied"
        else:
            normalized = cropped.copy()
            quality_flags["undistortion"] = "skipped_sparse_grid"
            logger.info(
                "Undistortion: pulado (%d keypoints < %d mínimo)",
                len(keypoints), min_kp_for_undistort,
            )

        # 4.5. Format detector — DESATIVADO. Layout agora é decidido pelo
        # LeadIdentifier do Stenhede (extract_signals_stenhede). Esse passo
        # rodava nosso detect_layout(), mas o resultado era apenas telemetria.
        quality_flags["layout"] = {"source": "deferred_to_stenhede_lead_identifier"}

        # 4.6. Calibrador — DESATIVADO. O px/mm calculado aqui era descartado
        # — o PixelSizeFinder do Stenhede (use_internal_pixel_size=True) é
        # quem define o px/mm efetivo no extract_signals_stenhede.
        # from .calibrator import calibrate  # DESATIVADO
        quality_flags["calibration"] = {
            "status": "skipped_uses_stenhede_pixel_size_finder"
        }
        calibrated_px_per_mm = grid_info["px_per_mm"]
        sampling_rate = int(round(calibrated_px_per_mm * PAPER_SPEED_DEFAULT))
        calibration = None  # marker — nenhum cal_dict do nosso calibrator

        # 5. Segmentação + extração FULL Stenhede (end-to-end)
        # ----------------------------------------------------------------
        # Caminho PADRÃO: U-Net Stenhede → SignalExtractor (imagem inteira)
        # → LeadIdentifier (atribui nomes + converte pra µV). Eliminamos a
        # lógica de cells/baseline do nosso lead_separator.
        # Fallback automático ao caminho cell-by-cell se o full pipeline
        # falhar.
        from .stenhede_adapter import extract_signals_stenhede

        try:
            cal_dict = (
                calibration
                if isinstance(calibration, dict)
                else {
                    "px_per_mm": calibrated_px_per_mm,
                    "uv_per_pixel": 1000.0 / (calibrated_px_per_mm * GAIN_DEFAULT),
                    "sampling_rate_hz": float(sampling_rate),
                }
            )
            stenhede_result = extract_signals_stenhede(
                image_bgr=normalized,
                px_per_mm=float(cal_dict["px_per_mm"]),
                paper_speed=float(PAPER_SPEED_DEFAULT),
                voltage_gain=float(GAIN_DEFAULT),
            )
            signals: dict[str, np.ndarray] = stenhede_result["signals"]
            # fs efetivo do Stenhede (target_num_samples / duracao)
            sampling_rate = int(round(stenhede_result["sampling_rate_hz"]))
            quality_flags["segmenter"] = "stenhede_full"
            quality_flags["stenhede_match"] = stenhede_result["match"]
            quality_flags["stenhede_n_lines"] = stenhede_result["n_lines_detected"]
            quality_flags["leads_extracted"] = sum(
                1 for k in signals if k != "II_rhythm" and not k.endswith("_long")
            )
        except Exception as e:
            # Fallback: caminho cell-by-cell (também usa Stenhede mas
            # respeita as bandas/baseline do lead_separator).
            logger.warning(
                "Stenhede FULL falhou (%s: %s) — fallback cell-by-cell",
                type(e).__name__, e,
            )
            from .lead_separator import separate_and_extract
            from .stenhede_adapter import (
                extract_signal_probabilities,
                signal_prob_to_binary_mask,
            )
            try:
                signal_prob = extract_signal_probabilities(normalized)
                lead_mask = signal_prob_to_binary_mask(signal_prob, threshold=0.1)
                cal_dict = (
                    calibration
                    if isinstance(calibration, dict)
                    else {
                        "px_per_mm": calibrated_px_per_mm,
                        "uv_per_pixel": 1000.0 / (calibrated_px_per_mm * GAIN_DEFAULT),
                        "sampling_rate_hz": float(sampling_rate),
                    }
                )
                sep = separate_and_extract(
                    mask=lead_mask,
                    normalized_image=normalized,
                    calibration=cal_dict,
                    signal_prob=signal_prob,
                )
                signals = {n: i["signal_uv"] for n, i in sep["leads"].items()}
                quality_flags["segmenter"] = (
                    f"stenhede_cellbycell_after_{type(e).__name__}"
                )
                quality_flags["lead_pixels"] = int(np.sum(lead_mask > 0))
                quality_flags["leads_extracted"] = sum(
                    1 for k in signals if k != "II_rhythm" and not k.endswith("_long")
                )
            except Exception as e2:
                # Último fallback: extrator legado puro
                logger.warning(
                    "Stenhede cell-by-cell tambem falhou (%s: %s) — "
                    "fallback ao extrator legado", type(e2).__name__, e2,
                )
                if self.use_mock:
                    lead_mask = self.leader_mock(normalized)
                else:
                    lead_mask = self.leader(normalized)
                signals_legacy, extraction_info = self.extract_signals(
                    lead_mask, calibrated_px_per_mm
                )
                quality_flags.update(extraction_info)
                quality_flags["segmenter"] = (
                    f"legacy_fallback_after_{type(e).__name__}_{type(e2).__name__}"
                )
                signals = signals_legacy

        # Compat com extrator legado: alias II_long ←→ II_rhythm
        if "II_rhythm" in signals and f"{RHYTHM_LEAD}_long" not in signals:
            signals[f"{RHYTHM_LEAD}_long"] = signals["II_rhythm"]

        if sampling_rate < 50:
            sampling_rate = SAMPLING_RATE  # fallback seguro

        return {
            "signals": signals,
            "sampling_rate": sampling_rate,
            "px_per_mm": calibrated_px_per_mm,
            "grid_shape": grid_info["shape"],
            "quality_flags": quality_flags,
        }

    # =====================================================================
    # Módulo 1: Pré-processamento (crop + correção de perspectiva)
    # =====================================================================

    def preprocess(self, img: np.ndarray) -> tuple[np.ndarray, dict]:
        """Detecta o papel ECG na foto e corrige perspectiva.

        1. Converte para grayscale, aplica blur e detecta bordas (Canny).
        2. Encontra o maior contorno retangular (≥4 lados → papel ECG).
        3. Aplica transformação de perspectiva para retificar.
        4. Fallback: crop de margens (5%).

        Returns:
            (imagem_cropada_BGR, info_dict)
        """
        h, w = img.shape[:2]
        info = {"method": "none", "original_size": (w, h)}

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        # Dilatar bordas para fechar gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=2)

        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        best_rect = None
        best_area = 0
        min_area = h * w * 0.15  # papel deve ser ≥15% da foto

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            peri = cv2.arcLength(cnt, True)
            # Epsilon menor (0.01 vs 0.02) -> cantos mais proximos da borda real
            approx = cv2.approxPolyDP(cnt, 0.01 * peri, True)
            if len(approx) == 4 and area > best_area:
                best_area = area
                best_rect = approx

        if best_rect is not None:
            pts = best_rect.reshape(4, 2).astype(np.float32)
            pts = self._order_points(pts)

            # Expandir os 4 cantos pra FORA do centroide por CROP_SAFETY_MARGIN_FRAC.
            # Isso compensa o approxPolyDP encolher pra dentro do papel real e
            # garante que nao perdemos tracado nas bordas. Clampar pra nao
            # sair da foto.
            center = pts.mean(axis=0)
            pts_expanded = (
                center + (pts - center) * (1.0 + CROP_SAFETY_MARGIN_FRAC)
            ).astype(np.float32)
            pts_expanded[:, 0] = np.clip(pts_expanded[:, 0], 0, w - 1)
            pts_expanded[:, 1] = np.clip(pts_expanded[:, 1], 0, h - 1)

            dst_w = int(max(
                np.linalg.norm(pts_expanded[1] - pts_expanded[0]),
                np.linalg.norm(pts_expanded[2] - pts_expanded[3]),
            ))
            dst_h = int(max(
                np.linalg.norm(pts_expanded[3] - pts_expanded[0]),
                np.linalg.norm(pts_expanded[2] - pts_expanded[1]),
            ))

            if dst_w > 100 and dst_h > 100:
                dst = np.float32([
                    [0, 0], [dst_w - 1, 0],
                    [dst_w - 1, dst_h - 1], [0, dst_h - 1],
                ])
                M = cv2.getPerspectiveTransform(pts_expanded, dst)
                warped = cv2.warpPerspective(img, M, (dst_w, dst_h))
                info["method"] = "perspective"
                info["crop_size"] = (dst_w, dst_h)
                info["safety_margin_frac"] = CROP_SAFETY_MARGIN_FRAC
                info["pts_detected"] = pts.tolist()
                info["pts_expanded"] = pts_expanded.tolist()
                logger.info(
                    "Pré-proc: perspectiva %dx%d → %dx%d (margem +%.0f%%)",
                    w, h, dst_w, dst_h, CROP_SAFETY_MARGIN_FRAC * 100,
                )
                return warped, info

        # Fallback: nenhum retangulo detectado -> usar imagem inteira sem
        # cortar (preserva todo o tracado; gridder/dotter lidam com o resto).
        info["method"] = "no_crop_fallback"
        info["crop_size"] = (w, h)
        logger.info("Pré-proc: fallback sem crop (imagem inteira %dx%d)", w, h)
        return img.copy(), info

    @staticmethod
    def _order_points(pts: np.ndarray) -> np.ndarray:
        """Ordena 4 pontos: top-left, top-right, bottom-right, bottom-left."""
        rect = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        d = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(d)]
        rect[3] = pts[np.argmax(d)]
        return rect

    # =====================================================================
    # Módulo 2: Dotter (detectar pontos de grid)
    # =====================================================================

    def dotter(self, img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Detecta pontos de interseção do grid usando UNet treinado.

        Divide a imagem em patches 256×256, roda o UNet em cada patch,
        recombina as máscaras e extrai os centroides dos blobs.

        Returns:
            (mask_H×W, keypoints_Nx2)
        """
        import torch

        if self._dotter_model is None:
            if self.dotter_weights is None or not Path(self.dotter_weights).is_file():
                logger.warning(
                    "Dotter UNet: pesos não encontrados (%s), fallback mock",
                    self.dotter_weights,
                )
                return self.dotter_mock(img)

            import sys
            # Adiciona training/ ao path para importar models.unet
            # Tenta local (dev) e container Modal (/root/training)
            training_dir = str(Path(__file__).resolve().parents[3] / "training")
            modal_training_dir = "/root/training"
            for d in [training_dir, modal_training_dir]:
                if d not in sys.path:
                    sys.path.insert(0, d)

            from models.unet import UNet

            self._dotter_model = UNet(in_channels=3, out_channels=1)
            checkpoint = torch.load(self.dotter_weights, map_location="cpu", weights_only=True)
            # Suporta tanto state_dict direto quanto checkpoint com model_state_dict
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                state = checkpoint["model_state_dict"]
            else:
                state = checkpoint
            self._dotter_model.load_state_dict(state)
            self._dotter_model.eval()
            if torch.cuda.is_available():
                self._dotter_model = self._dotter_model.cuda()
                logger.info(
                    "Dotter UNet carregado em GPU: %s", self.dotter_weights,
                )
            else:
                logger.info(
                    "Dotter UNet carregado em CPU: %s", self.dotter_weights,
                )

        h, w = img.shape[:2]
        patches = self._extract_patches(img)

        # Montar máscara full-size por acumulação (média nas sobreposições)
        accum = np.zeros((h, w), dtype=np.float32)
        count = np.zeros((h, w), dtype=np.float32)

        device = next(self._dotter_model.parameters()).device

        BATCH_SIZE = 32
        with torch.no_grad():
            for i in range(0, len(patches), BATCH_SIZE):
                batch = patches[i:i + BATCH_SIZE]
                tensors: list[torch.Tensor] = []
                for patch_bgr, _ox, _oy in batch:
                    # BGR → RGB, HWC → CHW, normalizar [0,1]
                    patch_rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB)
                    tensors.append(
                        torch.from_numpy(patch_rgb).permute(2, 0, 1).float() / 255.0
                    )
                stack = torch.stack(tensors).to(device)        # (B, 3, 256, 256)
                logits = self._dotter_model(stack)             # (B, 1, 256, 256)
                probs = torch.sigmoid(logits).cpu().numpy()    # (B, 1, 256, 256)
                for j, (_patch, ox, oy) in enumerate(batch):
                    prob = probs[j, 0]
                    ph, pw = prob.shape
                    accum[oy:oy + ph, ox:ox + pw] += prob
                    count[oy:oy + ph, ox:ox + pw] += 1.0

        count = np.maximum(count, 1.0)
        prob_map = accum / count

        # Binarizar com threshold 0.5
        mask = (prob_map > 0.5).astype(np.uint8) * 255

        keypoints = self._mask_to_keypoints(mask)
        logger.info("Dotter UNet: %d pontos de grid detectados", len(keypoints))
        return mask, keypoints

    def dotter_mock(self, img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Detecta pontos de grid com abordagem multi-estratégia.

        Estratégia 1: HSV — funciona para grids coloridos (saturação alta).
        Estratégia 2: Morfológica — funciona para grids de baixa saturação
                       (ex: Bionet rosa claro quase cinza). Isola linhas finas
                       retas (H/V) na faixa de intensidade do grid (entre o
                       traçado escuro e o fundo branco).

        Returns:
            (mask_H×W, keypoints_Nx2)  — keypoints[:, 0]=x, [:, 1]=y
        """
        h, w = img.shape[:2]

        # --- Estratégia 1: HSV (grids coloridos saturados) ---
        keypoints = self._dotter_hsv(img)
        if len(keypoints) >= 20:
            mask = self._keypoints_to_mask(keypoints, (h, w))
            logger.info("Dotter mock (HSV): %d pontos de grid", len(keypoints))
            return mask, keypoints

        # --- Estratégia 2: Morfológica (grids de baixa saturação) ---
        keypoints = self._dotter_morphological(img)
        mask = self._keypoints_to_mask(keypoints, (h, w))
        logger.info("Dotter mock (morph): %d pontos de grid", len(keypoints))
        return mask, keypoints

    def _dotter_hsv(self, img: np.ndarray) -> np.ndarray:
        """Detecta interseções do grid por cor HSV."""
        h, w = img.shape[:2]
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        grid_mask = np.zeros((h, w), dtype=np.uint8)

        color_ranges = [
            # Laranja/vermelho
            ((0, 30, 80), (15, 255, 220)),
            ((165, 30, 80), (180, 255, 220)),
            # Rosa/magenta
            ((140, 20, 80), (170, 255, 220)),
            # Azul
            ((90, 20, 80), (130, 200, 220)),
            # Verde
            ((35, 20, 80), (85, 200, 220)),
        ]

        for lower, upper in color_ranges:
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            grid_mask = cv2.bitwise_or(grid_mask, mask)

        # Fallback: saturação moderada
        if np.sum(grid_mask > 0) < h * w * 0.01:
            _, sat, _ = cv2.split(hsv)
            grid_mask = cv2.inRange(sat, 15, 120)

        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        grid_mask = cv2.morphologyEx(grid_mask, cv2.MORPH_OPEN, kernel_small)

        return self._find_grid_intersections(grid_mask)

    def _dotter_morphological(self, img: np.ndarray) -> np.ndarray:
        """Detecta interseções do grid por morfologia (grids de baixa saturação).

        Isola pixels na faixa de intensidade do grid (entre traçado escuro e
        fundo branco), depois extrai linhas H/V longas e finas.
        """
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()

        # Faixa de intensidade do grid: mais escuro que fundo, mais claro que traçado.
        # Calcular thresholds adaptativos usando percentis da imagem.
        p10 = np.percentile(gray, 10)   # ~traçado escuro
        p90 = np.percentile(gray, 90)   # ~fundo claro
        mid = (p10 + p90) / 2

        # Grid ocupa a faixa entre (mid - margem) e (p90 - margem_alta)
        grid_lo = int(max(mid - 10, p10 + 20))
        grid_hi = int(min(p90 - 5, 240))

        grid_band = cv2.inRange(gray, grid_lo, grid_hi)

        # Extrair linhas H e V longas e finas
        h_len = max(w // 30, 25)
        v_len = max(h // 30, 25)
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1))
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len))

        h_lines = cv2.morphologyEx(grid_band, cv2.MORPH_OPEN, h_kernel)
        v_lines = cv2.morphologyEx(grid_band, cv2.MORPH_OPEN, v_kernel)

        # Combinar
        grid_mask = cv2.bitwise_or(h_lines, v_lines)

        # Limpar ruído
        kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        grid_mask = cv2.morphologyEx(grid_mask, cv2.MORPH_CLOSE, kernel_clean)

        return self._find_grid_intersections(grid_mask)

    def _find_grid_intersections(self, grid_mask: np.ndarray) -> np.ndarray:
        """Encontra interseções H×V numa máscara de grid."""
        h, w = grid_mask.shape

        h_len = max(w // 30, 25)
        v_len = max(h // 30, 25)
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1))
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len))

        h_lines = cv2.morphologyEx(grid_mask, cv2.MORPH_OPEN, h_kernel)
        v_lines = cv2.morphologyEx(grid_mask, cv2.MORPH_OPEN, v_kernel)

        # Dilatar para garantir sobreposição nas interseções
        h_dilated = cv2.dilate(
            h_lines, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 7))
        )
        v_dilated = cv2.dilate(
            v_lines, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1))
        )
        intersections = cv2.bitwise_and(h_dilated, v_dilated)

        return self._mask_to_keypoints(intersections)

    def _keypoints_to_mask(
        self, keypoints: np.ndarray, shape: tuple[int, int]
    ) -> np.ndarray:
        """Gera máscara com pontos (círculos) nas posições dos keypoints."""
        mask = np.zeros(shape, dtype=np.uint8)
        for pt in keypoints:
            cx, cy = int(pt[0]), int(pt[1])
            cv2.circle(mask, (cx, cy), 3, 255, -1)
        return mask

    def _mask_to_keypoints(self, mask: np.ndarray) -> np.ndarray:
        """Converte máscara binária → array Nx2 de centroides (x, y)."""
        labeled, n_blobs = ndlabel(mask > 0)
        if n_blobs == 0:
            return np.empty((0, 2), dtype=np.float64)

        keypoints = []
        for blob_id in range(1, n_blobs + 1):
            ys, xs = np.where(labeled == blob_id)
            keypoints.append([np.mean(xs), np.mean(ys)])

        return np.array(keypoints, dtype=np.float64)

    def _extract_patches(
        self, img: np.ndarray
    ) -> list[tuple[np.ndarray, int, int]]:
        """Divide imagem em patches 256×256 com sobreposição.

        Returns:
            Lista de (patch_256x256, offset_x, offset_y)
        """
        h, w = img.shape[:2]
        stride = PATCH_SIZE - PATCH_OVERLAP
        patches = []

        for y in range(0, h, stride):
            for x in range(0, w, stride):
                y2 = min(y + PATCH_SIZE, h)
                x2 = min(x + PATCH_SIZE, w)
                y1 = max(0, y2 - PATCH_SIZE)
                x1 = max(0, x2 - PATCH_SIZE)

                patch = img[y1:y2, x1:x2]
                if patch.shape[0] < PATCH_SIZE or patch.shape[1] < PATCH_SIZE:
                    padded = np.zeros(
                        (PATCH_SIZE, PATCH_SIZE, 3), dtype=np.uint8
                    )
                    padded[: patch.shape[0], : patch.shape[1]] = patch
                    patch = padded

                patches.append((patch, x1, y1))

        return patches

    def _extract_patches_with_stride(
        self, img: np.ndarray, stride: int
    ) -> list[tuple[np.ndarray, int, int]]:
        """Divide imagem em patches 256×256 com stride customizável.

        Usado pelo Leader com stride=128 (50% overlap) para melhor cobertura.

        Returns:
            Lista de (patch_256x256, offset_x, offset_y)
        """
        h, w = img.shape[:2]
        patches = []

        for y in range(0, h, stride):
            for x in range(0, w, stride):
                y2 = min(y + PATCH_SIZE, h)
                x2 = min(x + PATCH_SIZE, w)
                y1 = max(0, y2 - PATCH_SIZE)
                x1 = max(0, x2 - PATCH_SIZE)

                patch = img[y1:y2, x1:x2]
                if patch.shape[0] < PATCH_SIZE or patch.shape[1] < PATCH_SIZE:
                    padded = np.zeros(
                        (PATCH_SIZE, PATCH_SIZE, 3), dtype=np.uint8
                    )
                    padded[: patch.shape[0], : patch.shape[1]] = patch
                    patch = padded

                patches.append((patch, x1, y1))

        return patches

    # =====================================================================
    # Módulo 3: Gridder (organizar pontos em matriz)
    # =====================================================================

    def gridder(
        self, keypoints: np.ndarray, img_shape: tuple[int, int]
    ) -> tuple[np.ndarray, dict]:
        """Constrói matriz de grid por conexão direta entre pontos vizinhos.

        Para cada ponto do Dotter, encontra o vizinho mais próximo em cada
        direção (direita, abaixo) dentro de spacing * 1.5. Depois percorre
        as conexões para montar a matriz grid[linha][coluna] = (x, y).
        """
        h, w = img_shape
        info = {"shape": (0, 0), "px_per_mm": 0.0, "interpolated": 0}

        if len(keypoints) < 4:
            return self._make_estimated_grid(img_shape, info)

        spacing = self._estimate_grid_spacing(keypoints)
        if spacing < 5:
            return self._make_estimated_grid(img_shape, info)

        px_per_mm = spacing / GRID_MAJOR_MM
        info["px_per_mm"] = px_per_mm
        self.px_per_mm = px_per_mm

        from scipy.spatial import KDTree
        tree = KDTree(keypoints)
        n_pts = len(keypoints)

        # Para cada ponto, encontrar vizinho em cada direção
        max_dist = spacing * 1.5
        angle_tol = 35  # graus de tolerância angular

        right_of = {}  # i -> j: j é vizinho à direita de i
        below_of = {}  # i -> j: j é vizinho abaixo de i

        k = min(12, n_pts)
        dists, idxs = tree.query(keypoints, k=k)

        for i in range(n_pts):
            for rank in range(1, k):
                j = idxs[i, rank]
                d = dists[i, rank]
                if d > max_dist:
                    break
                dx = keypoints[j, 0] - keypoints[i, 0]
                dy = keypoints[j, 1] - keypoints[i, 1]
                angle = np.degrees(np.arctan2(dy, dx))

                # Vizinho à direita
                if i not in right_of and abs(angle) <= angle_tol and dx > 0:
                    right_of[i] = j

                # Vizinho abaixo
                if i not in below_of and abs(angle - 90) <= angle_tol and dy > 0:
                    below_of[i] = j

        # Inversos: left_of e above_of
        left_of = {j: i for i, j in right_of.items()}
        above_of = {j: i for i, j in below_of.items()}

        # ----------------------------------------------------------
        # Atribuir coordenadas (row, col) a TODOS os pontos via BFS.
        # Propaga a partir de sementes usando as 4 conexões.
        # ----------------------------------------------------------
        from collections import deque

        coords: dict[int, tuple[int, int]] = {}  # i -> (row, col)

        def _bfs_from(seed: int, r0: int, c0: int) -> None:
            """Flood-fill de coordenadas a partir de seed=(r0, c0)."""
            if seed in coords:
                return
            coords[seed] = (r0, c0)
            queue = deque([seed])
            while queue:
                cur = queue.popleft()
                cr, cc = coords[cur]
                for neighbor, dr, dc in [
                    (right_of.get(cur), 0, 1),
                    (left_of.get(cur), 0, -1),
                    (below_of.get(cur), 1, 0),
                    (above_of.get(cur), -1, 0),
                ]:
                    if neighbor is not None and neighbor not in coords:
                        coords[neighbor] = (cr + dr, cc + dc)
                        queue.append(neighbor)

        # Semear componentes conectados: pegar o ponto sem coordenada
        # mais perto do canto superior esquerdo como próxima semente
        corner_order = np.argsort(keypoints[:, 0] + keypoints[:, 1])
        for seed_i in corner_order:
            seed_i = int(seed_i)
            if seed_i not in coords:
                _bfs_from(seed_i, 0, 0)

        assigned = len(coords)
        logger.debug(
            "Gridder BFS: %d/%d pontos com coordenada (%.0f%%)",
            assigned, n_pts, assigned / max(n_pts, 1) * 100,
        )

        if assigned < 4:
            logger.warning("Gridder BFS: poucos pontos atribuídos (%d), fallback", assigned)
            return self._make_estimated_grid(img_shape, info)

        # Normalizar: mínimo de row e col → 0
        all_rows = [rc[0] for rc in coords.values()]
        all_cols = [rc[1] for rc in coords.values()]
        min_r, min_c = min(all_rows), min(all_cols)

        n_rows = max(all_rows) - min_r + 1
        n_cols = max(all_cols) - min_c + 1

        if n_rows < 2 or n_cols < 2:
            logger.warning("Gridder: matriz insuficiente %dx%d, fallback", n_rows, n_cols)
            return self._make_estimated_grid(img_shape, info)

        # Montar matriz de coordenadas
        grid = np.full((n_rows, n_cols, 2), np.nan, dtype=np.float64)
        for pt_i, (r, c) in coords.items():
            gr, gc = r - min_r, c - min_c
            # Primeiro ponto a ocupar a célula vence (ignora conflitos)
            if np.isnan(grid[gr, gc, 0]):
                grid[gr, gc] = keypoints[pt_i]

        # Interpolar NaN restantes
        n_interpolated = self._interpolate_grid(grid)

        grid, n_rows, n_cols = self._trim_empty_borders(grid)
        info["shape"] = (n_rows, n_cols)
        info["interpolated"] = n_interpolated

        filled = int(np.sum(~np.isnan(grid[:, :, 0])))
        total = n_rows * n_cols
        occupancy = filled / max(total, 1) * 100

        logger.info(
            "Gridder por conexão direta: %dx%d, spacing=%.1f px, "
            "px_per_mm=%.2f, ocupação=%.0f%% (%d/%d)",
            n_rows, n_cols, spacing, px_per_mm,
            occupancy, filled, total
        )
        return grid, info

    def _trim_empty_borders(self, grid: np.ndarray) -> tuple[np.ndarray, int, int]:
        """Remove linhas e colunas totalmente vazias nas bordas da matriz."""
        n_rows, n_cols, _ = grid.shape

        # Encontrar linhas não-vazias
        row_has_data = np.any(~np.isnan(grid[:, :, 0]), axis=1)
        col_has_data = np.any(~np.isnan(grid[:, :, 0]), axis=0)

        if not np.any(row_has_data) or not np.any(col_has_data):
            return grid, n_rows, n_cols

        r_start = int(np.argmax(row_has_data))
        r_end = int(len(row_has_data) - np.argmax(row_has_data[::-1]))
        c_start = int(np.argmax(col_has_data))
        c_end = int(len(col_has_data) - np.argmax(col_has_data[::-1]))

        trimmed = grid[r_start:r_end, c_start:c_end, :]
        return trimmed, trimmed.shape[0], trimmed.shape[1]

    def _estimate_grid_spacing(self, keypoints: np.ndarray) -> float:
        """Estima espaçamento de 5mm via mediana de distâncias ao vizinho mais próximo."""
        from scipy.spatial import KDTree

        tree = KDTree(keypoints)
        # k=2: self + vizinho mais próximo
        dists, _ = tree.query(keypoints, k=2)
        neighbor_dists = dists[:, 1]  # distância ao 1-NN

        # Filtrar outliers (pontos isolados ou muito colados)
        neighbor_dists = neighbor_dists[
            (neighbor_dists > 5) & (neighbor_dists < 200)
        ]
        if len(neighbor_dists) < 10:
            return 0.0

        # Mediana é robusta a outliers
        spacing = float(np.median(neighbor_dists))
        return spacing

    def _bin_axis(
        self, values: np.ndarray, spacing: float, tolerance: float
    ) -> np.ndarray:
        """Agrupa coordenadas em bins regulares espaçados por `spacing`.

        Usa histograma com bins de largura `spacing` para encontrar picos,
        depois refina cada pico pela média dos pontos atribuídos.

        Returns:
            Array ordenado de centros dos bins (posições das linhas ou colunas).
        """
        vmin, vmax = float(values.min()), float(values.max())

        # Criar bins regulares cobrindo o range
        bin_edges = np.arange(vmin - spacing / 2, vmax + spacing, spacing)
        hist, edges = np.histogram(values, bins=bin_edges)

        # Pegar bins com pelo menos 2 pontos (não vazios)
        centers = []
        for i in range(len(hist)):
            if hist[i] >= 2:
                bin_lo = edges[i]
                bin_hi = edges[i + 1]
                pts_in_bin = values[(values >= bin_lo) & (values < bin_hi)]
                centers.append(float(np.mean(pts_in_bin)))

        if not centers:
            return np.array([])

        # Mesclar bins muito próximos (pode acontecer em bordas)
        merged = [centers[0]]
        for c in centers[1:]:
            if c - merged[-1] < spacing * 0.5:
                merged[-1] = (merged[-1] + c) / 2  # média
            else:
                merged.append(c)

        return np.array(sorted(merged))

    def _interpolate_grid(self, grid: np.ndarray) -> int:
        """Interpola posicoes vazias em duas passadas:

        Passada 1 — 4 vizinhos: para cada NaN cujos 4 vizinhos adjacentes
        diretos (acima, abaixo, esquerda, direita) existem na matriz
        original, insere a media (x, y) dos 4. Usa snapshot original ->
        nao depende de ordem de varredura.

        Passada 2 — 3 vizinhos: re-varre as posicoes que ainda estao
        vazias usando a matriz APOS a passada 1 (entao pontos
        interpolados na P1 ja contam). Se exatamente 3 dos 4 vizinhos
        adjacentes existem, o 4o e' inferido:
          - falta vertical (top/bottom): x = avg(left.x, right.x),
                                           y = vizinho_vertical.y +/- spacing_v
          - falta horizontal (left/right): y = avg(top.y, bottom.y),
                                             x = vizinho_horizontal.x +/- spacing_h
        spacing_v / spacing_h sao a mediana das distancias entre celulas
        vizinhas validas no estado pos-P1.

        Posicoes com <3 vizinhos ficam NaN (undistortion ja ignora).
        """
        n_rows, n_cols, _ = grid.shape
        if n_rows < 3 or n_cols < 3:
            return 0

        is_nan_orig = np.isnan(grid[:, :, 0])
        missing_before = int(is_nan_orig.sum())
        if missing_before == 0:
            return 0

        # ----- Passada 1: 4 vizinhos -----
        n_pass1 = 0
        for r in range(1, n_rows - 1):
            for c in range(1, n_cols - 1):
                if not is_nan_orig[r, c]:
                    continue
                if (is_nan_orig[r - 1, c] or is_nan_orig[r + 1, c]
                        or is_nan_orig[r, c - 1] or is_nan_orig[r, c + 1]):
                    continue
                grid[r, c, 0] = (
                    grid[r - 1, c, 0] + grid[r + 1, c, 0]
                    + grid[r, c - 1, 0] + grid[r, c + 1, 0]
                ) / 4.0
                grid[r, c, 1] = (
                    grid[r - 1, c, 1] + grid[r + 1, c, 1]
                    + grid[r, c - 1, 1] + grid[r, c + 1, 1]
                ) / 4.0
                n_pass1 += 1

        # ----- Passada 2: 3 vizinhos (usa snapshot pos-P1) -----
        is_nan = np.isnan(grid[:, :, 0])

        # Estimar spacing medio (mediana das distancias entre vizinhos validos)
        v_spacings = []
        h_spacings = []
        for r in range(n_rows - 1):
            for c in range(n_cols):
                if not is_nan[r, c] and not is_nan[r + 1, c]:
                    v_spacings.append(grid[r + 1, c, 1] - grid[r, c, 1])
        for r in range(n_rows):
            for c in range(n_cols - 1):
                if not is_nan[r, c] and not is_nan[r, c + 1]:
                    h_spacings.append(grid[r, c + 1, 0] - grid[r, c, 0])

        spacing_v = float(np.median(v_spacings)) if v_spacings else 0.0
        spacing_h = float(np.median(h_spacings)) if h_spacings else 0.0

        n_pass2 = 0
        if spacing_v > 0 and spacing_h > 0:
            for r in range(1, n_rows - 1):
                for c in range(1, n_cols - 1):
                    if not is_nan[r, c]:
                        continue
                    top_ok = not is_nan[r - 1, c]
                    bot_ok = not is_nan[r + 1, c]
                    left_ok = not is_nan[r, c - 1]
                    right_ok = not is_nan[r, c + 1]
                    if (int(top_ok) + int(bot_ok)
                            + int(left_ok) + int(right_ok)) != 3:
                        continue

                    if not top_ok:
                        x = (grid[r, c - 1, 0] + grid[r, c + 1, 0]) / 2.0
                        y = grid[r + 1, c, 1] - spacing_v
                    elif not bot_ok:
                        x = (grid[r, c - 1, 0] + grid[r, c + 1, 0]) / 2.0
                        y = grid[r - 1, c, 1] + spacing_v
                    elif not left_ok:
                        y = (grid[r - 1, c, 1] + grid[r + 1, c, 1]) / 2.0
                        x = grid[r, c + 1, 0] - spacing_h
                    else:  # not right_ok
                        y = (grid[r - 1, c, 1] + grid[r + 1, c, 1]) / 2.0
                        x = grid[r, c - 1, 0] + spacing_h

                    grid[r, c, 0] = x
                    grid[r, c, 1] = y
                    n_pass2 += 1

        n_interpolated = n_pass1 + n_pass2
        missing_after = missing_before - n_interpolated
        logger.info(
            "Interpolacao do grid: %d faltantes -> P1 (4 vizinhos)=%d, "
            "P2 (3 vizinhos)=%d, total=%d sinteticos, %d vazios",
            missing_before, n_pass1, n_pass2, n_interpolated, missing_after,
        )
        return n_interpolated

    def _make_estimated_grid(
        self, img_shape: tuple[int, int], info: dict
    ) -> tuple[np.ndarray, dict]:
        """Grid estimado quando detecção falha (assume largura ~250mm)."""
        h, w = img_shape
        estimated = w / 250.0
        info["px_per_mm"] = estimated
        info["shape"] = (0, 0)
        info["interpolated"] = 0
        self.px_per_mm = estimated
        grid = np.empty((0, 0, 2), dtype=np.float64)
        logger.warning("Gridder: estimativa px_per_mm=%.1f (fallback)", estimated)
        return grid, info

    # =====================================================================
    # Módulo 4: Undistortion (corrigir distorção do papel)
    # =====================================================================

    def undistort(
        self, img: np.ndarray, grid_matrix: np.ndarray, px_per_mm: float
    ) -> np.ndarray:
        """Corrige distorção aplicando homografia por quadrado do grid.

        Cada célula é renderizada com overlap de alguns pixels nas bordas,
        e as zonas sobrepostas são mescladas por média ponderada (blending).
        Pixels pretos residuais são preenchidos por inpainting.

        Returns:
            Imagem normalizada BGR com cores originais.
        """
        n_rows, n_cols = grid_matrix.shape[:2] if grid_matrix.size > 0 else (0, 0)

        if n_rows < 2 or n_cols < 2:
            logger.info("Undistortion: grid insuficiente, sem correção")
            return img.copy()

        nan_ratio = np.mean(np.isnan(grid_matrix[:, :, 0]))
        if nan_ratio > 0.5:
            logger.warning("Undistortion: %.0f%% NaN no grid, sem correção", nan_ratio * 100)
            return img.copy()

        cell_size = max(int(round(px_per_mm * GRID_MAJOR_MM)), 10)

        dst_h = (n_rows - 1) * cell_size
        dst_w = (n_cols - 1) * cell_size

        if dst_h < 10 or dst_w < 10:
            return img.copy()

        # Overlap: cada célula é renderizada maior por `ovl` pixels em cada borda
        ovl = max(cell_size // 6, 3)
        render_size = cell_size + 2 * ovl

        accum = np.zeros((dst_h, dst_w, 3), dtype=np.float64)
        weight = np.zeros((dst_h, dst_w), dtype=np.float64)

        # Peso com rampa nas bordas (o centro pesa 1.0, bordas rampam)
        cell_weight = self._make_blend_weight(render_size, render_size, ovl)

        # Pontos destino locais para o patch ampliado
        local_dst = np.float32([
            [ovl, ovl],
            [ovl + cell_size, ovl],
            [ovl + cell_size, ovl + cell_size],
            [ovl, ovl + cell_size],
        ])

        for r in range(n_rows - 1):
            for c in range(n_cols - 1):
                src_pts = np.float32([
                    grid_matrix[r, c],
                    grid_matrix[r, c + 1],
                    grid_matrix[r + 1, c + 1],
                    grid_matrix[r + 1, c],
                ])

                if np.any(np.isnan(src_pts)):
                    continue

                M = cv2.getPerspectiveTransform(src_pts, local_dst)
                patch = cv2.warpPerspective(
                    img, M, (render_size, render_size),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE,
                )

                # Posição no resultado final (com overlap saindo para os lados)
                y1 = r * cell_size - ovl
                x1 = c * cell_size - ovl
                # Clipar ao canvas
                py1 = max(0, -y1)
                px1 = max(0, -x1)
                cy1 = max(0, y1)
                cx1 = max(0, x1)
                cy2 = min(dst_h, y1 + render_size)
                cx2 = min(dst_w, x1 + render_size)
                py2 = py1 + (cy2 - cy1)
                px2 = px1 + (cx2 - cx1)

                if cy2 <= cy1 or cx2 <= cx1:
                    continue

                p = patch[py1:py2, px1:px2].astype(np.float64)
                w = cell_weight[py1:py2, px1:px2]

                accum[cy1:cy2, cx1:cx2] += p * w[:, :, np.newaxis]
                weight[cy1:cy2, cx1:cx2] += w

        # Normalizar
        valid = weight > 0
        for ch in range(3):
            accum[:, :, ch][valid] /= weight[valid]

        result = np.clip(accum, 0, 255).astype(np.uint8)

        # Inpainting: preencher pixels pretos residuais cercados por válidos
        black_mask = (weight == 0).astype(np.uint8)
        if np.any(black_mask):
            # Só inpaint pixels internos (não bordas externas)
            # Dilatar a máscara válida, interseção com black = gaps internos
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            dilated_valid = cv2.dilate(valid.astype(np.uint8), kernel, iterations=2)
            internal_gaps = black_mask & dilated_valid
            if np.any(internal_gaps):
                result = cv2.inpaint(result, internal_gaps, 3, cv2.INPAINT_TELEA)

        result = self._trim_black_borders(result)

        logger.info(
            "Undistortion: %dx%d → %dx%d (cell=%dpx, ovl=%dpx)",
            img.shape[1], img.shape[0],
            result.shape[1], result.shape[0], cell_size, ovl,
        )

        # Etapa final do undistortion: corrige a orientação (foto rotacionada
        # 90/180/270°) usando OCR sobre os labels de derivações e calibração.
        # OCR roda na imagem pré-undistort (img) que ainda tem labels e bordas
        # — o undistort costuma cortá-los. A rotação é aplicada no resultado.
        result = self._correct_orientation(result, ocr_source=img)

        return result

    def _correct_orientation(
        self,
        img: np.ndarray,
        ocr_source: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Detecta e corrige a orientação da imagem (0/90/180/270°) via OCR.

        ECGs têm labels (I, II, III, aVR, aVL, aVF, V1-V6) e calibração
        (25mm/s, 10mm/mV) que só ficam legíveis na orientação correta. Testa
        as 4 rotações, conta quantos tokens típicos cada uma detecta, e
        rotaciona a imagem para a melhor. Se nenhuma orientação detectar
        nada, mantém a original e loga warning.

        Args:
            img: imagem alvo onde a rotação será aplicada (saída do undistort).
            ocr_source: imagem opcional onde rodar o OCR. Se None, usa `img`.
                Útil quando o undistort recorta labels nas bordas — passa a
                imagem pré-undistort (que ainda tem labels e é maior em pixels).
        """
        try:
            import easyocr
        except ImportError:
            logger.warning("Orientação: EasyOCR indisponível — sem correção")
            return img

        if self._ocr_reader is None:
            try:
                # PT+EN: cabeçalhos de ECG brasileiros (Ritmo, Frequência, Padrão...)
                # frequentemente aparecem antes dos labels de derivações.
                import torch
                use_gpu = bool(torch.cuda.is_available())
                self._ocr_reader = easyocr.Reader(
                    ["pt", "en"], gpu=use_gpu, verbose=False,
                )
                logger.info("Orientação: EasyOCR em %s", "GPU" if use_gpu else "CPU")
            except Exception as e:
                logger.warning("Orientação: falha ao carregar EasyOCR (%s) — sem correção", e)
                return img

        ocr_img = ocr_source if ocr_source is not None else img

        # OCR é caro mas labels de derivações (I, V1, aVR) são pequenos:
        # rodamos com lado maior ≤ 2048px para preservar legibilidade.
        # A rotação final é aplicada na imagem original em resolução nativa.
        h, w = ocr_img.shape[:2]
        max_side = 2048
        scale = min(max_side / max(h, w), 1.0)
        if scale < 1.0:
            small = cv2.resize(ocr_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        else:
            small = ocr_img

        rotations = [
            (0, small),
            (90, cv2.rotate(small, cv2.ROTATE_90_CLOCKWISE)),
            (180, cv2.rotate(small, cv2.ROTATE_180)),
            (270, cv2.rotate(small, cv2.ROTATE_90_COUNTERCLOCKWISE)),
        ]

        scores: list[tuple[int, int]] = []
        for angle, rot in rotations:
            try:
                detections = self._ocr_reader.readtext(rot, detail=1, paragraph=False)
            except Exception as e:
                logger.warning("Orientação: OCR falhou em %d° (%s)", angle, e)
                detections = []

            seen: set[str] = set()
            for _bbox, text, conf in detections:
                if conf < 0.2:
                    continue
                norm = "".join(text.lower().split())
                if not norm:
                    continue
                for token in self._ECG_ORIENTATION_TOKENS:
                    if token in norm and token not in seen:
                        seen.add(token)
                        break
            scores.append((angle, len(seen)))

        best_angle, best_count = max(scores, key=lambda s: s[1])
        summary = ", ".join(f"{a}°={c}" for a, c in scores)

        if best_count == 0:
            logger.warning(
                "Orientação: nenhuma detecção em qualquer ângulo (%s) — mantendo original",
                summary,
            )
            return img

        logger.info(
            "Orientação: melhor %d° com %d detecções (%s)",
            best_angle, best_count, summary,
        )

        if best_angle == 0:
            return img
        if best_angle == 90:
            return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        if best_angle == 180:
            return cv2.rotate(img, cv2.ROTATE_180)
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    @staticmethod
    def _make_blend_weight(h: int, w: int, margin: int) -> np.ndarray:
        """Cria máscara de peso com rampa linear nas bordas para blending.

        Centro = 1.0, bordas rampam de 0 a 1 em `margin` pixels.
        Isso elimina costuras entre quadrados vizinhos.
        """
        wy = np.ones(h, dtype=np.float64)
        wx = np.ones(w, dtype=np.float64)

        if margin > 0 and margin < h // 2:
            ramp = np.linspace(0.0, 1.0, margin, endpoint=False)
            wy[:margin] = ramp
            wy[-margin:] = ramp[::-1]

        if margin > 0 and margin < w // 2:
            ramp = np.linspace(0.0, 1.0, margin, endpoint=False)
            wx[:margin] = ramp
            wx[-margin:] = ramp[::-1]

        return wy[:, np.newaxis] * wx[np.newaxis, :]

    @staticmethod
    def _trim_black_borders(img: np.ndarray) -> np.ndarray:
        """Remove bordas pretas (zeros) da imagem."""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        coords = cv2.findNonZero(thresh)
        if coords is None:
            return img
        x, y, w, h = cv2.boundingRect(coords)
        return img[y : y + h, x : x + w]

    # =====================================================================
    # Módulo 5: Leader (segmentar traçados dos leads)
    # =====================================================================

    def leader(self, img: np.ndarray) -> np.ndarray:
        """Segmenta traçados usando UNet treinado (pós-undistortion).

        Divide a imagem em patches 256×256 com overlap de 50%, roda cada
        patch pelo modelo, aplica sigmoid + threshold 0.5 e remonta a
        máscara completa. Fallback para mock se pesos não disponíveis.

        Returns:
            Máscara binária H×W (branco = traçado, preto = fundo/grid).
        """
        import torch

        if self._leader_model is None:
            if self.leader_weights is None or not Path(self.leader_weights).is_file():
                logger.warning(
                    "Leader UNet: pesos não encontrados (%s), fallback mock",
                    self.leader_weights,
                )
                return self.leader_mock(img)

            import sys
            training_dir = str(Path(__file__).resolve().parents[3] / "training")
            modal_training_dir = "/root/training"
            for d in [training_dir, modal_training_dir]:
                if d not in sys.path:
                    sys.path.insert(0, d)

            from models.unet import UNet

            self._leader_model = UNet(in_channels=3, out_channels=1)
            checkpoint = torch.load(self.leader_weights, map_location="cpu", weights_only=True)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                state = checkpoint["model_state_dict"]
            else:
                state = checkpoint
            self._leader_model.load_state_dict(state)
            self._leader_model.eval()
            logger.info("Leader UNet carregado: %s", self.leader_weights)

        h, w = img.shape[:2]
        # Patches com overlap 50% (stride = 128) para melhor cobertura de traçados
        leader_stride = PATCH_SIZE // 2  # 128
        patches = self._extract_patches_with_stride(img, leader_stride)

        accum = np.zeros((h, w), dtype=np.float32)
        count = np.zeros((h, w), dtype=np.float32)

        device = next(self._leader_model.parameters()).device

        with torch.no_grad():
            for patch_bgr, ox, oy in patches:
                patch_rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB)
                tensor = torch.from_numpy(patch_rgb).permute(2, 0, 1).float() / 255.0
                tensor = tensor.unsqueeze(0).to(device)

                logits = self._leader_model(tensor)
                prob = torch.sigmoid(logits).squeeze().cpu().numpy()

                # Clipar ao espaço real da imagem (patches de borda podem exceder)
                ph = min(prob.shape[0], h - oy)
                pw = min(prob.shape[1], w - ox)
                accum[oy:oy + ph, ox:ox + pw] += prob[:ph, :pw]
                count[oy:oy + ph, ox:ox + pw] += 1.0

        count = np.maximum(count, 1.0)
        prob_map = accum / count

        # Threshold 0.5 (padrão para segmentação binária)
        mask = (prob_map > 0.5).astype(np.uint8) * 255

        # Pós-processamento: remover blobs pequenos (ruído) e fechar micro-gaps
        # 1. Morphological open: remove pontos isolados e linhas finas do grid
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)

        # 2. Remover componentes conectados pequenos (< 0.01% da imagem)
        min_area = max(int(h * w * 0.0001), 50)
        labeled, n_comp = ndlabel(mask > 0)
        for comp_id in range(1, n_comp + 1):
            if np.sum(labeled == comp_id) < min_area:
                mask[labeled == comp_id] = 0

        # 3. Fechar micro-gaps no traçado
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

        logger.info("Leader UNet: %d px de traçado detectados", np.sum(mask > 0))
        return mask

    def leader_mock(self, img: np.ndarray) -> np.ndarray:
        """Segmenta traçados por threshold de intensidade.

        Abordagem em 2 fases:
        1. Threshold global agressivo: seleciona apenas pixels realmente escuros
           (traçado), excluindo o grid de intensidade intermediária.
        2. Remoção morfológica de linhas retas residuais (H/V do grid).
        3. Limpeza de componentes pequenos (ruído, texto, artefatos).

        Returns:
            Máscara binária H×W uint8 (255 = traçado).
        """
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        h, w = gray.shape

        # --- Fase 1: threshold adaptativo para encontrar traçado escuro ---
        # Excluir pixels pretos de borda (valor <10) do cálculo de percentis,
        # pois podem ser artefatos de undistortion ou padding.
        valid_pixels = gray[gray > 10]
        if len(valid_pixels) < 100:
            valid_pixels = gray.flatten()

        p5 = np.percentile(valid_pixels, 5)    # ~traçado escuro
        p50 = np.percentile(valid_pixels, 50)  # ~mediana (grid+fundo)
        p90 = np.percentile(valid_pixels, 90)  # ~fundo claro

        # Threshold: ponto médio entre traçado e grid.
        # Para grids de baixa saturação (fundo ~220, grid ~190, traçado ~80):
        # threshold fica em ~135, capturando o traçado sem o grid.
        trace_threshold = int((p5 + p50) / 2)
        trace_threshold = max(80, min(trace_threshold, 180))

        binary = np.zeros((h, w), dtype=np.uint8)
        binary[gray < trace_threshold] = 255

        logger.info(
            "Leader mock: p5=%d, p50=%d, p90=%d, threshold=%d, trace_px=%d",
            int(p5), int(p50), int(p90), trace_threshold, np.sum(binary > 0),
        )

        # --- Fase 2: remover linhas retas residuais do grid ---
        h_len = max(w // 30, 30)
        v_len = max(h // 30, 30)
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1))
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len))

        h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
        v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)

        binary = cv2.subtract(binary, h_lines)
        binary = cv2.subtract(binary, v_lines)

        # --- Fase 3: limpar componentes pequenos (ruído, labels, texto) ---
        # Labels como "I", "II", "V1" são ~100-500px. Traçados ECG reais
        # numa linha inteira formam componentes de milhares de pixels.
        min_area = max(int(h * w * 0.0003), 100)
        labeled, n_comp = ndlabel(binary > 0)
        for comp_id in range(1, n_comp + 1):
            if np.sum(labeled == comp_id) < min_area:
                binary[labeled == comp_id] = 0

        # --- Fase 4: mascarar margens (cabeçalho, rodapé, laterais) ---
        # Texto impresso fica no topo (~13%) e base (~4%).
        # Carimbos ficam tipicamente na margem direita (~8%).
        header_h = int(h * 0.13)
        footer_h = int(h * 0.04)
        right_margin = int(w * 0.05)
        binary[:header_h, :] = 0
        binary[h - footer_h:, :] = 0
        binary[:, w - right_margin:] = 0

        # Fechar micro-gaps no traçado
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        logger.info("Leader mock: %d px de traçado", np.sum(binary > 0))
        return binary

    # =====================================================================
    # Módulo 6: Extração de sinal (máscara → 12 arrays µV)
    # =====================================================================

    def extract_signals(
        self, lead_mask: np.ndarray, px_per_mm: float
    ) -> tuple[dict[str, np.ndarray], dict]:
        """Converte máscara de leads em 12 sinais digitais.

        Detecção automática de layout pelo número de faixas horizontais:
          - 3-4 faixas → layout 3×4+1 (3 linhas × 4 cols + rhythm)
          - 6-7 faixas → layout 6×2+1 (6 linhas × 2 cols + rhythm)
          - Outro      → fallback 3×4+1 com divisão uniforme

        Pipeline por derivação:
          1. Extrai sinal por varredura vertical (centro de massa)
          2. Converte pixel → µV
          3. Interpola gaps, filtra ruído
          4. Resample para 500 Hz

        Returns:
            (signals_dict, extraction_info)
        """
        h, w = lead_mask.shape[:2]
        info: dict = {"leads_extracted": 0, "nan_ratio": {}, "layout": "unknown"}

        # --- Dividir em faixas horizontais ---
        row_boundaries = self._find_row_boundaries(lead_mask)
        n_bands = len(row_boundaries)

        # --- Detectar layout pelo número de faixas ---
        if n_bands >= 6:
            layout_name = "6x2+1"
            lead_layout = LEAD_LAYOUT_6x2
            n_data_rows = 6
            n_cols = 2
            short_sec = SHORT_LEAD_SEC_6x2
            has_rhythm = n_bands >= 7
        elif n_bands >= 3:
            layout_name = "3x4+1"
            lead_layout = LEAD_LAYOUT_3x4
            n_data_rows = 3
            n_cols = 4
            short_sec = SHORT_LEAD_SEC_3x4
            has_rhythm = n_bands >= 4
        else:
            # Fallback: divisão uniforme assumindo 3×4+1
            layout_name = "3x4+1"
            lead_layout = LEAD_LAYOUT_3x4
            n_data_rows = 3
            n_cols = 4
            short_sec = SHORT_LEAD_SEC_3x4
            row_h = h // 4
            row_boundaries = [
                (0, row_h),
                (row_h, 2 * row_h),
                (2 * row_h, 3 * row_h),
                (3 * row_h, h),
            ]
            has_rhythm = True

        info["layout"] = layout_name
        logger.info(
            "Layout detectado: %s (%d faixas, rhythm=%s)",
            layout_name, n_bands, has_rhythm,
        )

        # --- Extrair derivações curtas ---
        signals: dict[str, np.ndarray] = {}

        for row_idx in range(min(n_data_rows, len(row_boundaries))):
            y1, y2 = row_boundaries[row_idx]
            row_mask = lead_mask[y1:y2, :]

            # Detectar gaps reais entre colunas via projeção horizontal
            col_boundaries = self._find_col_boundaries(row_mask, n_cols)

            for col_idx in range(min(n_cols, len(col_boundaries))):
                if row_idx >= len(lead_layout) or col_idx >= len(lead_layout[row_idx]):
                    continue
                lead_name = lead_layout[row_idx][col_idx]
                x1, x2 = col_boundaries[col_idx]

                roi = row_mask[:, x1:x2]
                raw_signal = self._extract_signal_from_roi(roi)
                signal_uv = self._pixel_to_uv(raw_signal, px_per_mm)

                signals[lead_name] = signal_uv
                nan_count = np.sum(np.isnan(signal_uv))
                info["nan_ratio"][lead_name] = nan_count / max(len(signal_uv), 1)

        # --- Extrair rhythm lead (DII longo) ---
        rhythm_row_idx = n_data_rows  # primeira faixa após as derivações curtas
        if has_rhythm and rhythm_row_idx < len(row_boundaries):
            y1, y2 = row_boundaries[rhythm_row_idx]
            rhythm_roi = lead_mask[y1:y2, :]
            raw_rhythm = self._extract_signal_from_roi(rhythm_roi)
            signals[f"{RHYTHM_LEAD}_long"] = self._pixel_to_uv(raw_rhythm, px_per_mm)

        # --- Limpar sinais ---
        for lead_name in signals:
            signals[lead_name] = self._clean_signal(signals[lead_name], px_per_mm)

        info["leads_extracted"] = len(
            [k for k in signals if not k.endswith("_long")]
        )

        # --- Resample para 500 Hz ---
        for lead_name in list(signals.keys()):
            sig = signals[lead_name]
            if lead_name.endswith("_long"):
                target_len = int(LONG_LEAD_SEC * SAMPLING_RATE)
            else:
                target_len = int(short_sec * SAMPLING_RATE)
            signals[lead_name] = self._resample(sig, target_len)

        native_sr = int(round(px_per_mm * PAPER_SPEED_DEFAULT))
        logger.info(
            "Extração: %s, %d derivações, px/mm=%.1f, native_sr=%d Hz",
            layout_name, info["leads_extracted"], px_per_mm, native_sr,
        )
        return signals, info

    def _find_col_boundaries(
        self, row_mask: np.ndarray, expected_cols: int
    ) -> list[tuple[int, int]]:
        """Detecta fronteiras de colunas em uma faixa horizontal via projeção.

        Garante cobertura total da largura: a primeira coluna sempre começa
        em x=0 e a última sempre termina em x=w. Os separadores internos são
        os pontos médios entre blocos de traçado detectados; se a detecção
        falhar, faz divisão uniforme de 0 a w.

        Returns:
            Lista de (x_start, x_end) para cada coluna.
        """
        h, w = row_mask.shape[:2]

        # Projeção horizontal: quantos pixels brancos em cada coluna X
        projection = np.sum(row_mask > 0, axis=0).astype(float)

        # Suavizar para evitar micro-gaps
        kernel_size = max(w // 200, 3)
        if kernel_size % 2 == 0:
            kernel_size += 1
        projection = medfilt(projection, kernel_size=kernel_size)

        # Threshold: coluna "ativa" se tem traçado significativo
        threshold = h * 0.02
        active = projection > threshold

        # Encontrar blocos contíguos de atividade
        blocks: list[tuple[int, int]] = []
        in_block = False
        start = 0
        min_block_w = max(w // (expected_cols * 4), 10)  # bloco mínimo

        for x in range(w):
            if active[x] and not in_block:
                start = x
                in_block = True
            elif not active[x] and in_block:
                if x - start >= min_block_w:
                    blocks.append((start, x))
                in_block = False
        if in_block and w - start >= min_block_w:
            blocks.append((start, w))

        # Se encontrou mais blocos que o esperado, mesclar os mais próximos
        if len(blocks) > expected_cols:
            while len(blocks) > expected_cols:
                min_gap = float("inf")
                min_idx = 0
                for i in range(len(blocks) - 1):
                    gap = blocks[i + 1][0] - blocks[i][1]
                    if gap < min_gap:
                        min_gap = gap
                        min_idx = i
                merged = (blocks[min_idx][0], blocks[min_idx + 1][1])
                blocks = blocks[:min_idx] + [merged] + blocks[min_idx + 2:]

        # Se encontrou o número esperado, usar pontos médios entre blocos
        # como separadores — primeira col começa em 0, última termina em w
        if len(blocks) == expected_cols:
            separators: list[int] = []
            for i in range(expected_cols - 1):
                mid = (blocks[i][1] + blocks[i + 1][0]) // 2
                separators.append(mid)
            cuts = [0] + separators + [w]
            result = [(cuts[i], cuts[i + 1]) for i in range(expected_cols)]
            logger.debug(
                "Col detection: %d blocos -> separadores %s",
                expected_cols, separators,
            )
            return result

        # Fallback: divisão uniforme de 0 a w
        logger.debug(
            "Col detection: %d blocos (esperado %d), fallback uniforme",
            len(blocks), expected_cols,
        )
        col_w = w / expected_cols
        return [
            (
                int(round(i * col_w)),
                w if i == expected_cols - 1 else int(round((i + 1) * col_w)),
            )
            for i in range(expected_cols)
        ]

    def _enforce_min_row_height(
        self,
        boundaries: list[tuple[int, int]],
        img_h: int,
        expected_rows: int = 4,
    ) -> list[tuple[int, int]]:
        """Expande faixas menores que (img_h / expected_rows) * 0.5.

        Expande simetricamente em torno do centro da faixa, respeitando
        vizinhos adjacentes e as bordas da imagem.
        """
        if not boundaries:
            return boundaries

        min_h = int((img_h / max(expected_rows, 1)) * 0.5)
        result = list(boundaries)

        for i, (y1, y2) in enumerate(result):
            cur_h = y2 - y1
            if cur_h >= min_h:
                continue
            center = (y1 + y2) / 2
            half = min_h / 2
            new_y1 = int(round(center - half))
            new_y2 = int(round(center + half))

            # Limite superior: borda ou vizinho de cima
            upper_limit = 0 if i == 0 else result[i - 1][1]
            # Limite inferior: borda ou vizinho de baixo
            lower_limit = img_h if i == len(result) - 1 else result[i + 1][0]

            new_y1 = max(new_y1, upper_limit)
            new_y2 = min(new_y2, lower_limit)

            # Se ainda cabe expansão assimétrica, tentar alcançar min_h
            if new_y2 - new_y1 < min_h:
                deficit = min_h - (new_y2 - new_y1)
                room_up = new_y1 - upper_limit
                room_down = lower_limit - new_y2
                take_up = min(deficit, room_up)
                new_y1 -= take_up
                deficit -= take_up
                take_down = min(deficit, room_down)
                new_y2 += take_down

            result[i] = (new_y1, new_y2)
            logger.info(
                "Row %d expandida de %dpx para %dpx (min=%d): y=[%d:%d]",
                i, cur_h, new_y2 - new_y1, min_h, new_y1, new_y2,
            )

        return result

    def _find_row_boundaries(self, mask: np.ndarray) -> list[tuple[int, int]]:
        """Encontra faixas horizontais com traçado via projeção vertical.

        Usa duas estratégias:
        1. Detecção de gaps (zonas com poucos pixels brancos) — funciona
           quando as faixas estão bem separadas.
        2. Se apenas 1-2 blocos detectados, tenta subdividir usando os
           vales (mínimos locais) na projeção — funciona quando faixas
           estão conectadas por picos altos (ex: QRS de V1-V3).
        """
        h, w = mask.shape[:2]
        projection = np.sum(mask > 0, axis=1).astype(float)

        kernel_size = max(h // 50, 5)
        if kernel_size % 2 == 0:
            kernel_size += 1
        proj_smooth = medfilt(projection, kernel_size=kernel_size)

        threshold = w * 0.01
        active = proj_smooth > threshold

        # --- Estratégia 1: gaps naturais ---
        boundaries = []
        in_block = False
        start = 0
        min_block_h = max(int(h * 0.03), 5)

        for y in range(h):
            if active[y] and not in_block:
                start = y
                in_block = True
            elif not active[y] and in_block:
                if y - start > min_block_h:
                    boundaries.append((start, y))
                in_block = False
        if in_block and h - start > min_block_h:
            boundaries.append((start, h))

        # Se encontrou 3+ faixas, ok
        if len(boundaries) >= 3:
            return self._enforce_min_row_height(boundaries, h)

        # --- Estratégia 2: subdividir por vales na projeção ---
        # Quando os traçados estão conectados, a projeção tem picos
        # por faixa com mínimos relativos entre eles.
        if len(boundaries) == 0:
            # Nenhuma faixa detectada — dividir uniformemente
            row_h = h // 4
            return [(i * row_h, (i + 1) * row_h if i < 3 else h)
                    for i in range(4)]

        # 1-2 blocos: tentar subdividir o maior bloco
        # Pegar o bloco que cobre mais altura
        main = max(boundaries, key=lambda b: b[1] - b[0])
        y_start, y_end = main
        block_h = y_end - y_start

        # Projeção dentro do bloco principal
        block_proj = proj_smooth[y_start:y_end].copy()

        # Suavizar mais para encontrar envoltória
        big_kernel = max(block_h // 15, 5)
        if big_kernel % 2 == 0:
            big_kernel += 1
        block_proj_smooth = medfilt(block_proj, kernel_size=big_kernel)

        # Encontrar mínimos locais (vales entre faixas)
        # Um vale é um ponto menor que seus vizinhos numa janela
        valley_window = max(block_h // 12, 5)
        valleys = []
        for y in range(valley_window, len(block_proj_smooth) - valley_window):
            local_min = np.min(block_proj_smooth[
                max(0, y - valley_window):y + valley_window + 1
            ])
            if block_proj_smooth[y] == local_min:
                # Vale significativo: abaixo de 60% da mediana
                if block_proj_smooth[y] < np.median(block_proj_smooth) * 0.6:
                    valleys.append(y)

        # Filtrar vales muito próximos (manter o mais profundo)
        if valleys:
            min_gap = max(block_h // 8, 10)
            filtered = [valleys[0]]
            for v in valleys[1:]:
                if v - filtered[-1] < min_gap:
                    # Manter o mais profundo
                    if block_proj_smooth[v] < block_proj_smooth[filtered[-1]]:
                        filtered[-1] = v
                else:
                    filtered.append(v)
            valleys = filtered

        if len(valleys) >= 2:
            # Usar os vales como separadores
            cuts = [y_start] + [y_start + v for v in valleys] + [y_end]
            new_boundaries = []
            for i in range(len(cuts) - 1):
                band_h = cuts[i + 1] - cuts[i]
                if band_h > min_block_h:
                    new_boundaries.append((cuts[i], cuts[i + 1]))
            if len(new_boundaries) >= 3:
                logger.info(
                    "Row detection: subdividido em %d faixas por vales",
                    len(new_boundaries),
                )
                return self._enforce_min_row_height(new_boundaries, h)

        # Fallback: dividir o bloco principal uniformemente
        # Estimar número de faixas pelo aspect ratio
        expected_rows = 4  # 3 derivações + rhythm
        row_h = block_h // expected_rows
        fallback = [
            (y_start + i * row_h,
             y_start + (i + 1) * row_h if i < expected_rows - 1 else y_end)
            for i in range(expected_rows)
        ]
        logger.info("Row detection: fallback uniforme %d faixas", expected_rows)
        return fallback

    def _extract_signal_from_roi(self, roi: np.ndarray) -> np.ndarray:
        """Extrai sinal 1D por varredura vertical (centro de massa dos pixels brancos)."""
        h, w = roi.shape[:2]
        signal = np.full(w, np.nan, dtype=np.float64)

        for x in range(w):
            col = roi[:, x]
            white_pixels = np.where(col > 0)[0]
            if len(white_pixels) > 0:
                weights = col[white_pixels].astype(np.float64)
                center = np.average(white_pixels, weights=weights)
                signal[x] = h - center  # inverter Y (ECG: amplitude cresce para cima)

        return signal

    def _pixel_to_uv(self, signal: np.ndarray, px_per_mm: float) -> np.ndarray:
        """Converte sinal de pixels para µV.

        amplitude_µV = (y_pixel - baseline) × 1000 / (px_per_mm × gain_mm_per_mV)
        Centraliza na mediana (baseline = 0).
        """
        if px_per_mm <= 0:
            return signal

        result = signal.copy()
        valid = ~np.isnan(result)
        if valid.sum() < 2:
            return result

        result = result - np.nanmedian(result)
        uv_per_pixel = 1000.0 / (px_per_mm * GAIN_DEFAULT)
        result = result * uv_per_pixel

        # Clipar artefatos extremos (ECG normal: ±5 mV = ±5000 µV)
        max_uv = 5000.0
        result = np.clip(result, -max_uv, max_uv)
        return result

    def _clean_signal(self, signal: np.ndarray, px_per_mm: float) -> np.ndarray:
        """Interpola gaps NaN e aplica filtro mediano anti-serrilhado.

        Gaps > 50ms são mantidos como NaN.
        """
        valid = ~np.isnan(signal)
        if valid.sum() < 2:
            return np.zeros_like(signal)

        x_valid = np.where(valid)[0]
        f = interp1d(
            x_valid, signal[x_valid],
            kind="linear", fill_value="extrapolate", bounds_error=False,
        )
        result = signal.copy()
        nan_mask = np.isnan(result)
        result[nan_mask] = f(np.where(nan_mask)[0])

        # Marcar gaps grandes (> 50ms) como NaN
        if px_per_mm > 0:
            samples_50ms = int(0.05 * px_per_mm * PAPER_SPEED_DEFAULT)
        else:
            samples_50ms = 25

        gap_runs = self._find_nan_runs(nan_mask)
        for start, length in gap_runs:
            if length > samples_50ms:
                result[start : start + length] = np.nan

        # Filtro mediano suave
        valid_final = ~np.isnan(result)
        if valid_final.sum() > 5:
            valid_values = result[valid_final]
            filtered = medfilt(valid_values, kernel_size=3)
            result[valid_final] = filtered

        # Clipar artefatos extremos (ECG normal: ±5 mV = ±5000 µV)
        result = np.clip(result, -5000.0, 5000.0)

        return result

    @staticmethod
    def _find_nan_runs(mask: np.ndarray) -> list[tuple[int, int]]:
        """Encontra runs contíguos de True."""
        runs = []
        in_run = False
        start = 0
        for i in range(len(mask)):
            if mask[i] and not in_run:
                start = i
                in_run = True
            elif not mask[i] and in_run:
                runs.append((start, i - start))
                in_run = False
        if in_run:
            runs.append((start, len(mask) - start))
        return runs

    @staticmethod
    def _resample(signal: np.ndarray, target_len: int) -> np.ndarray:
        """Resample para target_len via interpolação linear.

        Usa o último/primeiro valor válido para preencher bordas
        em vez de extrapolar (que produz artefatos extremos).
        """
        n = len(signal)
        if n == target_len:
            return signal
        if n < 2:
            return np.zeros(target_len, dtype=np.float64)

        valid = ~np.isnan(signal)
        if valid.sum() < 2:
            return np.zeros(target_len, dtype=np.float64)

        valid_vals = signal[valid]
        x_old = np.linspace(0, 1, n)
        x_new = np.linspace(0, 1, target_len)

        # fill_value = (primeiro válido, último válido) — sem extrapolação
        f = interp1d(
            x_old[valid], valid_vals,
            kind="linear", bounds_error=False,
            fill_value=(valid_vals[0], valid_vals[-1]),
        )
        return f(x_new)


# =========================================================================
# Função de conveniência — compatível com orchestrator existente
# =========================================================================

def digitize_ecg_pmcardio(image) -> np.ndarray:
    """Digitaliza ECG via pipeline PMcardio (mock).

    Aceita PIL Image ou caminho de arquivo.
    Retorna np.ndarray shape (12, 5000) em mV a 500 Hz.
    """
    import tempfile
    from PIL import Image

    if hasattr(image, "save"):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            image.save(tmp.name)
            image_path = tmp.name
    else:
        image_path = str(image)

    digitizer = ECGDigitizer(use_mock=True)
    result = digitizer.run(image_path)

    signal = np.zeros((12, OUTPUT_LENGTH), dtype=np.float64)

    for i, lead_name in enumerate(LEAD_ORDER):
        if lead_name in result["signals"]:
            sig = result["signals"][lead_name]
            sig_mv = sig / 1000.0  # µV → mV
            n = min(len(sig_mv), OUTPUT_LENGTH)
            signal[i, :n] = sig_mv[:n]
            if n < OUTPUT_LENGTH and n > 0:
                signal[i, n:] = sig_mv[n - 1]

    return np.nan_to_num(signal, nan=0.0)
