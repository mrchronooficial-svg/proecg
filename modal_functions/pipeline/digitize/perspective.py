"""
M2 — Correção de Perspectiva.

Input:  imagem float32 (H, W, 3) + dict de masks
Output: imagem e masks corrigidas (dewarped)

Usa Hough Lines na mask do grid para detectar ângulo de rotação.
Fallbacks progressivos: threshold relaxado -> sem correção.
"""

import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Limites de ângulo
_MIN_ANGLE_DEG = 0.3    # Abaixo disso, não vale corrigir
_MAX_SKEW_DEG = 15.0    # Acima disso, pode ser foto de lado (90°)
_NEAR_90_TOLERANCE = 10.0  # Tolerância para detectar rotação de ~90°


def _detect_angle_from_grid(grid_mask: np.ndarray,
                            hough_threshold: int = 100,
                            min_line_length: int = 80) -> Optional[float]:
    """Detecta ângulo de rotação via Hough Lines na mask do grid.

    Returns:
        Ângulo em graus (positivo = sentido anti-horário), ou None.
    """
    mask_u8 = (grid_mask.astype(np.uint8) * 255)

    lines = cv2.HoughLinesP(
        mask_u8,
        rho=1,
        theta=np.pi / 180,
        threshold=hough_threshold,
        minLineLength=min_line_length,
        maxLineGap=10,
    )

    if lines is None or len(lines) < 3:
        return None

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx = x2 - x1
        dy = y2 - y1
        length = np.sqrt(dx * dx + dy * dy)
        if length < 20:
            continue
        # Ângulo em relação à horizontal
        angle = np.degrees(np.arctan2(dy, dx))
        # Normalizar para [-45, 45] (linhas de grid são ~0° ou ~90°)
        angle = angle % 180
        if angle > 90:
            angle -= 180
        if abs(angle) < 45:
            angles.append(angle)

    if len(angles) < 3:
        return None

    # Mediana é robusta a outliers
    median_angle = float(np.median(angles))
    logger.info("Hough: %d linhas, angulo mediano = %.2f°", len(angles), median_angle)
    return median_angle


def _rotate_image(image: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotaciona imagem pelo centro, preenchendo bordas com a cor média."""
    H, W = image.shape[:2]
    center = (W / 2, H / 2)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)

    if image.ndim == 3:
        border_value = tuple(image.mean(axis=(0, 1)).tolist())
    else:
        border_value = float(image.mean())

    return cv2.warpAffine(
        image, M, (W, H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )


def _rotate_mask(mask: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotaciona mask booleana (nearest neighbor para não criar valores intermediários)."""
    H, W = mask.shape
    center = (W / 2, H / 2)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated = cv2.warpAffine(
        mask.astype(np.uint8), M, (W, H),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return rotated.astype(bool)


def correct_perspective(
    image: np.ndarray,
    masks: dict,
) -> tuple:
    """Corrige rotação/perspectiva do ECG.

    Args:
        image: float32 (H, W, 3)
        masks: dict com "background", "grid", "trace", "text"

    Returns:
        (image_corrected, masks_corrected)
        Se não for possível corrigir, retorna inputs inalterados.
    """
    grid_mask = masks.get("grid")
    if grid_mask is None or grid_mask.sum() < 100:
        logger.warning("Grid mask vazia ou ausente — sem correção de perspectiva")
        return image, masks

    # Tentar com thresholds padrão
    angle = _detect_angle_from_grid(grid_mask)

    # Fallback: thresholds mais relaxados
    if angle is None:
        logger.info("Hough padrão falhou, tentando com thresholds relaxados")
        angle = _detect_angle_from_grid(grid_mask, hough_threshold=50, min_line_length=40)

    if angle is None:
        logger.warning("Não foi possível detectar ângulo — prosseguindo sem correção")
        return image, masks

    # Detectar rotação de ~90° (foto de lado)
    if abs(angle) > (90 - _NEAR_90_TOLERANCE):
        correction = angle - 90 if angle > 0 else angle + 90
        logger.info("Detectada rotação ~90° (%.1f°), corrigindo por %.1f°", angle, correction)
        angle = correction

    # Não corrigir se ângulo muito pequeno
    if abs(angle) < _MIN_ANGLE_DEG:
        logger.info("Ângulo %.2f° < %.1f° — sem correção necessária", angle, _MIN_ANGLE_DEG)
        return image, masks

    # Ângulo muito grande pode indicar problema
    if abs(angle) > _MAX_SKEW_DEG:
        logger.warning("Ângulo %.1f° muito grande — prosseguindo sem correção", angle)
        return image, masks

    logger.info("Corrigindo rotação de %.2f°", angle)

    # Rotacionar imagem e todas as masks
    image_corrected = _rotate_image(image, angle)
    masks_corrected = {}
    for name, mask in masks.items():
        masks_corrected[name] = _rotate_mask(mask, angle)

    return image_corrected, masks_corrected
