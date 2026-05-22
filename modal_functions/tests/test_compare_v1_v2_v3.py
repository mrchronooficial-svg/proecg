"""
Investiga diferenca entre IMG_1407/ecg_digital.png (fina) e
IMG_1407/ecg_digital_v2.png (grossa). Gera V3 com MESMO metodo da V1
(alpha gradient pela prob).

Output:
  resultados_teste_v1/comparacao_v1_v2_1407.png — comparacao lado a lado
  resultados_teste_v1/ecg_digital_v3_1407.png   — V3 com alpha gradient
  resultados_teste_v1/relatorio_diferenca.txt   — explicacao
"""

import logging
import pickle
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
MODAL_ROOT = HERE.parent
sys.path.insert(0, str(MODAL_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("compare_v1_v2_v3")

RESULTS_ROOT = Path(r"C:\Users\rafae\Desktop\Projeto ECG\resultados_teste_v1")
CACHE_PATH = RESULTS_ROOT / "IMG_1407" / "_cache_pre_signal_extractor.pkl"

V1_PATH = RESULTS_ROOT / "IMG_1407" / "ecg_digital.png"
V2_PATH = RESULTS_ROOT / "IMG_1407" / "ecg_digital_v2.png"
V3_PATH = RESULTS_ROOT / "ecg_digital_v3_1407.png"
COMP_PATH = RESULTS_ROOT / "comparacao_v1_v2_1407.png"
RELATORIO_PATH = RESULTS_ROOT / "relatorio_diferenca.txt"


def make_v3_alpha(cache, output_size: tuple[int, int] = (1200, 600)):
    """V3: usa MESMO metodo da V1 — alpha gradient pela prob da UNet."""
    signal_prob = cache["signal_prob"]
    grid_prob = cache["grid_prob"]
    text_prob = cache["text_prob"]
    H, W = signal_prob.shape

    # Canvas RGB branco
    canvas = np.full((H, W, 3), 255, dtype=np.uint8)

    # GRID — alpha por intensidade (rosa-salmao papel ECG)
    GRID_THRESHOLD = 0.05
    grid_mask = grid_prob > GRID_THRESHOLD
    grid_intensity = np.clip(grid_prob / max(grid_prob.max(), 1e-6), 0, 1)
    grid_color = np.array([240, 180, 200], dtype=np.float32)
    alpha = (grid_intensity * 0.8)[..., np.newaxis]
    canvas_f = canvas.astype(np.float32)
    grid_pixels = grid_mask[..., np.newaxis] * alpha
    canvas_f = canvas_f * (1 - grid_pixels) + grid_color * grid_pixels
    canvas = canvas_f.astype(np.uint8)

    # TEXTO — preto puro onde prob > 0.3
    text_mask = text_prob > 0.3
    canvas[text_mask] = [0, 0, 0]

    # TRAÇADO — alpha gradient pela prob (MESMO METODO DA V1)
    TRACE_THRESHOLD = 0.05
    trace_mask = signal_prob > TRACE_THRESHOLD
    trace_intensity = np.clip(signal_prob / max(signal_prob.max(), 1e-6), 0, 1)
    canvas_f = canvas.astype(np.float32)
    alpha_t = (trace_intensity * 1.0)[..., np.newaxis]
    trace_pixels = trace_mask[..., np.newaxis] * alpha_t
    canvas_f = canvas_f * (1 - trace_pixels) + 0.0 * trace_pixels
    canvas = canvas_f.astype(np.uint8)

    # Grayscale + resize
    gray = cv2.cvtColor(canvas, cv2.COLOR_RGB2GRAY)
    target_w, target_h = output_size
    target_aspect = target_w / target_h
    src_aspect = W / H
    if src_aspect < target_aspect:
        new_w = int(round(H * target_aspect))
        pad_lef = (new_w - W) // 2
        pad_rig = new_w - W - pad_lef
        gray = cv2.copyMakeBorder(gray, 0, 0, pad_lef, pad_rig,
                                   cv2.BORDER_CONSTANT, value=255)
    else:
        new_h = int(round(W / target_aspect))
        pad_top = (new_h - H) // 2
        pad_bot = new_h - H - pad_top
        gray = cv2.copyMakeBorder(gray, pad_top, pad_bot, 0, 0,
                                   cv2.BORDER_CONSTANT, value=255)
    gray = cv2.resize(gray, (target_w, target_h), interpolation=cv2.INTER_AREA)
    return gray


def main() -> int:
    with CACHE_PATH.open("rb") as f:
        cache = pickle.load(f)

    # Le V1 e V2 (estavam em IMG_1407/)
    v1_color = cv2.imread(str(V1_PATH), cv2.IMREAD_UNCHANGED)
    v2_color = cv2.imread(str(V2_PATH), cv2.IMREAD_UNCHANGED)
    v1 = cv2.cvtColor(v1_color, cv2.COLOR_BGR2RGB) if v1_color is not None and v1_color.ndim == 3 else v1_color
    v2 = cv2.cvtColor(v2_color, cv2.COLOR_BGR2RGB) if v2_color is not None and v2_color.ndim == 3 else v2_color
    logger.info("V1 shape: %s, V2 shape: %s",
                v1.shape if v1 is not None else "n/a",
                v2.shape if v2 is not None else "n/a")

    # V3 — mesmo metodo da V1 (alpha gradient)
    v3 = make_v3_alpha(cache, output_size=(1200, 600))
    cv2.imwrite(str(V3_PATH), v3)
    logger.info("V3 salvo: %s (%dx%d)", V3_PATH.name, *v3.shape[::-1])

    # Comparacao
    fig, axes = plt.subplots(3, 1, figsize=(16, 18), dpi=120)
    axes[0].imshow(v1 if v1 is not None else np.zeros((100, 100, 3)))
    axes[0].set_title(
        f"V1: IMG_1407/ecg_digital.png — alpha gradient pela prob da UNet "
        f"(full resolution {v1.shape[1] if v1 is not None else '?'}x"
        f"{v1.shape[0] if v1 is not None else '?'} colorida)",
        fontsize=12, fontweight="bold",
    )
    axes[0].axis("off")
    axes[1].imshow(v2 if v2 is not None else np.zeros((100, 100, 3)))
    axes[1].set_title(
        f"V2: IMG_1407/ecg_digital_v2.png — solid black na mascara binarizada "
        f"+ grid matematico rosa "
        f"(full resolution {v2.shape[1] if v2 is not None else '?'}x"
        f"{v2.shape[0] if v2 is not None else '?'})",
        fontsize=12, fontweight="bold",
    )
    axes[1].axis("off")
    axes[2].imshow(v3, cmap="gray", vmin=0, vmax=255)
    axes[2].set_title(
        f"V3 (NOVO): mesmo metodo da V1 (alpha gradient) — 1200x600 grayscale",
        fontsize=12, fontweight="bold",
    )
    axes[2].axis("off")
    fig.tight_layout()
    fig.savefig(str(COMP_PATH), bbox_inches="tight", dpi=120, facecolor="white")
    plt.close(fig)
    logger.info("Comparacao salva: %s", COMP_PATH)

    # Relatorio
    n_mask = int((cache["signal_prob"] > 0.05).sum())
    n_high_prob = int((cache["signal_prob"] > 0.5).sum())
    report = f"""RELATORIO — Diferenca entre ecg_digital.png (V1) e ecg_digital_v2.png (V2)
====================================================================

ARQUIVOS:
  V1: IMG_1407/ecg_digital.png       (full res, RGB, alpha gradient)
  V2: IMG_1407/ecg_digital_v2.png    (full res, RGB, solid black + grid math)
  V3: ecg_digital_v3_1407.png        (NOVO — 1200x600 grayscale, MESMO metodo da V1)

GERADAS POR:
  V1 — modal_functions/tests/test_digital_ecg_1407.py
       (codigo da etapa "Tracado ECG: preto bold" — linhas 80-89)

  V2 — modal_functions/tests/test_digital_ecg_v2.py
       (codigo da etapa "3. Tracado: canal 2 binarizado, em preto" — linhas ~145)


DIFERENCA CHAVE — POR QUE V1 FICA FINA E V2 FICA GROSSA:

V1 (FINA) — usa ALPHA GRADIENT pela prob da UNet:
-----------------------------------------------------
  TRACE_THRESHOLD = 0.05
  trace_mask = signal_prob > TRACE_THRESHOLD
  trace_intensity = signal_prob / signal_prob.max()    ← intensidade gradual [0..1]
  alpha_t = trace_intensity * 1.0
  canvas = canvas * (1 - alpha_t * trace_mask) + 0.0 * alpha_t * trace_mask
  #             ^^^ alpha blending — pixels com prob baixa ficam PARCIALMENTE
  #             pretos, mixados com o cinza do grid embaixo.

  Visualmente: o "core" do trace (prob > 0.5, ~1px de espessura) fica
  preto opaco. A "halo" ao redor (prob 0.1-0.5, ~2-3px) fica cinza
  semi-transparente. RESULTADO: trace parece 1-2px de espessura.


V2 (GROSSA) — usa SOLID BLACK direto na mascara binaria:
---------------------------------------------------------
  TRACE_THRESHOLD = 0.05
  trace_mask = signal_prob > TRACE_THRESHOLD
  canvas[trace_mask] = COLOR_TRACE  # = (0, 0, 0) solid black
  #             ^^^ pixels com QUALQUER prob > 0.05 ficam totalmente pretos.
  #             Nao tem alpha, nao tem intensidade.

  Visualmente: tudo que esta acima do threshold vira PRETO PURO.
  Inclui o "halo" de prob baixa (~2-3px ao redor do trace real).
  RESULTADO: trace parece 3-5px de espessura.


ESTATISTICAS:
  Pixels com prob > 0.05 (V2 pinta TODOS):  {n_mask}
  Pixels com prob > 0.5  (V1 pinta opaco):   {n_high_prob}
  Reducao do "core" pra "halo":              {(1 - n_high_prob/max(n_mask,1)) * 100:.1f}%

V3 — RECONSTRUCAO USANDO METODO V1:
-----------------------------------
Aplica o mesmo alpha gradient + grayscale + resize pra 1200x600.
Resultado deve parecer FINO (como V1) mesmo em resolucao reduzida.


CONCLUSAO:
  V1 e V2 usam o MESMO INPUT (signal_prob > 0.05).
  A diferenca esta apenas no METODO DE PINTURA:
    - V1: alpha-blending → trace gradual, parece fino
    - V2: solid black → trace totalmente preto, parece grosso

PARA PRODUCAO MATCHED COM TREINO:
  Usar alpha gradient (V1) E NAO o solid black da V2.
  Codigo de referencia: test_digital_ecg_1407.py linhas 80-89.
"""
    RELATORIO_PATH.write_text(report, encoding="utf-8")
    logger.info("Relatorio salvo: %s", RELATORIO_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
