"""
Conversor CVAT → máscara binária do Leader
==========================================

Lê o export CVAT (annotations.xml) e gera, pra cada imagem que existe na
pasta de imagens normalizadas, uma máscara binária PNG (branco = traçado,
preto = fundo) — exatamente como seria a saída do módulo Leader.

CVAT RLE format:
    Cada `<mask>` contém:
      - left, top, width, height: bbox do mask na imagem
      - rle: lista de comprimentos de runs em row-major, alternando 0/1
        (começando com 0 = background), cobrindo width × height pixels.

Uso:
    python -m modal_functions.pipeline.digitize.cvat_to_masks
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import cv2
import numpy as np

CVAT_ZIP = Path(r"C:\Users\rafae\Downloads\teste leader.zip")
NORMALIZED_DIR = Path(r"C:\Users\rafae\Desktop\Projeto ECG\ECGs Normalizados Leader")
OUTPUT_DIR = Path(r"C:\Users\rafae\Desktop\Projeto ECG\Leader Masks Teste")


def _decode_cvat_rle(rle: list[int], box_w: int, box_h: int) -> np.ndarray:
    """Decodifica RLE do CVAT pra máscara binária (uint8) de (box_h, box_w).

    Os runs alternam 0/1 começando com 0 (bg). Row-major.
    """
    total = box_w * box_h
    flat = np.zeros(total, dtype=np.uint8)
    pos = 0
    val = 0
    for run in rle:
        run = int(run)
        if run > 0:
            end = min(pos + run, total)
            if val == 1:
                flat[pos:end] = 255
            pos = end
        val ^= 1  # alterna 0 ↔ 1
        if pos >= total:
            break
    return flat.reshape(box_h, box_w)


def _render_image_masks(image_elem: ET.Element) -> tuple[np.ndarray, int]:
    """Junta todas as máscaras 'lead_trace' de uma <image> numa máscara binária.

    Returns:
        (mask_uint8, n_polygons_renderizadas)
    """
    img_w = int(image_elem.get("width"))
    img_h = int(image_elem.get("height"))
    canvas = np.zeros((img_h, img_w), dtype=np.uint8)

    n_traces = 0
    for mask_elem in image_elem.findall("mask"):
        if mask_elem.get("label") != "lead_trace":
            continue

        left = int(mask_elem.get("left"))
        top = int(mask_elem.get("top"))
        bw = int(mask_elem.get("width"))
        bh = int(mask_elem.get("height"))
        rle_str = mask_elem.get("rle", "").strip()
        if not rle_str:
            continue
        rle = [int(x) for x in rle_str.split(",") if x.strip()]
        sub = _decode_cvat_rle(rle, bw, bh)

        # Recorta caso a bbox extrapole (defesa)
        x1, y1 = max(0, left), max(0, top)
        x2 = min(img_w, left + bw)
        y2 = min(img_h, top + bh)
        sub_x1 = x1 - left
        sub_y1 = y1 - top
        sub_x2 = sub_x1 + (x2 - x1)
        sub_y2 = sub_y1 + (y2 - y1)
        if sub_x2 <= sub_x1 or sub_y2 <= sub_y1:
            continue

        # OR no canvas
        canvas[y1:y2, x1:x2] = np.maximum(
            canvas[y1:y2, x1:x2], sub[sub_y1:sub_y2, sub_x1:sub_x2]
        )
        n_traces += 1

    return canvas, n_traces


def main() -> int:
    if not CVAT_ZIP.exists():
        print(f"ERRO: ZIP nao encontrado: {CVAT_ZIP}", file=sys.stderr)
        return 1
    if not NORMALIZED_DIR.exists():
        print(f"ERRO: pasta normalizadas nao encontrada: {NORMALIZED_DIR}", file=sys.stderr)
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Carrega o XML do ZIP
    with zipfile.ZipFile(CVAT_ZIP) as z:
        with z.open("annotations.xml") as f:
            tree = ET.parse(f)
    root = tree.getroot()

    # Inventário de imagens disponíveis na pasta normalizada (case-insensitive)
    available = {p.name: p for p in NORMALIZED_DIR.iterdir() if p.is_file()}

    image_elems = root.findall("image")
    print(f"Anotacoes no XML: {len(image_elems)}")
    print(f"Imagens normalizadas disponiveis: {len(available)}")
    print()

    n_processed = 0
    n_skipped = 0
    for image_elem in image_elems:
        name = image_elem.get("name")
        if name not in available:
            n_skipped += 1
            continue

        img_w = int(image_elem.get("width"))
        img_h = int(image_elem.get("height"))

        # Validação: tamanho do XML deve bater com a imagem real
        real = cv2.imread(str(available[name]))
        if real is None:
            print(f"  AVISO: nao abriu {name}, pulando")
            n_skipped += 1
            continue
        rh, rw = real.shape[:2]
        if (rh, rw) != (img_h, img_w):
            print(
                f"  AVISO: {name}: dimensoes XML ({img_w}x{img_h}) != imagem ({rw}x{rh}) "
                f"— ainda assim renderizando no tamanho da imagem"
            )

        mask, n_traces = _render_image_masks(image_elem)

        # Se imagem tem tamanho diferente do XML, redimensiona a mascara pra
        # casar com a imagem real
        if mask.shape != (rh, rw):
            mask = cv2.resize(mask, (rw, rh), interpolation=cv2.INTER_NEAREST)

        out_path = OUTPUT_DIR / name
        cv2.imwrite(str(out_path), mask)
        n_processed += 1

        coverage = float(np.sum(mask > 0)) / (rh * rw) * 100
        print(
            f"  OK  {name:55s}  {rw}x{rh}  {n_traces} traces  "
            f"cobertura={coverage:.2f}%"
        )

    print()
    print(f"Processadas: {n_processed}  |  Ignoradas (sem imagem): {n_skipped}")
    print(f"Mascaras salvas em: {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
