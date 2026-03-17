"""
debug_pipeline.py — Roda o pipeline passo a passo com debug visual.
Salva imagens de cada etapa para inspeção.
"""

import os
import sys

_modal_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _modal_dir)

import numpy as np
from PIL import Image

from pipeline.digitize.preprocess import preprocess
from pipeline.digitize.segmentation import segment, _load_model, _run_single_pass, _run_sliding_window
from pipeline.digitize.constants import UNET_PATCH_SIZE

OUT = os.path.join(_modal_dir, "scripts", "debug_output")
os.makedirs(OUT, exist_ok=True)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "../docs/ECG_teste.jpeg"
    img_pil = Image.open(path)
    print(f"Imagem: {img_pil.size} {img_pil.mode}")

    # M0
    img_np = preprocess(img_pil)
    H, W = img_np.shape[:2]
    print(f"Preprocessed: {img_np.shape}")

    # M1 — segmentação com logits raw
    import torch
    model = _load_model()
    tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float()

    if H <= UNET_PATCH_SIZE and W <= UNET_PATCH_SIZE:
        logits = _run_single_pass(model, tensor)
    else:
        logits = _run_sliding_window(model, tensor)

    # logits shape: (1, 4, H, W)
    probs = _softmax(logits[0])  # (4, H, W)
    class_map = np.argmax(logits[0], axis=0)

    # Salvar masks
    labels = ["background", "grid", "trace", "text"]
    for i, name in enumerate(labels):
        mask = (class_map == i)
        pct = mask.sum() / mask.size * 100
        print(f"  Mask {name}: {mask.sum()} px ({pct:.1f}%)")

        # Mask binária
        Image.fromarray((mask.astype(np.uint8) * 255)).save(
            os.path.join(OUT, f"debug_01_mask_{name}.png"))

        # Probabilidade (softmax)
        prob_img = (probs[i] * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(prob_img).save(
            os.path.join(OUT, f"debug_01_prob_{name}.png"))

    # Salvar grid prob como float para análise
    grid_prob = probs[1]  # classe 1 = grid
    grid_mask = (class_map == 1)
    trace_mask = (class_map == 2)

    # Debug autocorrelação
    print("\n--- Debug Autocorrelação ---")
    _debug_autocorrelation(grid_mask, grid_prob, H, W)

    # Salvar trace mask
    Image.fromarray((trace_mask.astype(np.uint8) * 255)).save(
        os.path.join(OUT, "debug_02_mask_trace.png"))

    print(f"\nImagens salvas em: {OUT}")


def _softmax(logits):
    """Softmax no eixo 0."""
    e = np.exp(logits - logits.max(axis=0, keepdims=True))
    return e / e.sum(axis=0, keepdims=True)


def _debug_autocorrelation(grid_mask, grid_prob, H, W):
    """Analisa autocorrelação tanto da mask binária quanto da probabilidade."""
    from scipy.signal import find_peaks

    for name, data in [("mask_binaria", grid_mask.astype(np.float64)),
                        ("probabilidade", grid_prob.astype(np.float64))]:
        print(f"\n  [{name}]")

        # Perfil horizontal
        h_profile = data.sum(axis=0)
        print(f"    Perfil H: len={len(h_profile)}, max={h_profile.max():.1f}, "
              f"mean={h_profile.mean():.1f}, nonzero={np.count_nonzero(h_profile)}")

        # Perfil vertical
        v_profile = data.sum(axis=1)
        print(f"    Perfil V: len={len(v_profile)}, max={v_profile.max():.1f}, "
              f"mean={v_profile.mean():.1f}, nonzero={np.count_nonzero(v_profile)}")

        # Autocorrelação horizontal
        _analyze_autocorr(h_profile, f"H_{name}", "horizontal")
        _analyze_autocorr(v_profile, f"V_{name}", "vertical")


def _analyze_autocorr(profile, label, axis_name):
    """Calcula e salva autocorrelação com debug."""
    from scipy.signal import find_peaks
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sig = profile.astype(np.float64)
    sig = sig - sig.mean()
    std = sig.std()
    if std < 1e-6:
        print(f"    Autocorr {axis_name}: std=0, skip")
        return

    sig = sig / std
    n = len(sig)
    autocorr = np.correlate(sig, sig, mode="full")
    autocorr = autocorr[n - 1:]
    autocorr = autocorr / autocorr[0]

    # Encontrar picos com thresholds progressivamente relaxados
    for h_thresh, p_thresh, desc in [
        (0.15, 0.02, "original"),
        (0.05, 0.01, "relaxado"),
        (0.02, 0.005, "muito_relaxado"),
    ]:
        peaks, props = find_peaks(autocorr, height=h_thresh, prominence=p_thresh, distance=3)
        if len(peaks) > 0:
            print(f"    Autocorr {axis_name} [{desc}]: {len(peaks)} picos, "
                  f"primeiro={peaks[0]}px, alturas={[f'{h:.3f}' for h in props['peak_heights'][:5]]}")
        else:
            print(f"    Autocorr {axis_name} [{desc}]: 0 picos")

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(14, 6))

    axes[0].plot(profile, linewidth=0.5)
    axes[0].set_title(f"Perfil {axis_name} ({label})")
    axes[0].set_ylabel("Soma pixels")

    max_lag = min(200, n // 2)
    axes[1].plot(autocorr[:max_lag], linewidth=0.8)
    axes[1].axhline(0.15, color="red", linestyle="--", alpha=0.5, label="h=0.15")
    axes[1].axhline(0.05, color="orange", linestyle="--", alpha=0.5, label="h=0.05")
    axes[1].set_title(f"Autocorrelação {axis_name}")
    axes[1].set_xlabel("Lag (pixels)")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"debug_03_autocorr_{label}.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
