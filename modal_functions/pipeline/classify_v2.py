"""Classificador CNN ResNet1d v2 — 24 classes (Ribeiro 2020).

Substitui classify.py para o pipeline novo. Carrega `best_ecg_model_v2.pth`
em `modal_functions/models/classifier/`. NAO mexe na arquitetura — qualquer
alteração faz o modelo produzir lixo.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Arquitetura (ResNet1d Ribeiro)
# ---------------------------------------------------------------------------


def _padding(downsample: int, kernel_size: int) -> int:
    return max(0, int(np.floor((kernel_size - downsample + 1) / 2)))


class ResBlock1d(nn.Module):
    """forward(x, y) — dois tensores separados (residual ANTES do bn2)."""

    def __init__(
        self,
        n_in: int,
        n_out: int,
        downsample: int,
        kernel_size: int = 17,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(
            n_in, n_out, kernel_size,
            padding=_padding(1, kernel_size), bias=False,
        )
        self.bn1 = nn.BatchNorm1d(n_out)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.conv2 = nn.Conv1d(
            n_out, n_out, kernel_size, stride=downsample,
            padding=_padding(downsample, kernel_size), bias=False,
        )
        self.bn2 = nn.BatchNorm1d(n_out)
        self.dropout2 = nn.Dropout(dropout_rate)
        skip: list[nn.Module] = []
        if downsample > 1:
            skip.append(nn.MaxPool1d(downsample, stride=downsample))
        if n_in != n_out:
            skip.append(nn.Conv1d(n_in, n_out, 1, bias=False))
        self.skip_connection = nn.Sequential(*skip) if skip else None

    def forward(self, x: torch.Tensor, y: torch.Tensor):  # type: ignore[override]
        if self.skip_connection is not None:
            y = self.skip_connection(y)
        x = self.dropout1(self.relu(self.bn1(self.conv1(x))))
        x = self.conv2(x)
        x = x + y
        y = x
        x = self.dropout2(self.relu(self.bn2(x)))
        return x, y


class ResNet1d(nn.Module):
    """Input: (batch, 12, 4096) @ 400Hz → Output: (batch, 24) logits."""

    def __init__(self, n_classes: int = 24) -> None:
        super().__init__()
        blocks_dim = [
            (64, 4096), (128, 1024), (196, 256), (256, 64), (320, 16),
        ]
        self.conv1 = nn.Conv1d(
            12, 64, 17, bias=False, stride=1, padding=_padding(1, 17),
        )
        self.bn1 = nn.BatchNorm1d(64)
        self.res_blocks = nn.ModuleList()
        n_out_prev, ns_prev = 64, 4096
        for i, (nf, ns) in enumerate(blocks_dim):
            ds = ns_prev // ns
            blk = ResBlock1d(n_out_prev, nf, ds, dropout_rate=0.0)
            self.res_blocks.append(blk)
            self.add_module(f"resblock1d_{i}", blk)  # registro duplo
            n_out_prev, ns_prev = nf, ns
        self.lin = nn.Linear(320 * 16, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.conv1(x)
        x = self.bn1(x)
        y = x
        for blk in self.res_blocks:
            x, y = blk(x, y)
        return self.lin(x.view(x.size(0), -1))


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

CLASSES_24 = [
    "NORM", "AF", "AFL", "1AVB", "2AVB", "3AVB",
    "RBBB", "IRBBB", "LBBB", "LAFB", "SB", "STach",
    "PAC", "PVC", "LVH", "RVH", "WPW", "STE",
    "STD", "TAb", "LQT", "LAD", "RAD", "MI",
]

THRESHOLDS: dict[str, float] = {c: 0.5 for c in CLASSES_24}
THRESHOLDS["STE"] = 0.35
THRESHOLDS["3AVB"] = 0.40
THRESHOLDS["MI"] = 0.40

ISQUEMIA_CODES = {"STE", "STD", "MI", "TAb"}
ARRITMIA_CODES = {
    "AF", "AFL", "SB", "STach", "PAC", "PVC", "WPW",
    "1AVB", "2AVB", "3AVB",
}

DIAG_TEXT: dict[str, str] = {
    "STE": "Supradesnivelamento de ST (sugestivo de SCA com supra)",
    "STD": "Infradesnivelamento de ST (sugestivo de isquemia)",
    "MI": "Área eletricamente inativa (infarto prévio)",
    "TAb": "Alteração da onda T",
    "AF": "Fibrilação atrial",
    "AFL": "Flutter atrial",
    "SB": "Bradicardia sinusal",
    "STach": "Taquicardia sinusal",
    "PAC": "Extrassístoles supraventriculares",
    "PVC": "Extrassístoles ventriculares",
    "WPW": "Pré-excitação ventricular (WPW)",
    "1AVB": "BAV de 1° grau",
    "2AVB": "BAV de 2° grau",
    "3AVB": "BAV total (3° grau)",
    "RBBB": "Bloqueio de ramo direito completo",
    "IRBBB": "Bloqueio de ramo direito incompleto",
    "LBBB": "Bloqueio de ramo esquerdo completo",
    "LAFB": "Bloqueio fascicular anterior esquerdo",
    "LVH": "Sobrecarga ventricular esquerda",
    "RVH": "Sobrecarga ventricular direita",
    "LQT": "QT prolongado",
    "LAD": "Desvio do eixo para a esquerda",
    "RAD": "Desvio do eixo para a direita",
    "NORM": "ECG dentro dos limites da normalidade",
}

DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "classifier"
    / "best_ecg_model_v2.pth"
)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

_model_cache: tuple[ResNet1d, str] | None = None


def load_cnn_model(
    model_path: str | Path = DEFAULT_MODEL_PATH,
    device: str | None = None,
) -> ResNet1d:
    """Carrega ResNet1d com workaround para BN running stats corrompidas."""
    global _model_cache
    p = str(model_path)
    if _model_cache is not None and _model_cache[1] == p:
        return _model_cache[0]

    model = ResNet1d(n_classes=24)
    ckpt = torch.load(p, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state_dict)

    # Reset BN stats corrompidas + usa batch stats em vez de running stats
    for m in model.modules():
        if isinstance(m, nn.BatchNorm1d):
            m.running_mean.zero_()
            m.running_var.fill_(1.0)
            m.num_batches_tracked.zero_()
            m.track_running_stats = False

    model.eval()
    if device:
        model = model.to(device)
    _model_cache = (model, p)
    return model


# ---------------------------------------------------------------------------
# Inferência
# ---------------------------------------------------------------------------

CNN_INPUT_SAMPLES = 4096
CNN_INPUT_FS = 400.0
CNN_INPUT_DURATION_S = CNN_INPUT_SAMPLES / CNN_INPUT_FS  # 10.24 s

LEAD_ORDER_12 = [
    "I", "II", "III", "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6",
]


def build_cnn_input(
    signals: dict[str, np.ndarray],
    sampling_rate: float,
    *,
    long_lead_key: str = "II_rhythm",
    long_lead_target: str = "II",
) -> np.ndarray:
    """Constrói matriz (12, 4096) @ 400Hz para a CNN sem distorcer o tempo.

    Estratégia:
      * Para `long_lead_target` (default "II"): se `long_lead_key`
        ("II_rhythm" ou "II_long") existe, USA esse sinal de 10s real.
      * Para qualquer lead com duração < ~9 s: aplica tile (repetição
        cíclica) até cobrir 10.24 s, preservando frequência cardíaca.
        Isso vale automaticamente para 3x4+1 (~2.5s -> 4 tiles) e
        6x2+1 (~5s -> 2 tiles).
      * Para 12x1 (cada lead já tem ~10s): só resample fino.

    Args:
        signals: dict {lead_name: np.ndarray (µV)}.
        sampling_rate: Hz do sinal de entrada (para os leads do grid).

    Returns:
        np.ndarray float32 shape (12, 4096) em µV.
    """
    from scipy.signal import resample as scipy_resample

    out = np.zeros((12, CNN_INPUT_SAMPLES), dtype=np.float32)
    target_samples_at_fs_in = int(round(CNN_INPUT_DURATION_S * sampling_rate))

    for i, name in enumerate(LEAD_ORDER_12):
        # Substituição especial: II preferencialmente usa rhythm strip
        if name == long_lead_target and long_lead_key in signals:
            src = signals[long_lead_key]
            src_fs = sampling_rate  # mesmo fs do Stenhede
        else:
            src = signals.get(name)
            src_fs = sampling_rate

        if src is None:
            continue
        arr = np.asarray(src, dtype=np.float32)
        if arr.size == 0:
            continue
        arr = np.nan_to_num(arr, nan=0.0)

        duration_s = len(arr) / float(src_fs)
        if duration_s >= 0.9 * CNN_INPUT_DURATION_S:
            # Sinal já tem ~10s — só resample direto pra 4096
            out[i] = scipy_resample(arr, CNN_INPUT_SAMPLES).astype(np.float32)
        else:
            # Tile até cobrir target_samples_at_fs_in, depois resample fino
            n_tile = int(np.ceil(target_samples_at_fs_in / max(len(arr), 1)))
            tiled = np.tile(arr, n_tile)[:target_samples_at_fs_in]
            out[i] = scipy_resample(
                tiled, CNN_INPUT_SAMPLES
            ).astype(np.float32)
    return out


def classify_signal(
    model: ResNet1d,
    signal_12lead: np.ndarray,
    original_hz: float = 500.0,
) -> dict:
    """Classifica sinal (12, N) em µV → dict com isquemia/arritmia/outras.

    NOTA: para máxima fidelidade temporal, prefira passar pela
    `build_cnn_input` antes (com o dict completo de leads, incluindo
    II_rhythm). Esta função aceita um array (12, N) genérico e faz
    resample direto — útil quando o sinal já chega em 10s.
    """
    from scipy.signal import resample as scipy_resample

    signal = np.asarray(signal_12lead, dtype=np.float32)
    if signal.ndim != 2:
        raise ValueError(f"Esperado 2D, recebi shape={signal.shape}")
    if signal.shape[0] != 12:
        signal = signal.T
    if signal.shape[0] != 12:
        raise ValueError(f"Esperado 12 derivações, recebi {signal.shape}")

    if signal.shape[1] != CNN_INPUT_SAMPLES:
        resampled = np.zeros((12, CNN_INPUT_SAMPLES), dtype=np.float32)
        for i in range(12):
            resampled[i] = scipy_resample(signal[i], CNN_INPUT_SAMPLES)
        signal = resampled

    # Z-score por derivação
    for i in range(12):
        std = float(np.std(signal[i]))
        if std > 1e-6:
            signal[i] = (signal[i] - float(np.mean(signal[i]))) / std
        else:
            signal[i] = signal[i] - float(np.mean(signal[i]))

    signal = np.nan_to_num(signal, nan=0.0)

    x = torch.from_numpy(signal).unsqueeze(0).float()
    device = next(model.parameters()).device
    x = x.to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits).cpu().numpy()[0]

    isquemia: list[dict] = []
    arritmia: list[dict] = []
    outras: list[dict] = []
    is_normal = False

    for i, cls in enumerate(CLASSES_24):
        prob = float(probs[i])
        if prob < THRESHOLDS.get(cls, 0.5):
            continue
        if cls == "NORM":
            is_normal = True
            continue
        entry = {"code": cls, "prob": prob}
        if cls in ISQUEMIA_CODES:
            isquemia.append(entry)
        elif cls in ARRITMIA_CODES:
            arritmia.append(entry)
        else:
            outras.append(entry)

    return {
        "isquemia": isquemia,
        "arritmia": arritmia,
        "outras": outras,
        "is_normal": (
            is_normal
            and len(isquemia) == 0
            and len(arritmia) == 0
            and len(outras) == 0
        ),
        "all_probs": {
            cls: float(probs[i]) for i, cls in enumerate(CLASSES_24)
        },
    }
