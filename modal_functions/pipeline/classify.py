"""
Classificação de ECG por CNN — ProECG

Carrega modelo ResNet-1D treinado (best_ecg_model.pth) e classifica
o sinal digital de 12 derivações em 13 classes diagnósticas.

Classes (índice → código):
  0: alteracao_st
  1: bav1
  2: bav2
  3: bav_total
  4: brd
  5: bre
  6: fa
  7: flutter
  8: hipertrofia
  9: normal
  10: outros
  11: sca_supra
  12: tv_mono
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "classifier"
MODEL_FILENAME = "best_ecg_model.pth"

CLASS_NAMES = [
    "alteracao_st",
    "bav1",
    "bav2",
    "bav_total",
    "brd",
    "bre",
    "fa",
    "flutter",
    "hipertrofia",
    "normal",
    "outros",
    "sca_supra",
    "tv_mono",
]

NUM_CLASSES = len(CLASS_NAMES)

# Mapeamento de código → descrição clínica em português
CLASS_DESCRIPTIONS: dict[str, str] = {
    "alteracao_st": "Alteração do segmento ST",
    "bav1": "Bloqueio atrioventricular de 1º grau",
    "bav2": "Bloqueio atrioventricular de 2º grau",
    "bav_total": "Bloqueio atrioventricular total (3º grau)",
    "brd": "Bloqueio de ramo direito",
    "bre": "Bloqueio de ramo esquerdo",
    "fa": "Fibrilação atrial",
    "flutter": "Flutter atrial",
    "hipertrofia": "Hipertrofia ventricular",
    "normal": "ECG dentro dos limites da normalidade",
    "outros": "Outras alterações",
    "sca_supra": "Sugestivo de síndrome coronariana aguda com supra de ST",
    "tv_mono": "Taquicardia ventricular monomórfica",
}

# Limiar de confiança mínimo para reportar um diagnóstico
CONFIDENCE_THRESHOLD = 0.5

# Comprimento esperado do sinal (em amostras, 10s a 500Hz)
EXPECTED_LENGTH = 5000
TARGET_FS = 500


# ---------------------------------------------------------------------------
# Modelo ResNet-1D
# ---------------------------------------------------------------------------

class ResidualBlock1D(nn.Module):
    """Bloco residual para sinal 1D (arquitetura do Colab)."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 7, stride, 3)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 7, 1, 3)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride),
                nn.BatchNorm1d(out_channels),
            )
        self.dropout = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.nn.functional.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return torch.nn.functional.relu(out)


class ECGResNet(nn.Module):
    """ResNet-1D para classificação de ECG de 12 derivações (arquitetura do Colab)."""

    def __init__(self, num_leads: int = 12, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.conv_in = nn.Sequential(
            nn.Conv1d(num_leads, 32, 15, 1, 7),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.layer1 = ResidualBlock1D(32, 64, stride=2)
        self.layer2 = ResidualBlock1D(64, 128, stride=2)
        self.layer3 = ResidualBlock1D(128, 256, stride=2)
        self.layer4 = ResidualBlock1D(256, 512, stride=2)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.fc(self.dropout(self.global_pool(x).squeeze(-1)))


# ---------------------------------------------------------------------------
# Cache do modelo (carrega uma vez por container Modal)
# ---------------------------------------------------------------------------

_model_cache: ECGResNet | None = None


def _load_model() -> ECGResNet:
    """Carrega o modelo do disco (ou retorna do cache)."""
    global _model_cache
    if _model_cache is not None:
        return _model_cache

    model_path = MODEL_DIR / MODEL_FILENAME
    if not model_path.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado em {model_path}. "
            f"Certifique-se de que best_ecg_model.pth está em "
            f"modal_functions/models/classifier/"
        )

    model = ECGResNet(num_leads=12, num_classes=NUM_CLASSES)
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    _model_cache = model
    return model


def _preprocess_signal(
    signal_12lead: np.ndarray, fs_in: int = 500,
) -> torch.Tensor:
    """Preprocessa sinal para entrada na CNN.

    Pipeline:
      1. Resample do fs_in → 500 Hz (TARGET_FS)
      2. Ajusta comprimento pra EXPECTED_LENGTH (5000 amostras = 10s @ 500Hz)
      3. Z-score por derivação
      4. Tensor (1, 12, 5000)
    """
    if signal_12lead.ndim != 2 or signal_12lead.shape[0] != 12:
        raise ValueError(f"Esperado (12, N), recebeu {signal_12lead.shape}")

    signal = signal_12lead.astype(np.float32).copy()

    # NaN → 0 (extração pode deixar NaN nas pontas)
    signal = np.where(np.isnan(signal), 0.0, signal)

    # ---- 1. Resample pra 500 Hz se necessário ----
    if fs_in != TARGET_FS and signal.shape[1] > 2:
        from scipy.signal import resample
        n_target = int(round(signal.shape[1] * TARGET_FS / fs_in))
        if n_target > 1:
            signal = resample(signal, n_target, axis=1).astype(np.float32)

    # ---- 2. Ajusta pra EXPECTED_LENGTH (5000) ----
    n_samples = signal.shape[1]
    if n_samples >= EXPECTED_LENGTH:
        # Center crop
        start = (n_samples - EXPECTED_LENGTH) // 2
        signal = signal[:, start:start + EXPECTED_LENGTH]
    else:
        # Pad zeros (centralizado)
        pad_total = EXPECTED_LENGTH - n_samples
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        signal = np.pad(signal, ((0, 0), (pad_left, pad_right)), mode="constant")

    # ---- 3. Z-score por derivação ----
    for i in range(12):
        lead = signal[i]
        std = float(np.std(lead))
        if std > 1e-6:
            signal[i] = (lead - float(np.mean(lead))) / std
        else:
            signal[i] = lead - float(np.mean(lead))

    return torch.from_numpy(signal).unsqueeze(0)


# ---------------------------------------------------------------------------
# Função principal
# ---------------------------------------------------------------------------

def classify_ecg(
    signal_12lead: np.ndarray,
    fs_in: int = TARGET_FS,
) -> list[dict]:
    """Classifica sinal de ECG usando ResNet-1D.

    Args:
        signal_12lead: array numpy (12, N) com o sinal das 12 derivações.
        fs_in: sampling rate do sinal de entrada (Hz). Será resampleado pra 500Hz.

    Returns:
        Lista de findings (acima do threshold), cada um:
            {code, description, source="cnn", score}.
        "normal" e "outros" não são reportados como achados.
    """
    model = _load_model()
    input_tensor = _preprocess_signal(signal_12lead, fs_in=fs_in)

    with torch.no_grad():
        logits = model(input_tensor)  # (1, NUM_CLASSES)
        scores = torch.sigmoid(logits).squeeze(0).numpy()

    findings: list[dict] = []
    for class_name, score in zip(CLASS_NAMES, scores):
        if class_name in ("normal", "outros"):
            continue
        if score >= CONFIDENCE_THRESHOLD:
            findings.append({
                "code": class_name,
                "description": CLASS_DESCRIPTIONS.get(class_name, class_name),
                "source": "cnn",
                "score": round(float(score), 3),
            })

    findings.sort(key=lambda f: f["score"], reverse=True)
    return findings


def classify_ecg_full(
    signal_12lead: np.ndarray,
    fs_in: int = TARGET_FS,
) -> dict:
    """Versão completa: retorna probabilidades de TODAS as 13 classes.

    Útil pro report combiner que precisa das probs por classe.
    """
    model = _load_model()
    input_tensor = _preprocess_signal(signal_12lead, fs_in=fs_in)
    with torch.no_grad():
        logits = model(input_tensor)
        scores = torch.sigmoid(logits).squeeze(0).numpy()
    return {
        cn: round(float(s), 3) for cn, s in zip(CLASS_NAMES, scores)
    }
