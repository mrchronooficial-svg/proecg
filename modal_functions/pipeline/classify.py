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


# ---------------------------------------------------------------------------
# Modelo ResNet-1D
# ---------------------------------------------------------------------------

class ResidualBlock1D(nn.Module):
    """Bloco residual para sinal 1D."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=7, stride=stride,
            padding=3, bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=7, stride=1,
            padding=3, bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(0.2)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class ECGResNet1D(nn.Module):
    """ResNet-1D para classificação de ECG de 12 derivações."""

    def __init__(self, num_leads: int = 12, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.conv1 = nn.Conv1d(num_leads, 64, kernel_size=15, stride=2, padding=7, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 64, blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, num_classes)

    @staticmethod
    def _make_layer(
        in_channels: int, out_channels: int, blocks: int, stride: int,
    ) -> nn.Sequential:
        layers = [ResidualBlock1D(in_channels, out_channels, stride)]
        for _ in range(1, blocks):
            layers.append(ResidualBlock1D(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


# ---------------------------------------------------------------------------
# Cache do modelo (carrega uma vez por container Modal)
# ---------------------------------------------------------------------------

_model_cache: ECGResNet1D | None = None


def _load_model() -> ECGResNet1D:
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

    model = ECGResNet1D(num_leads=12, num_classes=NUM_CLASSES)
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    _model_cache = model
    return model


def _preprocess_signal(signal_12lead: np.ndarray) -> torch.Tensor:
    """Preprocessa sinal para entrada na CNN.

    - Normaliza cada derivação (z-score)
    - Ajusta comprimento para EXPECTED_LENGTH (pad ou truncate)
    - Retorna tensor (1, 12, EXPECTED_LENGTH)
    """
    # Garantir shape (12, N)
    if signal_12lead.ndim != 2 or signal_12lead.shape[0] != 12:
        raise ValueError(f"Esperado (12, N), recebeu {signal_12lead.shape}")

    signal = signal_12lead.astype(np.float32).copy()

    # Normalização por derivação (z-score)
    for i in range(12):
        lead = signal[i]
        std = np.std(lead)
        if std > 1e-6:
            signal[i] = (lead - np.mean(lead)) / std
        else:
            signal[i] = lead - np.mean(lead)

    # Ajustar comprimento
    n_samples = signal.shape[1]
    if n_samples >= EXPECTED_LENGTH:
        # Pegar segmento central
        start = (n_samples - EXPECTED_LENGTH) // 2
        signal = signal[:, start:start + EXPECTED_LENGTH]
    else:
        # Pad com zeros
        pad_total = EXPECTED_LENGTH - n_samples
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        signal = np.pad(signal, ((0, 0), (pad_left, pad_right)), mode="constant")

    # Converter para tensor PyTorch: (1, 12, EXPECTED_LENGTH)
    return torch.from_numpy(signal).unsqueeze(0)


# ---------------------------------------------------------------------------
# Função principal
# ---------------------------------------------------------------------------

def classify_ecg(signal_12lead: np.ndarray) -> list[dict]:
    """Classifica sinal de ECG usando ResNet-1D.

    Args:
        signal_12lead: array numpy (12, N) com o sinal das 12 derivações.

    Returns:
        Lista de diagnósticos detectados, cada um com:
            - code (str): código da classe (ex: "fa", "sca_supra")
            - description (str): descrição clínica em português
            - source (str): sempre "cnn"
            - score (float): score do sigmoid (0-1)

        Apenas diagnósticos com score >= CONFIDENCE_THRESHOLD são retornados.
        A classe "normal" não é retornada como achado.
    """
    model = _load_model()
    input_tensor = _preprocess_signal(signal_12lead)

    with torch.no_grad():
        logits = model(input_tensor)  # (1, NUM_CLASSES)
        # Multi-label: sigmoid por classe (não softmax)
        scores = torch.sigmoid(logits).squeeze(0).numpy()

    findings: list[dict] = []

    for idx, (class_name, score) in enumerate(zip(CLASS_NAMES, scores)):
        # Não reportar "normal" como achado
        if class_name == "normal":
            continue

        # Não reportar "outros" — muito genérico
        if class_name == "outros":
            continue

        if score >= CONFIDENCE_THRESHOLD:
            findings.append({
                "code": class_name,
                "description": CLASS_DESCRIPTIONS.get(class_name, class_name),
                "source": "cnn",
                "score": round(float(score), 3),
            })

    # Ordenar por score decrescente
    findings.sort(key=lambda f: f["score"], reverse=True)

    return findings
