"""
Classificação de ECG por CNN — ProECG v5b

Carrega modelo ResNet-1D v5b treinado com 276k ECGs (7 datasets),
Focal Loss, dropout 0.3, Cosine Annealing.

24 classes diagnósticas com thresholds Youden otimizados por classe.
Input: 13 canais (12 leads + lead II rhythm), 4096 amostras @ 400Hz.

Macro AUC: 0.9854 | Macro Sensibilidade: 96% | Macro Especificidade: 95%
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
MODEL_FILENAME = "best_ecg_model_v5b_13ch_mv.pth"

# 24 classes diagnósticas (ordem do treino)
CLASS_NAMES = [
    "NORM", "AF", "AFL", "1AVB", "2AVB", "3AVB", "RBBB", "IRBBB",
    "LBBB", "LAFB", "SB", "STach", "PAC", "PVC", "LVH", "RVH",
    "WPW", "STE", "STD", "TAb", "LQT", "LAD", "RAD", "MI",
]

NUM_CLASSES = len(CLASS_NAMES)

# Mapeamento código → descrição clínica em português (padrão SBC)
CLASS_DESCRIPTIONS: dict[str, str] = {
    "NORM":  "ECG dentro dos limites da normalidade",
    "AF":    "Fibrilação atrial",
    "AFL":   "Flutter atrial",
    "1AVB":  "Bloqueio atrioventricular de 1º grau",
    "2AVB":  "Bloqueio atrioventricular de 2º grau",
    "3AVB":  "Bloqueio atrioventricular total (3º grau)",
    "RBBB":  "Bloqueio de ramo direito",
    "IRBBB": "Bloqueio incompleto de ramo direito",
    "LBBB":  "Bloqueio de ramo esquerdo",
    "LAFB":  "Bloqueio divisional anterossuperior esquerdo",
    "SB":    "Bradicardia sinusal",
    "STach": "Taquicardia sinusal",
    "PAC":   "Extrassístoles atriais",
    "PVC":   "Extrassístoles ventriculares",
    "LVH":   "Sobrecarga ventricular esquerda",
    "RVH":   "Sobrecarga ventricular direita",
    "WPW":   "Pré-excitação ventricular (Wolff-Parkinson-White)",
    "STE":   "Supradesnivelamento do segmento ST",
    "STD":   "Infradesnivelamento do segmento ST",
    "TAb":   "Alterações da onda T",
    "LQT":   "Intervalo QT prolongado",
    "LAD":   "Desvio do eixo elétrico para a esquerda",
    "RAD":   "Desvio do eixo elétrico para a direita",
    "MI":    "Sinais de infarto do miocárdio",
}

# Thresholds Youden otimizados por classe (maximiza sens + espec)
# Calculados no test set de 23.817 ECGs
YOUDEN_THRESHOLDS: dict[str, float] = {
    "NORM":  0.766,
    "AF":    0.020,
    "AFL":   0.226,
    "1AVB":  0.044,
    "2AVB":  0.002,
    "3AVB":  0.037,
    "RBBB":  0.104,
    "IRBBB": 0.055,
    "LBBB":  0.031,
    "LAFB":  0.043,
    "SB":    0.163,
    "STach": 0.022,
    "PAC":   0.020,
    "PVC":   0.075,
    "LVH":   0.013,
    "RVH":   0.002,
    "WPW":   0.008,
    "STE":   0.028,
    "STD":   0.041,
    "TAb":   0.140,
    "LQT":   0.007,
    "LAD":   0.049,
    "RAD":   0.040,
    "MI":    0.024,
}

# Red flags: diagnósticos que requerem atenção imediata
RED_FLAG_CODES = {"STE", "3AVB", "2AVB", "WPW", "LQT", "MI"}

# Comprimento esperado do sinal (amostras @ 400Hz)
EXPECTED_LENGTH = 4096
TARGET_FS = 400
N_CHANNELS = 13  # 12 leads + lead II duplicado/rhythm


# ---------------------------------------------------------------------------
# Modelo ResNet-1D v5b (idêntico ao treino)
# ---------------------------------------------------------------------------

def _padding(downsample: int, kernel_size: int) -> int:
    return max(0, int(np.floor((kernel_size - downsample + 1) / 2)))


class ResBlock1d(nn.Module):
    def __init__(self, n_in: int, n_out: int, downsample: int,
                 ks: int = 17, drop: float = 0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(n_in, n_out, ks,
                               padding=_padding(1, ks), bias=False)
        self.bn1 = nn.BatchNorm1d(n_out)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(drop)
        self.conv2 = nn.Conv1d(n_out, n_out, ks, stride=downsample,
                               padding=_padding(downsample, ks), bias=False)
        self.bn2 = nn.BatchNorm1d(n_out)
        self.dropout2 = nn.Dropout(drop)
        skip = []
        if downsample > 1:
            skip.append(nn.MaxPool1d(downsample, stride=downsample))
        if n_in != n_out:
            skip.append(nn.Conv1d(n_in, n_out, 1, bias=False))
        self.skip_connection = nn.Sequential(*skip) if skip else None

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        if self.skip_connection is not None:
            y = self.skip_connection(y)
        x = self.dropout1(self.relu(self.bn1(self.conv1(x))))
        x = self.conv2(x)
        x = x + y
        y = x
        x = self.dropout2(self.relu(self.bn2(x)))
        return x, y


class ResNet1d(nn.Module):
    """ResNet-1D para classificação de ECG — arquitetura v5b."""

    def __init__(self, n_channels: int = N_CHANNELS,
                 n_classes: int = NUM_CLASSES):
        super().__init__()
        blocks_dim = [(64, 4096), (128, 1024), (196, 256), (256, 64), (320, 16)]
        self.conv1 = nn.Conv1d(n_channels, 64, 17, bias=False,
                               padding=_padding(1, 17))
        self.bn1 = nn.BatchNorm1d(64)
        self.res_blocks = nn.ModuleList()
        n_out_prev, ns_prev = 64, 4096
        for i, (nf, ns) in enumerate(blocks_dim):
            ds = ns_prev // ns
            blk = ResBlock1d(n_out_prev, nf, ds)
            self.res_blocks.append(blk)
            self.add_module(f"resblock1d_{i}", blk)
            n_out_prev, ns_prev = nf, ns
        self.lin = nn.Linear(320 * 16, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        y = x
        for blk in self.res_blocks:
            x, y = blk(x, y)
        return self.lin(x.view(x.size(0), -1))


# ---------------------------------------------------------------------------
# Cache do modelo (carrega uma vez por container Modal)
# ---------------------------------------------------------------------------

_model_cache: ResNet1d | None = None
cnn_available: bool = False


def _load_model() -> ResNet1d:
    """Carrega o modelo do disco (ou retorna do cache)."""
    global _model_cache, cnn_available
    if _model_cache is not None:
        return _model_cache

    model_path = MODEL_DIR / MODEL_FILENAME
    if not model_path.exists():
        raise FileNotFoundError(
            f"Modelo CNN não encontrado em {model_path}. "
            f"Copie best_ecg_model_v5b_13ch_mv.pth para "
            f"modal_functions/models/classifier/"
        )

    model = ResNet1d(n_channels=N_CHANNELS, n_classes=NUM_CLASSES)
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

    # O checkpoint pode ter a key 'model' (salvo pelo treino) ou ser o state_dict direto
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()

    _model_cache = model
    cnn_available = True
    return model


def _preprocess_signal(signal: np.ndarray) -> torch.Tensor:
    """Preprocessa sinal para entrada na CNN v5b.

    Aceita:
      - (13, 4096): 12 leads + rhythm/lead II — usa direto
      - (12, 4096): 12 leads — duplica lead II como canal 13
      - (12, N): resamplea para 4096 + duplica lead II

    Returns:
        Tensor (1, 13, 4096)
    """
    sig = signal.astype(np.float32).copy()

    # NaN → 0
    sig = np.where(np.isnan(sig), 0.0, sig)

    n_leads, n_samples = sig.shape

    # Se já tem 13 canais e 4096 amostras, usa direto
    if n_leads == 13 and n_samples == EXPECTED_LENGTH:
        return torch.from_numpy(sig).unsqueeze(0)

    # Se tem 13 canais mas tamanho errado, ajusta
    if n_leads == 13:
        if n_samples > EXPECTED_LENGTH:
            sig = sig[:, :EXPECTED_LENGTH]
        elif n_samples < EXPECTED_LENGTH:
            padded = np.zeros((13, EXPECTED_LENGTH), dtype=np.float32)
            padded[:, :n_samples] = sig
            sig = padded
        return torch.from_numpy(sig).unsqueeze(0)

    # Se tem 12 canais, precisa adicionar canal 13 (lead II)
    if n_leads != 12:
        raise ValueError(f"Esperado (12, N) ou (13, N), recebeu {signal.shape}")

    # Resample se necessário
    if n_samples != EXPECTED_LENGTH:
        from scipy.signal import resample
        if n_samples > EXPECTED_LENGTH:
            sig = sig[:, :EXPECTED_LENGTH]
        elif n_samples > 0:
            sig = resample(sig, EXPECTED_LENGTH, axis=1).astype(np.float32)

    # Montar 13 canais: 12 leads + lead II (índice 1) como canal 13
    out = np.zeros((13, EXPECTED_LENGTH), dtype=np.float32)
    out[:12] = sig[:12]
    out[12] = sig[1]  # lead II

    return torch.from_numpy(out).unsqueeze(0)


# ---------------------------------------------------------------------------
# Funções principais
# ---------------------------------------------------------------------------

def classify_ecg(
    signal: np.ndarray,
    use_youden: bool = True,
) -> list[dict]:
    """Classifica sinal de ECG usando ResNet-1D v5b.

    Args:
        signal: array numpy (12, N), (13, N) ou (13, 4096)
        use_youden: se True, usa thresholds Youden por classe.
                    Se False, usa 0.5 fixo.

    Returns:
        Lista de findings (acima do threshold), cada um:
            {code, description, source, score, is_red_flag}
        "NORM" não é reportado como achado (é a ausência de achados).
    """
    model = _load_model()
    input_tensor = _preprocess_signal(signal)

    with torch.no_grad():
        logits = model(input_tensor)
        scores = torch.sigmoid(logits).squeeze(0).numpy()

    findings: list[dict] = []
    for i, (class_name, score) in enumerate(zip(CLASS_NAMES, scores)):
        if class_name == "NORM":
            continue

        threshold = YOUDEN_THRESHOLDS[class_name] if use_youden else 0.5

        if score >= threshold:
            findings.append({
                "code": class_name,
                "description": CLASS_DESCRIPTIONS.get(class_name, class_name),
                "source": "cnn",
                "score": round(float(score), 4),
                "threshold": round(threshold, 4),
                "is_red_flag": class_name in RED_FLAG_CODES,
            })

    # Red flags primeiro, depois por score decrescente
    findings.sort(key=lambda f: (-f["is_red_flag"], -f["score"]))
    return findings


def classify_ecg_full(signal: np.ndarray) -> dict:
    """Retorna probabilidades de TODAS as 24 classes.

    Útil pro report combiner e pra debug.
    """
    model = _load_model()
    input_tensor = _preprocess_signal(signal)
    with torch.no_grad():
        logits = model(input_tensor)
        scores = torch.sigmoid(logits).squeeze(0).numpy()
    return {
        cn: round(float(s), 4) for cn, s in zip(CLASS_NAMES, scores)
    }


def get_norm_probability(signal: np.ndarray) -> float:
    """Retorna a probabilidade de ser ECG normal (classe NORM)."""
    probs = classify_ecg_full(signal)
    return probs.get("NORM", 0.0)


def is_cnn_available() -> bool:
    """Verifica se o modelo CNN está disponível."""
    try:
        _load_model()
        return True
    except FileNotFoundError:
        return False
