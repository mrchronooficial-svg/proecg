"""
ProECG Digitizer — Constantes e Configuração
Padrão Brasil (25 mm/s, 10 mm/mV, 60 Hz)
"""

# --- Output ---
SAMPLING_RATE = 500
OUTPUT_LENGTH = 5000  # 10s × 500Hz
LEAD_ORDER = ["I", "II", "III", "aVR", "aVL", "aVF",
              "V1", "V2", "V3", "V4", "V5", "V6"]

# --- Padrão Brasil ---
PAPER_SPEED_DEFAULT = 25   # mm/s
GAIN_DEFAULT = 10          # mm/mV
GRID_MINOR_MM = 1
GRID_MAJOR_MM = 5

# --- Pré-processamento ---
MAX_INPUT_SIZE = 2048

# --- UNet ---
UNET_PATCH_SIZE = 1024
UNET_OVERLAP = 128

# UNet principal (segmentação)
UNET_ENCODER_WIDTHS = (32, 64, 128, 256, 320, 320, 320, 320)
UNET_IN_CHANNELS = 3
UNET_NUM_CLASSES = 4

# Lead Name UNet
LEAD_UNET_ENCODER_WIDTHS = (32, 64, 128, 256, 256)
LEAD_UNET_IN_CHANNELS = 1
LEAD_UNET_NUM_CLASSES = 13

# --- Segmentation classes ---
SEG_BACKGROUND = 0
SEG_GRID = 1
SEG_TRACE = 2
SEG_TEXT = 3

# --- Calibração (thresholds relaxados para BR) ---
AUTOCORR_HEIGHT_THRESH = 0.15
AUTOCORR_PROMINENCE_THRESH = 0.02

# --- Filtros (Brasil = 60Hz) ---
POWERLINE_FREQ = 60
BASELINE_HIGHPASS = 0.5
LOWPASS_FREQ = 150

# --- Caminhos dos pesos ---
UNET_WEIGHTS_FILENAME = "unet_weights.pt"
LEAD_UNET_WEIGHTS_FILENAME = "lead_name_unet_weights.pt"
