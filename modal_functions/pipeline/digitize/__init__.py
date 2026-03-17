"""
ProECG — Digitalizador Proprietário
Converte foto de ECG em papel → sinal digital de 12 derivações.

Pipeline: M0 (preprocess) → M1 (segment) → M2 (perspective) →
          M3 (calibrate) → M4 (detect leads) → M5 (extract) → M6 (postprocess)

Se o pipeline UNet falhar, cai para fallback clássico (binarização + morfologia).
"""

from .pipeline import digitize_ecg

__all__ = ["digitize_ecg"]
