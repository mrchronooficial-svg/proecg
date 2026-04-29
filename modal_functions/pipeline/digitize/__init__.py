"""
ProECG — Digitalizador Proprietário (PMcardio-style).

Pipeline em 6 módulos baseado no paper Demolder et al. 2025:
  Preprocess → Dotter → Gridder → Undistortion → Leader → Extract
"""

from .ecg_digitizer import ECGDigitizer, digitize_ecg_pmcardio

__all__ = ["ECGDigitizer", "digitize_ecg_pmcardio"]
