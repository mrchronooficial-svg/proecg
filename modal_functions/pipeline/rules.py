"""
Regras clínicas para interpretação de ECG -- ProECG

Implementa os critérios definidos em docs/REGRAS_CLINICAS.md.
Cada regra usa APENAS os limiares documentados pelo cardiologista.

Entrada: measurements dict com:
  - heart_rate (float): frequência cardíaca em bpm
  - axis (float): eixo elétrico em graus
  - pr_interval (float): intervalo PR em ms
  - qrs_duration (float): duração do QRS em ms
  - qt_interval (float): intervalo QT em ms
  - qtc_bazett (float): QTc corrigido por Bazett em ms
  - rhythm (str): "sinus", "atrial_fibrillation", etc.
  - sex (str): "M" ou "F"
  - p_wave_present (bool): onda P visível antes de cada QRS
  - leads (dict[str, LeadMorphology]): morfologia por derivação

LeadMorphology (dict por derivação):
  - qrs_amplitude (float): amplitude do QRS em mm
  - st_elevation (float): elevação ST em mm (positivo = supra)
  - st_depression (float): depressão ST em mm (positivo = infra)
  - t_wave (str): "positive", "negative", "biphasic", "peaked", "flat"
  - t_wave_amplitude (float): amplitude da onda T em mm
  - p_wave_present (bool): onda P presente
  - p_wave_amplitude (float): amplitude da onda P em mm
  - pr_depression (float): depressão do PR em mm
  - has_delta_wave (bool): presença de onda delta
  - has_rsr_prime (bool): padrão rSR' presente
  - has_wide_s (bool): onda S empastada/alargada
  - has_q_wave (bool): onda Q presente
  - has_monophasic_r (bool): onda R monofásica alargada
  - has_prominent_s (bool): onda S proeminente
  - has_u_wave (bool): onda U proeminente
  - st_morphology (str): "concave", "convex", "coved", "ascending"
  - qrs_alternans (bool): alternância elétrica do QRS
  - st_segment_ratio (float|None): razão ST/S (para critério modificado de Smith)
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Tipos auxiliares
# ---------------------------------------------------------------------------
Finding = dict[str, Any]  # code, description, source, leads_affected

LEAD_FRONTAL = ("DI", "DII", "DIII", "aVR", "aVL", "aVF")
LEAD_PRECORDIAL = ("V1", "V2", "V3", "V4", "V5", "V6")
ALL_LEADS = LEAD_FRONTAL + LEAD_PRECORDIAL


def _get_lead(leads: dict, name: str) -> dict:
    """Retorna dados de uma derivação, ou dict vazio se não existir."""
    return leads.get(name, {})


def _finding(
    code: str,
    description: str,
    leads_affected: list[str] | None = None,
) -> Finding:
    return {
        "code": code,
        "description": description,
        "source": "rules",
        "leads_affected": leads_affected or [],
    }


# ---------------------------------------------------------------------------
# Regras de medição básica
# ---------------------------------------------------------------------------

def _check_heart_rate(m: dict) -> list[Finding]:
    """FC -- bradicardia / taquicardia."""
    findings: list[Finding] = []
    hr = m.get("heart_rate")
    if hr is None:
        return findings

    if hr > 100:
        findings.append(_finding(
            "TACHYCARDIA",
            f"Taquicardia sinusal (FC {hr:.0f} bpm)",
        ))
    elif hr < 50:
        # Bradicardia grave tem prioridade (checada em regra própria)
        pass
    elif hr < 60:
        findings.append(_finding(
            "BRADYCARDIA",
            f"Bradicardia sinusal (FC {hr:.0f} bpm)",
        ))

    return findings


def _check_axis(m: dict) -> list[Finding]:
    """Eixo elétrico -- normal / desvio esquerdo / desvio direito."""
    findings: list[Finding] = []
    axis = m.get("axis")
    if axis is None:
        return findings

    if axis < -30:
        findings.append(_finding(
            "LEFT_AXIS_DEVIATION",
            f"Desvio do eixo elétrico para a esquerda ({axis:.0f}°)",
        ))
    elif axis > 90:
        findings.append(_finding(
            "RIGHT_AXIS_DEVIATION",
            f"Desvio do eixo elétrico para a direita ({axis:.0f}°)",
        ))

    return findings


# ---------------------------------------------------------------------------
# Grupo A -- Diagnósticos por Regras
# ---------------------------------------------------------------------------

def _check_bav_first_degree(m: dict) -> list[Finding]:
    """BAV 1º Grau: PR > 200ms com ritmo sinusal regular e condução 1:1.
    Referência: AHA/ACC."""
    pr = m.get("pr_interval")
    rhythm = m.get("rhythm", "")
    p_wave = m.get("p_wave_present", True)

    if pr is not None and pr > 200 and "sinus" in rhythm and p_wave:
        return [_finding(
            "AVB_FIRST_DEGREE",
            f"Bloqueio atrioventricular de 1º grau (PR {pr:.0f}ms)",
        )]
    return []


def _check_rbbb(m: dict) -> list[Finding]:
    """BRD Completo: QRS ≥ 120ms + rSR' em V1/V2 + S empastada em DI/V6.
    Referência: AHA/ACC."""
    qrs = m.get("qrs_duration")
    if qrs is None or qrs < 120:
        return []

    leads = m.get("leads", {})
    v1 = _get_lead(leads, "V1")
    v2 = _get_lead(leads, "V2")
    di = _get_lead(leads, "DI")
    v6 = _get_lead(leads, "V6")

    has_rsr = v1.get("has_rsr_prime", False) or v2.get("has_rsr_prime", False)
    has_wide_s = di.get("has_wide_s", False) or v6.get("has_wide_s", False)

    if has_rsr and has_wide_s:
        affected = []
        if v1.get("has_rsr_prime"):
            affected.append("V1")
        if v2.get("has_rsr_prime"):
            affected.append("V2")
        if di.get("has_wide_s"):
            affected.append("DI")
        if v6.get("has_wide_s"):
            affected.append("V6")

        return [_finding(
            "RBBB",
            f"Bloqueio de ramo direito completo (QRS {qrs:.0f}ms)",
            leads_affected=affected,
        )]
    return []


def _check_lbbb(m: dict) -> list[Finding]:
    """BRE Completo: QRS ≥ 120ms + ausência de Q em DI/V5/V6
    + R monofásica alargada em DI/aVL/V5/V6.
    Referência: AHA/ACC."""
    qrs = m.get("qrs_duration")
    if qrs is None or qrs < 120:
        return []

    leads = m.get("leads", {})
    di = _get_lead(leads, "DI")
    avl = _get_lead(leads, "aVL")
    v5 = _get_lead(leads, "V5")
    v6 = _get_lead(leads, "V6")

    # Ausência de Q em DI, V5, V6
    no_q = (
        not di.get("has_q_wave", False)
        and not v5.get("has_q_wave", False)
        and not v6.get("has_q_wave", False)
    )

    # R monofásica alargada em DI, aVL, V5, V6
    mono_r_leads = [
        lead_name
        for lead_name, lead_data in [("DI", di), ("aVL", avl), ("V5", v5), ("V6", v6)]
        if lead_data.get("has_monophasic_r", False)
    ]
    has_mono_r = len(mono_r_leads) >= 2  # pelo menos 2 derivações

    if no_q and has_mono_r:
        return [_finding(
            "LBBB",
            f"Bloqueio de ramo esquerdo completo (QRS {qrs:.0f}ms)",
            leads_affected=mono_r_leads,
        )]
    return []


def _check_long_qt(m: dict) -> list[Finding]:
    """QT Longo: QTc > 470ms (mulher) ou QTc > 450ms (homem).
    Referência: ESC 2023."""
    qtc = m.get("qtc_bazett")
    sex = m.get("sex", "M")
    if qtc is None:
        return []

    threshold = 470 if sex == "F" else 450
    if qtc > threshold:
        return [_finding(
            "LONG_QT",
            f"QT longo (QTc {qtc:.0f}ms, limiar {threshold}ms para sexo {'feminino' if sex == 'F' else 'masculino'})",
        )]
    return []


def _check_short_qt(m: dict) -> list[Finding]:
    """QT Curto: QTc < 340ms.
    Referência: ESC."""
    qtc = m.get("qtc_bazett")
    if qtc is not None and qtc < 340:
        return [_finding(
            "SHORT_QT",
            f"QT curto (QTc {qtc:.0f}ms)",
        )]
    return []


def _check_severe_sinus_bradycardia(m: dict) -> list[Finding]:
    """Bradicardia Sinusal Grave: FC < 50 bpm + ritmo sinusal + onda P presente.
    Referência: ACLS."""
    hr = m.get("heart_rate")
    rhythm = m.get("rhythm", "")
    p_wave = m.get("p_wave_present", True)

    if hr is not None and hr < 50 and "sinus" in rhythm and p_wave:
        return [_finding(
            "SEVERE_SINUS_BRADYCARDIA",
            f"Bradicardia sinusal grave (FC {hr:.0f} bpm)",
        )]
    return []


def _check_wpw(m: dict) -> list[Finding]:
    """WPW: PR < 120ms + onda delta + QRS > 100ms, com onda P presente.
    Referência: AHA/ACC."""
    pr = m.get("pr_interval")
    qrs = m.get("qrs_duration")
    p_wave = m.get("p_wave_present", True)

    if pr is None or qrs is None:
        return []

    if pr < 120 and qrs > 100 and p_wave:
        leads = m.get("leads", {})
        delta_leads = [
            name for name in ALL_LEADS
            if _get_lead(leads, name).get("has_delta_wave", False)
        ]

        if delta_leads:
            return [_finding(
                "WPW",
                f"Padrão de Wolf-Parkinson-White (PR {pr:.0f}ms, QRS {qrs:.0f}ms, onda delta presente)",
                leads_affected=delta_leads,
            )]
    return []


def _check_sgarbossa(m: dict) -> list[Finding]:
    """Critérios de Sgarbossa (no contexto de BRE):
    - Supra ≥ 1mm concordante com QRS = 5 pontos
    - Infra ≥ 1mm em V1/V2/V3 = 3 pontos
    - Supra ≥ 5mm discordante com QRS = 2 pontos
    - Score ≥ 3 = positivo
    - Critério modificado de Smith: razão ST/S > -0.25
    Referência: AHA/ACC, SBC."""

    # Só aplicar se BRE estiver presente
    qrs = m.get("qrs_duration")
    if qrs is None or qrs < 120:
        return []

    leads_data = m.get("leads", {})

    # Verificar se há BRE (simplificado: checar se a regra de LBBB detectaria)
    di = _get_lead(leads_data, "DI")
    v5 = _get_lead(leads_data, "V5")
    v6 = _get_lead(leads_data, "V6")
    no_q = (
        not di.get("has_q_wave", False)
        and not v5.get("has_q_wave", False)
        and not v6.get("has_q_wave", False)
    )
    mono_r_count = sum(
        1 for ld in ["DI", "aVL", "V5", "V6"]
        if _get_lead(leads_data, ld).get("has_monophasic_r", False)
    )
    is_lbbb = no_q and mono_r_count >= 2

    if not is_lbbb:
        return []

    score = 0
    affected: list[str] = []

    for lead_name in ALL_LEADS:
        lead = _get_lead(leads_data, lead_name)
        st_elev = lead.get("st_elevation", 0)
        st_depr = lead.get("st_depression", 0)
        qrs_amp = lead.get("qrs_amplitude", 0)

        # Supra ≥ 1mm concordante com QRS (QRS positivo + ST positivo)
        if st_elev >= 1 and qrs_amp > 0:
            score += 5
            affected.append(lead_name)

        # Infra ≥ 1mm em V1, V2 ou V3
        if lead_name in ("V1", "V2", "V3") and st_depr >= 1:
            score += 3
            if lead_name not in affected:
                affected.append(lead_name)

        # Supra ≥ 5mm discordante com QRS (QRS negativo + ST positivo)
        if st_elev >= 5 and qrs_amp < 0:
            score += 2
            if lead_name not in affected:
                affected.append(lead_name)

    findings: list[Finding] = []

    if score >= 3:
        findings.append(_finding(
            "SGARBOSSA_POSITIVE",
            f"Critérios de Sgarbossa positivos (score {score} pontos) -- sugestivo de SCA em contexto de BRE",
            leads_affected=affected,
        ))

    # Critério modificado de Smith: razão ST/S > -0.25 em qualquer derivação
    for lead_name in ALL_LEADS:
        lead = _get_lead(leads_data, lead_name)
        ratio = lead.get("st_segment_ratio")
        if ratio is not None and ratio > -0.25:
            # Só reportar se Sgarbossa clássico não foi positivo
            if score < 3:
                findings.append(_finding(
                    "SMITH_MODIFIED_SGARBOSSA",
                    f"Critério modificado de Smith positivo (razão ST/S > -0.25 em {lead_name}) -- sugestivo de SCA em contexto de BRE",
                    leads_affected=[lead_name],
                ))
            break  # reportar apenas uma vez

    return findings


# ---------------------------------------------------------------------------
# Grupo A/C -- Regras que complementam a CNN
# ---------------------------------------------------------------------------

def _check_s1q3t3(m: dict) -> list[Finding]:
    """S1Q3T3 (sugestivo de Embolia Pulmonar):
    - Onda S proeminente em DI
    - Onda Q em DIII
    - Inversão de T em DIII
    Referência: ESC."""
    leads = m.get("leads", {})
    di = _get_lead(leads, "DI")
    diii = _get_lead(leads, "DIII")

    s_in_di = di.get("has_prominent_s", False)
    q_in_diii = diii.get("has_q_wave", False)
    t_inv_diii = diii.get("t_wave") == "negative"

    if s_in_di and q_in_diii and t_inv_diii:
        return [_finding(
            "S1Q3T3",
            "Padrão S1Q3T3 -- sugestivo de embolia pulmonar, correlacionar com clínica",
            leads_affected=["DI", "DIII"],
        )]
    return []


def _check_brugada_type1(m: dict) -> list[Finding]:
    """Brugada Tipo 1: elevação ST ≥ 2mm tipo coved em V1 e/ou V2
    + onda T negativa.
    Referência: AHA/ACC, ESC."""
    leads = m.get("leads", {})
    affected: list[str] = []

    for lead_name in ("V1", "V2"):
        lead = _get_lead(leads, lead_name)
        st_elev = lead.get("st_elevation", 0)
        morphology = lead.get("st_morphology", "")
        t_wave = lead.get("t_wave", "")

        if st_elev >= 2 and morphology == "coved" and t_wave == "negative":
            affected.append(lead_name)

    if affected:
        return [_finding(
            "BRUGADA_TYPE1",
            "Padrão de Brugada tipo 1 (elevação ST coved ≥ 2mm com T negativa)",
            leads_affected=affected,
        )]
    return []


# ---------------------------------------------------------------------------
# Grupo A/C -- Distúrbios eletrolíticos e padrões específicos
# ---------------------------------------------------------------------------

def _check_hyperkalemia(m: dict) -> list[Finding]:
    """Hipercalemia:
    - Leve: T apiculadas simétricas, encurtamento QT
    - Moderada: achatamento P, prolongamento PR, alargamento QRS
    - Grave: ausência P, QRS muito alargado
    Referência: AHA/ACC, ESC."""
    leads = m.get("leads", {})
    pr = m.get("pr_interval")
    qrs = m.get("qrs_duration")

    # Checar T apiculadas
    peaked_t_leads = [
        name for name in ALL_LEADS
        if _get_lead(leads, name).get("t_wave") == "peaked"
    ]

    # Checar achatamento/ausência de onda P
    flat_p_leads = [
        name for name in ALL_LEADS
        if _get_lead(leads, name).get("p_wave_amplitude", 1) < 0.1
        and _get_lead(leads, name).get("p_wave_present") is not None
    ]

    absent_p = not m.get("p_wave_present", True)

    findings: list[Finding] = []

    # Grave: ausência de P + QRS muito alargado (> 160ms)
    if absent_p and qrs is not None and qrs > 160:
        findings.append(_finding(
            "HYPERKALEMIA_SEVERE",
            f"Padrão sugestivo de hipercalemia grave (ausência de onda P, QRS {qrs:.0f}ms)",
            leads_affected=peaked_t_leads,
        ))
    # Moderada: achatamento P + PR prolongado + QRS alargado
    elif len(flat_p_leads) >= 3 and pr is not None and pr > 200 and qrs is not None and qrs >= 120:
        findings.append(_finding(
            "HYPERKALEMIA_MODERATE",
            f"Padrão sugestivo de hipercalemia moderada (achatamento de P, PR {pr:.0f}ms, QRS {qrs:.0f}ms)",
            leads_affected=flat_p_leads + peaked_t_leads,
        ))
    # Leve: T apiculadas em múltiplas derivações
    elif len(peaked_t_leads) >= 3:
        findings.append(_finding(
            "HYPERKALEMIA_MILD",
            "Padrão sugestivo de hipercalemia leve (ondas T apiculadas e simétricas)",
            leads_affected=peaked_t_leads,
        ))

    return findings


def _check_hypokalemia(m: dict) -> list[Finding]:
    """Hipocalemia:
    - Achatamento ou inversão da onda T
    - Depressão ST
    - Onda U proeminente (V2-V3)
    - Prolongamento QT aparente
    Referência: AHA/ACC, ESC."""
    leads = m.get("leads", {})

    flat_t_leads = [
        name for name in ALL_LEADS
        if _get_lead(leads, name).get("t_wave") in ("flat", "negative")
    ]

    st_depression_leads = [
        name for name in ALL_LEADS
        if _get_lead(leads, name).get("st_depression", 0) >= 0.5
    ]

    u_wave_leads = [
        name for name in ("V2", "V3")
        if _get_lead(leads, name).get("has_u_wave", False)
    ]

    # Critério: pelo menos T achatada/invertida + (depressão ST OU onda U)
    if len(flat_t_leads) >= 3 and (len(st_depression_leads) >= 2 or len(u_wave_leads) >= 1):
        all_affected = list(set(flat_t_leads + st_depression_leads + u_wave_leads))
        return [_finding(
            "HYPOKALEMIA",
            "Padrão sugestivo de hipocalemia (achatamento de T, depressão ST"
            + (", onda U proeminente" if u_wave_leads else "")
            + ")",
            leads_affected=all_affected,
        )]
    return []


def _check_pericarditis(m: dict) -> list[Finding]:
    """Pericardite Aguda:
    - Supra ST difuso côncavo em múltiplas derivações
    - Infra PR (especialmente DII, DIII, aVF)
    - Sem supra em aVR
    - Sem imagem em espelho
    Referência: ESC 2015, AHA."""
    leads = m.get("leads", {})

    # Supra ST côncavo difuso
    st_elev_leads = [
        name for name in ALL_LEADS
        if name != "aVR"
        and _get_lead(leads, name).get("st_elevation", 0) >= 0.5
        and _get_lead(leads, name).get("st_morphology") == "concave"
    ]

    # Infra PR
    pr_depression_leads = [
        name for name in ("DII", "DIII", "aVF")
        if _get_lead(leads, name).get("pr_depression", 0) >= 0.5
    ]

    # Sem supra em aVR (diferencial com SCA)
    avr = _get_lead(leads, "aVR")
    avr_st = avr.get("st_elevation", 0)

    # Checar ausência de imagem em espelho (infra ST recíproca)
    st_depr_leads = [
        name for name in ALL_LEADS
        if _get_lead(leads, name).get("st_depression", 0) >= 1
    ]

    if (
        len(st_elev_leads) >= 4  # difuso = múltiplas derivações
        and avr_st < 1  # sem supra em aVR
        and len(st_depr_leads) <= 1  # sem imagem em espelho significativa
    ):
        affected = list(set(st_elev_leads + pr_depression_leads))
        desc = "Padrão sugestivo de pericardite aguda (supra ST difuso côncavo"
        if pr_depression_leads:
            desc += ", infra PR"
        desc += ", sem imagem em espelho)"
        return [_finding(
            "PERICARDITIS",
            desc,
            leads_affected=affected,
        )]
    return []


def _check_cardiac_tamponade(m: dict) -> list[Finding]:
    """Tamponamento Cardíaco:
    - Baixa voltagem difusa (QRS < 5mm no plano frontal)
    - Alternância elétrica
    - Taquicardia sinusal
    Referência: ESC, AHA."""
    leads = m.get("leads", {})
    hr = m.get("heart_rate")
    rhythm = m.get("rhythm", "")

    # Baixa voltagem no plano frontal (QRS < 5mm em todas as frontais)
    frontal_amps = [
        abs(_get_lead(leads, name).get("qrs_amplitude", 10))
        for name in LEAD_FRONTAL
        if _get_lead(leads, name).get("qrs_amplitude") is not None
    ]
    # Precisa de dados em pelo menos 4 derivações frontais para avaliar
    frontal_low_voltage = len(frontal_amps) >= 4 and all(a < 5 for a in frontal_amps)

    # Alternância elétrica
    has_alternans = any(
        _get_lead(leads, name).get("qrs_alternans", False)
        for name in ALL_LEADS
    )

    # Taquicardia sinusal
    has_sinus_tachy = hr is not None and hr > 100 and "sinus" in rhythm

    # Critério: baixa voltagem + (alternância elétrica OU taquicardia sinusal)
    if frontal_low_voltage and (has_alternans or has_sinus_tachy):
        features = ["baixa voltagem difusa"]
        if has_alternans:
            features.append("alternância elétrica")
        if has_sinus_tachy:
            features.append(f"taquicardia sinusal (FC {hr:.0f} bpm)")

        return [_finding(
            "CARDIAC_TAMPONADE",
            "Padrão sugestivo de tamponamento cardíaco (" + ", ".join(features) + ")",
        )]
    return []


# ---------------------------------------------------------------------------
# Função principal
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Grupo D — Supra ST por território (IAMCSST), Infra ST, Sobrecargas
# ---------------------------------------------------------------------------

ST_TERRITORIES = {
    "anterior":  ("V1", "V2", "V3", "V4"),
    "inferior":  ("DII", "DIII", "aVF"),
    "lateral":   ("DI", "aVL", "V5", "V6"),
    "septal":    ("V1", "V2"),
    "posterior": ("V7", "V8", "V9"),  # raramente disponíveis
}


def _check_st_elevation_territory(m: dict) -> list[Finding]:
    """Supra ST por território = IAMCSST.

    Critério: pelo menos 2 derivações contíguas do mesmo território com
    supra significativo (≥1mm membros / ≥2mm em V1-V3).
    """
    leads = m.get("leads", {})
    if not leads:
        return []

    findings: list[Finding] = []
    for territory, lead_names in ST_TERRITORIES.items():
        elevated: list[str] = []
        for ln in lead_names:
            lead = _get_lead(leads, ln)
            st_elev = lead.get("st_elevation", 0)
            thr = 2.0 if ln in ("V1", "V2", "V3") else 1.0
            if st_elev >= thr:
                elevated.append(ln)
        if len(elevated) >= 2:
            code = f"STEMI_{territory.upper()}"
            findings.append(_finding(
                code,
                f"Supradesnivelamento de ST em parede {territory} "
                f"({', '.join(elevated)}) — sugestivo de IAMCSST",
                leads_affected=elevated,
            ))
    return findings


def _check_st_depression_global(m: dict) -> list[Finding]:
    """Infradesnivelamento de ST > 0.5mm em ≥ 2 derivações = isquemia subendocárdica."""
    leads = m.get("leads", {})
    if not leads:
        return []
    depressed = [
        name for name in ALL_LEADS
        if _get_lead(leads, name).get("st_depression", 0) >= 0.5
    ]
    if len(depressed) >= 2:
        return [_finding(
            "ST_DEPRESSION",
            f"Infradesnivelamento de ST em {', '.join(depressed)} — "
            "sugestivo de isquemia subendocárdica",
            leads_affected=depressed,
        )]
    return []


def _check_lvh_sokolow(m: dict) -> list[Finding]:
    """Sobrecarga ventricular esquerda (SVE) — critério de Sokolow-Lyon:
    S(V1) + R(V5 ou V6) > 35mm. Critério clássico, amplamente usado.
    """
    leads = m.get("leads", {})
    if not leads:
        return []
    s_v1 = _get_lead(leads, "V1").get("s_amplitude_mm", 0)
    r_v5 = _get_lead(leads, "V5").get("r_amplitude_mm", 0)
    r_v6 = _get_lead(leads, "V6").get("r_amplitude_mm", 0)
    sokolow = s_v1 + max(r_v5, r_v6)
    if sokolow > 35.0:
        return [_finding(
            "LVH_SOKOLOW",
            f"Sobrecarga ventricular esquerda (Sokolow-Lyon = {sokolow:.0f} mm > 35 mm)",
            leads_affected=["V1", "V5", "V6"],
        )]
    return []


def _check_rvh(m: dict) -> list[Finding]:
    """Sobrecarga ventricular direita (SVD) — R/S > 1 em V1."""
    rs_v1 = m.get("r_s_ratio_v1")
    if rs_v1 is not None and rs_v1 > 1.0:
        return [_finding(
            "RVH",
            f"Sobrecarga ventricular direita (R/S em V1 = {rs_v1:.2f} > 1)",
            leads_affected=["V1"],
        )]
    return []


def _check_lae(m: dict) -> list[Finding]:
    """Sobrecarga atrial esquerda (SAE) — P bifásica em V1 com componente
    negativo > 1mm × 1mm (= 40ms × 100µV).
    """
    if m.get("p_bifasic_v1", False):
        return [_finding(
            "LAE",
            "Sobrecarga atrial esquerda (P bifásica em V1 com componente negativo proeminente)",
            leads_affected=["V1"],
        )]
    return []


def _check_rae(m: dict) -> list[Finding]:
    """Sobrecarga atrial direita (SAD) — P apiculada > 2.5mm em DII."""
    leads = m.get("leads", {})
    p_amp_dii = _get_lead(leads, "DII").get("p_wave_amplitude", 0)
    if p_amp_dii is not None and p_amp_dii > 2.5:
        return [_finding(
            "RAE",
            f"Sobrecarga atrial direita (P em DII = {p_amp_dii:.1f} mm > 2.5 mm)",
            leads_affected=["DII"],
        )]
    # Fallback: usa amplitude global de P (computada em DII no measure)
    p_amp_global = m.get("p_wave_amplitude")
    if p_amp_global is not None and p_amp_global / 100.0 > 2.5:
        return [_finding(
            "RAE",
            f"Sobrecarga atrial direita (P em DII = {p_amp_global / 100.0:.1f} mm > 2.5 mm)",
            leads_affected=["DII"],
        )]
    return []


# ---------------------------------------------------------------------------
# Função principal
# ---------------------------------------------------------------------------

ALL_RULES = [
    _check_heart_rate,
    _check_axis,
    _check_severe_sinus_bradycardia,
    _check_bav_first_degree,
    _check_rbbb,
    _check_lbbb,
    _check_long_qt,
    _check_short_qt,
    _check_wpw,
    _check_sgarbossa,
    _check_s1q3t3,
    _check_brugada_type1,
    _check_hyperkalemia,
    _check_hypokalemia,
    _check_pericarditis,
    _check_cardiac_tamponade,
    # Supra/Infra ST
    _check_st_elevation_territory,
    _check_st_depression_global,
    # Sobrecargas
    _check_lvh_sokolow,
    _check_rvh,
    _check_lae,
    _check_rae,
]


def apply_clinical_rules(measurements: dict) -> list[dict]:
    """Aplica todas as regras clínicas às medições do ECG.

    Aceita tanto o formato legacy (campos planos) quanto o formato novo
    estruturado (heart_rate como dict, intervals nested, etc.) — converte
    automaticamente.

    Args:
        measurements: medições do ECG. Pode estar em formato:
          - Legacy: {heart_rate: float, axis: float, pr_interval: float, ...}
          - Novo:   {heart_rate: {mean_bpm: float}, intervals: {pr_ms}, ...}

    Returns:
        Lista de achados (dicts) com code, description, source, leads_affected.
    """
    # Auto-detect: se heart_rate é dict, está no formato novo → converte
    if isinstance(measurements.get("heart_rate"), dict):
        from .measure import to_legacy_format
        measurements = to_legacy_format(measurements)

    findings: list[dict] = []
    for rule_fn in ALL_RULES:
        try:
            findings.extend(rule_fn(measurements))
        except Exception as e:  # defensive — não derruba o pipeline inteiro
            import logging
            logging.getLogger(__name__).warning(
                "Regra %s falhou: %s: %s", rule_fn.__name__, type(e).__name__, e,
            )
    return findings


# ---------------------------------------------------------------------------
# Teste com cenários clínicos simulados
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Teste apply_clinical_rules ===\n")

    # --- Cenário 1: ECG normal ---
    normal = {
        "heart_rate": 72,
        "axis": 45,
        "pr_interval": 160,
        "qrs_duration": 88,
        "qt_interval": 380,
        "qtc_bazett": 420,
        "rhythm": "sinus",
        "p_wave_present": True,
        "sex": "M",
        "leads": {},
    }
    print("Cenario 1 -- ECG normal:")
    findings = apply_clinical_rules(normal)
    print(f"  Achados: {len(findings)}")
    for f in findings:
        print(f"    [{f['code']}] {f['description']}")
    assert len(findings) == 0, "ECG normal nao deveria gerar achados"
    print("  OK\n")

    # --- Cenário 2: BAV 1o grau ---
    bav1 = {
        "heart_rate": 65,
        "axis": 30,
        "pr_interval": 240,
        "qrs_duration": 90,
        "qt_interval": 400,
        "qtc_bazett": 430,
        "rhythm": "sinus",
        "p_wave_present": True,
        "sex": "M",
        "leads": {},
    }
    print("Cenario 2 -- BAV 1o grau (PR=240ms):")
    findings = apply_clinical_rules(bav1)
    codes = [f["code"] for f in findings]
    print(f"  Achados: {codes}")
    assert "AVB_FIRST_DEGREE" in codes
    print("  OK\n")

    # --- Cenário 3: Bradicardia sinusal grave ---
    brady = {
        "heart_rate": 42,
        "axis": 60,
        "pr_interval": 170,
        "qrs_duration": 86,
        "qt_interval": 420,
        "qtc_bazett": 410,
        "rhythm": "sinus",
        "p_wave_present": True,
        "sex": "F",
        "leads": {},
    }
    print("Cenario 3 -- Bradicardia grave (FC=42):")
    findings = apply_clinical_rules(brady)
    codes = [f["code"] for f in findings]
    print(f"  Achados: {codes}")
    assert "SEVERE_SINUS_BRADYCARDIA" in codes
    print("  OK\n")

    # --- Cenário 4: QT longo (mulher) ---
    long_qt = {
        "heart_rate": 68,
        "axis": 50,
        "pr_interval": 160,
        "qrs_duration": 92,
        "qt_interval": 490,
        "qtc_bazett": 490,
        "rhythm": "sinus",
        "p_wave_present": True,
        "sex": "F",
        "leads": {},
    }
    print("Cenario 4 -- QT longo mulher (QTc=490):")
    findings = apply_clinical_rules(long_qt)
    codes = [f["code"] for f in findings]
    print(f"  Achados: {codes}")
    assert "LONG_QT" in codes
    print("  OK\n")

    # --- Cenário 5: QT longo limiar (homem 450, mulher 470 -- borda) ---
    borderline_m = {**long_qt, "sex": "M", "qtc_bazett": 455}
    print("Cenario 5a -- QTc=455 homem (> 450 -> longo):")
    findings = apply_clinical_rules(borderline_m)
    codes = [f["code"] for f in findings]
    print(f"  Achados: {codes}")
    assert "LONG_QT" in codes
    print("  OK")

    borderline_f = {**long_qt, "sex": "F", "qtc_bazett": 465}
    print("Cenario 5b -- QTc=465 mulher (< 470 -> normal):")
    findings = apply_clinical_rules(borderline_f)
    codes = [f["code"] for f in findings]
    print(f"  Achados: {codes}")
    assert "LONG_QT" not in codes
    print("  OK\n")

    # --- Cenário 6: BRD completo ---
    rbbb = {
        "heart_rate": 78,
        "axis": 100,
        "pr_interval": 160,
        "qrs_duration": 140,
        "qt_interval": 390,
        "qtc_bazett": 435,
        "rhythm": "sinus",
        "p_wave_present": True,
        "sex": "M",
        "leads": {
            "V1": {"has_rsr_prime": True, "qrs_amplitude": 8},
            "V2": {"has_rsr_prime": True, "qrs_amplitude": 10},
            "DI": {"has_wide_s": True, "qrs_amplitude": 6},
            "V6": {"has_wide_s": True, "qrs_amplitude": 7},
        },
    }
    print("Cenario 6 -- BRD completo:")
    findings = apply_clinical_rules(rbbb)
    codes = [f["code"] for f in findings]
    print(f"  Achados: {codes}")
    assert "RBBB" in codes
    assert "RIGHT_AXIS_DEVIATION" in codes
    print("  OK\n")

    # --- Cenário 7: WPW ---
    wpw = {
        "heart_rate": 80,
        "axis": 20,
        "pr_interval": 100,
        "qrs_duration": 130,
        "qt_interval": 380,
        "qtc_bazett": 425,
        "rhythm": "sinus",
        "p_wave_present": True,
        "sex": "M",
        "leads": {
            "V1": {"has_delta_wave": True},
            "V2": {"has_delta_wave": True},
            "DII": {"has_delta_wave": True},
        },
    }
    print("Cenario 7 -- WPW (PR=100, delta, QRS=130):")
    findings = apply_clinical_rules(wpw)
    codes = [f["code"] for f in findings]
    print(f"  Achados: {codes}")
    assert "WPW" in codes
    print("  OK\n")

    # --- Cenário 8: QT curto ---
    short_qt = {
        "heart_rate": 75,
        "axis": 40,
        "pr_interval": 150,
        "qrs_duration": 84,
        "qt_interval": 280,
        "qtc_bazett": 320,
        "rhythm": "sinus",
        "p_wave_present": True,
        "sex": "M",
        "leads": {},
    }
    print("Cenario 8 -- QT curto (QTc=320):")
    findings = apply_clinical_rules(short_qt)
    codes = [f["code"] for f in findings]
    print(f"  Achados: {codes}")
    assert "SHORT_QT" in codes
    print("  OK\n")

    # --- Cenário 9: Taquicardia ---
    tachy = {
        "heart_rate": 120,
        "axis": 70,
        "pr_interval": 140,
        "qrs_duration": 86,
        "qt_interval": 320,
        "qtc_bazett": 410,
        "rhythm": "sinus",
        "p_wave_present": True,
        "sex": "M",
        "leads": {},
    }
    print("Cenario 9 -- Taquicardia sinusal (FC=120):")
    findings = apply_clinical_rules(tachy)
    codes = [f["code"] for f in findings]
    print(f"  Achados: {codes}")
    assert "TACHYCARDIA" in codes
    print("  OK\n")

    print("=" * 50)
    print("Todos os cenarios passaram!")
