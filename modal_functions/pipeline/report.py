"""
Gerador de laudo descritivo -- ProECG

Combina medicoes (measure.py), regras clinicas (rules.py) e CNN (classify.py)
em um laudo de 5 secoes:

  1. DADOS TECNICOS     -- qualidade da digitalizacao
  2. MEDICOES           -- ritmo, FC, eixo, PR, QRS, QT, QTc
  3. ACHADOS            -- regras + CNN + analise ST por derivacao
  4. HIPOTESES          -- red flags primeiro, depois demais
  5. DISCLAIMER         -- obrigatorio em todo laudo

Linguagem:
  - NUNCA afirmativa ("O paciente TEM infarto")
  - SEMPRE sugestiva ("Sugestivo de...", "Compativel com...")
  - NUNCA mostrar probabilidade/confianca ao medico
"""

from __future__ import annotations

from typing import Any

# ------------------------------------------------------------------
# Constantes
# ------------------------------------------------------------------

DISCLAIMER = (
    "Ferramenta de apoio a decisao clinica "
    "-- nao substitui avaliacao medica. "
    "Correlacionar sempre com dados clinicos, exame fisico e contexto do paciente."
)

RHYTHM_LABELS: dict[str, str] = {
    "sinus": "sinusal",
    "irregular_sem_p": "irregular, sem ondas P identificaveis",
    "regular_sem_p": "regular, sem ondas P identificaveis",
    "indeterminado": "indeterminado",
}

# Codigos de red-flag (urgencia maxima) -- aparecem primeiro no laudo
RED_FLAG_CODES = {
    # Legados (regras clínicas — manter)
    "sca_supra", "sca_com_supra",
    "SGARBOSSA_POSITIVE", "SMITH_MODIFIED_SGARBOSSA",
    "tv_mono", "tv_poli", "torsades",
    "bav_total",
    "HYPERKALEMIA_SEVERE",
    "CARDIAC_TAMPONADE",
    "BRUGADA_TYPE1",
    "fa_pre_excitada",
    # Novos v5b (CNN)
    "STE", "3AVB", "2AVB", "WPW", "LQT", "MI",
}

# Mapeamento CNN code -> rules code (para deduplicacao)
# Para "STE" usamos lógica especial em _merge_findings (qualquer STEMI_* / SGARBOSSA).
CNN_TO_RULE_MAP: dict[str, str] = {
    # Legados (manter pra retrocompat)
    "bav1": "AVB_FIRST_DEGREE",
    "brd": "RBBB",
    "bre": "LBBB",
    "sca_supra": "SGARBOSSA_POSITIVE",
    # Novos v5b → regras existentes
    "1AVB": "AVB_FIRST_DEGREE",
    "RBBB": "RBBB",
    "LBBB": "LBBB",
    "STE": "STEMI_ANTERIOR",  # sentinel — _merge_findings testa prefixo STEMI_/SGARBOSSA
    "SB": "SEVERE_SINUS_BRADYCARDIA",
    "STach": "TACHYCARDIA",
    "LVH": "LVH_SOKOLOW",
    "LAD": "LEFT_AXIS_DEVIATION",
    "RAD": "RIGHT_AXIS_DEVIATION",
    "LQT": "LONG_QT",
    "WPW": "WPW",
}

# ------------------------------------------------------------------
# Secao 1 -- Dados tecnicos
# ------------------------------------------------------------------

def _build_technical_section(measurements: dict) -> str:
    """Qualidade da digitalizacao e metadados."""
    lines = ["DADOS TECNICOS:"]

    quality = measurements.get("quality_flags", {})
    if quality:
        grid_pts = quality.get("grid_points_detected")
        px_mm = quality.get("px_per_mm")
        layout = quality.get("layout")
        if grid_pts is not None:
            lines.append(f"  Pontos de grid detectados: {grid_pts}")
        if px_mm is not None:
            lines.append(f"  Resolucao: {px_mm:.1f} px/mm")
        if layout is not None:
            lines.append(f"  Layout: {layout}")
    else:
        lines.append("  Digitalizacao padrao.")

    return "\n".join(lines)


# ------------------------------------------------------------------
# Secao 2 -- Medicoes
# ------------------------------------------------------------------

def _build_measurements_section(m: dict) -> str:
    """Formata medicoes como texto."""
    lines = ["MEDICOES:"]

    # Linha 1: ritmo, FC, eixo
    parts: list[str] = []

    rhythm_raw = m.get("rhythm", "indeterminado")
    rhythm_label = RHYTHM_LABELS.get(rhythm_raw, rhythm_raw)
    parts.append(f"Ritmo: {rhythm_label}")

    hr = m.get("heart_rate")
    if hr is not None:
        parts.append(f"FC: {hr:.0f} bpm")

    axis = m.get("axis")
    if axis is not None:
        parts.append(f"Eixo: {axis:+.0f} graus")

    lines.append("  " + ". ".join(parts) + ".")

    # Linha 2: intervalos
    interval_parts: list[str] = []
    for key, label in [("pr_interval", "PR"), ("qrs_duration", "QRS"),
                       ("qt_interval", "QT")]:
        val = m.get(key)
        if val is not None:
            interval_parts.append(f"{label}: {val:.0f}ms")

    qtc = m.get("qtc_bazett")
    if qtc is not None:
        interval_parts.append(f"QTc: {qtc:.0f}ms (Bazett)")

    if interval_parts:
        lines.append("  " + ". ".join(interval_parts) + ".")

    return "\n".join(lines)


# ------------------------------------------------------------------
# Secao 3 -- Achados
# ------------------------------------------------------------------

def _build_st_analysis(measurements: dict) -> list[str]:
    """Analise ST por derivacao: so inclui se houver supra ou infra."""
    st_data = measurements.get("st_segment", {})
    if not st_data:
        return []

    supra_leads = []
    infra_leads = []
    for lead_name, st in st_data.items():
        cls = st.get("st_classification", "normal")
        amp = st.get("st_amplitude_mv")
        if cls == "supra" and amp is not None:
            supra_leads.append((lead_name, amp))
        elif cls == "infra" and amp is not None:
            infra_leads.append((lead_name, amp))

    lines = []
    if supra_leads:
        leads_str = ", ".join(f"{l} ({a:+.2f}mV)" for l, a in supra_leads)
        lines.append(f"  Supradesnivelamento do segmento ST em {leads_str}.")
    if infra_leads:
        leads_str = ", ".join(f"{l} ({a:+.2f}mV)" for l, a in infra_leads)
        lines.append(f"  Infradesnivelamento do segmento ST em {leads_str}.")

    return lines


def _merge_findings(
    rule_findings: list[dict],
    cnn_findings: list[dict],
) -> list[dict]:
    """Combina achados de regras + CNN, deduplicando.

    Lógica de confidence:
      - Detectado por regra E CNN  → source="cnn+rules", confidence=1.0 (alta)
      - Apenas regra               → source="rules",     confidence=1.0 (alta)
      - Apenas CNN, score ≥ 0.70   → source="cnn",       confidence=score (média)
      - Apenas CNN, score < 0.70   → source="cnn",       confidence=score (baixa)

    Cada achado retornado tem campo "confidence" (float 0–1).
    """
    rule_codes = {f["code"] for f in rule_findings}
    merged: list[dict] = []

    # 1. Achados das regras (todos confidence = 1.0)
    for rf in rule_findings:
        merged.append({
            **rf,
            "source": rf.get("source", "rules"),
            "confidence": 1.0,
        })

    # 2. Achados da CNN (deduplicados contra regras)
    for cnn_f in cnn_findings:
        cnn_code = cnn_f.get("code", "")
        mapped = CNN_TO_RULE_MAP.get(cnn_code)
        score = float(cnn_f.get("score", 0.0))

        # Caso especial — "STE" da CNN bate com QUALQUER STEMI_* ou SGARBOSSA_*
        # detectado pelas regras (que classificam por território).
        confirmed_code: str | None = None
        if cnn_code == "STE":
            for rc in rule_codes:
                if rc.startswith("STEMI_") or rc.startswith("SGARBOSSA") or rc == "SMITH_MODIFIED_SGARBOSSA":
                    confirmed_code = rc
                    break
        elif mapped and mapped in rule_codes:
            confirmed_code = mapped

        if confirmed_code is not None:
            # Confirmação cruzada → marca o existente
            for m in merged:
                if m["code"] == confirmed_code:
                    m["source"] = "cnn+rules"
                    m["cnn_score"] = score
                    if cnn_f.get("is_red_flag"):
                        m["is_red_flag"] = True
                    break
        else:
            merged.append({
                "code": cnn_code or "unknown",
                "description": cnn_f.get("description", ""),
                "source": "cnn",
                "leads_affected": cnn_f.get("leads_affected", []),
                "confidence": round(score, 3),
                "cnn_score": score,
                "is_red_flag": bool(cnn_f.get("is_red_flag", False)),
            })

    return merged


def _build_findings_section(
    findings: list[dict],
    measurements: dict,
) -> str:
    """Formata achados (regras + CNN + ST)."""
    lines = ["ACHADOS:"]

    if not findings and not measurements.get("st_segment"):
        return ""

    for f in findings:
        desc = f.get("description", "")
        leads = f.get("leads_affected", [])
        if leads:
            desc += f" ({', '.join(leads)})"
        lines.append(f"  - {desc}")

    # ST analysis
    st_lines = _build_st_analysis(measurements)
    lines.extend(st_lines)

    if len(lines) == 1:
        return ""  # so tinha o header, nada a reportar

    return "\n".join(lines)


# ------------------------------------------------------------------
# Secao 4 -- Hipoteses diagnosticas
# ------------------------------------------------------------------

def _derive_diagnoses(findings: list[dict]) -> list[dict]:
    """Deriva diagnosticos de alto nivel a partir dos achados.

    Linguagem SEMPRE 'sugestivo de' / 'compativel com'.
    """
    diagnoses: list[dict] = []
    codes = {f["code"] for f in findings}

    # Mapeamento: finding code -> (diag_code, descricao)
    diag_map: list[tuple[set[str], str, str]] = [
        # Red flags
        ({"sca_supra", "SGARBOSSA_POSITIVE", "SMITH_MODIFIED_SGARBOSSA"},
         "sca_com_supra",
         "Sugestivo de sindrome coronariana aguda com supra de ST"),
        ({"tv_mono"},
         "tv_mono",
         "Compativel com taquicardia ventricular monomórfica"),
        ({"bav_total"},
         "bav_total",
         "Sugestivo de bloqueio atrioventricular total (3o grau)"),
        ({"HYPERKALEMIA_SEVERE"},
         "hipercalemia_grave",
         "Padrao sugestivo de hipercalemia grave"),
        ({"CARDIAC_TAMPONADE"},
         "tamponamento",
         "Padrao sugestivo de tamponamento cardiaco"),
        ({"BRUGADA_TYPE1"},
         "brugada",
         "Compativel com padrao de Brugada tipo 1"),
        # Demais
        ({"fa"},
         "fibrilacao_atrial",
         "Compativel com fibrilacao atrial"),
        ({"flutter"},
         "flutter_atrial",
         "Compativel com flutter atrial"),
        ({"S1Q3T3"},
         "embolia_pulmonar",
         "Padrao S1Q3T3 -- sugestivo de embolia pulmonar, correlacionar com clinica"),
        ({"PERICARDITIS"},
         "pericardite",
         "Sugestivo de pericardite aguda"),
        ({"HYPERKALEMIA_MODERATE"},
         "hipercalemia_moderada",
         "Padrao sugestivo de hipercalemia moderada"),
        ({"HYPERKALEMIA_MILD"},
         "hipercalemia_leve",
         "Padrao sugestivo de hipercalemia leve"),
        ({"HYPOKALEMIA"},
         "hipocalemia",
         "Padrao sugestivo de hipocalemia"),
        ({"WPW"},
         "wpw",
         "Compativel com padrao de Wolf-Parkinson-White"),
        # CNN v5b (novos — só dispara se nenhum codigo do mesmo trigger ja virou diag)
        ({"STE"},
         "sca_com_supra",
         "Sugestivo de síndrome coronariana aguda com supra de ST"),
        ({"MI"},
         "infarto_miocardio",
         "Sinais sugestivos de infarto do miocárdio (onda Q patológica / alterações isquêmicas)"),
        ({"AF"},
         "fibrilacao_atrial",
         "Compatível com fibrilação atrial"),
        ({"AFL"},
         "flutter_atrial",
         "Compatível com flutter atrial"),
        ({"3AVB"},
         "bav_total",
         "Sugestivo de bloqueio atrioventricular total (3º grau)"),
        ({"2AVB"},
         "bav_segundo_grau",
         "Sugestivo de bloqueio atrioventricular de 2º grau"),
        ({"RBBB"},
         "bloqueio_ramo_direito",
         "Bloqueio de ramo direito"),
        ({"IRBBB"},
         "bloqueio_incompleto_ramo_direito",
         "Bloqueio incompleto de ramo direito"),
        ({"LBBB"},
         "bloqueio_ramo_esquerdo",
         "Bloqueio de ramo esquerdo"),
        ({"LAFB"},
         "bloqueio_divisional_anterossuperior",
         "Bloqueio divisional anterossuperior esquerdo"),
        ({"1AVB"},
         "bav_primeiro_grau",
         "Bloqueio atrioventricular de 1º grau"),
        ({"WPW"},
         "wpw",
         "Compatível com padrão de Wolff-Parkinson-White"),
        ({"LQT"},
         "qt_longo",
         "Intervalo QT prolongado"),
        ({"LVH"},
         "hipertrofia_ve",
         "Sobrecarga ventricular esquerda"),
        ({"RVH"},
         "hipertrofia_vd",
         "Sobrecarga ventricular direita"),
        ({"STD"},
         "isquemia_subendocardica",
         "Infradesnivelamento do segmento ST — sugestivo de isquemia subendocárdica"),
        ({"TAb"},
         "alteracao_onda_t",
         "Alterações inespecíficas da onda T"),
        ({"PAC"},
         "extrassistoles_atriais",
         "Extrassístoles supraventriculares"),
        ({"PVC"},
         "extrassistoles_ventriculares",
         "Extrassístoles ventriculares"),
        ({"SB"},
         "bradicardia_sinusal",
         "Bradicardia sinusal"),
        ({"STach"},
         "taquicardia_sinusal",
         "Taquicardia sinusal"),
        ({"LAD"},
         "desvio_eixo_esquerda",
         "Desvio do eixo elétrico para a esquerda"),
        ({"RAD"},
         "desvio_eixo_direita",
         "Desvio do eixo elétrico para a direita"),
    ]

    seen: set[str] = set()
    for trigger_codes, diag_code, desc in diag_map:
        if trigger_codes & codes and diag_code not in seen:
            diagnoses.append({
                "code": diag_code,
                "description": desc,
                "source": "rules",
            })
            seen.add(diag_code)

    return diagnoses


def _build_diagnoses_section(diagnoses: list[dict]) -> str:
    """Formata hipoteses com red flags primeiro."""
    if not diagnoses:
        return ""

    # Separar red-flags do resto
    red = [
        d for d in diagnoses
        if d["code"] in RED_FLAG_CODES
        or d.get("code", "") in (
            "sca_com_supra", "infarto_miocardio", "bav_total", "qt_longo",
        )
        or any(
            d["code"].startswith(r)
            for r in ("sca_", "tv_", "bav_total", "hipercalemia_grave", "tamponamento")
        )
    ]
    normal = [d for d in diagnoses if d not in red]

    lines = ["HIPOTESES DIAGNOSTICAS:"]

    for d in red:
        lines.append(f"  [!] {d['description']}")
    for d in normal:
        lines.append(f"  - {d['description']}")

    return "\n".join(lines)


# ------------------------------------------------------------------
# Funcao principal
# ------------------------------------------------------------------

def generate_report(
    measurements: dict,
    rule_findings: list[dict],
    cnn_findings: list[dict] | None = None,
) -> dict:
    """Gera laudo descritivo completo.

    Args:
        measurements: dict de medicoes (saida de measure.py).
        rule_findings: achados das regras clinicas (saida de rules.py).
        cnn_findings: achados da CNN (saida de classify.py). Opcional.

    Returns:
        dict com:
            - findings (list[dict]): achados combinados
            - diagnoses (list[dict]): hipoteses diagnosticas
            - report_text (str): laudo descritivo completo
    """
    if cnn_findings is None:
        cnn_findings = []

    # Auto-converte formato novo (estruturado) → legacy se necessário
    if isinstance(measurements.get("heart_rate"), dict):
        from .measure import to_legacy_format
        measurements = to_legacy_format(measurements)

    # Combinar achados
    all_findings = _merge_findings(rule_findings, cnn_findings)

    # Derivar diagnosticos
    diagnoses = _derive_diagnoses(all_findings)

    # Montar texto do laudo (5 secoes)
    sections: list[str] = []

    # 1. Dados tecnicos
    sections.append(_build_technical_section(measurements))

    # 2. Medicoes
    sections.append(_build_measurements_section(measurements))

    # 3. Achados
    findings_text = _build_findings_section(all_findings, measurements)
    if findings_text:
        sections.append(findings_text)

    # 4. Hipoteses diagnosticas
    diagnoses_text = _build_diagnoses_section(diagnoses)
    if diagnoses_text:
        sections.append(diagnoses_text)

    # Se nenhum achado nem diagnostico
    if not all_findings and not diagnoses:
        sections.append("ECG sem alteracoes significativas.")

    # 5. Disclaimer (sempre)
    sections.append(DISCLAIMER)

    report_text = "\n\n".join(sections)

    return {
        "findings": all_findings,
        "diagnoses": diagnoses,
        "report_text": report_text,
    }


# ------------------------------------------------------------------
# Saída estruturada pro frontend (red flags + severidade + JSON)
# ------------------------------------------------------------------

# Severidade por código de diagnóstico
_DIAGNOSIS_SEVERITY: dict[str, str] = {
    # Críticos (red flags)
    "sca_supra": "critical", "sca_com_supra": "critical",
    "STEMI_ANTERIOR": "critical", "STEMI_INFERIOR": "critical",
    "STEMI_LATERAL": "critical", "STEMI_SEPTAL": "critical",
    "STEMI_POSTERIOR": "critical",
    "SGARBOSSA_POSITIVE": "critical", "SMITH_MODIFIED_SGARBOSSA": "critical",
    "tv_mono": "critical", "tv_poli": "critical", "torsades": "critical",
    "bav_total": "critical", "BAV_TOTAL": "critical",
    "HYPERKALEMIA_SEVERE": "critical",
    "CARDIAC_TAMPONADE": "critical",
    "BRUGADA_TYPE1": "critical",
    "fa_pre_excitada": "critical",
    # Moderados
    "AVB_FIRST_DEGREE": "moderate",
    "RBBB": "moderate", "LBBB": "moderate",
    "LONG_QT": "moderate", "SHORT_QT": "moderate",
    "WPW": "moderate",
    "LVH_SOKOLOW": "moderate", "RVH": "moderate",
    "LAE": "moderate", "RAE": "moderate",
    "ST_DEPRESSION": "moderate",
    "S1Q3T3": "moderate",
    "HYPOKALEMIA": "moderate",
    # Leves
    "TACHYCARDIA": "low", "BRADYCARDIA": "low",
    "SEVERE_SINUS_BRADYCARDIA": "moderate",
    "LEFT_AXIS_DEVIATION": "low", "RIGHT_AXIS_DEVIATION": "low",
    # CNN v5b
    "STE": "critical",
    "3AVB": "critical",
    "MI": "critical",
    "2AVB": "moderate",
    "WPW": "moderate",
    "LQT": "moderate",
    "AF": "moderate",
    "AFL": "moderate",
    "RBBB": "moderate",
    "LBBB": "moderate",
    "IRBBB": "low",
    "LAFB": "low",
    "1AVB": "moderate",
    "LVH": "moderate",
    "RVH": "moderate",
    "STD": "moderate",
    "TAb": "low",
    "PAC": "low",
    "PVC": "low",
    "SB": "low",
    "STach": "low",
    "LAD": "low",
    "RAD": "low",
    "NORM": "normal",
    # Diagnoses derivados (saída de _derive_diagnoses)
    "sca_com_supra": "critical",
    "infarto_miocardio": "critical",
    "bav_total": "critical",
    "bav_segundo_grau": "moderate",
    "qt_longo": "moderate",
    "isquemia_subendocardica": "moderate",
    "hipertrofia_ve": "moderate",
    "hipertrofia_vd": "moderate",
    "bloqueio_ramo_direito": "moderate",
    "bloqueio_ramo_esquerdo": "moderate",
    "bloqueio_incompleto_ramo_direito": "low",
    "bloqueio_divisional_anterossuperior": "low",
    "bav_primeiro_grau": "moderate",
    "bradicardia_sinusal": "low",
    "taquicardia_sinusal": "low",
    "alteracao_onda_t": "low",
    "extrassistoles_atriais": "low",
    "extrassistoles_ventriculares": "low",
    "desvio_eixo_esquerda": "low",
    "desvio_eixo_direita": "low",
}


def _detect_red_flags(measurements: dict, findings: list[dict]) -> list[str]:
    """Identifica red flags (alertas críticos) que exigem atenção imediata.

    Critérios:
      - Supra ST por território (STEMI)
      - FC < 30 ou > 180
      - QRS > 160ms (possível TV / bloqueio severo)
      - BAV total
      - QTc > 550ms (risco de torsades)
    """
    flags: list[str] = []

    # Converte pra legacy se necessário (pra ler campos planos)
    if isinstance(measurements.get("heart_rate"), dict):
        from .measure import to_legacy_format
        m = to_legacy_format(measurements)
    else:
        m = measurements

    # STEMI (qualquer território)
    for f in findings:
        code = f.get("code", "")
        if code.startswith("STEMI_") or code in ("sca_supra", "sca_com_supra"):
            terr = code.replace("STEMI_", "").lower() if code.startswith("STEMI_") else "supra"
            flags.append(f"STEMI {terr}")
        if code in ("SGARBOSSA_POSITIVE", "SMITH_MODIFIED_SGARBOSSA"):
            flags.append("Sgarbossa positivo (SCA em BRE)")
        if code in ("BAV_TOTAL", "bav_total"):
            flags.append("BAV total (3º grau)")
        if code == "BRUGADA_TYPE1":
            flags.append("Brugada tipo 1")
        if code == "HYPERKALEMIA_SEVERE":
            flags.append("Hipercalemia grave")
        if code == "CARDIAC_TAMPONADE":
            flags.append("Tamponamento cardíaco")
        if code in ("tv_mono", "tv_poli", "torsades"):
            flags.append(code.upper())
        # CNN v5b — red flags por código direto
        if code == "STE":
            flags.append("Supra de ST detectado pela IA (possível SCA)")
        if code == "3AVB":
            flags.append("BAV total (3º grau)")
        if code == "2AVB":
            flags.append("BAV 2º grau")
        if code == "MI":
            flags.append("Sinais de infarto do miocárdio")
        if code == "WPW":
            flags.append("Pré-excitação ventricular (WPW)")
        if code == "LQT":
            flags.append("QT prolongado (risco de arritmia)")

    # FC extrema
    hr = m.get("heart_rate")
    if hr is not None:
        if hr < 30:
            flags.append(f"Bradicardia extrema (FC {hr:.0f} bpm)")
        elif hr > 180:
            flags.append(f"Taquicardia extrema (FC {hr:.0f} bpm)")

    # QRS muito largo (possível TV)
    qrs = m.get("qrs_duration")
    if qrs is not None and qrs > 160:
        flags.append(f"QRS muito largo ({qrs:.0f}ms — possível TV ou bloqueio severo)")

    # QTc muito longo (risco torsades)
    qtc = m.get("qtc_bazett")
    if qtc is not None and qtc > 550:
        flags.append(f"QTc muito longo ({qtc:.0f}ms — risco de torsades)")

    # Dedup mantendo ordem
    seen = set()
    deduped = []
    for f in flags:
        if f not in seen:
            seen.add(f)
            deduped.append(f)
    return deduped


def _classify_overall_severity(red_flags: list[str], diagnoses: list[dict]) -> str:
    """Severidade geral do laudo: critical / moderate / normal."""
    if red_flags:
        return "critical"
    for d in diagnoses:
        sev = _DIAGNOSIS_SEVERITY.get(d.get("code", ""), "low")
        if sev in ("critical", "moderate"):
            return "moderate"
    if diagnoses:
        return "moderate"
    return "normal"


def generate_frontend_report(
    measurements: dict,
    rule_findings: list[dict],
    cnn_findings: list[dict] | None = None,
) -> dict:
    """Gera o JSON estruturado pro frontend.

    Returns:
        {
            "text_report": str,
            "measurements": dict (original, formato novo ou legacy),
            "diagnoses": [{name, confidence, source, severity, code}],
            "red_flags": [str],
            "severity": "critical" | "moderate" | "normal",
            "warnings": [str]
        }
    """
    rep = generate_report(measurements, rule_findings, cnn_findings)

    red_flags = _detect_red_flags(measurements, rep["findings"])
    severity = _classify_overall_severity(red_flags, rep["diagnoses"])

    # Mapa code → confidence/source dos findings (que já têm confidence)
    finding_lookup = {f.get("code"): f for f in rep["findings"]}

    # Diagnoses com severidade + confidence numérica
    diagnoses_out = []
    for d in rep["diagnoses"]:
        code = d.get("code", "")
        # Tenta achar achado correspondente pra herdar confidence/source
        related = finding_lookup.get(code)
        if related:
            confidence = float(related.get("confidence", 1.0))
            source = related.get("source", "rules")
        else:
            confidence = 1.0 if d.get("source") == "rules" else 0.7
            source = d.get("source", "rules")
        diagnoses_out.append({
            "name": d.get("description", code),
            "code": code,
            "confidence": round(confidence, 3),
            "source": source,
            "severity": _DIAGNOSIS_SEVERITY.get(code, "low"),
        })

    # Warnings — combina warnings do measure_ecg + warnings de qualidade
    warnings_: list[str] = []
    m_warnings = measurements.get("warnings")
    if isinstance(m_warnings, list):
        warnings_.extend(m_warnings)
    quality = measurements.get("quality")
    if isinstance(quality, dict) and quality.get("overall") == "poor":
        warnings_.append("Qualidade do sinal baixa — medições podem ser imprecisas")

    return {
        "text_report": rep["report_text"],
        "measurements": measurements,
        "diagnoses": diagnoses_out,
        "red_flags": red_flags,
        "severity": severity,
        "warnings": warnings_,
        # Campos auxiliares (debugging / pipeline)
        "_findings_raw": rep["findings"],
    }


# ------------------------------------------------------------------
# Testes com cenarios clinicos
# ------------------------------------------------------------------

if __name__ == "__main__":
    import textwrap

    def print_report(title: str, result: dict) -> None:
        print("=" * 60)
        print(f"  {title}")
        print("=" * 60)
        print(result["report_text"])
        print()

    # ---- Cenario 1: ECG normal ----
    m_normal = {
        "heart_rate": 72, "heart_rate_unit": "bpm",
        "axis": 45, "axis_unit": "graus",
        "pr_interval": 160, "pr_unit": "ms",
        "qrs_duration": 88, "qrs_unit": "ms",
        "qt_interval": 380, "qt_unit": "ms",
        "qtc_bazett": 420, "qtc_unit": "ms",
        "rhythm": "sinus",
        "p_wave_present": True,
        "st_segment": {
            "DI": {"st_amplitude_mv": 0.02, "st_classification": "normal"},
            "DII": {"st_amplitude_mv": 0.01, "st_classification": "normal"},
            "V1": {"st_amplitude_mv": -0.03, "st_classification": "normal"},
            "V2": {"st_amplitude_mv": 0.05, "st_classification": "normal"},
        },
    }
    r1 = generate_report(m_normal, [], [])
    print_report("CENARIO 1: ECG NORMAL", r1)
    assert len(r1["findings"]) == 0
    assert len(r1["diagnoses"]) == 0
    assert "sem alteracoes" in r1["report_text"].lower()
    print("  [OK] Normal: sem achados, sem diagnosticos\n")

    # ---- Cenario 2: BAV 1o + bradicardia ----
    m_bav = {
        "heart_rate": 48, "heart_rate_unit": "bpm",
        "axis": 30, "axis_unit": "graus",
        "pr_interval": 280, "pr_unit": "ms",
        "qrs_duration": 92, "qrs_unit": "ms",
        "qt_interval": 440, "qt_unit": "ms",
        "qtc_bazett": 395, "qtc_unit": "ms",
        "rhythm": "sinus",
        "p_wave_present": True,
        "st_segment": {},
    }
    rule_bav = [
        {"code": "AVB_FIRST_DEGREE",
         "description": "Bloqueio atrioventricular de 1o grau (PR 280ms)",
         "source": "rules", "leads_affected": []},
        {"code": "SEVERE_SINUS_BRADYCARDIA",
         "description": "Bradicardia sinusal grave (FC 48 bpm)",
         "source": "rules", "leads_affected": []},
    ]
    r2 = generate_report(m_bav, rule_bav, [])
    print_report("CENARIO 2: BAV 1o + BRADICARDIA", r2)
    assert len(r2["findings"]) == 2
    codes2 = [f["code"] for f in r2["findings"]]
    assert "AVB_FIRST_DEGREE" in codes2
    assert "SEVERE_SINUS_BRADYCARDIA" in codes2
    print("  [OK] BAV 1o + bradicardia detectados\n")

    # ---- Cenario 3: Supra ST anterior (SCA) ----
    m_supra = {
        "heart_rate": 95, "heart_rate_unit": "bpm",
        "axis": 60, "axis_unit": "graus",
        "pr_interval": 150, "pr_unit": "ms",
        "qrs_duration": 86, "qrs_unit": "ms",
        "qt_interval": 360, "qt_unit": "ms",
        "qtc_bazett": 430, "qtc_unit": "ms",
        "rhythm": "sinus",
        "p_wave_present": True,
        "st_segment": {
            "V1": {"st_amplitude_mv": 0.25, "st_classification": "supra"},
            "V2": {"st_amplitude_mv": 0.35, "st_classification": "supra"},
            "V3": {"st_amplitude_mv": 0.30, "st_classification": "supra"},
            "V4": {"st_amplitude_mv": 0.15, "st_classification": "supra"},
            "DII": {"st_amplitude_mv": -0.08, "st_classification": "infra"},
            "DIII": {"st_amplitude_mv": -0.10, "st_classification": "infra"},
            "aVF": {"st_amplitude_mv": -0.07, "st_classification": "infra"},
        },
    }
    # CNN detectou SCA
    cnn_supra = [
        {"code": "sca_supra",
         "description": "Sindrome coronariana aguda com supra de ST em parede anterior",
         "leads_affected": ["V1", "V2", "V3", "V4"]},
    ]
    r3 = generate_report(m_supra, [], cnn_supra)
    print_report("CENARIO 3: SUPRA ST ANTERIOR (SCA)", r3)
    assert any(d["code"] == "sca_com_supra" for d in r3["diagnoses"])
    assert "Supradesnivelamento" in r3["report_text"]
    assert "Infradesnivelamento" in r3["report_text"]
    assert "[!]" in r3["report_text"]  # red flag marker
    print("  [OK] SCA com supra + infra reciproca detectada, red flag marcado\n")

    print("=" * 60)
    print("  Todos os cenarios passaram!")
    print("=" * 60)
