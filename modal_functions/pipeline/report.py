"""
Gerador de laudo descritivo — ProECG

Recebe medições, achados de regras e achados da CNN e monta um laudo
descritivo em texto (template fixo, em português).

Linguagem:
  - NUNCA afirmativa ("O paciente TEM infarto")
  - SEMPRE sugestiva ("Sugestivo de...", "Compatível com...")
  - SEMPRE com disclaimer
  - NUNCA mostrar probabilidade/confiança ao médico
"""

from __future__ import annotations

from typing import Any

DISCLAIMER = (
    "⚠️ Ferramenta de apoio à decisão clínica — não substitui avaliação médica. "
    "Correlacionar sempre com dados clínicos, exame físico e contexto do paciente."
)

# Mapeamento de ritmo interno → texto do laudo
RHYTHM_LABELS: dict[str, str] = {
    "sinus": "sinusal",
    "irregular_sem_p": "irregular, sem ondas P identificáveis",
    "regular_sem_p": "regular, sem ondas P identificáveis",
    "indeterminado": "indeterminado",
}


def _format_measurement_line(measurements: dict) -> str:
    """Formata a linha de medições do laudo."""
    m = measurements
    parts: list[str] = []

    # Ritmo
    rhythm_raw = m.get("rhythm", "indeterminado")
    rhythm_label = RHYTHM_LABELS.get(rhythm_raw, rhythm_raw)
    parts.append(f"Ritmo: {rhythm_label}")

    # FC
    hr = m.get("heart_rate")
    if hr is not None:
        parts.append(f"FC: {hr:.0f} bpm")

    # Eixo
    axis = m.get("axis")
    if axis is not None:
        parts.append(f"Eixo: {axis:+.0f}°")

    line1 = ". ".join(parts) + "."

    # Intervalos
    interval_parts: list[str] = []
    pr = m.get("pr_interval")
    if pr is not None:
        interval_parts.append(f"PR: {pr:.0f}ms")

    qrs = m.get("qrs_duration")
    if qrs is not None:
        interval_parts.append(f"QRS: {qrs:.0f}ms")

    qt = m.get("qt_interval")
    if qt is not None:
        interval_parts.append(f"QT: {qt:.0f}ms")

    qtc = m.get("qtc_bazett")
    if qtc is not None:
        interval_parts.append(f"QTc: {qtc:.0f}ms (Bazett)")

    line2 = ". ".join(interval_parts) + "." if interval_parts else ""

    return f"{line1}\n{line2}" if line2 else line1


def _merge_findings(
    rule_findings: list[dict],
    cnn_findings: list[dict],
) -> list[dict]:
    """Combina achados de regras e CNN, removendo duplicatas.

    Se a mesma condição aparece em ambas as fontes, prioriza a regra
    (mais específica) e marca source como "cnn+rules".
    """
    # Mapeamento de códigos equivalentes (CNN code → rules code)
    cnn_to_rule_map: dict[str, str] = {
        "bav1": "AVB_FIRST_DEGREE",
        "brd": "RBBB",
        "bre": "LBBB",
        "fa": "fa",  # sem equivalente direto em regras
        "flutter": "flutter",
        "sca_supra": "SGARBOSSA_POSITIVE",
        "bav_total": "bav_total",
        "bav2": "bav2",
        "tv_mono": "tv_mono",
        "alteracao_st": "alteracao_st",
        "hipertrofia": "hipertrofia",
    }

    rule_codes = {f["code"] for f in rule_findings}
    merged: list[dict] = list(rule_findings)

    for cnn_f in cnn_findings:
        cnn_code = cnn_f["code"]
        mapped_rule_code = cnn_to_rule_map.get(cnn_code)

        # Se a regra já detectou a mesma condição, marcar como cnn+rules
        if mapped_rule_code and mapped_rule_code in rule_codes:
            for f in merged:
                if f["code"] == mapped_rule_code:
                    f["source"] = "cnn+rules"
                    break
        else:
            # Adicionar achado da CNN (sem duplicar)
            merged.append({
                "code": cnn_f["code"],
                "description": cnn_f["description"],
                "source": "cnn",
                "leads_affected": cnn_f.get("leads_affected", []),
            })

    return merged


def _build_findings_text(findings: list[dict]) -> str:
    """Formata achados como texto para o laudo."""
    if not findings:
        return ""

    lines: list[str] = []
    for f in findings:
        desc = f["description"]
        leads = f.get("leads_affected", [])
        if leads:
            desc += f" ({', '.join(leads)})"
        lines.append(f"  • {desc}")

    return "Achados:\n" + "\n".join(lines)


def _build_diagnoses_text(diagnoses: list[dict]) -> str:
    """Formata diagnósticos como texto para o laudo."""
    if not diagnoses:
        return ""

    lines: list[str] = []
    for d in diagnoses:
        lines.append(f"  • {d['description']}")

    return "Hipóteses diagnósticas:\n" + "\n".join(lines)


def _derive_diagnoses(findings: list[dict]) -> list[dict]:
    """Deriva diagnósticos de alto nível a partir dos achados combinados.

    Nunca usar linguagem afirmativa — sempre "sugestivo de", "compatível com".
    """
    diagnoses: list[dict] = []
    codes = {f["code"] for f in findings}

    # SCA com supra
    if "sca_supra" in codes or "SGARBOSSA_POSITIVE" in codes or "SMITH_MODIFIED_SGARBOSSA" in codes:
        source = "cnn+rules" if ("sca_supra" in codes and
                                  ("SGARBOSSA_POSITIVE" in codes or "SMITH_MODIFIED_SGARBOSSA" in codes)) else "cnn" if "sca_supra" in codes else "rules"
        diagnoses.append({
            "code": "sca_com_supra",
            "description": "Sugestivo de síndrome coronariana aguda com supra de ST",
            "source": source,
        })

    # FA
    if "fa" in codes:
        diagnoses.append({
            "code": "fibrilacao_atrial",
            "description": "Compatível com fibrilação atrial",
            "source": "cnn",
        })

    # Flutter
    if "flutter" in codes:
        diagnoses.append({
            "code": "flutter_atrial",
            "description": "Compatível com flutter atrial",
            "source": "cnn",
        })

    # TV monomórfica
    if "tv_mono" in codes:
        diagnoses.append({
            "code": "tv_mono",
            "description": "Compatível com taquicardia ventricular monomórfica",
            "source": "cnn",
        })

    # BAV total
    if "bav_total" in codes:
        diagnoses.append({
            "code": "bav_total",
            "description": "Sugestivo de bloqueio atrioventricular total (3º grau)",
            "source": "cnn+rules" if "bav_total" in codes else "cnn",
        })

    # Brugada
    if "BRUGADA_TYPE1" in codes:
        diagnoses.append({
            "code": "brugada",
            "description": "Compatível com padrão de Brugada tipo 1",
            "source": "rules",
        })

    # EP (S1Q3T3)
    if "S1Q3T3" in codes:
        diagnoses.append({
            "code": "embolia_pulmonar",
            "description": "Padrão S1Q3T3 — sugestivo de embolia pulmonar, correlacionar com clínica",
            "source": "rules",
        })

    # Pericardite
    if "PERICARDITIS" in codes:
        diagnoses.append({
            "code": "pericardite",
            "description": "Sugestivo de pericardite aguda",
            "source": "rules",
        })

    # Tamponamento
    if "CARDIAC_TAMPONADE" in codes:
        diagnoses.append({
            "code": "tamponamento",
            "description": "Padrão sugestivo de tamponamento cardíaco",
            "source": "rules",
        })

    # Hipercalemia
    for sev in ("HYPERKALEMIA_SEVERE", "HYPERKALEMIA_MODERATE", "HYPERKALEMIA_MILD"):
        if sev in codes:
            nivel = {"HYPERKALEMIA_SEVERE": "grave", "HYPERKALEMIA_MODERATE": "moderada", "HYPERKALEMIA_MILD": "leve"}[sev]
            diagnoses.append({
                "code": "hipercalemia",
                "description": f"Padrão sugestivo de hipercalemia {nivel}",
                "source": "rules",
            })
            break

    # Hipocalemia
    if "HYPOKALEMIA" in codes:
        diagnoses.append({
            "code": "hipocalemia",
            "description": "Padrão sugestivo de hipocalemia",
            "source": "rules",
        })

    # WPW
    if "WPW" in codes:
        diagnoses.append({
            "code": "wpw",
            "description": "Compatível com padrão de Wolf-Parkinson-White",
            "source": "rules",
        })

    return diagnoses


def generate_report(
    measurements: dict,
    rule_findings: list[dict],
    cnn_findings: list[dict],
) -> dict:
    """Gera o laudo descritivo completo.

    Args:
        measurements: dict de medições (saída de measure.py).
        rule_findings: lista de achados das regras clínicas (saída de rules.py).
        cnn_findings: lista de achados da CNN (saída de classify.py).

    Returns:
        dict com:
            - findings (list[dict]): achados combinados
            - diagnoses (list[dict]): diagnósticos derivados
            - report_text (str): laudo descritivo em texto
    """
    # Combinar achados
    all_findings = _merge_findings(rule_findings, cnn_findings)

    # Derivar diagnósticos
    diagnoses = _derive_diagnoses(all_findings)

    # Montar texto do laudo
    sections: list[str] = []

    # Seção 1: Medições
    sections.append(_format_measurement_line(measurements))

    # Seção 2: Achados
    findings_text = _build_findings_text(all_findings)
    if findings_text:
        sections.append(findings_text)

    # Seção 3: Diagnósticos
    diagnoses_text = _build_diagnoses_text(diagnoses)
    if diagnoses_text:
        sections.append(diagnoses_text)

    # Se nenhum achado nem diagnóstico
    if not all_findings and not diagnoses:
        sections.append("Sem alterações significativas ao eletrocardiograma.")

    # Disclaimer (sempre)
    sections.append(DISCLAIMER)

    report_text = "\n\n".join(sections)

    return {
        "findings": all_findings,
        "diagnoses": diagnoses,
        "report_text": report_text,
    }
