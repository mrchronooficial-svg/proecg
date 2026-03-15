# Diagnósticos — ProECG

## Status dos Diagnósticos

| # | Diagnóstico | Método | Status MVP | Grupo |
|---|------------|--------|------------|-------|
| 1 | SCA com supra de ST | CNN | Ativo | B |
| 2 | SCA sem supra de ST | CNN + contexto | Em validação | C |
| 3 | BRE completo | Regra (QRS≥120ms + morfologia) | Ativo | A |
| 4 | BRD completo | Regra (QRS≥120ms + rSR' V1) | Ativo | A |
| 5 | Fibrilação atrial | CNN | Ativo | B |
| 6 | Flutter atrial | CNN | Ativo | B |
| 7 | TSV por reentrada nodal | CNN | Em validação | C |
| 8 | TV monomórfica | CNN | Ativo | B |
| 9 | TV polimórfica | CNN | Em validação | C |
| 10 | Torsades de pointes | CNN | Em validação | C |
| 11 | BAV total (3º grau) | CNN + Regra | Ativo | B |
| 12 | BAV 2º grau Mobitz 1 | CNN + Regra | Em validação | C |
| 13 | BAV 2º grau Mobitz 2 | CNN + Regra | Em validação | C |
| 14 | BAV 1º grau | Regra (PR>200ms) | Ativo | A |
| 15 | Hipercalemia | CNN + Regra | Ativo | B |
| 16 | Hipocalemia | CNN + Regra | Em validação | C |
| 17 | Padrão de Winter | CNN | Em validação | C |
| 18 | Supra de aVR com infra difuso | CNN + Regra | Em validação | C |
| 19 | Infarto posterior | CNN | Em validação | C |
| 20 | BRE novo + Sgarbossa positivo | Regra (score ≥3) | Ativo | A |
| 21 | Wellens | CNN | Em validação | C |
| 22 | Taquicardia atrial | CNN | Em validação | C |
| 23 | Wolf-Parkinson-White | Regra (PR<120ms + delta + QRS>100ms) | Ativo | A |
| 24 | FA pré-excitada | CNN | Em validação | C |
| 25 | Bradicardia sinusal grave | Regra (FC<50 + ritmo sinusal) | Ativo | A |
| 26 | Síndrome de Brugada | CNN + Regra | Em validação | C |
| 27 | QT longo | Regra (QTc>470ms F / >450ms M) | Ativo | A |
| 28 | QT curto | Regra (QTc<340ms) | Ativo | A |
| 29 | Embolia pulmonar (S1Q3T3) | Regra + CNN | Em validação | C |
| 30 | Pericardite aguda | CNN + Regra | Ativo | B |
| 31 | Tamponamento cardíaco | CNN + Regra | Em validação | C |

## Legenda

- **Grupo A:** Resolvido por regras numéricas (critérios objetivos). Sem treino de CNN.
- **Grupo B:** CNN tem dados suficientes nos datasets (PTB-XL, CODE-15%). Treino viável.
- **Grupo C:** Poucos exemplos nos datasets. CNN tenta, mas precisão pode ser baixa. Complementado por regras quando possível.
- **Ativo:** Aparece no laudo do MVP.
- **Em validação:** Modelo treina com os dados disponíveis, mas só aparece no laudo quando atingir precisão aceitável (definida pelo cardiologista).

## Laudo Descritivo (quando nenhum diagnóstico detectado)

Se nenhum dos diagnósticos acima for identificado, o laudo é puramente descritivo:
- Ritmo (sinusal, etc.)
- FC
- Eixo elétrico
- Intervalo PR
- Duração QRS
- Intervalo QT
- QTc corrigido (Bazett)
- "Sem alterações significativas ao eletrocardiograma."

## Linguagem do Laudo

- NUNCA usar linguagem afirmativa: "O paciente TEM infarto"
- SEMPRE usar linguagem sugestiva: "Achados sugestivos de...", "Compatível com..."
- SEMPRE incluir disclaimer: "Ferramenta de apoio — correlacionar com dados clínicos"
- NUNCA mostrar probabilidade ou nível de confiança ao médico
