# Regras Clínicas — ProECG

> **Este documento deve ser preenchido pelo cardiologista (Rafael + sócio).**
> Cada regra será transformada em código Python pelo Claude Code.
> Use linguagem médica normal — o dev transforma em "if/then".

## Como Preencher

Para cada diagnóstico baseado em regras (Grupo A), escreva:
1. O nome do diagnóstico
2. Os critérios objetivos (medições, limiares)
3. Exceções ou condições adicionais
4. Referência da diretriz

---

## Regras de Medição Básica

### Frequência Cardíaca
- Bradicardia: FC < 60 bpm
- Bradicardia grave: FC < 50 bpm
- Taquicardia: FC > 100 bpm
- Cálculo: 60 / intervalo RR médio (em segundos)

### Eixo Elétrico
- Normal: -30° a +90°
- Desvio esquerdo: < -30°
- Desvio direito: > +90°
- Indeterminado: complexo QRS negativo em DI e aVF
- Cálculo: baseado na amplitude do QRS em DI e aVF

---

## Grupo A — Diagnósticos por Regras

### BAV 1º Grau
- Critério: PR > 200ms com ritmo sinusal regular
- Cada onda P seguida de QRS (condução 1:1)
- Referência: AHA/ACC

### BRD Completo
- Critério: QRS ≥ 120ms
- Padrão rSR' em V1 e V2
- Onda S empastada e alargada em DI e V6
- Referência: AHA/ACC

### BRE Completo
- Critério: QRS ≥ 120ms
- Ausência de onda Q em DI, V5, V6
- Onda R monofásica alargada em DI, aVL, V5, V6
- Referência: AHA/ACC

### QT Longo
- Critério: QTc > 470ms (mulher) ou QTc > 450ms (homem)
- Correção por Bazett: QTc = QT / √RR
- Referência: ESC 2023

### QT Curto
- Critério: QTc < 340ms
- Referência: ESC

### Bradicardia Sinusal Grave
- Critério: FC < 50 bpm + ritmo sinusal (onda P presente antes de cada QRS)
- Referência: ACLS

### Wolf-Parkinson-White (WPW)
- Critério: PR < 120ms + presença de onda delta + QRS > 100ms
- Intervalo PR curto sem ausência de onda P
- Referência: AHA/ACC

### Sgarbossa (no contexto de BRE)
- Supra ≥ 1mm concordante com QRS = 5 pontos
- Infra ≥ 1mm em V1, V2 ou V3 = 3 pontos
- Supra ≥ 5mm discordante com QRS = 2 pontos
- Score ≥ 3 = positivo (sugestivo de SCA em contexto de BRE)
- Critério modificado de Smith: razão ST/S > -0.25 em qualquer derivação
- Referência: AHA/ACC, SBC

---

## Grupo A/C — Regras que Complementam a CNN

### BAV 2º Grau Mobitz 1 (Wenckebach)
- PR progressivamente mais longo até onda P bloqueada (sem QRS)
- Ciclo se repete
- Diferencial com Mobitz 2: PR NÃO é fixo antes do bloqueio
- **Nota:** difícil de detectar só por regra, CNN complementa

### BAV 2º Grau Mobitz 2
- PR constante (fixo) + onda P bloqueada súbita (sem QRS)
- Diferencial com Mobitz 1: PR fixo antes do bloqueio
- **Nota:** difícil de detectar só por regra, CNN complementa

### S1Q3T3 (sugestivo de Embolia Pulmonar)
- Onda S proeminente em DI
- Onda Q em DIII
- Inversão de T em DIII
- **Nota:** baixa especificidade — sempre "sugestivo de EP, correlacionar com clínica"
- Referência: ESC

### Brugada (Tipo 1)
- Elevação ST ≥ 2mm tipo coved (convexa) em V1 e/ou V2
- Seguida de onda T negativa
- **Nota:** CNN auxilia na detecção do padrão morfológico
- Referência: AHA/ACC, ESC

---

## Grupo A/C — Distúrbios Eletrolíticos e Padrões Específicos

### Hipercalemia
- Leve (5.5-6.5 mEq/L): ondas T apiculadas e simétricas, encurtamento do QT
- Moderada (6.5-8.0 mEq/L): achatamento de onda P, prolongamento do PR, alargamento do QRS
- Grave (>8.0 mEq/L): ausência de onda P, QRS muito alargado (padrão sinusoidal), risco de FV/assistolia
- **Nota:** CNN auxilia na detecção de T apiculada e padrão sinusoidal
- Referência: AHA/ACC, ESC

### Hipocalemia
- Achatamento ou inversão da onda T
- Depressão do segmento ST
- Presença de onda U proeminente (melhor vista em V2-V3)
- Prolongamento do QT aparente (fusão T-U)
- Casos graves: alargamento do QRS, arritmias (extrassístoles, TV)
- **Nota:** CNN auxilia na detecção de onda U e morfologia de T
- Referência: AHA/ACC, ESC

### Padrão de Winter
- Onda T hiperaguda e simétrica nas precordiais anteriores (V2-V4)
- Segmento ST ascendente com concavidade superior (sem supra clássico)
- Infradesnivelamento do segmento ST no ponto J em V2-V4
- Considerado equivalente de SCA com oclusão de DA
- **Nota:** CNN principal método de detecção; padrão sutil
- Referência: Winter et al. 2008, NEJM

### Wellens
- Tipo A (25%): ondas T bifásicas (positiva-negativa) em V2-V3
- Tipo B (75%): ondas T profundamente invertidas e simétricas em V2-V3 (pode estender V1-V6)
- Presente em intervalo livre de dor (sem alteração durante dor)
- Sem perda de progressão de R
- Troponina normal ou minimamente elevada
- Sugestivo de estenose crítica de DA proximal
- **Nota:** CNN principal método de detecção
- Referência: AHA/ACC

### Pericardite Aguda
- Supra de ST difuso, côncavo (forma de "colher"), em múltiplas derivações
- Infradesnivelamento do segmento PR (especialmente em DII, DIII, aVF)
- Supra de ST em aVR ausente (diferencial com SCA)
- Sem imagem em espelho (diferente de SCA com supra)
- Evolução em 4 estágios (supra difuso → normalização ST → inversão T → normalização)
- **Nota:** CNN auxilia; regra complementa checando ausência de imagem em espelho
- Referência: ESC 2015, AHA

### Tamponamento Cardíaco
- Baixa voltagem difusa (QRS < 5mm nas derivações do plano frontal)
- Alternância elétrica (variação da amplitude do QRS batimento a batimento)
- Taquicardia sinusal
- Pode ter alterações inespecíficas de ST
- **Nota:** CNN auxilia na detecção de alternância elétrica; regra checa baixa voltagem
- Referência: ESC, AHA
