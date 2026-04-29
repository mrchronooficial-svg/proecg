# Análise Técnica Exaustiva — Paper PMcardio (Demolder et al., 2025)
## "High Precision ECG Digitization Using Artificial Intelligence"
### Documento base para construção do digitalizador ProECG

---

## 1. PIPELINE COMPLETO — VISÃO GERAL

O sistema PMcardio opera em **dois estágios sequenciais**, cada um com sub-módulos:

```
ESTÁGIO 1: ECG Normalization
  Foto bruta → [Dotter] → [Gridder] → [Undistortion] → Imagem normalizada

ESTÁGIO 2: ECG Reconstruction  
  Imagem normalizada → [Leader] → [Post-processing] → [Format Detection] → Sinais digitais (12 leads)
```

**Fluxo detalhado (confirmado pela Figura 1):**

1. **Input:** Foto de ECG em papel (JPEG ou PNG, RGB)
2. **Dotter (DL):** Detecta pontos de interseção do grid milimetrado
3. **Gridder (algoritmo):** Constrói matriz de grid a partir dos pontos
4. **Undistortion (algoritmo):** Corrige distorção não-linear quadrado a quadrado
5. **Leader (DL):** Segmenta traçados como máscara binária na imagem normalizada
6. **Post-processing (algoritmo):** Refina leads, conecta gaps, filtra ruído
7. **Format Detection (heurística):** Identifica layout, rotação, ordem dos leads
8. **Conversão final:** Pixel → µV usando paper speed e voltage gain

**[inferência visual - Figura 1]:** O pipeline é estritamente sequencial. A etapa 2 (Reconstruction) recebe a imagem já normalizada — não a foto bruta. A imagem normalizada mostra grid perfeitamente regular com fundo limpo.

---

## 2. MÓDULO DOTTER — Detecção de Keypoints do Grid

### Arquitetura da rede neural
- **Tipo:** U-Net com Residual Blocks (ResBlocks)
- **U-Net base:** Ronneberger et al. (Ref 11) — arquitetura encoder-decoder com skip connections para segmentação biomédica
- **ResBlocks:** He et al. (Ref 12, 13) — o paper cita AMBOS os artigos: o original de 2016 (deep residual learning) e o de identity mappings. Isso sugere que usam **pre-activation ResBlocks** (batch norm → activation → conv), que é a versão de identity mappings
- **Optimizer:** ADAM (Ref 14, Kingma & Ba 2014)
- **Loss function:** Binary cross-entropy with logits (BCEWithLogitsLoss no PyTorch)
- **Activation function:** SiLU (Sigmoid-weighted Linear Unit, Ref 15, Elfwing et al. 2018) — equivalente a `x * sigmoid(x)`, também chamada "Swish"
- **Epochs:** 300
- **Learning rate:** 0.005
- **Input size:** 256×256 pixels (patches)
- **Output:** Máscara binária preta-e-branca com dots (interseções de grid) em branco

### Processo de patching (input 256×256)
O paper diz que "input patches, each measuring 256×256 pixels, were obtained by segmenting the ECG images into smaller sections." Isso significa:

1. A imagem ECG original (resolução variável) é dividida em patches de 256×256px
2. Cada patch é processado independentemente pelo Dotter
3. As máscaras são recombinadas para formar a máscara completa da imagem

**[não documentado]:** Overlap entre patches, stride do patching, como as bordas são tratadas na recombinação, e se há normalização de tamanho da imagem antes do patching.

### Dataset de treino
- **Total:** 176 imagens ECG anotadas
- **Split:** 140 treino / 36 validação
- **Anotações:** Média de 1.944 (±649) interseções de grid marcadas por ECG
- **O que é um keypoint:** Cada ponto onde duas linhas de grid se cruzam (interseção milimétrica)
- **Augmentation:** >10.000 patches augmented gerados a partir das 176 imagens originais

### Variações cobertas no dataset de treino
- Posições extremas de câmera
- Diferentes qualidades de papel (dobras, distorções, arranhões)
- Efeitos visuais (brilho, contraste, desfoque, tons de cor, sombras, efeito Moiré)
- Diferentes formatos de impressão (layouts de leads, formas de grid, cores de tinta, tamanhos de papel)

### Augmentations aplicadas aos patches (confirmado Figura 3)
- **Brightness adjustment** — simula sub/superexposição
- **Color tone variation** — simula diferentes temperaturas de cor/iluminação
- **Rotation** — simula foto tirada em ângulo
- **Image flipping** — espelhamento horizontal/vertical

**[inferência visual - Figura 3]:** Os patches mostram regiões de ~3-5 quadrados de grid cada, com traçados ECG visíveis. As augmentations mantêm o grid visível mas mudam significativamente a aparência visual. A rotação é de ângulo moderado (estimativa: ±15-20°). O flipping produz espelhamento completo.

### Output do Dotter
- **Formato:** Máscara binária (preto e branco) do mesmo tamanho que o input
- **Preto:** Background (não é interseção)
- **Branco:** Dots (centros das interseções de grid)
- **Interpretação:** Cada blob branco na máscara corresponde a um ponto de interseção do grid milimetrado. O centroide de cada blob dá a coordenada sub-pixel da interseção.

---

## 3. MÓDULO GRIDDER — Construção da Grade

### O que o paper documenta
O Gridder é descrito como um "algoritmo" (não deep learning) que:
1. Recebe o output do Dotter (conjunto de pontos de interseção)
2. Constrói uma **matriz de grid** organizada (linhas × colunas)
3. Trata **gaps** (pontos não detectados) via interpolação
4. Trata **distorções** do papel

### O que a Figura 1 (painel Normalization) revela
**[inferência visual]:** O painel "Grid point matrix generated by the Gridder algorithm" mostra a foto original do ECG amassado com pontos coloridos sobrepostos formando uma grade. Os pontos seguem as curvaturas do papel — ou seja, a grade resultante NÃO é regular, ela mapeia fielmente as distorções do papel. Há pontos coloridos diferentemente (possivelmente vermelho, verde, azul) sugerindo que o grid é organizado em linhas e colunas indexadas.

### Tratamento de gaps e distorções
O paper menciona que o Gridder "handles distortions and filling gaps through interpolation." Isso implica:
1. Os pontos detectados pelo Dotter podem ter falhas (interseções não detectadas)
2. O Gridder interpola as posições faltantes usando os vizinhos
3. A grade final é uma representação completa e contínua do grid do papel

**[não documentado]:** O algoritmo exato de interpolação (linear, spline, bicúbica), como as linhas e colunas são identificadas (RANSAC? Hough? agrupamento por proximidade?), como lida com grids parcialmente visíveis nas bordas, e se usa a periodicidade esperada (1mm/5mm) como constraint.

---

## 4. MÓDULO UNDISTORTION — Correção de Distorção

### O que significa "4-point linear undistortion per grid square"
Matematicamente, cada quadrado individual do grid é tratado como uma transformação separada:

1. **Grid distorcido:** Cada quadrado no paper amassado tem 4 cantos em posições irregulares (determinados pelo Gridder)
2. **Template regular:** Um template com quadrados perfeitamente regulares e espaçados uniformemente é gerado
3. **Mapeamento:** Para cada quadrado, uma **transformação projetiva (homografia) de 4 pontos** mapeia os 4 cantos do quadrado distorcido para os 4 cantos do quadrado regular correspondente
4. **Resultado:** A imagem é "desembrulhada" quadrado por quadrado, produzindo uma imagem com grid perfeitamente regular

### O que a Figura 4a mostra
**[inferência visual]:** Um template em branco com quadrados regulares azuis (wireframe). Este é o "alvo" — a imagem destino onde o ECG será projetado. Cada quadrado tem o mesmo tamanho e espaçamento uniforme. Há também um quadrado verde sólido, possivelmente indicando um quadrado específico sendo mapeado.

### O que a Figura 4b mostra
**[inferência visual]:** A mesma imagem distorcida com linhas azuis curvadas conectando os cantos do grid distorcido ao template regular. As linhas mostram a "trajetória" do mapeamento — claramente curvas, não retas, demonstrando que o paper tem distorção não-linear. A distorção total é decomposta em distorções locais lineares (uma por quadrado).

### Por que o fundo fica preto nas bordas
**[inferência visual - Figura 1, imagem undistorted]:** A imagem normalizada tem bordas pretas porque:
1. O paper original fotografado em ângulo ou amassado tem uma forma irregular quando projetado
2. O template regular é retangular
3. Pixels do template que não têm correspondente na foto original ficam pretos (sem dados)
4. Artefato inevitável da correção projetiva — equivalente a padding com zeros

### Referência da patente
- **US Patent App. 18/280,116** — Rovder et al., "Methods for generating an image of a graphical representation" (2024)
- O paper diz: "Further details can be found in the patent document"
- **O que está omitido:** Todo o algoritmo detalhado do Undistortion, incluindo como a homografia é computada, como os quadrados são blendados nas fronteiras, como trata artefatos de aliasing, e como lida com oclusões

---

## 5. MÓDULO LEADER — Segmentação de Leads

### Por que precisa da imagem normalizada
O Leader processa a imagem APÓS normalização (confirmado pela Figura 1). Razões:
1. Na imagem normalizada, o grid é regular → mais fácil separar traçado de grid
2. Os leads estão alinhados horizontalmente → mais fácil segmentar colunas
3. Distorções de perspectiva já foram removidas → traçados sem curvatura artificial

**[inferência visual - Figura 1]:** O input do Leader é claramente a imagem após undistortion (fundo com bordas pretas, grid regular). O output é uma máscara B&W onde as linhas brancas correspondem aos traçados dos leads.

### Arquitetura
- **Tipo:** U-Net com ResBlocks — **mesma arquitetura do Dotter**
- **Input:** Patches 256×256 da imagem normalizada
- **Output:** Máscara binária com leads em branco
- **Métrica:** IoU (Intersection over Union / Jaccard Index)

### Por que IoU foi escolhida
IoU é a métrica padrão para segmentação semântica:
- IoU = (Interseção ÷ União) entre pixels preditos e reais
- Range 0-1. Mais rigorosa que accuracy para segmentação
- Penaliza tanto falsos positivos (grid marcado como lead) quanto falsos negativos (lead não detectado)

**[não documentado]:** O valor de IoU alcançado no validation set.

### Dataset de treino
- **Total:** 232 imagens ECG
- **Split:** 185 treino / 47 validação
- **Anotação por layers:** Cada lead em camada separada (como layers no Photoshop)
- **Verificação:** Layers plotados sobre foto original para checar precisão
- **Augmentation:** >10.000 patches, mesmas augmentations do Dotter

### Output do Leader
- Máscara binária: branco = pixel pertence a um lead, preto = background/grid
- A máscara NÃO diferencia entre leads individuais — classificação é no post-processing

---

## 6. POST-PROCESSING DA RECONSTRUÇÃO

### Lista completa e ordenada de etapas
1. **Filtragem de ruído** — Remove artefatos pequenos da máscara
2. **Identificação da ROI** — Encontra região útil com leads
3. **Segmentação de colunas de leads** — Divide em colunas por grupo de leads
4. **Identificação da baseline** — Para cada lead, identifica linha de base (0 mV)
5. **Extensão de paths** — Estende traçados que terminam prematuramente
6. **Conexão de endpoints** — Em regiões complexas, conecta pontos finais
7. **Maximização de brilho** — Otimização visual [detalhes não claros]
8. **Correção de overlapping** — Trata sobreposição entre leads adjacentes
9. **Correção de gaps** — Preenche lacunas no traçado

### O que a Figura 10 (V2 problemático) revela
Métricas específicas documentadas para o lead V2:
- Target count: 2500 amostras
- Predicted count: 2431 amostras (69 faltando = 2.8%)
- RMSE: 0.389 mV (muito alto)
- PCC: 0.676 (moderado)
- SNR: 2.57 dB (péssimo)

**Causa:** Artefatos no final do registro geram NaN → preenchidos com zeros → métricas distorcidas.

**Para o ProECG:** Truncar sinal na última amostra válida. Reportar duração efetiva.

### Como NaNs são tratados
- **No pipeline:** NaNs onde não há traçado detectável
- **Tabela 2 (só sucessos):** NaNs excluídos
- **Tabela 4 (zero-filled):** NaNs → zeros — penaliza fortemente

### Fórmula de conversão pixel → µV
```
Amplitude_mV = (y_baseline - y_pixel) × (1 / gain_mm_per_mV) × (1 / px_per_mm)
Amplitude_µV = Amplitude_mV × 1000

Tempo_s = x_pixel × (1 / speed_mm_per_s) × (1 / px_per_mm)

Padrão brasileiro:
- gain_mm_per_mV = 10 mm/mV
- speed_mm_per_s = 25 mm/s  
- px_per_mm = determinado pela calibração do grid
```

---

## 7. FORMAT DETECTION

### "Limb-lead logic" — Exploração algorítmica
Relações matemáticas fixas das derivações dos membros (Lei de Einthoven):
- **II = I + III**
- **aVR = -(I + II)/2**
- **aVL = (I - III)/2**
- **aVF = (II + III)/2**

O sistema testa combinações de leads candidatos para verificar qual atribuição satisfaz essas relações. A combinação que melhor satisfaz Einthoven identifica corretamente qual lead é qual.

### "Rhythm lead information"
O rhythm lead (DII longo) é detectável por:
- Comprimento excepcional (toda a largura do ECG)
- Posição na última linha
- Mais ciclos cardíacos visíveis

### Por que não funciona para ECGs multipágina
Sem rhythm lead contínuo para referência, e Einthoven pode não ser verificável se derivações estão distribuídas entre páginas.

---

## 8. DATASET PM-ECG-ID

### Construção
1. **Origem:** PTB-XL (público), 100 ECGs selecionados (75 patológicos + 25 normais), 500 Hz
2. **Impressão:** Waveforms em grid milimetrado → impressão em papel
3. **Fotografia:** Sob várias condições
4. **Resolução waveform images:** 7.483×5.291 pixels (~40 MP, geradas digitalmente)

### Câmeras usadas
| Câmera | Resolução | Abertura |
|--------|-----------|----------|
| Samsung Galaxy A32 | 64.2 MP | f/1.8 |
| iPhone 11 | 12 MP | f/1.8 |
| Doogee S55 | 13 MP | f/2.0 |

Distâncias: base (ideal), medium (~35cm), large (~53cm)

### 60 cenários completos

**Photo Variations (7 × 100 = 700):**
Bends, Crumples, iPhone, Doogee, Samsung, Scans, Screen photos

**Samsung Photo Augmentations (31 × 100 = 3.100):**
Blur {11,21,31,41}, Brightness {±40,±80,±120,±160}, Contrast gamma {0.125,0.25,0.5,0.667,1.5,2,4,8}, JPEG {90,45,22,11,5,2}, Scale {1/2,1/4,1/8,1/16,1/32}

**Distance Photo Augmentations (11 × 100 = 1.100):**
Padding {small,medium,large,extra_large}, Perspective {20°,30°,40°}, Rotation {5°,15°,25°,35°}

**Waveform Transformations (11 × 100 = 1.100):**
HF noise {low,mid,large} (20-40Hz), LF noise {low,mid,large} (0.05-0.3Hz), Lead opacity {1.0,0.6,0.4,0.2,0.15}

**Total: 6.000 imagens**

### Diferença entre augmentations
- **Foto (verde):** Pós-impressão. Simulam degradação de captura (blur, brilho, compressão)
- **Waveform (azul):** Pré-impressão. Modificam o ECG (ruído, opacidade). Re-renderizados em JPEG 7.483×5.291

---

## 9. PERFORMANCE POR LEAD (Tabela 2 — só sucessos)

| Lead | Fail Rate | RMSE (mV) | PCC | SNR (dB) |
|------|-----------|-----------|-----|----------|
| I | 6.63% | 0.034 | 0.953 | 15.574 |
| II | 6.65% | 0.040 | 0.957 | 15.717 |
| III | 6.65% | 0.043 | 0.957 | 15.762 |
| aVR | 6.62% | 0.031 | 0.949 | 15.232 |
| aVL | 6.62% | 0.034 | 0.955 | 15.446 |
| aVF | 6.62% | 0.035 | 0.958 | 15.758 |
| V1 | 6.57% | 0.045 | 0.957 | 16.445 |
| V2 | 6.57% | 0.079 | 0.949 | 15.117 |
| V3 | 6.57% | 0.095 | 0.936 | 13.799 |
| **V4** | **6.67%** | **0.099** | **0.918** | **12.758** |
| V5 | 6.67% | 0.081 | 0.935 | 13.484 |
| V6 | 6.65% | 0.051 | 0.952 | 15.505 |
| **Todos** | **6.62%** | **0.056** | **0.948** | **15.050** |

### Interpretação clínica

**V3-V5 piores por:**
- Maior amplitude QRS → overlap com leads adjacentes
- Na disposição 3×4, ficam na região central com maior densidade
- Onda R em V4 pode atingir 2-3 mV (20-30mm), cruzando lead acima

**Risco clínico V4:** RMSE 0.099 mV ≈ 1mm em papel. Supra limítrofe de 1mm em V4 pode ser indistinguível do erro de digitalização. IAMCSST anterior depende de V1-V4.

**Fail rate uniforme (~6.6%):** Quando falha, falha para TODOS os leads — problema na normalização, não na reconstrução.

---

## 10. PERFORMANCE POR CENÁRIO (Tabela 3)

### Easy (12 subsets): PCC 0.977, RMSE 0.041, SNR 16.682
Digital original, JPEG 90-45%, Brightness ±40, Contrast 0.5-1.5, Scale 1/2, Scans, Samsung, iPhone

### Medium (29 subsets): PCC 0.970, RMSE 0.047, SNR 15.785
JPEG 22-11%, Blur 11-21, Brightness ±80/±120, Contrast 0.125-0.25/2-4, Zoom small-medium, Scale 1/4, Rotation 5-15°, HF/LF noise small-medium, Lead opacity 60-40%, Doogee, Screens, Bent, Crumpled

### Hard (19 subsets): PCC 0.817, RMSE 0.098, SNR 11.101
JPEG 5-2%, Blur 31-41, Brightness -160/+160, Contrast g=8, Zoom large-extreme, Scale 1/8-1/32, Rotation 25-35°, HF/LF noise large, Lead opacity 20-15%

### Cenários com falha crítica (>5%)
| Cenário | Fail Rate | PCC | RMSE |
|---------|-----------|-----|------|
| Scale down 1/32 | 100% | — | — |
| Blur kernel 41 | 96% | 0.087 | 0.226 |
| Blur kernel 31 | 69% | 0.365 | 0.240 |
| Zoom out extreme | 47.25% | 0.888 | 0.077 |
| Lead opacity 15% | 31% | 0.949 | 0.061 |
| Scale down 1/16 | 17.75% | 0.290 | 0.254 |

### Timing
- Média geral: 4.86s (±0.6s)
- Easy: ~4.5s (±0.15s), Hard: até 6.35s
- **Sempre < 7 segundos**

---

## 11. TABELA 2 vs TABELA 4 (ZERO-FILLED)

| Métrica | Tabela 2 (só sucessos) | Tabela 4 (zero-filled) | Delta |
|---------|----------------------|----------------------|-------|
| PCC | 0.948 | 0.885 | -6.6% |
| RMSE | 0.056 mV | 0.067 mV | +19.6% |
| SNR | 15.050 | 14.053 | -6.6% |

Para cenários difíceis a diferença é maior (ex: Zoom extreme: PCC 0.888 → 0.468).

---

## 12. ESPECIFICAÇÕES TÉCNICAS (Supplemental Table 1)

| Feature | Suportado |
|---------|-----------|
| Formatos | JPEG, PNG (RGB) |
| Voltage gains | 2.5, 5, 10, 20 mm/mV |
| Paper speeds | 5, 12.5, 25, 50 mm/s |
| Rotação | 0°, 90°, 180°, 270° + até 25° |
| Grid | 1mm minor + 5mm major |
| Contraste grid | Variedade de cores (grid mais escuro que fundo) |
| Origem | Scans, fotos de tela, fotos de celular |
| Imperfeições | Amassados, dobras, perspectiva leve |
| Óptica | Blur, contraste, brilho, zoom variáveis |
| Leads | Parcialmente faltantes ou desbotados |

---

## 13. LAYOUTS SUPORTADOS (Supplemental Table 2)

| Formato | Descrição | PMcardio | Outros |
|---------|-----------|----------|--------|
| **3×4+1** | **Padrão brasileiro** | ✓ | Santamónica, Wu, Badilini |
| 3×4+0 | Sem rhythm | ✓ | Santamónica, Wu, Badilini, Ravichandran |
| 3×4+2 | +2 rhythms | ✓ | Santamónica |
| 3×4+3 | +3 rhythms | ✓ | Santamónica |
| 6×2+0/+1 | 6 linhas de 2 | ✓ | Santamónica |
| 12×1+0 | Lead contínuo | ✓ | Santamónica, Wu, Badilini |
| Multipágina | Vários formatos | ✓ | Nenhum outro (exclusivo) |

---

## 14. LIMITAÇÕES DOCUMENTADAS

### Pacemaker spikes
Spikes de ~2ms, 2-20 mV. No papel: ~0.5mm (1-2px). O UNet não consegue distinguir do traçado normal. Pacientes com marcapasso ≈ 5% dos ECGs de emergência.

### V3-V5
Maior amplitude → overlapping → segmentação falha. Região central com maior densidade.

### Thresholds de rejeição implícitos
- Scale 1/32: 100% falha → resolução mínima obrigatória
- Blur kernel 41: 96% → blur extremo irrecuperável
- Blur kernel 31: 69% → blur severo crítico

**[não documentado]:** Critérios formais de rejeição automática.

---

## 15. REFERÊNCIAS CRÍTICAS

| Ref | O que é | O que usar |
|-----|---------|------------|
| 11 | U-Net (Ronneberger 2015) | Arquitetura encoder-decoder com skip connections |
| 12 | ResNet original (He 2016a) | Shortcut connections |
| 13 | Identity Mappings (He 2016b) | Pre-activation: BN→SiLU→Conv→BN→SiLU→Conv + shortcut |
| 14 | ADAM (Kingma 2014) | Optimizer adaptativo, LR 0.005 |
| 15 | SiLU (Elfwing 2018) | x × sigmoid(x), substitui ReLU |
| 16 | Patente Undistortion | Inacessível para implementação |
| 17/18 | PTB-XL | 21.837 ECGs, 500Hz, base do PM-ECG-ID |
| 19 | PM-ECG-ID | 6.000 imagens públicas para benchmark |

---

## 16. O QUE PODEMOS REPLICAR vs. O QUE FALTA

### Replicável fielmente
- Arquitetura Dotter/Leader (UNet+ResBlocks+SiLU+BCELoss+ADAM)
- Format Detection (Einthoven + comprimento do rhythm lead)
- Métricas (PCC, RMSE, SNR)
- Benchmark com PM-ECG-ID (público)

### Parcialmente replicável
- Dotter/Leader treino (precisamos de nosso dataset anotado)
- Gridder (conceito sim, algoritmo não)
- Undistortion (conceito sim, detalhes na patente)
- Post-processing (lista de etapas sem algoritmos)

### Precisamos resolver por conta própria
- Algoritmo do Gridder (organização de pontos em matriz)
- Algoritmo do Undistortion (blending, bordas, aliasing)
- Baseline detection
- Overlap correction
- Cores de grid brasileiras (HSV ranges)
- Thresholds de grid detection para ECGs brasileiros
- Layouts específicos de fabricantes brasileiros
- Critérios de rejeição de qualidade

---

## 17. ARQUITETURA RECOMENDADA PARA O ProECG

### Pipeline (6 módulos)

```
Foto celular
    │
    ▼
[1. PRÉ-PROCESSAMENTO]          ← OpenCV
    │ Crop auto do papel ECG
    │ Correção de perspectiva  
    ▼
[2. GRID DETECTION]              ← UNet (pesos Ahus-AIM ou treinar)
    │ Patches 256×256 → máscara keypoints
    ▼
[3. GRID CONSTRUCTION]           ← Algoritmo próprio
    │ Pontos → matriz → interpolação
    ▼
[4. UNDISTORTION]                ← Homografia por quadrado (OpenCV)
    │ cv2.getPerspectiveTransform por célula
    ▼
[5. LEAD SEGMENTATION]           ← UNet (pesos Ahus-AIM ou treinar)
    │ Patches 256×256 → máscara leads
    ▼
[6. SIGNAL EXTRACTION]           ← Algoritmo próprio
    │ Layout 3×4+1 → baseline → px→µV → interpolação
    ▼
12 arrays µV × tempo
```

### Fases de implementação

**Fase 1 — MVP (2-3 semanas):**
Pesos Ahus-AIM, autocorrelação grid, layout fixo 3×4+1, 25mm/s 10mm/mV, homografia global

**Fase 2 — Robustez (1-2 meses):**
Fine-tuning brasileiro, undistortion por quadrado, detecção automática layout/speed, overlap V3-V5

**Fase 3 — Produção (2-3 meses):**
Dataset brasileiro >500 ECGs, métricas clínicas por diagnóstico, quality scoring input, pacemaker
