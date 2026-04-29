# ProECG — Mapa Completo do Projeto
## Todas as etapas, sub-etapas e sub-sub-etapas em detalhe

---

# PILAR 1: DIGITALIZAÇÃO
**Objetivo:** Transformar a foto de um ECG de papel (tirada com celular) em 12 sinais digitais (números que representam as ondas do ECG em cada derivação).

---

## 1. PREPARAÇÃO DOS DADOS DE TREINO

Antes de construir qualquer coisa, precisamos de "material de estudo" para ensinar a IA. São dois conjuntos de dados: um sintético (gerado por computador) e um real (fotos de celular).

### 1.1 Dataset sintético (gerado por computador)

#### 1.1.1 Obter sinais ECG digitais do PTB-XL
- Baixar o banco de dados PTB-XL (banco público com 21.837 ECGs digitais gravados em hospitais)
- Selecionar ~500 ECGs variados (normais + diversas patologias)
- Cada ECG tem 12 derivações, 10 segundos de duração, 500 amostras por segundo

#### 1.1.2 Renderizar os sinais em "papel virtual"
- Criar um programa que desenha o ECG em uma imagem, como se fosse impresso em papel milimetrado
- O programa desenha: o grid (quadriculado) de fundo + os traçados das 12 derivações por cima
- Configurar para padrão brasileiro: 25mm/s (velocidade horizontal), 10mm/mV (escala vertical)
- Layout 3×4+1 (3 linhas com 4 derivações cada + DII longo embaixo)
- Variar as cores do grid: laranja, rosa, azul, verde, cinza (simular fabricantes brasileiros)
- Resolução: gerar em alta resolução (~4000×3000 pixels) para simular foto de celular boa

#### 1.1.3 Aplicar "sujeiras" nas imagens (augmentations)
- **Rotação:** girar a imagem de 0° a 35° (simula foto torta)
- **Perspectiva:** distorcer como se a foto fosse tirada de ângulo (não de frente)
- **Brilho:** mais claro e mais escuro (simula iluminação ruim de UBS)
- **Contraste:** mais e menos contraste (simula papel desbotado)
- **Blur (desfoque):** simula foto tremida ou fora de foco
- **Compressão JPEG:** simula perda de qualidade quando a foto é compartilhada por WhatsApp
- **Redução de resolução:** simula câmera ruim de celular barato
- **Rugas e dobras:** simula papel que foi dobrado ou amassado (deformação geométrica)
- **Sombras:** simula sombra de mão ou objeto sobre o papel
- **Padding (bordas):** adicionar mesa/fundo ao redor do papel (simula a foto real onde o ECG não ocupa a imagem toda)

#### 1.1.4 Gerar as "respostas certas" automaticamente (ground truth)
- Como NÓS criamos a imagem, sabemos exatamente:
  - Onde está cada cruzamento do grid (keypoints) → resposta certa para o Dotter
  - Onde está cada traçado (pixels do lead) → resposta certa para o Leader
  - Qual é o sinal original em µV → resposta certa para validar o pipeline todo
- Salvar tudo organizado: imagem + máscara de grid + máscara de leads + sinal original

#### 1.1.5 Meta final do dataset sintético
- ~5.000 imagens com todas as variações
- Cada imagem com suas "respostas certas" prontas
- Tempo estimado para criar o programa gerador: ~1 semana
- Tempo para gerar as 5.000 imagens: algumas horas (automático)

### 1.2 Dataset real (fotos de celular)

#### 1.2.1 Coletar fotos reais de ECGs brasileiros
- Precisamos de ~100-200 fotos de ECGs tiradas com celular em condições reais
- Variedade necessária:
  - Diferentes fabricantes (TEB, Atrys, Biomet, ECGMAC, CardECG, Micromed, etc.)
  - Diferentes cores de grid (laranja, rosa, azul, verde, cinza)
  - Diferentes qualidades de foto (boa, razoável, ruim)
  - Diferentes ambientes (UBS, emergência, consultório)
  - Com e sem sujeiras (papel amassado, dobrado, com carimbo, com anotações manuscritas)
- IMPORTANTE: remover qualquer dado de paciente (nome, CPF) antes de usar

#### 1.2.2 Anotar as fotos reais — Pontos de grid
- Usar uma ferramenta de anotação (CVAT ou Label Studio — softwares gratuitos que rodam no navegador)
- Para cada foto: clicar em cada cruzamento visível do grid milimetrado
- ~1.000-2.000 cliques por foto
- Tempo estimado: ~20 minutos por foto
- Atalho: treinar um modelo "rascunho" primeiro (com dados sintéticos) e usar ele para pré-marcar os pontos, aí a pessoa só corrige os erros

#### 1.2.3 Anotar as fotos reais — Traçados dos leads
- Na mesma ferramenta de anotação
- Para cada foto: "pintar" por cima de cada traçado de derivação
- Cada derivação em uma camada separada (como folhas transparentes empilhadas)
- Tempo estimado: ~15 minutos por foto
- Mesmo atalho: pré-anotar com modelo rascunho e só corrigir

#### 1.2.4 Meta final do dataset real
- ~100-200 fotos anotadas com pontos de grid E traçados de leads
- Tempo estimado total de anotação: ~50-80 horas de trabalho manual (com pré-anotação)
- Pode ser terceirizado para uma pessoa treinada (não precisa ser médico)

---

## 2. CONSTRUÇÃO DA ARQUITETURA DA IA (código dos modelos)

Aqui escrevemos o "projeto do cérebro" — a estrutura da rede neural. Ainda não é treino, é só o desenho.

### 2.1 Modelo Dotter (detecta pontos de grid)

#### 2.1.1 Escrever a arquitetura UNet + ResBlocks
- UNet: rede em formato de U — comprime a imagem para entender o contexto, depois expande de volta ao tamanho original para classificar cada pixel
- ResBlocks: blocos com "atalhos" que ajudam a rede a aprender melhor (evitam que o aprendizado "estacione")
- Activation SiLU: a "decisão" de cada neurônio — mais suave que alternativas mais antigas
- Tudo seguindo exatamente o que o paper do PMcardio documenta

#### 2.1.2 Configurar o treino
- Loss function: BCEWithLogitsLoss (a "nota" que diz quão errada a rede está — ela tenta minimizar essa nota)
- Optimizer: ADAM (o "método de estudo" — como a rede ajusta seus pesos a cada rodada)
- Learning rate: 0.005 (a "velocidade de aprendizado" — quão grandes são os ajustes a cada passo)
- Epochs: 300 (quantas vezes a rede vê o dataset inteiro durante o treino)
- Input: patches de 256×256 pixels (pedacinhos da imagem)

#### 2.1.3 Escrever o script de treino
- Código que: carrega as imagens, recorta em patches, alimenta a rede, calcula o erro, ajusta os pesos
- Inclui validação: a cada X rodadas, testa em imagens que a rede nunca viu para ver se está melhorando
- Salva os "pesos" (o que a rede aprendeu) em um arquivo .pt

### 2.2 Modelo Leader (segmenta traçados dos leads)

#### 2.2.1 Escrever a arquitetura
- Mesma arquitetura do Dotter (UNet + ResBlocks + SiLU)
- Mesmo tamanho de input (256×256)
- Diferença: o Dotter detecta pontos, o Leader detecta linhas (traçados)

#### 2.2.2 Configurar o treino
- Mesmos hiperparâmetros do Dotter
- Métrica de avaliação: IoU (Intersection over Union — mede quão bem a máscara prevista se sobrepõe à máscara real, de 0 a 1)

#### 2.2.3 Escrever o script de treino
- Mesmo processo do Dotter, mas usando as anotações de traçados em vez de pontos de grid

---

## 3. TREINO DOS MODELOS (rodar em GPU)

O treino é pesado computacionalmente — precisa de GPU (placa de vídeo potente). Não roda no Claude Code, roda no Google Colab ou similar.

### 3.1 Treino do Dotter

#### 3.1.1 Fase 1 — Treinar com dados sintéticos
- Subir o código e o dataset sintético para o Google Colab
- Rodar 300 epochs (~2-6 horas dependendo da GPU)
- Ao final: modelo "rascunho" que sabe detectar grid em imagens limpas

#### 3.1.2 Fase 2 — Refinar com dados reais (fine-tuning)
- Pegar o modelo rascunho e continuar treinando, agora com as fotos reais anotadas
- Menos epochs (~50-100), learning rate menor (para não "esquecer" o que já aprendeu)
- Ao final: modelo que funciona com fotos de celular de ECGs brasileiros

#### 3.1.3 Avaliar o modelo
- Testar em fotos reais que o modelo NUNCA viu durante o treino
- Medir: quantos pontos de grid foram encontrados corretamente vs. quantos faltaram vs. quantos foram falsos positivos
- Se os resultados forem ruins: ajustar augmentations, coletar mais dados reais, tentar mais epochs

### 3.2 Treino do Leader

#### 3.2.1 Fase 1 — Treinar com dados sintéticos
- Mesmo processo do Dotter, mas agora a rede aprende a identificar traçados, não pontos
- ~2-6 horas no Colab

#### 3.2.2 Fase 2 — Refinar com dados reais
- Fine-tuning com as fotos reais onde os traçados foram anotados
- ~50-100 epochs extras

#### 3.2.3 Avaliar o modelo
- Medir IoU (sobreposição entre máscara prevista e real)
- Meta: IoU > 0.8 (80% de sobreposição)
- Atenção especial a V3-V5 (derivações onde traçados se sobrepõem mais)

---

## 4. PIPELINE DE DIGITALIZAÇÃO (os 6 módulos em código)

Com os modelos treinados, construímos o pipeline completo que transforma foto → sinal digital. Cada módulo abaixo é um pedaço de código Python.

### 4.1 Módulo 1: Pré-processamento da foto

#### 4.1.1 Detectar o papel ECG dentro da foto
- A foto do celular tem o ECG + mesa + mão + teclado + outros objetos
- O código precisa encontrar onde começa e termina o papel do ECG
- Técnica: detectar bordas retas (contornos) e encontrar o retângulo maior → esse é o papel
- Fallback: se não encontrar bordas claras, usar detecção por textura (região com padrão repetitivo = grid)

#### 4.1.2 Corrigir a perspectiva
- A foto pode estar torta (tirada de ângulo, não de cima)
- O código pega os 4 cantos do papel e "estica" a imagem para ficar reta
- Técnica: transformação de perspectiva do OpenCV (biblioteca de visão computacional)
- Resultado: imagem retangular com o papel ECG ocupando toda a área

#### 4.1.3 Normalizar o tamanho
- Redimensionar a imagem para um tamanho padrão que os modelos esperam
- Manter a proporção original (não distorcer)

### 4.2 Módulo 2: Dotter (detectar pontos de grid)

#### 4.2.1 Dividir a imagem em patches
- Cortar a imagem grande em pedacinhos de 256×256 pixels
- Os pedacinhos podem ter sobreposição entre si (para não perder informação nas bordas)

#### 4.2.2 Rodar o modelo Dotter em cada patch
- Cada patch entra no modelo treinado
- O modelo retorna uma máscara de probabilidades (cada pixel tem uma nota de 0 a 1)
- Converter para binário: nota > 0.5 = é ponto de grid, nota ≤ 0.5 = não é

#### 4.2.3 Remontar a máscara completa
- Juntar todas as máscaras dos patches de volta na imagem original
- Nas áreas de sobreposição: fazer média das probabilidades

#### 4.2.4 Encontrar os centros dos pontos
- Cada "blob" branco na máscara é um ponto de grid detectado
- Calcular o centro (centroide) de cada blob → coordenada precisa do ponto

### 4.3 Módulo 3: Gridder (organizar pontos em grade)

#### 4.3.1 Agrupar pontos em linhas horizontais
- Os pontos de grid formam linhas horizontais e verticais
- Algoritmo: agrupar pontos que estão na mesma altura (Y similar) → cada grupo é uma linha

#### 4.3.2 Agrupar pontos em colunas verticais
- Dentro de cada linha, ordenar da esquerda para a direita
- Verificar quais pontos de linhas diferentes estão na mesma posição horizontal → cada grupo é uma coluna

#### 4.3.3 Montar a matriz de grid
- Resultado: uma tabela onde cada célula é uma interseção do grid
- Formato: grid[linha][coluna] = coordenada (x, y) do ponto
- Algumas células podem estar vazias (pontos não detectados)

#### 4.3.4 Interpolar pontos faltantes
- Para cada célula vazia: estimar a posição baseada nos vizinhos
- Técnica: interpolação linear (traçar linha entre dois vizinhos conhecidos e achar o ponto no meio)
- Resultado: grade completa sem buracos

#### 4.3.5 Calcular o espaçamento do grid em pixels
- Medir a distância média entre pontos consecutivos
- Esse valor = quantos pixels equivalem a 1mm no papel
- Confirmar: distância de 5 em 5 pontos deveria ser 5× maior (linhas grossas do grid a cada 5mm)

### 4.4 Módulo 4: Undistortion (corrigir distorção do papel)

#### 4.4.1 Criar o template regular (alvo)
- Gerar uma grade perfeita com quadrados todos iguais e espaçados uniformemente
- Cada quadrado tem o tamanho do espaçamento do grid calculado no passo anterior

#### 4.4.2 Mapear cada quadrado distorcido para o quadrado regular
- Para cada quadrado do grid:
  - Pegar os 4 cantos no grid distorcido (posições reais na foto)
  - Pegar os 4 cantos correspondentes no template regular (posições ideais)
  - Calcular a transformação de perspectiva entre eles (homografia — a "receita" de como deformar um no outro)

#### 4.4.3 Aplicar a correção quadrado por quadrado
- Para cada quadrado: aplicar a transformação e copiar os pixels da foto para o template
- Resultado: imagem "desembrulhada" onde o grid é perfeitamente regular
- Bordas ficam pretas onde não havia informação na foto original

#### 4.4.4 Limpeza final
- Recortar a imagem para remover bordas pretas excessivas
- Resultado: imagem normalizada, pronta para o próximo estágio

### 4.5 Módulo 5: Leader (segmentar traçados)

#### 4.5.1 Dividir a imagem normalizada em patches
- Mesmo processo do Dotter: cortar em pedacinhos de 256×256

#### 4.5.2 Rodar o modelo Leader em cada patch
- Cada patch entra no Leader treinado
- Retorna máscara: branco = pixel pertence a um traçado, preto = fundo/grid

#### 4.5.3 Remontar a máscara completa de leads
- Juntar os patches de volta
- Resultado: uma imagem binária onde TODOS os traçados estão em branco e todo o resto (grid, fundo, texto, labels) está em preto

### 4.6 Módulo 6: Extração de sinal (máscara → números)

#### 4.6.1 Identificar as linhas de leads (separação vertical)
- Na máscara de leads, encontrar as faixas horizontais onde estão os traçados
- Procurar gaps horizontais largos (espaços sem traçado entre as linhas)
- Resultado: dividir em 3 ou 4 faixas horizontais (linhas do ECG)

#### 4.6.2 Identificar as colunas dentro de cada linha
- Dentro de cada faixa horizontal, dividir em colunas (geralmente 4)
- Cada coluna = uma derivação
- Para o layout 3×4+1: 12 derivações nas 3 linhas superiores + DII longo na linha de baixo

#### 4.6.3 Atribuir nomes às derivações (Format Detection)
- Layout padrão brasileiro 3×4:
  - Linha 1: I, aVR, V1, V4
  - Linha 2: II, aVL, V2, V5
  - Linha 3: III, aVF, V3, V6
  - Linha 4: DII longo (rhythm strip)
- Validação: usar a Lei de Einthoven (II = I + III) para confirmar que a atribuição está correta
- Se não bater: tentar outras combinações

#### 4.6.4 Extrair o sinal de cada derivação
- Para cada derivação (cada retângulo na imagem):
  - Percorrer da esquerda para a direita, coluna por coluna de pixels
  - Em cada coluna: encontrar os pixels brancos (traçado) e calcular o centro (posição Y)
  - Se nenhum pixel branco na coluna: marcar como NaN (dado faltante)
  - Resultado: um array de posições Y ao longo do eixo X

#### 4.6.5 Converter pixel para µV
- Inverter o eixo Y (na imagem, Y cresce para baixo; no ECG, mV cresce para cima)
- Centralizar: subtrair a mediana (para que a baseline fique em zero)
- Converter: posição em pixels → amplitude em µV usando o espaçamento do grid
  - 1mm no papel = 0.1 mV (com ganho de 10mm/mV)
  - Se 1mm = N pixels, então: amplitude_µV = posição_pixels × (100 / N)

#### 4.6.6 Calcular o eixo temporal (sampling rate)
- Se 1mm = N pixels e a velocidade é 25mm/s:
  - Sampling rate = N × 25 amostras por segundo
- Cada ponto do array corresponde a um instante de tempo

#### 4.6.7 Interpolar gaps (preencher NaNs)
- Onde não havia traçado detectável: interpolar linearmente usando os vizinhos
- Se o gap for muito grande (> 50ms): manter como NaN e alertar

#### 4.6.8 Filtrar ruído
- Aplicar filtro suave para remover serrilhado (artefatos de pixelização)
- Cuidado para NÃO remover informação clínica (ondas P pequenas, notches no QRS)

#### 4.6.9 Gerar o output final
- Formato: dicionário com 12 arrays (um por derivação) em µV
- Incluir: sampling rate, duração, flag de qualidade do grid, flag de confiança por lead
- Este output vai para o Pilar 2 (Diagnóstico)

---

## 5. INTEGRAÇÃO E DEPLOY DO DIGITALIZADOR

### 5.1 Empacotar o pipeline para rodar no Modal (serverless)

#### 5.1.1 Criar a Modal function
- Modal = serviço que roda código Python sob demanda (só quando alguém usa, sem servidor ligado 24h)
- Empacotar todos os módulos (Dotter, Gridder, Undistortion, Leader, extração) em uma única function
- Incluir os pesos dos modelos treinados no pacote

#### 5.1.2 Definir o contrato de entrada/saída
- Entrada: URL da foto no Cloudflare R2 (onde o celular do médico fez upload)
- Saída: JSON com os 12 sinais digitais + metadados (sampling rate, confiança, etc.)

#### 5.1.3 Otimizar performance
- Meta: < 10 segundos do recebimento da foto até o output
- Carregar modelos na memória do Modal (para não recarregar a cada chamada)
- Se necessário: reduzir resolução da imagem para acelerar inferência

### 5.2 Testes e validação do digitalizador

#### 5.2.1 Testar com dataset PM-ECG-ID (benchmark público)
- Rodar nosso pipeline nas 6.000 imagens do PM-ECG-ID (dataset do PMcardio, público)
- Comparar nosso sinal digitalizado com o sinal original
- Medir PCC, RMSE, SNR por lead (mesmas métricas do paper)
- Meta: PCC > 0.90, RMSE < 0.10 mV, fail rate < 10%

#### 5.2.2 Testar com fotos reais de ECGs brasileiros
- Rodar em 20-50 fotos reais que NÃO foram usadas no treino
- Inspeção visual: olhar cada digitalização e comparar com o papel
- Verificar especificamente: V3-V5 (overlap), derivações dos membros (Einthoven bate?)

#### 5.2.3 Testar em condições extremas
- Fotos muito escuras, muito claras, muito desfocadas
- Papel muito amassado ou dobrado
- ECGs com anotações manuscritas por cima
- Metade do ECG coberta por carimbo
- Definir: quando o sistema deve REJEITAR a foto e pedir uma nova

---

---

# PILAR 2: DIAGNÓSTICO
**Objetivo:** Pegar os 12 sinais digitais produzidos pelo Pilar 1 e gerar um laudo médico com medições, achados e hipóteses diagnósticas.

---

## 6. MEDIÇÕES AUTOMÁTICAS (código matemático, sem IA)

São cálculos feitos diretamente nos sinais digitais. Código determinístico (sempre dá o mesmo resultado para o mesmo input).

### 6.1 Frequência cardíaca

#### 6.1.1 Detectar picos R
- O pico R é o ponto mais alto do complexo QRS — o "espetão" do ECG
- Usar algoritmo de detecção de picos (biblioteca NeuroKit2 ou scipy)
- Resultado: lista de posições (em ms) de cada pico R

#### 6.1.2 Calcular intervalos RR
- Medir a distância entre cada par de picos R consecutivos
- Resultado: array de intervalos RR em milissegundos

#### 6.1.3 Calcular FC
- FC = 60.000 / RR_médio (em ms)
- Se ritmo irregular (variação > 20%): reportar FC média + "ritmo irregular"

### 6.2 Eixo elétrico

#### 6.2.1 Medir amplitude do QRS em DI e aVF
- Para cada derivação: amplitude = pico positivo - pico negativo do QRS
- Estas duas derivações definem o eixo no plano frontal

#### 6.2.2 Calcular o eixo
- Usar fórmula trigonométrica: eixo = arctan(aVF / DI) em graus
- Classificar: normal (-30° a +90°), desvio esquerdo (< -30°), desvio direito (> +90°)

### 6.3 Intervalo PR

#### 6.3.1 Detectar início da onda P
- A onda P é a "lombinha" pequena antes do QRS
- Encontrar o ponto onde o sinal começa a subir antes do QRS

#### 6.3.2 Detectar início do QRS
- Ponto onde o sinal sobe abruptamente (início do complexo rápido)

#### 6.3.3 Calcular PR
- PR = tempo entre início da P e início do QRS, em milissegundos
- Normal: 120-200ms

### 6.4 Duração do QRS

#### 6.4.1 Detectar início e fim do QRS
- Início: subida abrupta
- Fim: retorno à baseline após o espetão
- Usar derivação com QRS mais largo (geralmente V1 ou V5)

#### 6.4.2 Calcular duração
- QRS = tempo entre início e fim, em milissegundos
- Normal: < 120ms

### 6.5 Intervalo QT e QTc

#### 6.5.1 Detectar fim da onda T
- A onda T é a "lombada" depois do QRS
- Encontrar onde a onda T volta à baseline
- Difícil automatizar com precisão — fonte comum de erros

#### 6.5.2 Calcular QT
- QT = tempo entre início do QRS e fim da onda T

#### 6.5.3 Calcular QTc (corrigido pela frequência)
- Fórmula de Bazett: QTc = QT / √(RR em segundos)
- Normal: < 450ms (homem), < 470ms (mulher)

### 6.6 Análise do segmento ST

#### 6.6.1 Medir o segmento ST em cada derivação
- Ponto J: fim do QRS
- Medir a amplitude do ST 60-80ms após o ponto J
- Comparar com a baseline (nível isoelétrico entre T e P)

#### 6.6.2 Classificar por derivação
- Supra de ST: elevação > 1mm (0.1 mV) em derivações dos membros, > 2mm (0.2 mV) em precordiais (V1-V3)
- Infra de ST: depressão > 0.5mm (0.05 mV)
- Registrar em quais derivações tem supra/infra

---

## 7. REGRAS CLÍNICAS (if/then baseado nas medições)

Código que aplica critérios diagnósticos objetivos — os mesmos que um cardiologista usa na prática.

### 7.1 Diagnósticos puramente por regras (Grupo A)

#### 7.1.1 BAV 1° grau
- SE PR > 200ms E ritmo sinusal (onda P antes de cada QRS) → BAV 1° grau

#### 7.1.2 BRD completo
- SE QRS ≥ 120ms E padrão rSR' em V1/V2 E onda S alargada em DI/V6 → BRD completo

#### 7.1.3 BRE completo
- SE QRS ≥ 120ms E ausência de Q em DI/V5/V6 E R monofásica alargada em DI/aVL/V5/V6 → BRE completo

#### 7.1.4 QT longo
- SE QTc > 450ms (homem) ou > 470ms (mulher) → QT longo

#### 7.1.5 QT curto
- SE QTc < 340ms → QT curto

#### 7.1.6 WPW (Wolff-Parkinson-White)
- SE PR < 120ms E onda delta presente E QRS > 100ms → WPW

#### 7.1.7 Bradicardia sinusal grave
- SE FC < 50 bpm E ritmo sinusal → bradicardia grave

#### 7.1.8 Sgarbossa (infarto com BRE)
- Pontuar: supra concordante (5pts) + infra V1-V3 (3pts) + supra discordante >5mm (2pts)
- SE score ≥ 3 → sugestivo de SCA com BRE

### 7.2 Diagnósticos complementados por regras (Grupo A/C)

#### 7.2.1 Hipercalemia
- Regra: T apiculada E simétrica + QT curto + possível alargamento QRS
- CNN complementa a detecção morfológica

#### 7.2.2 S1Q3T3 (embolia pulmonar)
- Regra: S proeminente em DI + Q em DIII + T invertida em DIII
- Sempre "sugestivo, correlacionar com clínica"

#### 7.2.3 Pericardite aguda
- Regra: supra ST difuso côncavo + infra PR em DII/DIII/aVF + ausência de imagem em espelho

---

## 8. CNN DE CLASSIFICAÇÃO (IA para padrões visuais)

Para diagnósticos onde o "formato" da onda importa mais que medições numéricas.

### 8.1 Preparação dos dados de treino

#### 8.1.1 Baixar datasets públicos de ECG digital
- PTB-XL: 21.837 ECGs com diagnósticos anotados por cardiologistas
- CODE-15%: dataset brasileiro da UFMG/Telehealth (~300.000 ECGs, parcialmente rotulado)
- Cada ECG já é digital (não precisa digitalizar) — são sinais prontos

#### 8.1.2 Organizar por diagnóstico
- Separar os ECGs por categoria: FA, flutter, TV, STEMI, Brugada, etc.
- Verificar quantos exemplos existem de cada (alguns serão raros)
- Diagnósticos com poucos exemplos (< 100): marcar como "em validação" (não ativamos no MVP)

#### 8.1.3 Dividir em treino/validação/teste
- 70% treino / 15% validação / 15% teste
- Garantir que o mesmo paciente não apareça em mais de um grupo

### 8.2 Construção do modelo CNN

#### 8.2.1 Definir a arquitetura
- Tipo: ResNet-1D (rede residual aplicada a sinais unidimensionais — o ECG é uma série temporal)
- Input: 12 canais (uma derivação por canal) × 5000 amostras (10 segundos a 500Hz)
- Output: probabilidades para cada diagnóstico (ex: FA 95%, flutter 3%, normal 2%)

#### 8.2.2 Definir a loss function
- Como um ECG pode ter múltiplos diagnósticos ao mesmo tempo (ex: FA + BRE):
  - Usar BCEWithLogitsLoss (cada diagnóstico é uma decisão independente: sim ou não)
  - NÃO usar CrossEntropyLoss (que serve para "uma resposta só")

#### 8.2.3 Configurar augmentations de sinal
- Adicionar ruído aleatório (simula interferência elétrica)
- Variar a baseline (simula artefato de respiração)
- Variar a amplitude (simula diferenças de ganho entre máquinas)
- Shift temporal (deslocar o sinal para frente/trás aleatoriamente)

### 8.3 Treino da CNN

#### 8.3.1 Treinar no Google Colab ou similar
- Rodar por ~100-200 epochs
- Tempo estimado: 4-12 horas dependendo do tamanho do dataset e GPU

#### 8.3.2 Avaliar por diagnóstico
- Para CADA diagnóstico, calcular separadamente:
  - Sensibilidade (de todos os casos reais, quantos a CNN detectou?)
  - Especificidade (de todos os casos normais, quantos a CNN acertou como normal?)
  - AUC (área sob a curva — nota geral de 0 a 1)
- Meta mínima para ativar um diagnóstico no MVP: AUC > 0.85 e sensibilidade > 0.80

#### 8.3.3 Decidir quais diagnósticos ativar
- Diagnósticos que atingiram a meta: "Ativo" no MVP → aparecem no laudo
- Diagnósticos que não atingiram: "Em validação" → não aparecem no laudo (ficam desligados)
- O cardiologista (você) valida caso a caso

### 8.4 Heatmap de explicabilidade (Grad-CAM)

#### 8.4.1 Implementar Grad-CAM
- Técnica que "olha pra trás" na CNN e identifica quais trechos do ECG mais influenciaram o diagnóstico
- Usa biblioteca pronta: pytorch-grad-cam (~20-30 linhas de código)

#### 8.4.2 Gerar heatmap por derivação
- Para cada diagnóstico positivo: gerar um "mapa de calor" sobre cada derivação
- Onde o mapa é mais quente/colorido = "a IA prestou mais atenção aqui"
- Semelhante ao ECGxplain™ do PMcardio (as faixas azuis que você viu no app)

#### 8.4.3 Calcular score de confiança por lead
- Cada derivação recebe uma nota (%) de quanto contribuiu para o diagnóstico
- Exibir para o médico: "Relevant (87%)" ou "Low confidence (12%)"

---

## 9. MOTOR DE LAUDO (combina tudo)

### 9.1 Template do laudo

#### 9.1.1 Seção 1: Dados técnicos
- Qualidade da digitalização (boa/razoável/ruim)
- Derivações com confiança baixa (se houver)

#### 9.1.2 Seção 2: Medições
- Ritmo, FC, eixo, PR, QRS, QT, QTc
- Formato: "Ritmo: sinusal. FC: 78 bpm. Eixo: +60°. PR: 180ms. QRS: 88ms. QT: 380ms. QTc: 420ms (Bazett)."

#### 9.1.3 Seção 3: Achados
- O que foi detectado pelas regras e/ou CNN
- Formato: "Supradesnivelamento do segmento ST em V1-V4 (2-3mm)."

#### 9.1.4 Seção 4: Hipóteses diagnósticas
- Conclusões com grau de confiança
- Red flags (alertas vermelhos) no topo: STEMI, TV, BAV total, hipercalemia grave
- Formato: "⚠️ Sugestivo de síndrome coronariana aguda com supra de ST em parede anterior."

#### 9.1.5 Seção 5: Disclaimer
- "Ferramenta de apoio à decisão clínica — não substitui avaliação médica."
- Sempre presente, obrigatório

### 9.2 Lógica de red flags (alertas de emergência)

#### 9.2.1 Definir condições de red flag
- STEMI / STEMI equivalente → 🔴 "ALERTA: Possível oclusão coronariana"
- TV sustentada → 🔴 "ALERTA: Taquicardia ventricular"
- BAV total → 🔴 "ALERTA: Bloqueio atrioventricular total"
- Hipercalemia grave → 🔴 "ALERTA: Padrão de hipercalemia grave"
- QTc > 500ms → 🟡 "ATENÇÃO: QTc muito prolongado — risco de Torsades"

#### 9.2.2 Exibir red flags com destaque visual
- Vermelho, ícone de alerta, no topo do laudo
- O médico vê PRIMEIRO o que pode matar

### 9.3 Geração do output final

#### 9.3.1 Montar o JSON de resposta
- Todas as medições + achados + diagnósticos + confiança + heatmaps
- Este JSON é enviado do Modal de volta para o Next.js (o app no celular)

#### 9.3.2 Renderizar o laudo na tela do médico
- Tela mobile-first (celular em primeiro lugar)
- Seções expansíveis (toca para ver mais detalhe)
- Botão de exportar PDF
- Botão de compartilhar via WhatsApp

---

## 10. VALIDAÇÃO CLÍNICA FINAL

### 10.1 Validação interna

#### 10.1.1 Testar o pipeline completo (foto → laudo) em 100+ ECGs
- Incluir normais e patológicos variados
- Verificar cada laudo manualmente (você como cardiologista)
- Anotar: correto, parcialmente correto, incorreto, perigoso

#### 10.1.2 Calcular métricas por diagnóstico
- Sensibilidade e especificidade de cada diagnóstico ativado
- Atenção especial a falsos negativos em red flags (STEMI não detectado = erro grave)

#### 10.1.3 Ajustar thresholds
- Se muitos falsos positivos: aumentar o limiar de confiança
- Se muitos falsos negativos em red flags: diminuir o limiar (preferir "alarme falso" a "perder infarto")

### 10.2 Validação com médicos beta-testers

#### 10.2.1 Selecionar 5-10 médicos para teste beta
- Emergencistas, intensivistas e generalistas de UBS
- Dar acesso gratuito por 30 dias

#### 10.2.2 Coletar feedback
- O laudo faz sentido clinicamente?
- As medições batem com o que o médico lê manualmente?
- Os diagnósticos estão corretos?
- A interface é fácil de usar no plantão (sob pressão, com pressa)?

#### 10.2.3 Iterar baseado no feedback
- Corrigir erros encontrados
- Ajustar linguagem do laudo
- Adicionar/remover informações conforme necessidade real

---

# RESUMO DAS DEPENDÊNCIAS

```
O que vem primeiro → O que depende dele

1. Dataset sintético (1.1) → Treino do Dotter e Leader (3.1, 3.2)
2. Dataset real (1.2) → Fine-tuning dos modelos (3.1.2, 3.2.2)
3. Modelos treinados (3) → Pipeline de digitalização (4)
4. Pipeline de digitalização (4) → Saída: 12 sinais digitais
5. Datasets PTB-XL/CODE (8.1) → Treino da CNN (8.3)
6. Sinais digitais (4) + Medições (6) + Regras (7) + CNN (8) → Motor de laudo (9)
7. Motor de laudo (9) → Validação (10)
```

# ESTIMATIVA DE TEMPO

| Fase | Tempo estimado |
|------|---------------|
| Dataset sintético (1.1) | 1-2 semanas |
| Coleta de fotos reais (1.2.1) | Em paralelo, 2-3 semanas |
| Anotação das fotos reais (1.2.2-1.2.3) | 2-3 semanas (pode terceirizar) |
| Código dos modelos Dotter/Leader (2) | 1 semana |
| Treino dos modelos (3) | 1-2 semanas (inclui iterações) |
| Pipeline de digitalização (4) | 2-3 semanas |
| Integração e deploy (5) | 1 semana |
| Medições automáticas (6) | 1-2 semanas |
| Regras clínicas (7) | 1 semana |
| CNN de classificação (8) | 2-3 semanas |
| Motor de laudo (9) | 1 semana |
| Validação (10) | 2-3 semanas |
| **TOTAL ESTIMADO** | **~3-4 meses** |

Nota: várias etapas rodam em paralelo (ex: coleta de fotos + código dos modelos + treino da CNN).
