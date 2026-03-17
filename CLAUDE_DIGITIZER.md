# CLAUDE_DIGITIZER.md — Especificação Completa do Digitalizador ProECG

> **Este documento é a especificação completa e a única fonte de verdade para implementar o digitalizador proprietário do ProECG.**
>
> Leia TUDO antes de escrever qualquer código.
> Siga a ordem das fases EXATAMENTE.
> Teste cada módulo antes de avançar para o próximo.
> Em caso de dúvida, pergunte — não assuma.

---

## ÍNDICE

1. [Contexto do Projeto](#1-contexto-do-projeto)
2. [Estado Atual — O que já existe](#2-estado-atual)
3. [Contrato de Interface — O que NÃO pode mudar](#3-contrato-de-interface)
4. [Padrão Brasil — Parâmetros obrigatórios](#4-padrão-brasil)
5. [Paper de Referência — O que usar e o que NÃO usar](#5-paper-de-referência)
6. [Arquitetura do Digitalizador — 7 Módulos](#6-arquitetura-do-digitalizador)
7. [Arquitetura Exata do UNet — Verificada contra pesos reais](#7-arquitetura-exata-do-unet)
8. [Especificação Módulo a Módulo](#8-especificação-módulo-a-módulo)
9. [Constantes e Configuração (config)](#9-constantes-e-configuração)
10. [Estrutura de Arquivos — O que criar](#10-estrutura-de-arquivos)
11. [Como Obter e Carregar os Pesos do UNet](#11-como-obter-e-carregar-os-pesos)
12. [Fases de Implementação — Ordem obrigatória](#12-fases-de-implementação)
13. [Estratégia de Testes](#13-estratégia-de-testes)
14. [Riscos e Mitigações](#14-riscos-e-mitigações)
15. [Regras Invioláveis — O que NUNCA fazer](#15-regras-invioláveis)

---

## 1. CONTEXTO DO PROJETO

O ProECG é uma plataforma SaaS mobile-first para médicos de emergência, UTI e UBS. O fluxo principal:

1. Médico tira foto de um ECG em papel (12 derivações) com o celular
2. O sistema **digitaliza** a foto — detecta derivações, corrige perspectiva, extrai traçados como sinal digital
3. Um motor de interpretação (regras clínicas + CNN) analisa os traçados e gera laudo
4. Médico recebe o laudo em segundos

**O gargalo hoje é o passo 2 — a digitalização.** O digitalizador atual (`modal_functions/pipeline/digitize.py`) tenta usar o Open-ECG-Digitizer (repo externo), mas:
- Dá erro de compatibilidade numpy/PyTorch (`numpy.ndarray * Tensor`)
- Grid detection falha com ECGs brasileiros (thresholds calibrados para ECGs noruegueses a 50mm/s)
- Resultado: ritmo "indeterminado", zero medições, laudo inútil

**A decisão:** construir um digitalizador proprietário que usa os **pesos UNet pré-treinados** do Ahus-AIM (Open-ECG-Digitizer) mas com **pipeline 100% nosso**, calibrado para o padrão Brasil.

### Stack técnica relevante
- **Backend IA:** Python via Modal (serverless)
- **Libs:** PyTorch, OpenCV, NumPy, SciPy, Pillow
- **Deploy:** `modal deploy analyze.py`
- **Sem acesso a GPU em dev local** (Modal provisiona GPU em produção)

---

## 2. ESTADO ATUAL

### Estrutura existente em `modal_functions/`

```
modal_functions/
├── analyze.py                    # Modal function: endpoint HTTP
├── requirements.txt
├── models/
│   ├── classifier/
│   │   └── best_ecg_model.pth   # CNN classificação (15 MB) — JÁ EXISTE
│   └── digitizer/               # ← CRIAR esta pasta, colocar pesos UNet aqui
│       ├── unet_weights.pt              # ← BAIXAR (ver seção 11)
│       └── lead_name_unet_weights.pt    # ← BAIXAR (ver seção 11)
├── pipeline/
│   ├── __init__.py
│   ├── digitize.py              # ← SUBSTITUIR por pacote digitize/
│   ├── measure.py               # NÃO MEXER
│   ├── rules.py                 # NÃO MEXER (21 KB, regras clínicas completas)
│   ├── classify.py              # NÃO MEXER
│   ├── orchestrator.py          # NÃO MEXER
│   └── report.py                # NÃO MEXER
└── tests/
```

### O que mudar

1. **RENOMEAR** `pipeline/digitize.py` → `pipeline/digitize_old.py` (backup)
2. **CRIAR** `pipeline/digitize/` (pacote Python com `__init__.py`)
3. **CRIAR** `models/digitizer/` (pasta para pesos UNet)
4. O import `from .digitize import digitize_ecg` no orchestrator.py continua funcionando — Python trata pacote e módulo da mesma forma

---

## 3. CONTRATO DE INTERFACE

### O que o `orchestrator.py` espera (NÃO PODE MUDAR)

```python
# orchestrator.py, linha relevante:
from .digitize import digitize_ecg

# Chamada:
signal_12lead = digitize_ecg(image)  
# image: PIL.Image.Image (RGB)
# retorno: np.ndarray shape (12, 5000) em mV, dtype float64
```

### Assinatura exata que o novo digitize/ DEVE expor

```python
def digitize_ecg(image: PIL.Image.Image) -> np.ndarray:
    """Converte foto de ECG em papel → sinal digital de 12 derivações.

    Args:
        image: PIL Image (RGB) da foto do ECG.

    Returns:
        np.ndarray shape (12, 5000) com sinal em mV a 500 Hz.
        Ordem das derivações: I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6
        Duração: 10 segundos (5000 amostras a 500 Hz)
        NaN substituídos por 0.0

    Raises:
        RuntimeError: se não foi possível digitalizar.
    """
```

### Constantes que o orchestrator usa

```python
SAMPLING_RATE = 500       # Hz
LEAD_ORDER = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
LONG_SIGNAL_SEC = 10.0    # 10 segundos de output
# Output: shape (12, 5000)
```

---

## 4. PADRÃO BRASIL

**TODOS os defaults, thresholds e conversões DEVEM seguir o padrão brasileiro.** O paper do Ahus-AIM usa padrão norueguês (50mm/s) — serve APENAS como referência de arquitetura e pesos UNet, NÃO como padrão de parâmetros.

### Defaults obrigatórios

| Parâmetro | Default BR | Variação a detectar |
|-----------|-----------|---------------------|
| Velocidade do papel | **25 mm/s** | 50 mm/s (emergência/UTI) |
| Ganho (calibração vertical) | **10 mm/mV** | 5 mm/mV, 20 mm/mV |
| Layout | **3×4 + DII longo** | 3×4 sem ritmo, 3×3 |
| Grid | **1mm pequeno, 5mm grande** | Cores variadas |
| Frequência da rede elétrica | **60 Hz** | (Europa usa 50 Hz — NÃO é nosso caso) |

### Conversões a 25 mm/s e 10 mm/mV

| O que | Conversão |
|-------|-----------|
| 1mm horizontal | 0.04 s = 40 ms |
| 5mm horizontal (quadrado grande) | 0.20 s = 200 ms |
| 25mm horizontal | 1.0 s |
| 250mm (largura total) | 10.0 s |
| 1mm vertical | 0.1 mV = 100 µV |
| 5mm vertical (quadrado grande) | 0.5 mV = 500 µV |
| 10mm vertical | 1.0 mV = 1000 µV |

### Se a velocidade for 50 mm/s

| O que | Conversão |
|-------|-----------|
| 1mm horizontal | 0.02 s = 20 ms |
| Largura total | 500mm para 10s (ou 250mm para 5s) |

### Layouts brasileiros

```
Layout "3×4 + DII longo" (DEFAULT — ~80% dos ECGs brasileiros):
Linha 1: I    | aVR  | V1  | V4    ← 2.5s cada
Linha 2: II   | aVL  | V2  | V5    ← 2.5s cada
Linha 3: III  | aVF  | V3  | V6    ← 2.5s cada
Linha 4: DII longo (rhythm strip)   ← 10.0s

Layout "3×4 sem ritmo":
Mesmas 3 linhas, sem linha 4

Layout "3×3" (CardECG e similares):
Linha 1: I    | II   | III
Linha 2: aVR  | aVL  | aVF
Linha 3: V1   | V2   | V3
Linha 4: V4   | V5   | V6
```

### Fabricantes de ECG comuns no Brasil

GE, Philips, Nihon Kohden, Dixtal, TEB, Bionet, Biomet/EKG2000, Micromed/Global Telemedicina, CardECG, ECGMAC, Atrys. Cada um tem grid com cor, fonte e layout diferentes.

### Tipos de papel/grid

| Tipo | Cor do grid | Particularidades |
|------|------------|------------------|
| Papel laranja/salmon | Laranja | Clássico brasileiro, mais comum |
| Papel rosa (Biomet) | Rosa | Mais antigo, texto manuscrito |
| Papel branco (Atrys) | Cinza/preto | Impresso, pode vir rotacionado 90° |
| Grid azul (ECGMAC) | Azul claro | Formato paisagem |
| Grid verde (Micromed) | Verde claro | Formato mais "digital" |
| Sem grid (CardECG) | Quase sem grid | Bordas pretas separando leads |

**Conclusão:** Separar grid por cor HSV NÃO funciona para todos os tipos. O UNet (agnóstico à cor) é obrigatório.

### Riscos clínicos da calibração errada

| Erro | Consequência |
|------|-------------|
| Assumir 25mm/s quando é 50mm/s | Todos os intervalos medidos com DOBRO do valor real. PR=160ms vira 320ms → falso BAV 1º grau |
| Assumir 10mm/mV quando é 5mm/mV | Amplitudes subestimadas pela metade. HVE real não detectado |
| Troca de derivações (V1 no lugar de V3) | Supra de ST localizado na parede errada → erro de localização do infarto |
| Filtro 50Hz em vez de 60Hz | Ruído de rede elétrica não removido (Brasil usa 60Hz) |

---

## 5. PAPER DE REFERÊNCIA

### Dados do paper
- **Título:** "Digitizing Paper ECGs at Scale: An Open-Source Algorithm for Clinical Research"
- **Autores:** Elias Stenhede, Agnar Martin Bjørnstad, Arian Ranjbar (Akershus University Hospital, Noruega)
- **Publicado:** npj Digital Medicine, Janeiro 2026
- **Repo:** https://github.com/Ahus-AIM/Open-ECG-Digitizer
- **Licença:** CC BY 4.0 (pode usar livremente com citação)

### O que USAR do paper/repo

| Item | O que pegar | Onde no repo |
|------|------------|-------------|
| Pesos UNet segmentação | Arquivo .pt pré-treinado | `weights/unet_weights_07072025.pt` |
| Pesos UNet lead names | Arquivo .pt pré-treinado | `weights/lead_name_unet_weights_07072025.pt` |
| Arquitetura UNet | Classe ResidualUNet | `src/model/unet.py` |
| Ideia da dupla Hough Transform | Conceito para correção de perspectiva | Paper, seção "Perspective correction" |
| Ideia de autocorrelação para grid | Conceito para calibração | Paper, seção "Grid size extraction" |
| Snipping algorithm | Conceito para traces sobrepostos | Paper, seção "Segmentation-to-trace conversion" |
| Template matching de layout | Conceito para identificação de leads | Paper, Algorithm 1 |

### O que NÃO usar do repo

| Item | Por quê |
|------|---------|
| `inference_wrapper.py` | Complexo demais, bugado com ECGs brasileiros |
| `dewarper.py` | Experimental, falta `super().__init__()`, causa crash |
| Qualquer import de `ray.tune` | Não necessário para inferência |
| Configs YAML do repo | Vamos fazer as nossas com padrão Brasil |
| Thresholds de autocorrelação (h>0.3, p>0.05) | Calibrados para 50mm/s norueguês — relaxar para BR |

### Resultados do paper (expectativas de performance)

| Cenário | SNR (dB) | Relevância para nós |
|---------|---------|---------------------|
| Scanner 600dpi, 50mm/s | 19.65 | NÃO é nosso caso |
| Celular OnePlus, 50mm/s | 12.19 | Parcialmente relevante (celular sim, 50mm/s não) |
| Celular iPhone, 50mm/s | 10.47 | Parcialmente relevante |
| **Emory scan, 25mm/s** | **7.34** | **MAIS próximo do nosso caso** |
| Emory mobile, 25mm/s | 1.05 - 2.03 | Pior caso esperado para nós |

**Nota:** O SNR mais baixo no Emory (25mm/s) vs Ahus (50mm/s) é porque a 25mm/s mais informação é comprimida no papel, dificultando a digitalização. ECGs brasileiros são 25mm/s.

### Dados de treino do UNet

O UNet foi treinado com imagens sintéticas geradas a partir do **CODE-15%** (dataset brasileiro da UFMG) via ECG-Image-Kit. Isso significa que os pesos já têm exposição a ECGs brasileiros, o que é bom para nós.

---

## 6. ARQUITETURA DO DIGITALIZADOR

### Pipeline completo (7 módulos)

```
Foto (celular)
     │
     ▼
[M0] Pré-processamento
     │  - Corrigir orientação EXIF
     │  - Converter para RGB
     │  - Resize (lado maior → 2048px)
     │  - Normalizar para float32 [0, 1]
     │
     ▼
[M1] Segmentação UNet
     │  - Input: imagem RGB [0, 1]
     │  - Output: 4 masks (background, grid, trace, texto)
     │  - Pesos: unet_weights.pt (Ahus-AIM)
     │  - Sliding window com overlap para imagens grandes
     │  - CRÍTICO: converter output para numpy IMEDIATAMENTE
     │
     ▼
[M2] Correção de Perspectiva
     │  - Input: imagem + masks
     │  - Hough Transform na mask do grid → detectar ângulo
     │  - Rotacionar/warp imagem e masks
     │  - Fallback: se grid fraco, tentar por contornos do papel
     │  - Fallback final: prosseguir sem correção
     │
     ▼
[M3] Calibração (MÓDULO MAIS CRÍTICO)
     │  - Método 1 (primário): Autocorrelação na mask do grid → px/mm
     │  - Método 2 (fallback): OCR no cabeçalho → velocidade e ganho
     │  - Método 3 (fallback): Pulso de calibração (quadradinho 1mV)
     │  - Método 4 (fallback final): Assumir padrão BR (250mm largura)
     │  - Output: px_per_mm_h, px_per_mm_v, paper_speed, gain, confidence
     │
     ▼
[M4] Identificação de Leads
     │  - Detectar linhas (gaps horizontais na mask de trace)
     │  - Detectar colunas (gaps verticais dentro de cada linha)
     │  - Mapear para derivações via:
     │    A) UNet lead names (se detectar nomes na imagem)
     │    B) Template matching com layouts BR
     │    C) Fallback: assumir layout default 3×4+DII longo
     │
     ▼
[M5] Extração de Sinal
     │  - Para cada lead: mask trace na bbox → centroide por coluna → array Y
     │  - Snipping algorithm se traces sobrepostos
     │  - Inverter eixo Y (pixels ↓, mV ↑)
     │  - Converter px → mV usando calibração do M3:
     │    mV = px × (1 / px_per_mm_v) × (1 / gain)
     │  - Interpolar gaps NaN
     │  - Resamplear para 500 Hz
     │
     ▼
[M6] Pós-processamento
     │  - Filtro high-pass 0.5Hz (baseline wander)
     │  - Filtro notch 60Hz (powerline brasileira)
     │  - Filtro low-pass 150Hz (anti-aliasing)
     │  - Validação de sanidade (amplitudes plausíveis?)
     │  - Substituir NaN residuais por 0.0
     │
     ▼
[OUTPUT] np.ndarray shape (12, 5000), dtype float64, em mV
```

### Princípio fundamental

> Todo ECG é uma conversão de pixels → milímetros → tempo (s) e voltagem (mV).
> Se a calibração estiver correta, o resto funciona independente de papel, cor ou fabricante.
> Se a calibração falhar, usar fallbacks progressivos — nunca rejeitar a imagem.
> Um resultado impreciso com aviso é infinitamente melhor que nenhum resultado.

### Filosofia de NaN/fallback

- **NUNCA rejeitar a imagem inteira** — sempre tentar extrair o que puder
- Leads com >50% NaN: setar como array de zeros (lead indisponível)
- Leads com <50% NaN: interpolar gaps
- Se calibração falhou: usar fallback e marcar confidence como "low"
- Manter o fallback clássico (binarização + morfologia) do `digitize.py` antigo como ÚLTIMO recurso

---

## 7. ARQUITETURA EXATA DO UNet

### Verificação contra pesos reais

A arquitetura foi verificada por **inspeção direta dos pesos** (`lead_name_unet_weights_07072025.pt`). Os nomes de parâmetros confirmam a estrutura exata.

### Estrutura dos pesos (nomes de parâmetros)

```
_orig_mod.encoders.{0..N-1}.{0,1}.{0=Conv2d, 1=InstanceNorm2d, 2=LeakyReLU}
_orig_mod.encoder_skips.{0..N-1}.0 = Conv2d (kernel 1×1)
_orig_mod.encoder_downscaling.{0..N-1} = Conv2d (kernel 2×2, stride 2)
_orig_mod.decoders.{0..N-2}.{0,1}.{0=Conv2d, 1=InstanceNorm2d, 2=LeakyReLU}
_orig_mod.decoder_skips.{0..N-2}.0 = Conv2d (kernel 1×1)
_orig_mod.final_conv = Conv2d (kernel 3×3)
```

**Prefixo `_orig_mod.`**: aparece porque o modelo foi salvo com `torch.compile()`. Deve ser removido ao carregar os pesos.

### Dois modelos

| Modelo | In Channels | Out Classes | Encoder Widths | Parâmetros |
|--------|------------|-------------|----------------|------------|
| UNet principal (segmentação) | 3 (RGB) | 4 (bg/grid/trace/text) | (32, 64, 128, 256, 320, 320, 320, 320) | ~22.6M |
| Lead Name UNet | 1 (prob texto) | 13 (12 leads + bg) | (32, 64, 128, 256, 256) | ~6M |

### Arquitetura detalhada

```python
class ResidualUNet(nn.Module):
    """
    Residual U-Net. Funciona para ambos os modelos.
    
    Encoder:
      Para cada nível i (0..N-1):
        residual = encoder_skip[i](x)          # Conv1×1, sem bias
        x = encoder[i](x) + residual           # 2×(Conv3×3 → InstanceNorm → LeakyReLU) + residual
        features[i] = x                         # salvar para skip connection
        x = encoder_downscaling[i](x)           # StridedConv 2×2, sem bias
    
    Decoder:
      Para cada nível j (0..N-2):
        enc_level = N-1-j
        x = upsample_bilinear(x, size=features[enc_level])
        x = concat(x, features[enc_level])
        decoded = decoder[j](x)                 # 2×(Conv3×3 → InstanceNorm → LeakyReLU)
        x = decoded + decoder_skip[j](decoded)  # Conv1×1 residual
    
    Final:
      x = upsample_bilinear(x, size=input_original)
      x = final_conv(x)                         # Conv3×3, com bias → num_classes
    
    NOTAS CRÍTICAS:
    - TODOS os N encoder levels têm downscaling (inclusive o último)
    - O decoder tem N-1 blocos (não usa features[0] como skip)
    - A final_conv recebe encoder_widths[1] canais (output do último decoder)
    - Conv3×3 nos blocos: sem bias (seguido de InstanceNorm com affine=True)
    - final_conv: COM bias
    - Upsampling: bilinear, align_corners=False
    """
```

### Encoder block (repetido em encoder e decoder)

```python
nn.Sequential(
    nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.InstanceNorm2d(out_ch, affine=True),   # TEM weight e bias
        nn.LeakyReLU(inplace=True),
    ),
    nn.Sequential(
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.InstanceNorm2d(out_ch, affine=True),
        nn.LeakyReLU(inplace=True),
    ),
)
```

### Carregamento de pesos

```python
def load_weights(model, path, device="cpu"):
    state_dict = torch.load(path, map_location=device, weights_only=True)
    # OBRIGATÓRIO: remover prefixo _orig_mod. do torch.compile()
    cleaned = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(cleaned, strict=True)
    model.eval()
    return model
```

**Se `strict=True` der erro:** a arquitetura não bate com os pesos. Revisar nomes de camadas e shapes.

---

## 8. ESPECIFICAÇÃO MÓDULO A MÓDULO

### M0 — Pré-processamento (`preprocess.py`)

**Input:** PIL.Image.Image (qualquer modo/tamanho/orientação)
**Output:** np.ndarray float32 shape (H, W, 3), range [0, 1]

Etapas:
1. **Corrigir orientação EXIF** — celulares salvam rotação como metadata, não na imagem. Se não corrigir, ECG pode entrar de lado.
2. **Converter para RGB** — se grayscale ou RGBA, converter.
3. **Resize** — lado maior para 2048px mantendo aspect ratio. UNet foi treinado com patches 1024×1024. Fotos de celular vêm com 4000-6000px.
4. **Normalizar** — dividir por 255.0 para range [0, 1].
5. **Opcional: CLAHE** — equalização de histograma adaptativa se foto muito escura. Tentar sem primeiro; se segmentação falhar, tentar com CLAHE.

### M1 — Segmentação (`segmentation.py`)

**Input:** np.ndarray float32 (H, W, 3), range [0, 1]
**Output:** dict com 4 masks booleanas shape (H, W): "background", "grid", "trace", "text"

Etapas:
1. Converter para tensor PyTorch: `(1, 3, H, W)`
2. Se imagem ≤ 1024×1024: rodar UNet direto (single pass)
3. Se imagem > 1024×1024: sliding window com patches 1024×1024 e overlap 128px
   - Acumular logits nos patches
   - Média nas regiões de overlap
4. **CRÍTICO:** converter output para numpy IMEDIATAMENTE após `model(tensor)`:
   ```python
   with torch.no_grad():
       output = model(tensor)
       output_np = output.detach().cpu().numpy()  # SEMPRE aqui
   ```
5. Argmax → class_map (H, W)
6. Separar em 4 masks booleanas

**Se UNet não disponível (pesos não encontrados):** retornar None. O pipeline deve usar fallback clássico.

**Cache:** carregar modelo uma vez por container Modal, guardar em variável global.

### M2 — Correção de Perspectiva (`perspective.py`)

**Input:** imagem float32 (H, W, 3) + dict de masks
**Output:** imagem e masks corrigidas (dewarped)

Etapas:
1. Na mask do grid, aplicar Hough Lines (cv2.HoughLinesP)
2. Calcular ângulos de todas as linhas detectadas
3. Mediana dos ângulos = ângulo de rotação do papel
4. Se ângulo > 0.3°: rotacionar imagem e todas as masks
5. Se ângulo > 10°: pode ser foto de lado (90°) — detectar e corrigir

Fallbacks:
- Se grid muito fraco (poucas linhas): tentar Hough com thresholds mais baixos
- Se nenhuma linha detectada: prosseguir sem correção

### M3 — Calibração (`calibration.py`)

**O módulo mais crítico.** Erro aqui = erro em TODA medição.

**Input:** masks dewarped (especialmente grid), imagem dewarped
**Output:** CalibrationResult (px_per_mm_h, px_per_mm_v, paper_speed, gain, confidence)

**4 métodos em cascata (usar o primeiro que funcionar):**

#### Método 1: Autocorrelação no grid (primário)
```
1. Pegar mask do grid → converter para float
2. Soma de colunas → sinal 1D horizontal (density signal)
3. Autocorrelação (np.correlate ou scipy)
4. Encontrar picos periódicos:
   - Thresholds RELAXADOS para BR: height > 0.15, prominence > 0.02
   - (Paper usa h>0.3, p>0.05 — muito restritivo para ECGs BR)
5. Primeiro pico significativo = espaçamento do minor grid (1mm) em pixels
6. Validar: deve haver pico em ~5× (major grid)
7. px_per_mm = posição_do_pico
8. Repetir para eixo vertical (soma de linhas)
```

#### Método 2: OCR do cabeçalho
```
1. Na mask de texto, identificar região do topo da imagem
2. OCR (easyocr ou tesseract) para extrair texto
3. Regex para encontrar: "25 mm/s", "50 mm/s", "10 mm/mV", "5 mm/mV", "20 mm/mV"
4. Se encontrado, usar para definir/corrigir velocidade e ganho
```

#### Método 3: Pulso de calibração
```
1. Na mask de trace, borda esquerda de cada lead
2. Buscar padrão retangular (subida abrupta → plateau → descida)
3. Medir altura em pixels
4. Comparar com altura esperada (10mm a 10mm/mV = CALIB_PULSE_HEIGHT × px_per_mm)
5. Se desvio > 30%: ganho provavelmente diferente do padrão
```

#### Método 4: Fallback final
```
1. Assumir largura do ECG = 250mm (10s × 25mm/s)
2. px_per_mm_h = largura_da_imagem / 250
3. px_per_mm_v = px_per_mm_h (grid quadrado)
4. paper_speed = 25, gain = 10 (defaults BR)
5. confidence = "low"
```

### M4 — Identificação de Leads (`lead_detection.py`)

**Input:** masks dewarped, CalibrationResult
**Output:** lista de LeadRegion (name, bbox, row, col)

Etapas:
1. **Detectar linhas horizontais:**
   - Projetar mask de trace no eixo vertical (soma por linha → perfil vertical)
   - Encontrar gaps (vales) → separadores entre linhas do ECG
   - Dividir em N linhas (tipicamente 3 ou 4)

2. **Detectar colunas dentro de cada linha:**
   - Para cada linha, projetar mask de trace no eixo horizontal
   - Encontrar gaps verticais
   - Se gaps claros: dividir por gaps
   - Se sem gaps: dividir igualmente (4 colunas para 3×4)

3. **Mapear nomes das derivações (prioridade decrescente):**
   - **A) UNet lead names:** rodar segundo UNet na mask de texto, detectar nomes, matching por posição
   - **B) Template matching:** comparar grid (rows, cols) com LAYOUTS da config, score por distância
   - **C) Fallback:** assumir layout default 3×4+DII longo (I, aVR, V1, V4 / II, aVL, V2, V5 / III, aVF, V3, V6 / DII longo)

### M5 — Extração de Sinal (`signal_extraction.py`)

**Input:** mask de trace dewarped, lista de LeadRegion, CalibrationResult
**Output:** dict {lead_name: np.ndarray em mV} + metadata

Para cada lead:
```
1. Recortar mask de trace na bbox
2. Para cada coluna x (esquerda → direita):
   a. Pixels de trace nesta coluna
   b. Se encontrou: centroide Y (center of mass)
   c. Se não encontrou: NaN
3. Se traces sobrepostos (mais de 1 componente conectado na vertical):
   → Snipping algorithm: caminho de menor resistência horizontal
   → Dividir componentes
   → Re-atribuir via linear sum assignment (scipy.optimize.linear_sum_assignment)
4. Inverter eixo Y (pixels crescem pra baixo, mV cresce pra cima)
5. Centrar em zero (subtrair mediana)
6. Converter px → mV:
   mV = pixels × (1.0 / calibration.px_per_mm_v) × (1.0 / calibration.gain)
7. Interpolar gaps NaN (scipy.interpolate.interp1d, linear)
   Gaps > 50% do lead → manter como NaN (lead falhou)
8. Resamplear para 500 Hz:
   native_sr = px_per_mm_h × paper_speed (ex: 3 px/mm × 25 mm/s = 75 Hz)
   scipy.signal.resample para 500 Hz
```

### M6 — Pós-processamento (`postprocess.py`)

**Input:** dict {lead_name: np.ndarray em mV a 500 Hz}
**Output:** np.ndarray shape (12, 5000) em mV, dtype float64

Etapas:
1. **Filtro high-pass 0.5 Hz** (Butterworth 2ª ordem) — remove baseline wander
2. **Filtro notch 60 Hz** (IIR notch, Q=30) — remove ruído da rede elétrica BRASILEIRA
3. **Filtro low-pass 150 Hz** (Butterworth 4ª ordem) — anti-aliasing
4. **Validação:** se amplitude máxima > 5 mV → provavelmente artefato → clipar
5. **Montar array final (12, 5000):**
   - Para cada lead na ordem: I, II, III, aVR, aVL, aVF, V1-V6
   - Se lead disponível: usar sinal filtrado, pad/truncar para 5000 amostras
   - Se lead falhou: array de zeros
6. **Substituir NaN residuais por 0.0**
7. **Retornar como float64** (orchestrator.py espera isso)

---

## 9. CONSTANTES E CONFIGURAÇÃO

Criar um arquivo `constants.py` dentro do pacote `digitize/` com TODAS as constantes. Centralizar tudo num lugar — nunca hardcodar valores nos módulos.

```python
# Constantes essenciais (consolidar num único arquivo):

SAMPLING_RATE = 500
OUTPUT_LENGTH = 5000  # 10s × 500Hz
LEAD_ORDER = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

PAPER_SPEED_DEFAULT = 25   # mm/s
GAIN_DEFAULT = 10          # mm/mV
GRID_MINOR_MM = 1
GRID_MAJOR_MM = 5

MAX_INPUT_SIZE = 2048
UNET_PATCH_SIZE = 1024
UNET_OVERLAP = 128

# UNet principal
UNET_ENCODER_WIDTHS = (32, 64, 128, 256, 320, 320, 320, 320)
UNET_IN_CHANNELS = 3
UNET_NUM_CLASSES = 4

# Lead Name UNet
LEAD_UNET_ENCODER_WIDTHS = (32, 64, 128, 256, 256)
LEAD_UNET_IN_CHANNELS = 1
LEAD_UNET_NUM_CLASSES = 13

# Segmentation classes
SEG_BACKGROUND = 0
SEG_GRID = 1
SEG_TRACE = 2
SEG_TEXT = 3

# Calibração (thresholds relaxados para BR)
AUTOCORR_HEIGHT_THRESH = 0.15
AUTOCORR_PROMINENCE_THRESH = 0.02

# Filtros (Brasil = 60Hz)
POWERLINE_FREQ = 60
BASELINE_HIGHPASS = 0.5
LOWPASS_FREQ = 150

# Caminhos dos pesos (relativos à raiz de modal_functions/)
UNET_WEIGHTS_FILENAME = "unet_weights.pt"
LEAD_UNET_WEIGHTS_FILENAME = "lead_name_unet_weights.pt"
```

---

## 10. ESTRUTURA DE ARQUIVOS

### O que CRIAR

```
modal_functions/
├── models/
│   └── digitizer/                        # ← CRIAR
│       ├── unet_weights.pt               # ← BAIXAR (ver seção 11)
│       └── lead_name_unet_weights.pt     # ← BAIXAR (ver seção 11)
│
├── pipeline/
│   ├── digitize_old.py                   # ← RENOMEAR de digitize.py
│   │
│   ├── digitize/                         # ← CRIAR (pacote)
│   │   ├── __init__.py                   # Expõe digitize_ecg() — MESMA assinatura
│   │   ├── constants.py                  # Todas as constantes (padrão Brasil)
│   │   ├── unet.py                       # Arquitetura ResidualUNet
│   │   ├── preprocess.py                 # [M0] Pré-processamento
│   │   ├── segmentation.py              # [M1] UNet → 4 masks
│   │   ├── perspective.py               # [M2] Correção de perspectiva
│   │   ├── calibration.py               # [M3] Grid → px/mm + velocidade + ganho
│   │   ├── lead_detection.py            # [M4] Detectar e mapear derivações
│   │   ├── signal_extraction.py         # [M5] Mask → série temporal mV
│   │   ├── postprocess.py              # [M6] Filtros + montagem final
│   │   └── classic_fallback.py          # Fallback clássico (do digitize.py antigo)
│   │
│   ├── orchestrator.py                   # NÃO MEXER
│   ├── measure.py                        # NÃO MEXER
│   ├── rules.py                          # NÃO MEXER
│   ├── classify.py                       # NÃO MEXER
│   └── report.py                         # NÃO MEXER
│
└── scripts/
    └── verify_weights.py                 # ← CRIAR (testa carregamento dos pesos)
```

### `__init__.py` do pacote digitize/

```python
"""
ProECG — Digitalizador Proprietário
Converte foto de ECG em papel → sinal digital de 12 derivações.
"""

from .pipeline import digitize_ecg

__all__ = ["digitize_ecg"]
```

Onde `pipeline.py` (ou o nome que escolher) orquestra M0→M6 e expõe `digitize_ecg()` com a assinatura exata que o orchestrator.py espera. Também deve tentar o pipeline UNet completo e, se falhar, cair para o `classic_fallback`.

---

## 11. COMO OBTER E CARREGAR OS PESOS

### Download dos pesos

Os pesos estão no GitHub do Ahus-AIM. Dois arquivos:

**1. Peso principal (segmentação) — ~90MB:**
```
https://github.com/Ahus-AIM/Open-ECG-Digitizer/raw/main/weights/unet_weights_07072025.pt
```
Salvar como: `modal_functions/models/digitizer/unet_weights.pt`

**2. Peso lead names — ~23MB:**
```
https://github.com/Ahus-AIM/Open-ECG-Digitizer/raw/main/weights/lead_name_unet_weights_07072025.pt
```
Salvar como: `modal_functions/models/digitizer/lead_name_unet_weights.pt`

**Alternativa (Kaggle):**
```bash
pip install kaggle
kaggle models download eliasstenhede/open-ecg-digitizer
# Extrair e copiar os .pt para models/digitizer/
```

**Alternativa (clonar repo):**
```bash
git clone --depth 1 https://github.com/Ahus-AIM/Open-ECG-Digitizer.git /tmp/oecg
cp /tmp/oecg/weights/unet_weights_07072025.pt modal_functions/models/digitizer/unet_weights.pt
cp /tmp/oecg/weights/lead_name_unet_weights_07072025.pt modal_functions/models/digitizer/lead_name_unet_weights.pt
rm -rf /tmp/oecg
```

### Carregamento no código

```python
import torch
import os

def _resolve_weights_path(filename):
    """Encontra o arquivo de pesos a partir da raiz do modal_functions/."""
    # Subir do digitize/ → pipeline/ → modal_functions/
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(base, "models", "digitizer", filename)
```

**IMPORTANTE para Modal (serverless):**
- Os pesos devem ser incluídos no container Modal (via `modal.Image` ou `modal.Mount`)
- Cache do modelo: carregar UMA VEZ por container (variável global `_model_cache`)
- Device: usar `"cuda"` se disponível, senão `"cpu"`

### Script de verificação (`scripts/verify_weights.py`)

Criar um script que:
1. Instancia ResidualUNet com a arquitetura correta
2. Carrega os pesos com `load_weights()`
3. Roda forward pass com input dummy (tensor aleatório 1×3×256×256)
4. Verifica que output tem shape (1, 4, 256, 256)
5. Printa "✅ Sucesso" ou "❌ Erro" com detalhes

---

## 12. FASES DE IMPLEMENTAÇÃO

### Fase 1 — Fundação (config + UNet + verificação)

```
1.1 Criar pasta models/digitizer/
1.2 Baixar os 2 arquivos de pesos
1.3 Criar digitize/constants.py com todas as constantes
1.4 Criar digitize/unet.py com arquitetura EXATA (seção 7)
1.5 Criar scripts/verify_weights.py
1.6 TESTAR: rodar verify_weights.py — ambos modelos devem carregar com sucesso
1.7 Se der erro de strict=True: comparar nomes de parâmetros do modelo vs pesos
```

**Critério de sucesso:** `verify_weights.py` printa "✅" para ambos os modelos.

### Fase 2 — Segmentação (M0 + M1)

```
2.1 Criar digitize/preprocess.py (M0)
2.2 Criar digitize/segmentation.py (M1)
2.3 TESTAR: rodar com uma foto de ECG real
    - Visualizar as 4 masks (salvar como imagens para inspeção)
    - Mask de trace deve ter pixels onde há sinal visível
    - Mask de grid deve cobrir as linhas do quadriculado
2.4 Se segmentação falhar: tentar com CLAHE no pré-processamento
```

**Critério de sucesso:** masks visualmente corretas numa foto de ECG.

### Fase 3 — Perspectiva + Calibração (M2 + M3)

```
3.1 Criar digitize/perspective.py (M2)
3.2 Criar digitize/calibration.py (M3 — método 1: autocorrelação)
3.3 TESTAR autocorrelação: px_per_mm deve ser razoável (tipicamente 2-5 px/mm)
3.4 Implementar M3 método 2 (OCR) — pode ser simplificado no MVP
3.5 Implementar M3 método 4 (fallback final) — SEMPRE implementar
3.6 TESTAR: comparar px_per_mm com medição manual na foto
```

**Critério de sucesso:** px_per_mm com erro < 20% vs medição manual.

### Fase 4 — Leads + Extração (M4 + M5)

```
4.1 Criar digitize/lead_detection.py (M4 — sem UNet lead names, só template matching)
4.2 Criar digitize/signal_extraction.py (M5 — sem snipping)
4.3 TESTAR: extrair sinais de ECG com layout 3×4 conhecido
    - Visualizar sinais extraídos (plotar os 12 leads)
    - QRS deve ser visível em pelo menos 8 leads
4.4 Implementar snipping algorithm (traces sobrepostos) se necessário
4.5 Implementar UNet lead names se template matching não for suficiente
```

**Critério de sucesso:** 12 sinais extraídos com morfologia reconhecível.

### Fase 5 — Pós-processamento + Integração (M6 + __init__.py)

```
5.1 Criar digitize/postprocess.py (M6 — filtros + montagem final)
5.2 Criar digitize/classic_fallback.py (copiar lógica do digitize.py antigo)
5.3 Criar digitize/__init__.py que expõe digitize_ecg()
5.4 Renomear pipeline/digitize.py → pipeline/digitize_old.py
5.5 TESTAR: rodar orchestrator.analyze() end-to-end com uma foto real
    - Deve retornar JSON com measurements, findings, diagnoses
    - Comparar medições com laudo de cardiologista
5.6 TESTAR: se UNet falhar (remover pesos temporariamente), fallback clássico deve funcionar
```

**Critério de sucesso:** `orchestrator.analyze()` retorna laudo plausível.

### Fase 6 — Robustez (testes com múltiplas fotos)

```
6.1 Testar com fotos de diferentes fabricantes (se disponíveis)
6.2 Testar com foto torta (rotação >10°)
6.3 Testar com foto escura/mal iluminada
6.4 Testar com ECG a 50mm/s (se disponível)
6.5 Ajustar thresholds conforme resultados
```

---

## 13. ESTRATÉGIA DE TESTES

### Teste unitário por módulo

| Módulo | O que testar | Critério |
|--------|-------------|----------|
| M0 | Foto 4000px → array 2048px, float32, [0,1] | Shape e range corretos |
| M1 | Imagem → 4 masks | Masks não são todas vazias; trace tem pixels |
| M2 | Imagem torta → imagem reta | Grid alinhado (visual) |
| M3 | Mask grid → px_per_mm | Valor entre 1 e 10 (plausível) |
| M4 | Masks → 12 regiões | Regiões não se sobrepõem; cobrem a imagem |
| M5 | Mask + calib → sinais | QRS visível; amplitudes < 5mV |
| M6 | Sinais brutos → (12, 5000) | Shape correto; sem NaN; valores plausíveis |

### Teste end-to-end

```python
# Testar que a interface não mudou:
from pipeline.digitize import digitize_ecg
from PIL import Image

img = Image.open("test_ecg.jpg")
signal = digitize_ecg(img)

assert signal.shape == (12, 5000)
assert signal.dtype == np.float64
assert not np.any(np.isnan(signal))
assert np.max(np.abs(signal)) < 10.0  # mV, plausível
```

---

## 14. RISCOS E MITIGAÇÕES

| # | Risco | Gravidade | Mitigação |
|---|-------|-----------|-----------|
| 1 | Pesos não carregam (arquitetura errada) | BLOQUEANTE | Verificar com verify_weights.py PRIMEIRO. Se strict=True falhar, comparar nomes param por param |
| 2 | Bug numpy × Tensor | BLOQUEANTE | Converter para numpy IMEDIATAMENTE após model(). Nunca misturar operações |
| 3 | Grid invisível (papel térmico desbotado) | ALTA | 4 métodos de calibração em cascata. Fallback final sempre funciona |
| 4 | Velocidade 50mm/s não detectada | ALTA (erro clínico) | OCR do cabeçalho + regra: se grid spacing ≈ dobro do esperado → provavelmente 50mm/s |
| 5 | Traces sobrepostos (V3/V4) | ALTA | Snipping algorithm. Se muito complexo para MVP, aceitar perda de 1-2 leads |
| 6 | Foto rotacionada 90° (Atrys) | MÉDIA | Orientação EXIF + detecção de orientação do grid |
| 7 | Cold start do Modal | BAIXA | Cache de modelo em variável global |
| 8 | Fallback clássico também falha | BAIXA | RuntimeError com mensagem clara para o médico |

---

## 15. REGRAS INVIOLÁVEIS

### NÃO FAZER

- **NÃO modificar** orchestrator.py, measure.py, rules.py, classify.py, report.py
- **NÃO mudar** a assinatura de `digitize_ecg(image) → np.ndarray (12, 5000)`
- **NÃO usar** o repo Open-ECG-Digitizer como dependência runtime (só pegar pesos e arquitetura)
- **NÃO hardcodar** velocidade como 50mm/s (padrão BR é 25mm/s)
- **NÃO usar** filtro notch 50Hz (Brasil é 60Hz)
- **NÃO rejeitar** imagens — sempre tentar, usar fallbacks, retornar o que puder
- **NÃO misturar** operações NumPy com tensores PyTorch
- **NÃO instalar** ray, ray.tune, ou qualquer dependência do repo original além de PyTorch
- **NÃO criar** API routes fora do padrão existente
- **NÃO armazenar** dados de pacientes

### SEMPRE FAZER

- **SEMPRE** converter output do UNet para numpy imediatamente: `output.detach().cpu().numpy()`
- **SEMPRE** usar `torch.no_grad()` em inferência
- **SEMPRE** testar cada módulo antes de avançar para o próximo
- **SEMPRE** manter o fallback clássico como último recurso
- **SEMPRE** logar warnings/erros com `logging` (não print)
- **SEMPRE** usar `strict=True` ao carregar pesos (detecta erros de arquitetura)
- **SEMPRE** tratar NaN (interpolar ou substituir por 0.0 antes de retornar)

---

## APÊNDICE A — Código do digitize.py atual (para referência do fallback clássico)

O fallback clássico (binarização + morfologia + varredura vertical) deve ser preservado em `classic_fallback.py`. A lógica relevante está nas funções:
- `_digitize_classic_fallback()` — pipeline principal do fallback
- `_remove_grid()` — remoção de grid por morfologia
- `_extract_signal_from_roi()` — extração por varredura vertical (center of mass)
- `_resample_signal()` — resample para 500Hz
- `_normalize_to_mv()` — normalização heurística para mV

Copiar essas funções para `classic_fallback.py` e expor como `digitize_classic_fallback(image: PIL.Image.Image) -> np.ndarray`.

---

## APÊNDICE B — Contrato JSON da Modal function (para contexto)

O `orchestrator.analyze()` retorna:
```json
{
  "success": true,
  "measurements": {
    "heart_rate": 78,
    "axis": 60,
    "pr_interval": 180,
    "qrs_duration": 88,
    "qt_interval": 380,
    "qtc_bazett": 420,
    "rhythm": "sinusal"
  },
  "findings": [...],
  "diagnoses": [...],
  "report_text": "Ritmo: sinusal. FC: 78 bpm...",
  "processing_time_ms": 1850
}
```

---

## APÊNDICE C — Referências

- Paper: https://arxiv.org/abs/2510.19590
- Repo: https://github.com/Ahus-AIM/Open-ECG-Digitizer
- Pesos (Kaggle): https://www.kaggle.com/models/eliasstenhede/open-ecg-digitizer
- Dataset treino: https://huggingface.co/datasets/Ahus-AIM/Open-ECG-Digitizer-Development-Dataset
- ECG-Image-Kit: https://github.com/alphanumericslab/ecg-image-kit
- CODE-15%: https://zenodo.org/records/4916206

---

*Documento gerado em 17/03/2026. Versão 1.0.*
*Qualquer dúvida: pergunte antes de assumir.*
