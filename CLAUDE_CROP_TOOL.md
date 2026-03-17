# CLAUDE_CROP_TOOL.md — Ferramenta de Seleção de Bordas do ECG

> **Especificação completa para implementar a ferramenta de seleção de bordas (crop + perspectiva) do ProECG.**
> Este componente é inserido ENTRE a captura da foto e o envio para processamento.
> Leia TUDO antes de implementar. Siga a ordem das fases.

---

## ÍNDICE

1. [Contexto e Por Quê](#1-contexto-e-por-quê)
2. [Fluxo do Usuário](#2-fluxo-do-usuário)
3. [Arquitetura — Frontend vs Backend](#3-arquitetura)
4. [Componente Frontend — ECG Crop Tool](#4-componente-frontend)
5. [Auto-detecção de Cantos](#5-auto-detecção-de-cantos)
6. [Backend — Perspective Warp](#6-backend-perspective-warp)
7. [Integração com Pipeline Existente](#7-integração-com-pipeline-existente)
8. [Especificação Visual / UI](#8-especificação-visual)
9. [Fases de Implementação](#9-fases-de-implementação)
10. [Regras Invioláveis](#10-regras-invioláveis)

---

## 1. CONTEXTO E POR QUÊ

### O problema atual

O digitalizador ProECG recebe uma foto de celular e precisa:
1. Encontrar onde está o papel ECG na foto
2. Corrigir a perspectiva (papel fotografado em ângulo)
3. Recortar só o papel, ignorando mesa/objetos/mãos/carimbos

Hoje isso é feito 100% automaticamente pelo pipeline UNet, mas **falha frequentemente**:
- Bounding boxes escapam do papel (pegam carimbo, mesa, objetos)
- Grid detection confunde bordas do papel com grid do ECG
- Perspectiva não corrigida quando grid é fraco

### A solução

Adicionar uma **tela intermediária** entre a captura da foto e o processamento, onde o médico vê a foto e pode **confirmar ou ajustar os 4 cantos do papel ECG**.

Isso resolve de uma vez:
- ✅ Crop preciso (só o papel, sem lixo)
- ✅ Correção de perspectiva exata (4 pontos → warp perfeito)
- ✅ Calibração mais precisa (imagem contém SÓ o ECG)
- ✅ Leads não escapam do papel

### Referência de UX

O gesto é idêntico ao de apps de scanner (CamScanner, Adobe Scan, Apple Notes scanner):
1. Foto tirada
2. 4 pontos aparecem nas quinas detectadas
3. Usuário ajusta se necessário
4. Confirma → imagem é corrigida e processada

Médicos já conhecem esse padrão. Gesto de 3 segundos, sem curva de aprendizado.

---

## 2. FLUXO DO USUÁRIO

### Fluxo completo (atualizado)

```
1. Médico toca "Novo ECG" no dashboard
2. Câmera do celular abre
3. Médico tira foto do ECG em papel
4. ★ TELA DE CROP aparece:
   │  - Foto exibida em tela cheia
   │  - 4 pontos nos cantos do papel (auto-detectados)
   │  - Overlay escurecido fora da área selecionada
   │  - Médico pode arrastar os pontos para ajustar
   │  - Botão "Confirmar" e "Tirar nova foto"
   │
5. Médico toca "Confirmar"
6. Frontend envia: foto original + coordenadas dos 4 pontos
7. Backend: warp perspectiva → crop → pipeline UNet normal
8. Resultado (laudo) aparece na tela
```

### Casos de uso

| Caso | O que acontece |
|------|---------------|
| Foto bem enquadrada, cantos detectados ok | Médico só toca "Confirmar" (1 segundo) |
| Cantos detectados errado (ex: pegou a mesa) | Médico arrasta 1-2 pontos, toca "Confirmar" (3-5 segundos) |
| Carimbo/objeto cobrindo parte do ECG | Médico posiciona cantos na área limpa |
| Auto-detecção falhou completamente | Médico posiciona os 4 cantos manualmente |
| Foto ficou ruim (tremida, escura) | Médico toca "Tirar nova foto" |

---

## 3. ARQUITETURA

### Divisão frontend / backend

```
FRONTEND (Next.js / React)                    BACKEND (Python / Modal)
┌──────────────────────────┐                  ┌──────────────────────────┐
│                          │                  │                          │
│  Câmera → Foto capturada │                  │  Recebe: foto + 4 pontos │
│          │               │                  │          │               │
│          ▼               │     HTTP POST    │          ▼               │
│  Auto-detecção cantos    │ ───────────────▶ │  cv2.getPerspective      │
│  (JavaScript, no browser)│   foto_url +     │  Transform + warpPersp.  │
│          │               │   corners[4]     │          │               │
│          ▼               │                  │          ▼               │
│  Tela de crop (canvas)   │                  │  Imagem corrigida+cropada│
│  Médico ajusta cantos    │                  │          │               │
│          │               │                  │          ▼               │
│          ▼               │                  │  Pipeline UNet normal    │
│  "Confirmar" → envia     │                  │  (M0→M6, sem M2)        │
│                          │                  │                          │
└──────────────────────────┘                  └──────────────────────────┘
```

### O que roda onde

| Tarefa | Onde | Por quê |
|--------|------|---------|
| Exibir foto | Frontend | UX instantânea |
| Auto-detectar cantos | Frontend (JS) | Feedback imediato, sem round-trip ao servidor |
| Arrastar pontos | Frontend (canvas) | Interação touch nativa |
| Preview do crop (área selecionada) | Frontend | Visual feedback em tempo real |
| Perspective warp | Backend (Python/OpenCV) | Preciso, usa cv2.warpPerspective |
| Pipeline de digitalização | Backend | UNet + PyTorch |

### Dados transmitidos

```typescript
// Frontend → Backend
interface CropRequest {
  image_url: string;            // URL da foto no R2
  corners: {
    top_left: [number, number];     // [x, y] em pixels da imagem ORIGINAL
    top_right: [number, number];
    bottom_right: [number, number];
    bottom_left: [number, number];
  };
}
```

**IMPORTANTE:** As coordenadas dos cantos devem ser em pixels da **imagem original** (não da imagem exibida na tela). O frontend deve converter coordenadas de tela → coordenadas de imagem original considerando o scaling.

---

## 4. COMPONENTE FRONTEND — ECG CROP TOOL

### Componente React: `<EcgCropTool />`

```
Localização: src/components/ecg/EcgCropTool.tsx
```

### Props

```typescript
interface EcgCropToolProps {
  imageUrl: string;                    // URL ou data URL da foto capturada
  imageWidth: number;                  // Largura original da imagem em pixels
  imageHeight: number;                 // Altura original da imagem em pixels
  onConfirm: (corners: Corners) => void;   // Chamado quando médico confirma
  onRetake: () => void;                // Chamado quando médico quer nova foto
}

interface Corners {
  top_left: [number, number];
  top_right: [number, number];
  bottom_right: [number, number];
  bottom_left: [number, number];
}
```

### Comportamento

1. **Ao montar:**
   - Exibir foto em tela cheia (fit no viewport mobile, mantendo aspect ratio)
   - Rodar auto-detecção de cantos (seção 5)
   - Posicionar 4 pontos arrastáveis nos cantos detectados
   - Se auto-detecção falhar: posicionar nos cantos da imagem com margem de 10%

2. **Interação:**
   - Cada ponto é um **círculo arrastável** (~44px de diâmetro — tamanho mínimo touch)
   - Arrastar com dedo move o ponto
   - Linhas conectam os 4 pontos formando um quadrilátero
   - Área FORA do quadrilátero tem overlay escurecido (opacity 0.5)
   - Área DENTRO fica clara (preview do que será processado)

3. **Ao confirmar:**
   - Converter coordenadas de tela → coordenadas da imagem original
   - Chamar `onConfirm(corners)`

4. **Ao tirar nova foto:**
   - Chamar `onRetake()`

### Implementação técnica

Usar **HTML Canvas** para o overlay e os pontos. Alternativa: SVG overlay sobre `<img>`.

```
Canvas approach (recomendada para mobile):
1. <img> com a foto como background (object-fit: contain)
2. <canvas> overlay do mesmo tamanho, posição absolute
3. No canvas: desenhar linhas entre pontos, overlay escurecido, círculos dos pontos
4. Touch events no canvas: detectar qual ponto está sendo arrastado, mover
```

**Performance mobile:** O canvas deve redesenhar a cada touchmove. Usar `requestAnimationFrame` para throttle. Não redesenhar a foto — ela é background estático.

### Conversão de coordenadas

```typescript
// A foto é exibida com object-fit: contain dentro do viewport
// Precisa converter coordenadas de touch → coordenadas da imagem original

function screenToImageCoords(
  screenX: number,        // Posição do touch na tela
  screenY: number,
  imgElement: HTMLImageElement,
  originalWidth: number,  // Dimensões da imagem original
  originalHeight: number,
): [number, number] {
  // Calcular offset e scale do object-fit: contain
  const rect = imgElement.getBoundingClientRect();
  const displayedAspect = rect.width / rect.height;
  const imageAspect = originalWidth / originalHeight;

  let scale: number, offsetX: number, offsetY: number;

  if (imageAspect > displayedAspect) {
    // Imagem mais larga — barras em cima/baixo
    scale = rect.width / originalWidth;
    offsetX = 0;
    offsetY = (rect.height - originalHeight * scale) / 2;
  } else {
    // Imagem mais alta — barras nas laterais
    scale = rect.height / originalHeight;
    offsetX = (rect.width - originalWidth * scale) / 2;
    offsetY = 0;
  }

  const imageX = (screenX - rect.left - offsetX) / scale;
  const imageY = (screenY - rect.top - offsetY) / scale;

  return [
    Math.max(0, Math.min(originalWidth, imageX)),
    Math.max(0, Math.min(originalHeight, imageY)),
  ];
}
```

---

## 5. AUTO-DETECÇÃO DE CANTOS

### No frontend (JavaScript, sem dependência externa)

A auto-detecção roda no browser para feedback instantâneo. Não precisa ser perfeita — é só um ponto de partida que o médico pode ajustar.

### Algoritmo (simples e rápido para mobile)

```
1. Redimensionar imagem para ~500px de largura (performance)
2. Converter para grayscale
3. Aplicar Gaussian blur (kernel 5×5)
4. Canny edge detection (threshold 50/150)
5. Encontrar contornos (findContours)
6. Filtrar: o maior contorno que tem ~4 lados (approxPolyDP)
7. Se encontrou quadrilátero: usar como cantos
8. Se não: usar cantos da imagem com margem de 10%
```

### Implementação com Canvas API (sem OpenCV.js)

```typescript
async function autoDetectCorners(
  imageUrl: string,
  width: number,
  height: number,
): Promise<Corners> {
  // Criar canvas offscreen
  const canvas = document.createElement('canvas');
  const scale = 500 / Math.max(width, height);
  canvas.width = Math.round(width * scale);
  canvas.height = Math.round(height * scale);
  const ctx = canvas.getContext('2d')!;

  // Desenhar imagem reduzida
  const img = await loadImage(imageUrl);
  ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

  // Pegar pixels
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const gray = toGrayscale(imageData);

  // Edge detection simplificado (Sobel)
  const edges = sobelEdgeDetect(gray, canvas.width, canvas.height);

  // Encontrar retângulo dominante
  const corners = findLargestRectangle(edges, canvas.width, canvas.height);

  if (corners) {
    // Converter de volta para coordenadas originais
    return {
      top_left: [corners.tl[0] / scale, corners.tl[1] / scale],
      top_right: [corners.tr[0] / scale, corners.tr[1] / scale],
      bottom_right: [corners.br[0] / scale, corners.br[1] / scale],
      bottom_left: [corners.bl[0] / scale, corners.bl[1] / scale],
    };
  }

  // Fallback: cantos da imagem com 10% de margem
  const mx = width * 0.1;
  const my = height * 0.1;
  return {
    top_left: [mx, my],
    top_right: [width - mx, my],
    bottom_right: [width - mx, height - my],
    bottom_left: [mx, height - my],
  };
}
```

### Alternativa mais robusta: usar OpenCV.js

Se a detecção simples com Sobel não for boa o suficiente, considerar importar OpenCV.js (~8MB). Nesse caso:

```typescript
// Com OpenCV.js
function autoDetectWithOpenCV(imageElement: HTMLImageElement): Corners {
  const src = cv.imread(imageElement);
  const gray = new cv.Mat();
  const blurred = new cv.Mat();
  const edges = new cv.Mat();

  cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
  cv.GaussianBlur(gray, blurred, new cv.Size(5, 5), 0);
  cv.Canny(blurred, edges, 50, 150);

  // Dilatar para conectar bordas quebradas
  const kernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(3, 3));
  cv.dilate(edges, edges, kernel);

  // Encontrar contornos
  const contours = new cv.MatVector();
  const hierarchy = new cv.Mat();
  cv.findContours(edges, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

  // Encontrar o maior contorno com ~4 lados
  let bestContour = null;
  let bestArea = 0;

  for (let i = 0; i < contours.size(); i++) {
    const contour = contours.get(i);
    const area = cv.contourArea(contour);
    if (area < src.rows * src.cols * 0.1) continue; // Ignorar contornos pequenos (<10% da imagem)

    const peri = cv.arcLength(contour, true);
    const approx = new cv.Mat();
    cv.approxPolyDP(contour, approx, 0.02 * peri, true);

    if (approx.rows === 4 && area > bestArea) {
      bestContour = approx;
      bestArea = area;
    }
  }

  // Extrair cantos e ordenar (top-left, top-right, bottom-right, bottom-left)
  // ... (ordenar por posição)

  // Limpar memória
  src.delete(); gray.delete(); blurred.delete(); edges.delete();
  contours.delete(); hierarchy.delete();

  return corners;
}
```

### Decisão de implementação

- **Fase 1:** Usar detecção simples (Sobel puro em JS, sem OpenCV.js) — mais leve, mais rápido
- **Se não for bom o suficiente:** migrar para OpenCV.js
- **O importante é:** se a auto-detecção falhar, o médico ajusta manualmente em 3 segundos. Não precisa ser perfeita.

---

## 6. BACKEND — PERSPECTIVE WARP

### Novo endpoint ou parâmetro

O backend precisa receber os 4 cantos junto com a URL da imagem. Duas opções:

**Opção A (recomendada):** Adicionar `corners` ao request existente do `analyze()`:

```python
# modal_functions/pipeline/orchestrator.py — modificar analyze()

def analyze(image_url: str, corners: dict | None = None, use_placeholder: bool = False) -> dict:
    """
    Args:
        image_url: URL da imagem no R2
        corners: dict com 4 cantos {"top_left": [x,y], "top_right": [x,y], ...}
                 Se None, pipeline tenta detectar automaticamente (comportamento atual)
    """
    image = _download_image(image_url)

    if corners is not None:
        image = _apply_perspective_crop(image, corners)

    signal_12lead = _digitize_ecg(image, use_placeholder=use_placeholder)
    # ... resto do pipeline
```

**Opção B:** Endpoint separado que faz o warp e retorna URL da imagem corrigida.

### Função de perspective warp

```python
# Pode ficar em pipeline/digitize/perspective.py ou em um novo arquivo

import cv2
import numpy as np
from PIL import Image

def apply_perspective_crop(image: Image.Image, corners: dict) -> Image.Image:
    """Aplica correção de perspectiva e crop baseado nos 4 cantos fornecidos pelo usuário.

    Args:
        image: PIL Image original
        corners: dict com keys "top_left", "top_right", "bottom_right", "bottom_left"
                 Cada valor é [x, y] em pixels da imagem original

    Returns:
        PIL Image corrigida e cropada (retangular, perspectiva corrigida)
    """
    img_np = np.array(image)
    h, w = img_np.shape[:2]

    # Pontos de origem (cantos marcados pelo usuário)
    src_pts = np.float32([
        corners["top_left"],
        corners["top_right"],
        corners["bottom_right"],
        corners["bottom_left"],
    ])

    # Calcular dimensões do retângulo de destino
    # Largura = média das distâncias top e bottom
    width_top = np.linalg.norm(src_pts[1] - src_pts[0])
    width_bottom = np.linalg.norm(src_pts[2] - src_pts[3])
    dst_width = int(max(width_top, width_bottom))

    # Altura = média das distâncias left e right
    height_left = np.linalg.norm(src_pts[3] - src_pts[0])
    height_right = np.linalg.norm(src_pts[2] - src_pts[1])
    dst_height = int(max(height_left, height_right))

    # Pontos de destino (retângulo perfeito)
    dst_pts = np.float32([
        [0, 0],
        [dst_width - 1, 0],
        [dst_width - 1, dst_height - 1],
        [0, dst_height - 1],
    ])

    # Calcular e aplicar transformação de perspectiva
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(img_np, M, (dst_width, dst_height),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REPLICATE)

    return Image.fromarray(warped)
```

### O que muda no pipeline de digitalização

Quando o backend recebe `corners`:
1. **M0 (pré-processamento):** recebe imagem já cropada e corrigida → só faz resize e normalização
2. **M1 (segmentação UNet):** funciona normal, mas recebe imagem MUITO mais limpa
3. **M2 (perspectiva automática):** PULAR — já foi feita pelo warp dos 4 cantos
4. **M3 (calibração):** funciona normal, mas grid detection muito mais fácil (sem mesa/objetos)
5. **M4 (leads):** funciona normal, bounding boxes não escapam (imagem contém só o papel)
6. **M5-M6:** sem mudança

Quando NÃO recebe `corners` (compatibilidade):
- Pipeline funciona exatamente como hoje (fallback automático)
- Nenhuma mudança no comportamento atual

---

## 7. INTEGRAÇÃO COM PIPELINE EXISTENTE

### tRPC route (ou API route)

O endpoint que o frontend chama para analisar ECG precisa aceitar o campo `corners`:

```typescript
// Exemplo tRPC (adaptar ao router existente)

// Input schema
const analyzeInput = z.object({
  imageUrl: z.string().url(),
  corners: z.object({
    top_left: z.tuple([z.number(), z.number()]),
    top_right: z.tuple([z.number(), z.number()]),
    bottom_right: z.tuple([z.number(), z.number()]),
    bottom_left: z.tuple([z.number(), z.number()]),
  }).optional(),   // ← OPTIONAL — se não enviado, pipeline detecta automaticamente
});
```

### Modal function

```python
# modal_functions/analyze.py — ajustar para aceitar corners

@app.function(...)
def analyze_ecg(image_url: str, corners: dict | None = None) -> dict:
    from pipeline.orchestrator import analyze
    return analyze(image_url, corners=corners)
```

### Fluxo de dados completo

```
Celular:
  Câmera → foto → upload R2 → URL

Frontend (React):
  EcgCropTool(imageUrl) → corners

tRPC:
  ecg.analyze({ imageUrl, corners })

Modal function:
  analyze(image_url, corners)

Pipeline:
  if corners: apply_perspective_crop(image, corners)
  digitize_ecg(image) → signal → measure → rules → classify → report
```

---

## 8. ESPECIFICAÇÃO VISUAL / UI

### Layout da tela de crop (mobile-first)

```
┌──────────────────────────────┐
│         Status bar           │
├──────────────────────────────┤
│                              │
│  ┌────────────────────────┐  │
│  │ █████████████████████ │  │
│  │ █┌──────────────────┐█ │  │
│  │ █│                  │█ │  │
│  │ █│   FOTO DO ECG    │█ │  │
│  │ █│   (área clara)   │█ │  │
│  │ █│                  │█ │  │
│  │ █│  ●───────────●   │█ │  │
│  │ █│  │           │   │█ │  │
│  │ █│  │           │   │█ │  │
│  │ █│  ●───────────●   │█ │  │
│  │ █│                  │█ │  │
│  │ █└──────────────────┘█ │  │
│  │ █████████████████████ │  │
│  └────────────────────────┘  │
│                              │
│  "Ajuste os cantos do ECG"   │
│                              │
│  ┌──────────┐ ┌────────────┐ │
│  │ ↻ Nova   │ │ ✓ Confirmar│ │
│  │   foto   │ │            │ │
│  └──────────┘ └────────────┘ │
│                              │
└──────────────────────────────┘

█ = overlay escurecido (fora da seleção)
● = pontos arrastáveis (cantos)
```

### Especificação dos elementos

| Elemento | Especificação |
|----------|--------------|
| **Foto** | Centralizada, object-fit: contain, ocupa ~75% da altura da tela |
| **Overlay escurecido** | rgba(0, 0, 0, 0.5) fora do quadrilátero selecionado |
| **Pontos arrastáveis** | Círculos brancos, 44×44px (mínimo touch), borda 3px accent (#5B65DC), sombra |
| **Linhas entre pontos** | 2px, cor accent (#5B65DC), semi-transparente |
| **Área selecionada** | Sem overlay (transparente), borda 2px accent |
| **Texto instrução** | "Ajuste os cantos do papel ECG" — 16px, cor text-light |
| **Botão "Nova foto"** | Outline, secundário |
| **Botão "Confirmar"** | Preenchido, accent, primário |
| **Background** | Preto (#000) — contraste com a foto |

### Interação touch

| Gesto | Ação |
|-------|------|
| Tap num ponto | Seleciona (highlight) |
| Drag (arrastar) | Move o ponto selecionado |
| Tap fora dos pontos | Nada |
| Pinch zoom | NÃO implementar (complica coordenadas) |

### Feedback visual ao arrastar

- Ponto arrastado aumenta levemente (scale 1.2)
- Linhas e overlay se atualizam em tempo real
- Cor do ponto muda para accent-hover enquanto arrastando

### Edge cases

| Caso | Comportamento |
|------|--------------|
| Pontos cruzados (quadrilátero inválido) | Permitir, mas mostrar aviso sutil. Backend lida com isso. |
| Ponto arrastado fora da imagem | Clipar nas bordas da imagem |
| Imagem muito grande (>8MP) | Exibir reduzida, mas coordenadas finais em pixels originais |
| Orientação paisagem | Tela rotaciona com a foto, layout se adapta |

---

## 9. FASES DE IMPLEMENTAÇÃO

### Fase 1 — Componente de crop (frontend only)

```
1.1 Criar src/components/ecg/EcgCropTool.tsx
    - Canvas com foto + overlay + 4 pontos arrastáveis
    - Touch/drag funcional
    - Overlay escurecido fora da seleção
    - Botões "Nova foto" e "Confirmar"

1.2 Posição inicial dos pontos: cantos da imagem com 10% de margem (sem auto-detecção)

1.3 TESTAR no celular:
    - Foto carrega e exibe
    - Pontos são arrastáveis com o dedo
    - Overlay atualiza em tempo real
    - "Confirmar" retorna coordenadas corretas (console.log)
```

**Critério de sucesso:** Componente funciona no celular, pontos arrastáveis, coordenadas corretas.

### Fase 2 — Auto-detecção de cantos

```
2.1 Implementar autoDetectCorners() em JavaScript puro (Sobel + contornos)
2.2 Ao montar o componente, rodar auto-detecção
2.3 Posicionar pontos nos cantos detectados (ou fallback 10% margem)

2.4 TESTAR com fotos variadas:
    - Foto bem enquadrada: cantos devem estar perto das bordas do papel
    - Foto com objetos: cantos devem estar no papel, não nos objetos
    - Foto escura: fallback para margens
```

**Critério de sucesso:** Auto-detecção acerta em >60% dos casos. Nos outros, médico ajusta rápido.

### Fase 3 — Backend (perspective warp)

```
3.1 Criar função apply_perspective_crop() no backend
3.2 Modificar orchestrator.analyze() para aceitar corners (optional)
3.3 Se corners presente: aplicar warp ANTES de digitize_ecg()
3.4 Se corners ausente: comportamento atual (retrocompatível)

3.5 TESTAR:
    - Com corners: imagem resultante deve ser retangular, só o papel
    - Sem corners: pipeline funciona como antes
```

**Critério de sucesso:** Imagem warped contém só o papel ECG, retangular, perspectiva correta.

### Fase 4 — Integração end-to-end

```
4.1 Ajustar a tRPC route (ou API route) para aceitar corners
4.2 Conectar EcgCropTool → upload → analyze com corners
4.3 TESTAR fluxo completo no celular:
    - Tirar foto → crop tool → confirmar → laudo aparece
    
4.4 Comparar resultado:
    - SEM crop tool: resultado do pipeline atual
    - COM crop tool: resultado deve ser melhor (mais leads, menos artefatos)
```

**Critério de sucesso:** Fluxo completo funciona. Qualidade do laudo melhora vs sem crop.

### Fase 5 — Polimento

```
5.1 Animação suave dos pontos ao posicionar (auto-detecção)
5.2 Haptic feedback no celular ao começar a arrastar um ponto
5.3 Loading state enquanto auto-detecção roda
5.4 Se OpenCV.js for necessário: importar lazy (code split)
5.5 Testar em iOS Safari e Android Chrome
```

---

## 10. REGRAS INVIOLÁVEIS

### NÃO FAZER

- **NÃO bloquear** o fluxo se auto-detecção falhar — sempre mostrar pontos default
- **NÃO exigir** precisão perfeita nos pontos — o warp funciona bem com erro de ±20px
- **NÃO implementar** zoom/pinch na tela de crop — complica as coordenadas sem necessidade
- **NÃO remover** o pipeline automático — `corners` é OPTIONAL, sem ele tudo funciona como antes
- **NÃO importar** OpenCV.js como dependência obrigatória (performance mobile)
- **NÃO rodar** o warp no frontend — rodar no backend com cv2 (mais preciso, mais rápido)
- **NÃO adicionar** mais de 2 botões na tela (Nova foto + Confirmar) — simplicidade

### SEMPRE FAZER

- **SEMPRE** 44px mínimo nos pontos arrastáveis (guidelines de touch mobile)
- **SEMPRE** converter coordenadas para pixels da imagem original (não da tela)
- **SEMPRE** manter retrocompatibilidade (corners=None funciona normalmente)
- **SEMPRE** validar que os 4 pontos formam um quadrilátero válido antes do warp
- **SEMPRE** testar em celular (não só desktop) — este é um componente mobile-first

---

## APÊNDICE A — Referências de UI

- Apple Notes scanner: https://support.apple.com/guide/iphone/scan-documents-iphbc95d30b8/ios
- CamScanner: abrir o app e ver o fluxo de scan
- Adobe Scan: mesmo fluxo de 4 pontos

Todos seguem o mesmo padrão: foto → 4 pontos → ajustar → confirmar.

---

## APÊNDICE B — Bibliotecas JavaScript para considerar

| Lib | O que faz | Tamanho | Quando usar |
|-----|-----------|---------|-------------|
| **Nenhuma (Canvas API puro)** | Desenho + touch | 0 KB | Fase 1 — recomendado |
| **OpenCV.js** | Edge detection robusto | ~8 MB | Se auto-detecção simples falhar |
| **fabric.js** | Canvas interativo | ~300 KB | Se precisar de interação mais complexa |
| **react-image-crop** | Crop retangular | ~50 KB | NÃO usar — é crop retangular, não perspectiva |

Recomendação: começar com Canvas API puro. Só adicionar OpenCV.js se necessário.

---

*Documento gerado em 17/03/2026. Versão 1.0.*
*Qualquer dúvida: pergunte antes de assumir.*
