"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Button } from "@proecg/ui/components/button";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface Corners {
  top_left: [number, number];
  top_right: [number, number];
  bottom_right: [number, number];
  bottom_left: [number, number];
}

interface EcgCropToolProps {
  imageUrl: string;
  imageWidth: number;
  imageHeight: number;
  onConfirm: (corners: Corners) => void;
  onRetake: () => void;
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const POINT_RADIUS = 22; // 44px diameter — minimum touch target
const POINT_BORDER = 3;
const ACCENT = "#5B65DC";
const ACCENT_ACTIVE = "#7B85FC";
const OVERLAY_COLOR = "rgba(0,0,0,0.50)";
const LINE_WIDTH = 2;
const HIT_RADIUS = 30; // Larger than visual for easier touch

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Convert screen touch coords to image-original coords. */
function screenToImage(
  clientX: number,
  clientY: number,
  imgRect: DOMRect,
  imgOrigW: number,
  imgOrigH: number,
): [number, number] {
  const displayedAspect = imgRect.width / imgRect.height;
  const imageAspect = imgOrigW / imgOrigH;

  let scale: number, offsetX: number, offsetY: number;
  if (imageAspect > displayedAspect) {
    scale = imgRect.width / imgOrigW;
    offsetX = 0;
    offsetY = (imgRect.height - imgOrigH * scale) / 2;
  } else {
    scale = imgRect.height / imgOrigH;
    offsetX = (imgRect.width - imgOrigW * scale) / 2;
    offsetY = 0;
  }

  const x = (clientX - imgRect.left - offsetX) / scale;
  const y = (clientY - imgRect.top - offsetY) / scale;

  return [
    Math.max(0, Math.min(imgOrigW, x)),
    Math.max(0, Math.min(imgOrigH, y)),
  ];
}

/** Convert image coords to canvas-local coords. */
function imageToCanvas(
  ix: number,
  iy: number,
  imgRect: DOMRect,
  canvasRect: DOMRect,
  imgOrigW: number,
  imgOrigH: number,
): [number, number] {
  const displayedAspect = imgRect.width / imgRect.height;
  const imageAspect = imgOrigW / imgOrigH;

  let scale: number, offsetX: number, offsetY: number;
  if (imageAspect > displayedAspect) {
    scale = imgRect.width / imgOrigW;
    offsetX = 0;
    offsetY = (imgRect.height - imgOrigH * scale) / 2;
  } else {
    scale = imgRect.height / imgOrigH;
    offsetX = (imgRect.width - imgOrigW * scale) / 2;
    offsetY = 0;
  }

  const sx = ix * scale + offsetX + (imgRect.left - canvasRect.left);
  const sy = iy * scale + offsetY + (imgRect.top - canvasRect.top);
  return [sx, sy];
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function EcgCropTool({
  imageUrl,
  imageWidth,
  imageHeight,
  onConfirm,
  onRetake,
}: EcgCropToolProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const imgRef = useRef<HTMLImageElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // 4 corners in IMAGE coordinates (not screen)
  const [corners, setCorners] = useState<Corners>(() => {
    const mx = imageWidth * 0.1;
    const my = imageHeight * 0.1;
    return {
      top_left: [mx, my],
      top_right: [imageWidth - mx, my],
      bottom_right: [imageWidth - mx, imageHeight - my],
      bottom_left: [mx, imageHeight - my],
    };
  });

  const [dragging, setDragging] = useState<keyof Corners | null>(null);
  const [detecting, setDetecting] = useState(true);

  // -----------------------------------------------------------------------
  // Auto-detect corners on mount
  // -----------------------------------------------------------------------
  useEffect(() => {
    autoDetectCorners(imageUrl, imageWidth, imageHeight).then((detected) => {
      setCorners(detected);
      setDetecting(false);
    });
  }, [imageUrl, imageWidth, imageHeight]);

  // -----------------------------------------------------------------------
  // Draw overlay
  // -----------------------------------------------------------------------
  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    const img = imgRef.current;
    if (!canvas || !img) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const canvasRect = canvas.getBoundingClientRect();
    const imgRect = img.getBoundingClientRect();

    // Match canvas pixel size to display size
    const dpr = window.devicePixelRatio || 1;
    canvas.width = canvasRect.width * dpr;
    canvas.height = canvasRect.height * dpr;
    ctx.scale(dpr, dpr);

    ctx.clearRect(0, 0, canvasRect.width, canvasRect.height);

    // Convert image coords to canvas coords
    const keys: (keyof Corners)[] = [
      "top_left",
      "top_right",
      "bottom_right",
      "bottom_left",
    ];
    const pts = keys.map((k) =>
      imageToCanvas(
        corners[k][0],
        corners[k][1],
        imgRect,
        canvasRect,
        imageWidth,
        imageHeight,
      ),
    );

    // Dark overlay (full canvas)
    ctx.fillStyle = OVERLAY_COLOR;
    ctx.fillRect(0, 0, canvasRect.width, canvasRect.height);

    // Clear inside the quadrilateral (make it bright)
    ctx.save();
    ctx.beginPath();
    ctx.moveTo(pts[0][0], pts[0][1]);
    for (let i = 1; i < 4; i++) ctx.lineTo(pts[i][0], pts[i][1]);
    ctx.closePath();
    ctx.globalCompositeOperation = "destination-out";
    ctx.fill();
    ctx.restore();

    // Draw border lines
    ctx.strokeStyle = ACCENT;
    ctx.lineWidth = LINE_WIDTH;
    ctx.beginPath();
    ctx.moveTo(pts[0][0], pts[0][1]);
    for (let i = 1; i < 4; i++) ctx.lineTo(pts[i][0], pts[i][1]);
    ctx.closePath();
    ctx.stroke();

    // Draw corner points
    for (let i = 0; i < 4; i++) {
      const isActive = dragging === keys[i];
      ctx.beginPath();
      ctx.arc(
        pts[i][0],
        pts[i][1],
        POINT_RADIUS * (isActive ? 1.2 : 1),
        0,
        Math.PI * 2,
      );
      ctx.fillStyle = "white";
      ctx.fill();
      ctx.strokeStyle = isActive ? ACCENT_ACTIVE : ACCENT;
      ctx.lineWidth = POINT_BORDER;
      ctx.stroke();
    }
  }, [corners, dragging, imageWidth, imageHeight]);

  useEffect(() => {
    draw();
  }, [draw]);

  // Redraw on resize
  useEffect(() => {
    const handleResize = () => draw();
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, [draw]);

  // Also redraw when image loads
  const handleImageLoad = () => draw();

  // -----------------------------------------------------------------------
  // Touch / pointer handlers
  // -----------------------------------------------------------------------
  const getClosestCorner = (
    clientX: number,
    clientY: number,
  ): keyof Corners | null => {
    const img = imgRef.current;
    const canvas = canvasRef.current;
    if (!img || !canvas) return null;

    const canvasRect = canvas.getBoundingClientRect();
    const imgRect = img.getBoundingClientRect();
    const keys: (keyof Corners)[] = [
      "top_left",
      "top_right",
      "bottom_right",
      "bottom_left",
    ];

    let closest: keyof Corners | null = null;
    let minDist = Infinity;

    for (const k of keys) {
      const [cx, cy] = imageToCanvas(
        corners[k][0],
        corners[k][1],
        imgRect,
        canvasRect,
        imageWidth,
        imageHeight,
      );
      const dx = clientX - canvasRect.left - cx;
      const dy = clientY - canvasRect.top - cy;
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (dist < HIT_RADIUS && dist < minDist) {
        minDist = dist;
        closest = k;
      }
    }
    return closest;
  };

  const handlePointerDown = (e: React.PointerEvent) => {
    const corner = getClosestCorner(e.clientX, e.clientY);
    if (corner) {
      setDragging(corner);
      e.currentTarget.setPointerCapture(e.pointerId);
    }
  };

  const handlePointerMove = (e: React.PointerEvent) => {
    if (!dragging) return;
    const img = imgRef.current;
    if (!img) return;

    const imgRect = img.getBoundingClientRect();
    const [ix, iy] = screenToImage(
      e.clientX,
      e.clientY,
      imgRect,
      imageWidth,
      imageHeight,
    );

    setCorners((prev) => ({ ...prev, [dragging]: [ix, iy] }));
  };

  const handlePointerUp = () => {
    setDragging(null);
  };

  // -----------------------------------------------------------------------
  // Render
  // -----------------------------------------------------------------------
  return (
    <div className="flex h-full w-full flex-col bg-black">
      {/* Image + canvas area */}
      <div ref={containerRef} className="relative flex-1 overflow-hidden">
        <img
          ref={imgRef}
          src={imageUrl}
          alt="ECG para recorte"
          className="h-full w-full object-contain"
          onLoad={handleImageLoad}
          draggable={false}
        />
        <canvas
          ref={canvasRef}
          className="absolute inset-0 h-full w-full touch-none"
          onPointerDown={handlePointerDown}
          onPointerMove={handlePointerMove}
          onPointerUp={handlePointerUp}
          onPointerCancel={handlePointerUp}
        />
        {detecting && (
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="rounded-lg bg-black/70 px-4 py-2 text-sm text-white">
              Detectando bordas...
            </div>
          </div>
        )}
      </div>

      {/* Instructions + buttons */}
      <div className="flex flex-col items-center gap-3 px-4 py-4">
        <p className="text-sm text-neutral-400">
          Ajuste os cantos do papel ECG
        </p>
        <div className="flex gap-3">
          <Button variant="outline" onClick={onRetake} className="border-neutral-600 text-neutral-300">
            Nova foto
          </Button>
          <Button onClick={() => onConfirm(corners)}>
            Confirmar
          </Button>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Auto-detection (simple Sobel edge detection in pure JS)
// ---------------------------------------------------------------------------

async function autoDetectCorners(
  imageUrl: string,
  origW: number,
  origH: number,
): Promise<Corners> {
  const fallback = (): Corners => {
    const mx = origW * 0.1;
    const my = origH * 0.1;
    return {
      top_left: [mx, my],
      top_right: [origW - mx, my],
      bottom_right: [origW - mx, origH - my],
      bottom_left: [mx, origH - my],
    };
  };

  try {
    const img = await loadImage(imageUrl);

    // Resize for performance
    const maxDim = 500;
    const scale = maxDim / Math.max(img.width, img.height);
    const sw = Math.round(img.width * scale);
    const sh = Math.round(img.height * scale);

    const canvas = document.createElement("canvas");
    canvas.width = sw;
    canvas.height = sh;
    const ctx = canvas.getContext("2d");
    if (!ctx) return fallback();

    ctx.drawImage(img, 0, 0, sw, sh);
    const imageData = ctx.getImageData(0, 0, sw, sh);

    // Grayscale
    const gray = new Uint8Array(sw * sh);
    for (let i = 0; i < gray.length; i++) {
      const idx = i * 4;
      gray[i] = Math.round(
        imageData.data[idx] * 0.299 +
        imageData.data[idx + 1] * 0.587 +
        imageData.data[idx + 2] * 0.114,
      );
    }

    // Sobel edge detection
    const edges = sobelEdges(gray, sw, sh);

    // Find bounding rectangle of the strongest edge region
    const rect = findEdgeRect(edges, sw, sh);
    if (!rect) return fallback();

    // Convert back to original coords with small margin
    const margin = 0.02;
    return {
      top_left: [
        Math.max(0, (rect.x1 / scale) - origW * margin),
        Math.max(0, (rect.y1 / scale) - origH * margin),
      ],
      top_right: [
        Math.min(origW, (rect.x2 / scale) + origW * margin),
        Math.max(0, (rect.y1 / scale) - origH * margin),
      ],
      bottom_right: [
        Math.min(origW, (rect.x2 / scale) + origW * margin),
        Math.min(origH, (rect.y2 / scale) + origH * margin),
      ],
      bottom_left: [
        Math.max(0, (rect.x1 / scale) - origW * margin),
        Math.min(origH, (rect.y2 / scale) + origH * margin),
      ],
    };
  } catch {
    return fallback();
  }
}

function loadImage(url: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = url;
  });
}

function sobelEdges(gray: Uint8Array, w: number, h: number): Uint8Array {
  const out = new Uint8Array(w * h);
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const idx = y * w + x;
      const gx =
        -gray[(y - 1) * w + x - 1] + gray[(y - 1) * w + x + 1] +
        -2 * gray[y * w + x - 1] + 2 * gray[y * w + x + 1] +
        -gray[(y + 1) * w + x - 1] + gray[(y + 1) * w + x + 1];
      const gy =
        -gray[(y - 1) * w + x - 1] - 2 * gray[(y - 1) * w + x] - gray[(y - 1) * w + x + 1] +
        gray[(y + 1) * w + x - 1] + 2 * gray[(y + 1) * w + x] + gray[(y + 1) * w + x + 1];
      out[idx] = Math.min(255, Math.sqrt(gx * gx + gy * gy));
    }
  }
  return out;
}

function findEdgeRect(
  edges: Uint8Array,
  w: number,
  h: number,
): { x1: number; y1: number; x2: number; y2: number } | null {
  // Threshold: top 15% of edge strength
  const sorted = [...edges].sort((a, b) => b - a);
  const threshold = sorted[Math.floor(sorted.length * 0.15)] || 50;

  // Project edges onto axes
  const hProj = new Float32Array(w);
  const vProj = new Float32Array(h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (edges[y * w + x] >= threshold) {
        hProj[x]++;
        vProj[y]++;
      }
    }
  }

  // Find bounds (10th and 90th percentile of edge mass)
  const findBounds = (proj: Float32Array): [number, number] => {
    const total = proj.reduce((a, b) => a + b, 0);
    if (total === 0) return [0, proj.length - 1];

    let cumul = 0;
    let lo = 0;
    let hi = proj.length - 1;
    for (let i = 0; i < proj.length; i++) {
      cumul += proj[i];
      if (cumul >= total * 0.08) { lo = i; break; }
    }
    cumul = 0;
    for (let i = proj.length - 1; i >= 0; i--) {
      cumul += proj[i];
      if (cumul >= total * 0.08) { hi = i; break; }
    }
    return [lo, hi];
  };

  const [x1, x2] = findBounds(hProj);
  const [y1, y2] = findBounds(vProj);

  // Must cover at least 20% of the image
  if ((x2 - x1) * (y2 - y1) < w * h * 0.15) return null;

  return { x1, y1, x2, y2 };
}
