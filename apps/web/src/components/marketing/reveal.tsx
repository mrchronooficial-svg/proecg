"use client";

import { useEffect, useRef, useState, type ReactNode } from "react";

interface Props {
  children: ReactNode;
  className?: string;
  delay?: number;
  y?: number;
}

/**
 * IntersectionObserver + CSS transitions. Mais leve que framer-motion em
 * mobile (sem JS animando frame a frame). Honra prefers-reduced-motion e
 * reduz a distância de translate em telas pequenas pra repaints menores.
 */
export function Reveal({ children, className = "", delay = 0, y = 30 }: Props) {
  const ref = useRef<HTMLDivElement | null>(null);
  const [shown, setShown] = useState(false);
  const [reduced, setReduced] = useState(false);
  const [mobile, setMobile] = useState(false);

  useEffect(() => {
    const mqReduced = window.matchMedia("(prefers-reduced-motion: reduce)");
    const mqMobile = window.matchMedia("(max-width: 767px)");
    setReduced(mqReduced.matches);
    setMobile(mqMobile.matches);

    const onR = () => setReduced(mqReduced.matches);
    const onM = () => setMobile(mqMobile.matches);
    mqReduced.addEventListener("change", onR);
    mqMobile.addEventListener("change", onM);
    return () => {
      mqReduced.removeEventListener("change", onR);
      mqMobile.removeEventListener("change", onM);
    };
  }, []);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    if (reduced) {
      setShown(true);
      return;
    }
    const io = new IntersectionObserver(
      (entries) => {
        for (const e of entries) {
          if (e.isIntersecting) {
            setShown(true);
            io.disconnect();
            break;
          }
        }
      },
      { rootMargin: "0px 0px -80px 0px", threshold: 0.05 },
    );
    io.observe(el);
    return () => io.disconnect();
  }, [reduced]);

  // Em mobile, distancia menor (paint area menor) e duração menor.
  const ty = mobile ? Math.min(y, 14) : y;
  const duration = mobile ? 450 : 700;

  const style: React.CSSProperties = reduced
    ? { opacity: 1 }
    : {
        opacity: shown ? 1 : 0,
        transform: shown ? "translate3d(0,0,0)" : `translate3d(0,${ty}px,0)`,
        transition: `opacity ${duration}ms cubic-bezier(0.21,0.47,0.32,0.98) ${delay}s, transform ${duration}ms cubic-bezier(0.21,0.47,0.32,0.98) ${delay}s`,
        willChange: shown ? "auto" : "opacity, transform",
      };

  return (
    <div ref={ref} className={className} style={style}>
      {children}
    </div>
  );
}
