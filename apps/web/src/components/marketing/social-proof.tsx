"use client";

import { Reveal } from "./reveal";
import { useEffect, useRef, useState } from "react";

function Counter({ target, suffix = "" }: { target: number; suffix?: string }) {
  const [count, setCount] = useState(0);
  const ref = useRef<HTMLSpanElement>(null);
  const started = useRef(false);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting && !started.current) {
          started.current = true;
          const duration = 1500;
          const start = performance.now();
          function tick(now: number) {
            const elapsed = now - start;
            const progress = Math.min(elapsed / duration, 1);
            const eased = 1 - Math.pow(1 - progress, 3);
            setCount(Math.round(eased * target));
            if (progress < 1) requestAnimationFrame(tick);
          }
          requestAnimationFrame(tick);
        }
      },
      { threshold: 0.5 },
    );
    observer.observe(el);
    return () => observer.disconnect();
  }, [target]);

  return (
    <span ref={ref}>
      {count.toLocaleString("pt-BR")}
      {suffix}
    </span>
  );
}

const metrics = [
  { value: 2500, prefix: "+", suffix: "", label: "Laudos gerados" },
  { value: 30, prefix: "< ", suffix: "s", label: "Tempo médio do laudo" },
  { value: 30, prefix: "~", suffix: "", label: "Diagnósticos cobertos" },
];

export function SocialProof() {
  return (
    <section className="border-y border-[#E2E4F0] bg-white px-4 py-14">
      <div className="mx-auto max-w-[1200px]">
        <Reveal>
          <p className="mb-10 text-center text-lg font-medium text-[#4A5078]">
            A confiança de quem já usa
          </p>
        </Reveal>
        <div className="grid grid-cols-1 gap-8 sm:grid-cols-3">
          {metrics.map((m, i) => (
            <Reveal key={m.label} delay={i * 0.1}>
              <div className="flex flex-col items-center gap-1">
                <p className="text-4xl font-extrabold text-[#122056] sm:text-5xl">
                  {m.prefix}
                  <Counter target={m.value} suffix={m.suffix} />
                </p>
                <p className="text-base text-[#4A5078]">{m.label}</p>
              </div>
            </Reveal>
          ))}
        </div>
      </div>
    </section>
  );
}
