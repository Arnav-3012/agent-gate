"use client";
import { useEffect } from "react";

// Cursor-follow light: sets --mx/--my on <html> (page glow) and --px/--py on the hovered .panel/.card (edge sheen).
export function Spotlight() {
  useEffect(() => {
    let raf = 0;
    const h = (e: PointerEvent) => {
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(() => {
        document.documentElement.style.setProperty("--mx", `${e.clientX}px`);
        document.documentElement.style.setProperty("--my", `${e.clientY}px`);
        const el = (e.target as HTMLElement).closest<HTMLElement>(".panel, .card");
        if (el) {
          const r = el.getBoundingClientRect();
          el.style.setProperty("--px", `${e.clientX - r.left}px`);
          el.style.setProperty("--py", `${e.clientY - r.top}px`);
        }
      });
    };
    window.addEventListener("pointermove", h, { passive: true });
    return () => { window.removeEventListener("pointermove", h); cancelAnimationFrame(raf); };
  }, []);
  return <div className="bg" aria-hidden><i className="aurora a1" /><i className="aurora a2" /><i className="aurora a3" /><i className="dots" /><i className="cursorglow" /><i className="grain" /></div>;
}
