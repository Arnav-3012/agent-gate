"use client";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";

const PAGES = ["/", "/galaxy", "/battle", "/benchmark"];

// 1-4 jump between pages, P toggles presentation mode (chrome hidden, larger type), Esc leaves it. Ignored while typing.
export function Hotkeys() {
  const router = useRouter();
  const [on, setOn] = useState(false);
  useEffect(() => {
    document.body.classList.toggle("present", on);
  }, [on]);
  useEffect(() => {
    const h = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement;
      if (t.closest("input, select, textarea") || e.metaKey || e.ctrlKey || e.altKey) return;
      if (e.key >= "1" && e.key <= "4") router.push(PAGES[+e.key - 1]);
      else if (e.key === "p" || e.key === "P") setOn((v) => !v);
      else if (e.key === "Escape") setOn(false);
    };
    window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
  }, [router]);
  return on ? <div className="presenthint" aria-hidden>Presentation · 1-4 pages · Esc exit</div> : null;
}
