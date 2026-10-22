"use client";
import { MotionConfig } from "framer-motion";
import type { ReactNode } from "react";

// reducedMotion="user": framer animations honour the OS reduced-motion setting.
export function Providers({ children }: { children: ReactNode }) {
  return <MotionConfig reducedMotion="user">{children}</MotionConfig>;
}
