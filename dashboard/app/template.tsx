"use client";
import { motion } from "framer-motion";
import type { ReactNode } from "react";

// A template remounts on every navigation, so each page gets an enter transition.
export default function Template({ children }: { children: ReactNode }) {
  return <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4, ease: [0.2, 0.8, 0.2, 1] }}>{children}</motion.div>;
}
