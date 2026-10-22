"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { motion } from "framer-motion";

const LINKS = [["/", "Command Center"], ["/galaxy", "Galaxy"], ["/battle", "Battle"], ["/benchmark", "Benchmark"]] as const;

export function Nav() {
  const path = usePathname();
  return (
    <nav>
      {LINKS.map(([href, label]) => (
        <Link key={href} href={href} className={path === href ? "active" : ""}>
          {path === href && <motion.span layoutId="navpill" className="navpill" transition={{ type: "spring", stiffness: 500, damping: 36 }} />}
          <span className="navlabel">{label}</span>
        </Link>
      ))}
    </nav>
  );
}
