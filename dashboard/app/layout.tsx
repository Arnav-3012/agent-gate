import "./globals.css";
import type { ReactNode } from "react";
import Link from "next/link";
import { HealthPill } from "@/components/ui/HealthPill";

export const metadata = { title: "AgentGate" };

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>
        <header>
          <h1>AgentGate</h1>
          <nav>
            <Link href="/">Command Center</Link>
            <Link href="/galaxy">Galaxy</Link>
            <Link href="/battle">Battle</Link>
            <Link href="/benchmark">Benchmark</Link>
          </nav>
          <HealthPill />
        </header>
        {children}
      </body>
    </html>
  );
}
