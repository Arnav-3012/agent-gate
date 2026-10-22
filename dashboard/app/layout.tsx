import "./globals.css";
import type { ReactNode } from "react";
import { Nav } from "@/components/ui/Nav";
import { Hotkeys } from "@/components/ui/Hotkeys";
import { Spotlight } from "@/components/ui/Spotlight";
import { Providers } from "@/components/ui/Providers";
import { HealthPill } from "@/components/ui/HealthPill";

export const metadata = { title: "AgentGate" };

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>
        <Providers>
        <Spotlight />
        <Hotkeys />
        <div className="progress" aria-hidden />
        <header>
          <h1><i className="logo" />AgentGate</h1>
          <Nav />
          <HealthPill />
        </header>
        {children}
        <footer>AgentGate · cost-aware routing across local and hosted models</footer>
        </Providers>
      </body>
    </html>
  );
}
