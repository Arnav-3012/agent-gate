import type { ReactNode } from "react";

export function PageHead({ kicker, title, children }: { kicker: string; title: string; children?: ReactNode }) {
  return (
    <section className="pagehead">
      <span className="kicker">{kicker}</span>
      <h1>{title}</h1>
      {children && <p>{children}</p>}
    </section>
  );
}
