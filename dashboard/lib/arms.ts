// Arm ids (config/arms.yaml + the D52 cross-tier arms). Order = cheapest first, as the cascade ladder.
// usdPer1k: modelled $/1k queries, "true" cost view (D53); replace from results.json once wired.
export const ARM_META = [
  { id: "qwen7b-closed", label: "7B closed-book", tier: "local", usdPer1k: 0.004, color: "#7dd3fc" },
  { id: "qwen7b-rag", label: "7B RAG", tier: "local", usdPer1k: 0.004, color: "#6c8cff" },
  { id: "qwen7b-agent", label: "7B agent", tier: "local", usdPer1k: 0.01, color: "#a78bfa" },
  { id: "qwen14b-rag", label: "14B RAG", tier: "local", usdPer1k: 0.009, color: "#f0abfc" },
  { id: "qwen14b-agent", label: "14B agent", tier: "local", usdPer1k: 0.02, color: "#fb7185" },
  { id: "haiku-rag", label: "Haiku 5.5 RAG", tier: "hosted", usdPer1k: 0.19, color: "#34d399" },
  { id: "sonnet-rag", label: "Sonnet 5.5 RAG", tier: "hosted", usdPer1k: 3.76, color: "#fbbf24" },
] as const;
export type ArmId = (typeof ARM_META)[number]["id"];
export const ARMS = ARM_META.map((a) => a.id);
export const armMeta = (id: string) => ARM_META.find((a) => a.id === id);

// results.json names arms "<model>/<mode>" (e.g. "default/rag" = the 7B default); map them to ARM_META ids.
export function resultArmId(name: string): string {
  const mode = name.split("/").pop() ?? "rag";
  const n = name.toLowerCase();
  if (n.includes("haiku")) return "haiku-rag";
  if (n.includes("sonnet")) return "sonnet-rag";
  if (n.includes("14b")) return `qwen14b-${mode}`;
  if (n.startsWith("default") || n.includes("7b")) return `qwen7b-${mode}`;
  return name;
}
export const isHosted = (name: string) => armMeta(resultArmId(name))?.tier === "hosted";
export const armLabel = (name: string) => armMeta(resultArmId(name))?.label ?? name;
