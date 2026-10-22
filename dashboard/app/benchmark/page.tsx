import { PageHead } from "@/components/ui/PageHead";
import { BenchmarkView } from "@/components/charts/BenchmarkView";

export default function Page() {
  return <main><PageHead kicker="Evidence" title="Benchmark">Cost vs quality for every arm, with bootstrap intervals and the oracle ceiling.</PageHead><BenchmarkView /></main>;
}
