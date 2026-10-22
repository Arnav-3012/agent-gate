"use client";
import { useEffect, useState } from "react";
import { MOCK_RESULTS, type Results } from "./results";

// /data/results.json is a copy of eval/results.json (synced by predev/prebuild); labelled mock until it exists.
export function useResults(): { results: Results; mock: boolean } {
  const [state, setState] = useState({ results: MOCK_RESULTS, mock: true });
  useEffect(() => {
    fetch("/data/results.json").then((r) => (r.ok ? r.json() : Promise.reject())).then((results: Results) => setState({ results, mock: false })).catch(() => {});
  }, []);
  return state;
}
