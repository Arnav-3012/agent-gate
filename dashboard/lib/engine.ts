import { EngineClient } from "./api";

// NEXT_PUBLIC_API_URL (AWS HTTP API) wins over NEXT_PUBLIC_ENGINE_URL (engine direct, local dev).
const API_URL = process.env.NEXT_PUBLIC_API_URL;
export const ENGINE_URL = API_URL ?? process.env.NEXT_PUBLIC_ENGINE_URL ?? "http://localhost:8000";
export const client = new EngineClient(ENGINE_URL, undefined, API_URL ? "/query" : "/run");
