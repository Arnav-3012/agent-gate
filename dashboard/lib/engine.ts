import { EngineClient } from "./api";

export const ENGINE_URL = process.env.NEXT_PUBLIC_ENGINE_URL ?? "http://localhost:8000";
export const client = new EngineClient(ENGINE_URL);
