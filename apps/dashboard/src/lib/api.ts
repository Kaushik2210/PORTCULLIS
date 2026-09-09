/**
 * Typed client for the PORTCULLIS gateway (packages/gateway) - the same
 * API surface the Python client SDK wraps, just from TypeScript. Every
 * shape here mirrors packages/gateway/src/portcullis/gateway/schemas.py
 * and decision_log.py exactly; this file has no independent logic of its
 * own beyond "fetch and parse."
 */

export const GATEWAY_BASE_URL =
  process.env.NEXT_PUBLIC_GATEWAY_URL ?? "http://127.0.0.1:8001";

export type Verdict = "allow" | "flag" | "sanitise" | "challenge" | "block";
export type ConversationState =
  | "normal"
  | "probing"
  | "establishing"
  | "exploiting";
export type Scope = "user" | "system" | "retrieved" | "tool_result" | "file";

export interface MatchedRule {
  rule_id: string;
  name: string;
  severity: string;
  labels: string[];
  weight: number;
  span: [number, number];
  matched_text: string;
  rationale: string;
}

export interface LatencyBreakdown {
  l0_ms: number;
  l1_ms: number;
  l2_ms: number;
  knn_ms: number;
  fusion_ms: number;
  total_ms: number;
}

export interface DetectResponse {
  verdict: Verdict;
  enforced: boolean;
  score: number;
  rationale: string;
  contributions: Record<string, number>;
  matched_rules: MatchedRule[];
  taxonomy_labels: string[];
  obfuscation_score: number;
  max_decode_depth: number;
  nearest_known_attack: string | null;
  nearest_known_attack_family: string | null;
  latency_ms: LatencyBreakdown;
  conversation_state: ConversationState | null;
}

export interface DecisionLogEntry {
  id: string;
  timestamp: number;
  text_preview: string;
  verdict: Verdict;
  enforced: boolean;
  score: number;
  taxonomy_labels: string[];
  conversation_state: ConversationState | null;
  latency_ms: number;
  source: "detect" | "detect_batch" | "chat_completions";
}

export interface BootstrapStat {
  tpr: number;
  ci_low: number;
  ci_high: number;
  ci_level: number;
  n_resamples: number;
}

export interface HeadlineReport {
  auprc: number;
  "tpr_at_fpr_0.100%": BootstrapStat;
  "tpr_at_fpr_1.0%": BootstrapStat;
}

export interface AblationReport extends HeadlineReport {
  coef: Record<string, number>;
}

export interface EvalResults {
  test_set: { n: number; n_attack: number };
  fusion_weights: { intercept: number; coef: Record<string, number> };
  headline: HeadlineReport;
  baselines: { max: HeadlineReport; regex_only: HeadlineReport };
  ablations: { remove_l0: AblationReport; remove_knn: AblationReport };
  latency_ms: { p50: number; p95: number; p99: number; n: number };
  adaptive: {
    paraphrase: {
      n_seeds: number;
      n_seeds_with_no_paraphrase_available: number;
      n_variants_tried: number;
      variant_evasion_rate: number;
      seed_evasion_rate: number;
      block_threshold: number;
    };
    blackbox_query: {
      n_seeds: number;
      evasion_rate: number;
      block_threshold: number;
      max_queries: number;
      mean_queries_used: number;
    };
  };
  elapsed_s: number;
}

export interface EvalScores {
  labels: number[];
  fusion: number[];
  remove_knn: number[];
}

async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(`${GATEWAY_BASE_URL}${path}`);
  if (!response.ok) {
    throw new Error(`GET ${path} -> ${response.status}`);
  }
  return (await response.json()) as T;
}

export async function postDetect(
  text: string,
  opts?: { scope?: Scope; shadow?: boolean; conversation_id?: string }
): Promise<DetectResponse> {
  const response = await fetch(`${GATEWAY_BASE_URL}/v1/detect`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      text,
      scope: opts?.scope ?? "user",
      shadow: opts?.shadow ?? false,
      conversation_id: opts?.conversation_id,
    }),
  });
  if (!response.ok) {
    throw new Error(`POST /v1/detect -> ${response.status}`);
  }
  return (await response.json()) as DetectResponse;
}

export function getRecentDecisions(n = 50): Promise<DecisionLogEntry[]> {
  return getJson<DecisionLogEntry[]>(`/v1/decisions/recent?n=${n}`);
}

export function getEvalScores(): Promise<EvalScores> {
  return getJson<EvalScores>("/v1/eval/scores");
}

export function getEvalResults(): Promise<EvalResults> {
  return getJson<EvalResults>("/v1/eval/results");
}

/**
 * Subscribes to the live decision feed. Returns an unsubscribe function.
 * Uses the browser's native EventSource rather than a library - one
 * long-lived GET with a well-defined `text/event-stream` body is exactly
 * what EventSource exists for, and pulling in a dependency to replace
 * something the platform already does well would be the wrong trade here.
 */
export function subscribeToDecisions(
  onDecision: (entry: DecisionLogEntry) => void
): () => void {
  const source = new EventSource(`${GATEWAY_BASE_URL}/v1/decisions/stream`);
  source.onmessage = (event) => {
    onDecision(JSON.parse(event.data) as DecisionLogEntry);
  };
  return () => source.close();
}
