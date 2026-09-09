"use client";

import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { cn } from "cn";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { Slider } from "@/components/ui/slider";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  getEvalResults,
  getEvalScores,
  type BootstrapStat,
  type EvalResults,
  type HeadlineReport,
} from "@/lib/api";
import { computeConfusionMatrix } from "@/lib/confusion-matrix";

/** "0.191" -> "19.1%", tolerant of 0-1 fractions used throughout EvalResults. */
function formatPct(value: number, digits = 1): string {
  return `${(value * 100).toFixed(digits)}%`;
}

/** A bootstrap point estimate is never shown without its CI (ADR-0011/working agreement). */
function formatBootstrap(stat: BootstrapStat): string {
  return `${formatPct(stat.tpr)} (${formatPct(stat.ci_level, 0)} CI ${formatPct(
    stat.ci_low
  )}–${formatPct(stat.ci_high)}, n=${stat.n_resamples.toLocaleString()})`;
}

function StatTile({
  label,
  value,
  emphasis,
}: {
  label: string;
  value: string;
  emphasis?: "block" | "muted";
}) {
  return (
    <div className="rounded-lg border border-border p-3">
      <div className="text-[11px] uppercase tracking-wide text-muted-foreground">
        {label}
      </div>
      <div
        className={cn(
          "font-mono text-lg tabular-nums",
          emphasis === "block" && "text-verdict-block"
        )}
      >
        {value}
      </div>
    </div>
  );
}

function LoadingPanel({ label }: { label: string }) {
  return <p className="p-4 text-sm text-muted-foreground">{label}</p>;
}

function MissingArtifactPanel({ what }: { what: string }) {
  return (
    <p className="p-4 text-sm text-verdict-block">
      {what} not available - run <code className="font-mono">just eval</code>{" "}
      first.
    </p>
  );
}

/**
 * Headline replay-vs-baselines table. Deliberately shows the regex-only
 * baseline matching (or beating) the full fusion system at these operating
 * points - a real M9 finding, not something to soften or hide.
 */
function HeadlineTable({ results }: { results: EvalResults }) {
  const rows: { name: string; report: HeadlineReport; note?: string }[] = [
    { name: "Fusion (deployed)", report: results.headline },
    {
      name: "Baseline: regex-only (L1)",
      report: results.baselines.regex_only,
      note: "Matches or slightly beats fusion at these operating points.",
    },
    {
      name: "Baseline: max()",
      report: results.baselines.max,
      note: "Naive max() over layer scores instead of learned fusion.",
    },
  ];

  return (
    <div className="overflow-x-auto">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Variant</TableHead>
            <TableHead>AUPRC</TableHead>
            <TableHead>TPR @ 0.1% FPR</TableHead>
            <TableHead>TPR @ 1% FPR</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {rows.map((row) => (
            <TableRow key={row.name}>
              <TableCell className="align-top">
                <div className="font-medium">{row.name}</div>
                {row.note && (
                  <div className="mt-0.5 max-w-[220px] text-xs text-muted-foreground">
                    {row.note}
                  </div>
                )}
              </TableCell>
              <TableCell className="font-mono text-xs tabular-nums">
                {row.report.auprc.toFixed(3)}
              </TableCell>
              <TableCell className="font-mono text-xs tabular-nums">
                {formatBootstrap(row.report["tpr_at_fpr_0.100%"])}
              </TableCell>
              <TableCell className="font-mono text-xs tabular-nums">
                {formatBootstrap(row.report["tpr_at_fpr_1.0%"])}
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}

function HeadlineChart({ results }: { results: EvalResults }) {
  const chartData = [
    {
      name: "Fusion",
      "TPR@0.1%FPR": results.headline["tpr_at_fpr_0.100%"].tpr,
      "TPR@1%FPR": results.headline["tpr_at_fpr_1.0%"].tpr,
    },
    {
      name: "Regex-only",
      "TPR@0.1%FPR": results.baselines.regex_only["tpr_at_fpr_0.100%"].tpr,
      "TPR@1%FPR": results.baselines.regex_only["tpr_at_fpr_1.0%"].tpr,
    },
    {
      name: "max()",
      "TPR@0.1%FPR": results.baselines.max["tpr_at_fpr_0.100%"].tpr,
      "TPR@1%FPR": results.baselines.max["tpr_at_fpr_1.0%"].tpr,
    },
  ];

  return (
    <div className="h-[240px] w-full">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={chartData} margin={{ top: 4, right: 8, bottom: 4, left: 0 }}>
          <CartesianGrid vertical={false} stroke="var(--color-border)" />
          <XAxis
            dataKey="name"
            tick={{ fontSize: 11, fill: "var(--color-muted-foreground)" }}
            axisLine={false}
            tickLine={false}
          />
          <YAxis
            tickFormatter={(v: number) => formatPct(v, 0)}
            tick={{ fontSize: 11, fill: "var(--color-muted-foreground)" }}
            axisLine={false}
            tickLine={false}
            width={40}
          />
          <Tooltip
            formatter={(v) => formatPct(Number(v))}
            cursor={{ fill: "var(--color-muted)" }}
            contentStyle={{
              background: "var(--color-card)",
              border: "1px solid var(--color-border)",
              borderRadius: 8,
              fontSize: 12,
            }}
          />
          <Legend wrapperStyle={{ fontSize: 11 }} />
          <Bar dataKey="TPR@0.1%FPR" fill="var(--color-chart-3)" radius={[3, 3, 0, 0]} maxBarSize={36} />
          <Bar dataKey="TPR@1%FPR" fill="var(--color-chart-5)" radius={[3, 3, 0, 0]} maxBarSize={36} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

function HeadlineTab({ results }: { results: EvalResults }) {
  return (
    <div className="grid gap-4 lg:grid-cols-[3fr_2fr]">
      <Card>
        <CardHeader>
          <CardTitle>Corpus replay: headline vs baselines</CardTitle>
          <CardDescription>
            Milestone 9&apos;s full offline evaluation run over the real test
            set ({results.test_set.n.toLocaleString()} rows,{" "}
            {results.test_set.n_attack.toLocaleString()} attacks) - not a live
            re-run from this page.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <HeadlineTable results={results} />
        </CardContent>
      </Card>
      <Card>
        <CardHeader>
          <CardTitle>TPR by operating point</CardTitle>
          <CardDescription>Point estimates only - see table for CIs.</CardDescription>
        </CardHeader>
        <CardContent>
          <HeadlineChart results={results} />
        </CardContent>
      </Card>
    </div>
  );
}

function ConfusionMatrixPanel({
  title,
  matrix,
}: {
  title: string;
  matrix: ReturnType<typeof computeConfusionMatrix>;
}) {
  return (
    <div className="space-y-2">
      <div className="text-sm font-medium">{title}</div>
      <div className="grid grid-cols-2 gap-2">
        <StatTile label="TP" value={matrix.truePositive.toLocaleString()} />
        <StatTile label="FP" value={matrix.falsePositive.toLocaleString()} />
        <StatTile label="FN" value={matrix.falseNegative.toLocaleString()} />
        <StatTile label="TN" value={matrix.trueNegative.toLocaleString()} />
        <StatTile label="TPR" value={matrix.tpr.toFixed(3)} />
        <StatTile label="FPR" value={matrix.fpr.toFixed(3)} />
      </div>
    </div>
  );
}

function AblationTab({
  results,
  threshold,
  onThresholdChange,
}: {
  results: EvalResults;
  threshold: number;
  onThresholdChange: (t: number) => void;
}) {
  const {
    data: scores,
    isLoading: scoresLoading,
    isError: scoresError,
  } = useQuery({
    queryKey: ["eval-scores"],
    queryFn: getEvalScores,
    staleTime: Infinity,
    retry: false,
  });

  const fusionMatrix = useMemo(
    () => (scores ? computeConfusionMatrix(scores.labels, scores.fusion, threshold) : null),
    [scores, threshold]
  );
  const removeKnnMatrix = useMemo(
    () =>
      scores ? computeConfusionMatrix(scores.labels, scores.remove_knn, threshold) : null,
    [scores, threshold]
  );

  const rows: { name: string; report: HeadlineReport }[] = [
    { name: "Fusion (deployed)", report: results.headline },
    { name: "Fusion without kNN sidecar", report: results.ablations.remove_knn },
  ];

  return (
    <div className="grid gap-4 lg:grid-cols-[3fr_2fr]">
      <Card>
        <CardHeader>
          <CardTitle>Ablation diff: fusion vs. fusion without kNN</CardTitle>
          <CardDescription>
            Both are real, already-computed M9 fits scored on the same test
            set - not two trained model versions. Removing the kNN sidecar
            here actually improves the headline numbers slightly.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Variant</TableHead>
                  <TableHead>AUPRC</TableHead>
                  <TableHead>TPR @ 0.1% FPR</TableHead>
                  <TableHead>TPR @ 1% FPR</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {rows.map((row) => (
                  <TableRow key={row.name}>
                    <TableCell className="font-medium">{row.name}</TableCell>
                    <TableCell className="font-mono text-xs tabular-nums">
                      {row.report.auprc.toFixed(3)}
                    </TableCell>
                    <TableCell className="font-mono text-xs tabular-nums">
                      {formatBootstrap(row.report["tpr_at_fpr_0.100%"])}
                    </TableCell>
                    <TableCell className="font-mono text-xs tabular-nums">
                      {formatBootstrap(row.report["tpr_at_fpr_1.0%"])}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Confusion matrices at shared threshold</CardTitle>
          <CardDescription>
            Same {results.test_set.n.toLocaleString()}-row eval set, same
            threshold, both variants&apos; real per-row scores.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {scoresLoading ? (
            <LoadingPanel label="Loading eval scores..." />
          ) : scoresError ? (
            <MissingArtifactPanel what="Eval scores" />
          ) : (
            <>
              <div className="flex items-center gap-3">
                <Slider
                  min={0}
                  max={1}
                  step={0.01}
                  value={[threshold]}
                  onValueChange={(value) =>
                    onThresholdChange(Array.isArray(value) ? value[0] : value)
                  }
                  className="flex-1"
                />
                <span className="w-14 shrink-0 text-right font-mono text-sm tabular-nums">
                  {threshold.toFixed(2)}
                </span>
              </div>
              <div className="grid gap-4 sm:grid-cols-2">
                {fusionMatrix && (
                  <ConfusionMatrixPanel title="Fusion (deployed)" matrix={fusionMatrix} />
                )}
                {removeKnnMatrix && (
                  <ConfusionMatrixPanel
                    title="Fusion without kNN"
                    matrix={removeKnnMatrix}
                  />
                )}
              </div>
            </>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

function AdaptiveTab({ results }: { results: EvalResults }) {
  const { paraphrase, blackbox_query: blackbox } = results.adaptive;
  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <Card>
        <CardHeader>
          <CardTitle>Paraphrase attack</CardTitle>
          <CardDescription>
            Seed attacks rewritten by a paraphraser, retried against the
            deployed block threshold ({paraphrase.block_threshold.toFixed(3)}
            ).
          </CardDescription>
        </CardHeader>
        <CardContent className="grid grid-cols-2 gap-2">
          <StatTile label="Seeds" value={paraphrase.n_seeds.toLocaleString()} />
          <StatTile
            label="Seeds w/o paraphrase"
            value={paraphrase.n_seeds_with_no_paraphrase_available.toLocaleString()}
          />
          <StatTile
            label="Variants tried"
            value={paraphrase.n_variants_tried.toLocaleString()}
          />
          <StatTile
            label="Variant evasion rate"
            value={formatPct(paraphrase.variant_evasion_rate)}
          />
          <StatTile
            label="Seed evasion rate"
            value={formatPct(paraphrase.seed_evasion_rate)}
            emphasis={paraphrase.seed_evasion_rate > 0.3 ? "block" : undefined}
          />
        </CardContent>
      </Card>
      <Card>
        <CardHeader>
          <CardTitle>Black-box query attack</CardTitle>
          <CardDescription>
            Adaptive, query-limited search against the deployed block
            threshold ({blackbox.block_threshold.toFixed(3)}). Reported
            honestly - this is a genuinely weak result.
          </CardDescription>
        </CardHeader>
        <CardContent className="grid grid-cols-2 gap-2">
          <StatTile label="Seeds" value={blackbox.n_seeds.toLocaleString()} />
          <StatTile
            label="Evasion rate"
            value={formatPct(blackbox.evasion_rate)}
            emphasis="block"
          />
          <StatTile label="Max queries" value={blackbox.max_queries.toLocaleString()} />
          <StatTile
            label="Mean queries used"
            value={blackbox.mean_queries_used.toFixed(1)}
          />
        </CardContent>
      </Card>
    </div>
  );
}

/** Human-readable Markdown summary of whatever is currently loaded. */
function buildReportMarkdown(
  results: EvalResults,
  activeTab: string,
  ablationThreshold: number
): string {
  const lines: string[] = [];
  lines.push("# PORTCULLIS red-team console report");
  lines.push("");
  lines.push(`Generated: ${new Date().toISOString()}`);
  lines.push(`Active tab at export time: ${activeTab}`);
  lines.push(`Ablation-diff threshold at export time: ${ablationThreshold.toFixed(2)}`);
  lines.push("");
  lines.push(
    `Test set: ${results.test_set.n.toLocaleString()} rows, ${results.test_set.n_attack.toLocaleString()} attacks.`
  );
  lines.push("");
  lines.push("## Headline vs baselines");
  lines.push("");
  lines.push("| Variant | AUPRC | TPR@0.1%FPR | TPR@1%FPR |");
  lines.push("| --- | --- | --- | --- |");
  const headlineRows: [string, HeadlineReport][] = [
    ["Fusion (deployed)", results.headline],
    ["Baseline: regex-only", results.baselines.regex_only],
    ["Baseline: max()", results.baselines.max],
  ];
  for (const [name, report] of headlineRows) {
    lines.push(
      `| ${name} | ${report.auprc.toFixed(3)} | ${formatBootstrap(
        report["tpr_at_fpr_0.100%"]
      )} | ${formatBootstrap(report["tpr_at_fpr_1.0%"])} |`
    );
  }
  lines.push("");
  lines.push(
    "Note: the regex-only baseline matches or slightly beats the full fusion system at these operating points."
  );
  lines.push("");
  lines.push("## Ablation diff: fusion vs. fusion without kNN sidecar");
  lines.push("");
  lines.push("| Variant | AUPRC | TPR@0.1%FPR | TPR@1%FPR |");
  lines.push("| --- | --- | --- | --- |");
  const ablationRows: [string, HeadlineReport][] = [
    ["Fusion (deployed)", results.headline],
    ["Fusion without kNN sidecar", results.ablations.remove_knn],
  ];
  for (const [name, report] of ablationRows) {
    lines.push(
      `| ${name} | ${report.auprc.toFixed(3)} | ${formatBootstrap(
        report["tpr_at_fpr_0.100%"]
      )} | ${formatBootstrap(report["tpr_at_fpr_1.0%"])} |`
    );
  }
  lines.push("");
  lines.push(
    "Note: this is an ablation comparison between two real, already-computed fusion variants, not two different trained model versions."
  );
  lines.push("");
  lines.push("## Latency (full cascade)");
  lines.push("");
  lines.push(
    `p50 ${results.latency_ms.p50.toFixed(1)}ms, p95 ${results.latency_ms.p95.toFixed(
      1
    )}ms, p99 ${results.latency_ms.p99.toFixed(1)}ms (n=${results.latency_ms.n.toLocaleString()}).`
  );
  lines.push("");
  lines.push("## Adaptive attacks");
  lines.push("");
  lines.push(
    `Paraphrase: seed evasion rate ${formatPct(
      results.adaptive.paraphrase.seed_evasion_rate
    )} across ${results.adaptive.paraphrase.n_seeds.toLocaleString()} seeds.`
  );
  lines.push(
    `Black-box query: evasion rate ${formatPct(
      results.adaptive.blackbox_query.evasion_rate
    )} across ${results.adaptive.blackbox_query.n_seeds.toLocaleString()} seeds (max ${results.adaptive.blackbox_query.max_queries} queries each).`
  );
  lines.push("");
  lines.push("## Raw eval-results.json");
  lines.push("");
  lines.push("```json");
  lines.push(JSON.stringify(results, null, 2));
  lines.push("```");
  return lines.join("\n");
}

function downloadTextFile(filename: string, contents: string, mimeType: string) {
  const blob = new Blob([contents], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  document.body.removeChild(anchor);
  URL.revokeObjectURL(url);
}

export default function RedTeamConsolePage() {
  const [activeTab, setActiveTab] = useState("headline");
  const [ablationThreshold, setAblationThreshold] = useState(0.5);

  const {
    data: results,
    isLoading: resultsLoading,
    isError: resultsError,
  } = useQuery({
    queryKey: ["eval-results"],
    queryFn: getEvalResults,
    staleTime: Infinity,
    retry: false,
  });

  function handleExport() {
    if (!results) return;
    const markdown = buildReportMarkdown(results, activeTab, ablationThreshold);
    const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
    downloadTextFile(`portcullis-redteam-report-${timestamp}.md`, markdown, "text/markdown");
  }

  return (
    <div className="p-4">
      <div className="mb-4 flex flex-wrap items-start justify-between gap-3">
        <div>
          <h1 className="font-mono text-lg font-semibold tracking-tight">
            Red-Team Console
          </h1>
          <p className="max-w-2xl text-sm text-muted-foreground">
            Real results from Milestone 9&apos;s offline evaluation harness
            against the ~20k-row test corpus - not a live re-run from this
            page. The model diff below is an ablation comparison between two
            real, already-fit fusion variants; only one trained checkpoint
            exists.
          </p>
        </div>
        <div className="flex items-center gap-2">
          {results && (
            <Badge variant="outline" className="font-mono text-[11px]">
              elapsed {results.elapsed_s.toFixed(1)}s
            </Badge>
          )}
          <Button
            variant="outline"
            size="sm"
            onClick={handleExport}
            disabled={!results}
          >
            Export report
          </Button>
        </div>
      </div>

      <Separator className="mb-4" />

      {resultsLoading ? (
        <LoadingPanel label="Loading eval results..." />
      ) : resultsError || !results ? (
        <MissingArtifactPanel what="Eval results" />
      ) : (
        <Tabs value={activeTab} onValueChange={(v) => setActiveTab(String(v))}>
          <TabsList>
            <TabsTrigger value="headline">Headline &amp; Baselines</TabsTrigger>
            <TabsTrigger value="ablation">Ablation Diff</TabsTrigger>
            <TabsTrigger value="adaptive">Adaptive Attacks</TabsTrigger>
          </TabsList>
          <TabsContent value="headline">
            <HeadlineTab results={results} />
          </TabsContent>
          <TabsContent value="ablation">
            <AblationTab
              results={results}
              threshold={ablationThreshold}
              onThresholdChange={setAblationThreshold}
            />
          </TabsContent>
          <TabsContent value="adaptive">
            <AdaptiveTab results={results} />
          </TabsContent>
        </Tabs>
      )}
    </div>
  );
}
