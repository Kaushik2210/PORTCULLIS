"use client";

import { useEffect, useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { cn } from "cn";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Slider } from "@/components/ui/slider";
import { Separator } from "@/components/ui/separator";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  getEvalScores,
  getRecentDecisions,
  subscribeToDecisions,
  type DecisionLogEntry,
  type Verdict,
} from "@/lib/api";
import { computeConfusionMatrix } from "@/lib/confusion-matrix";

const MAX_FEED_LENGTH = 100;

const VERDICT_BADGE_CLASS: Record<Verdict, string> = {
  allow: "bg-verdict-allow/10 text-verdict-allow",
  flag: "bg-verdict-flag/10 text-verdict-flag",
  sanitise: "bg-verdict-sanitise/10 text-verdict-sanitise",
  challenge: "bg-verdict-challenge/10 text-verdict-challenge",
  block: "bg-verdict-block/15 text-verdict-block",
};

/** Relative-time formatter for the feed - "3s ago", "2m ago", etc. */
function formatRelativeTime(timestampSeconds: number, nowMs: number): string {
  const deltaMs = nowMs - timestampSeconds * 1000;
  const deltaS = Math.max(0, Math.round(deltaMs / 1000));
  if (deltaS < 1) return "now";
  if (deltaS < 60) return `${deltaS}s ago`;
  const deltaMin = Math.round(deltaS / 60);
  if (deltaMin < 60) return `${deltaMin}m ago`;
  const deltaHr = Math.round(deltaMin / 60);
  return `${deltaHr}h ago`;
}

/**
 * Merges freshly-streamed decisions into the existing feed, de-duplicating
 * by id and keeping the newest MAX_FEED_LENGTH entries at the front.
 */
function prependDecision(
  existing: DecisionLogEntry[],
  incoming: DecisionLogEntry
): DecisionLogEntry[] {
  const withoutDuplicate = existing.filter((entry) => entry.id !== incoming.id);
  return [incoming, ...withoutDuplicate].slice(0, MAX_FEED_LENGTH);
}

function DecisionFeed({
  feed,
  isLoading,
  isError,
}: {
  feed: DecisionLogEntry[];
  isLoading: boolean;
  isError: boolean;
}) {
  const [nowMs, setNowMs] = useState(() => Date.now());

  // Tick the clock every second so "Ns ago" labels stay fresh.
  useEffect(() => {
    const interval = setInterval(() => setNowMs(Date.now()), 1000);
    return () => clearInterval(interval);
  }, []);

  return (
    <Card className="flex h-full flex-col">
      <CardHeader>
        <CardTitle>Decision feed</CardTitle>
        <CardDescription>
          Live detections from the gateway, newest first.
        </CardDescription>
      </CardHeader>
      <CardContent className="flex-1 min-h-0 px-0">
        {isLoading ? (
          <p className="px-4 text-sm text-muted-foreground">Loading recent decisions...</p>
        ) : isError ? (
          <p className="px-4 text-sm text-verdict-block">
            Could not reach the gateway for recent decisions.
          </p>
        ) : feed.length === 0 ? (
          <p className="px-4 text-sm text-muted-foreground">
            No decisions yet - send a request through the gateway.
          </p>
        ) : (
          <ScrollArea className="h-[560px]">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Time</TableHead>
                  <TableHead>Verdict</TableHead>
                  <TableHead>Score</TableHead>
                  <TableHead>Taxonomy</TableHead>
                  <TableHead>Source</TableHead>
                  <TableHead className="text-right">Latency</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {feed.map((entry) => (
                  <TableRow key={entry.id}>
                    <TableCell className="whitespace-nowrap font-mono text-xs text-muted-foreground">
                      {formatRelativeTime(entry.timestamp, nowMs)}
                    </TableCell>
                    <TableCell>
                      <Badge
                        className={cn(
                          "uppercase",
                          VERDICT_BADGE_CLASS[entry.verdict]
                        )}
                        variant="outline"
                      >
                        {entry.verdict}
                      </Badge>
                    </TableCell>
                    <TableCell className="font-mono text-xs">
                      {entry.score.toFixed(3)}
                    </TableCell>
                    <TableCell className="max-w-[220px]">
                      {entry.taxonomy_labels.length === 0 ? (
                        <span className="text-xs text-muted-foreground">-</span>
                      ) : (
                        <div className="flex flex-wrap gap-1">
                          {entry.taxonomy_labels.slice(0, 3).map((label) => (
                            <Badge key={label} variant="secondary" className="text-[10px]">
                              {label}
                            </Badge>
                          ))}
                          {entry.taxonomy_labels.length > 3 && (
                            <span className="text-xs text-muted-foreground">
                              +{entry.taxonomy_labels.length - 3}
                            </span>
                          )}
                        </div>
                      )}
                    </TableCell>
                    <TableCell className="text-xs text-muted-foreground">
                      {entry.source}
                    </TableCell>
                    <TableCell className="text-right font-mono text-xs">
                      {entry.latency_ms.toFixed(0)}ms
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </ScrollArea>
        )}
      </CardContent>
    </Card>
  );
}

function TaxonomyBreakdown({ decisions }: { decisions: DecisionLogEntry[] }) {
  const chartData = useMemo(() => {
    const counts = new Map<string, number>();
    for (const decision of decisions) {
      for (const label of decision.taxonomy_labels) {
        counts.set(label, (counts.get(label) ?? 0) + 1);
      }
    }
    return Array.from(counts.entries())
      .map(([label, count]) => ({ label, count }))
      .sort((a, b) => b.count - a.count)
      .slice(0, 8);
  }, [decisions]);

  return (
    <Card>
      <CardHeader>
        <CardTitle>Attack taxonomy</CardTitle>
        <CardDescription>
          Label distribution across the loaded decisions.
        </CardDescription>
      </CardHeader>
      <CardContent>
        {chartData.length === 0 ? (
          <p className="text-sm text-muted-foreground">
            No taxonomy labels in the current feed.
          </p>
        ) : (
          <div className="h-[220px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={chartData}
                layout="vertical"
                margin={{ top: 4, right: 16, bottom: 4, left: 4 }}
              >
                <CartesianGrid horizontal={false} stroke="var(--color-border)" />
                <XAxis
                  type="number"
                  allowDecimals={false}
                  tick={{ fontSize: 11, fill: "var(--color-muted-foreground)" }}
                  axisLine={false}
                  tickLine={false}
                />
                <YAxis
                  type="category"
                  dataKey="label"
                  width={110}
                  tick={{ fontSize: 11, fill: "var(--color-muted-foreground)" }}
                  axisLine={false}
                  tickLine={false}
                />
                <Tooltip
                  cursor={{ fill: "var(--color-muted)" }}
                  contentStyle={{
                    background: "var(--color-card)",
                    border: "1px solid var(--color-border)",
                    borderRadius: 8,
                    fontSize: 12,
                  }}
                />
                <Bar
                  dataKey="count"
                  fill="var(--color-chart-3)"
                  radius={[0, 3, 3, 0]}
                  maxBarSize={14}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function StatTile({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-border p-3">
      <div className="text-[11px] uppercase tracking-wide text-muted-foreground">
        {label}
      </div>
      <div className="font-mono text-lg tabular-nums">{value}</div>
    </div>
  );
}

function ThresholdExplorer() {
  const [threshold, setThreshold] = useState(0.5);
  const {
    data: evalScores,
    isLoading,
    isError,
  } = useQuery({
    queryKey: ["eval-scores"],
    queryFn: getEvalScores,
    staleTime: Infinity,
    retry: false,
  });

  const matrix = useMemo(() => {
    if (!evalScores) return null;
    return computeConfusionMatrix(evalScores.labels, evalScores.fusion, threshold);
  }, [evalScores, threshold]);

  return (
    <Card>
      <CardHeader>
        <CardTitle>Threshold explorer</CardTitle>
        <CardDescription>
          Confusion matrix against the M9 eval set at the selected threshold.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {isLoading ? (
          <p className="text-sm text-muted-foreground">Loading eval scores...</p>
        ) : isError ? (
          <p className="text-sm text-verdict-block">
            Eval scores not available - run <code className="font-mono">just eval</code> first.
          </p>
        ) : (
          <>
            <div className="flex items-center gap-3">
              <Slider
                min={0}
                max={1}
                step={0.01}
                value={[threshold]}
                onValueChange={(value) =>
                  setThreshold(Array.isArray(value) ? value[0] : value)
                }
                className="flex-1"
              />
              <span className="w-14 shrink-0 text-right font-mono text-sm tabular-nums">
                {threshold.toFixed(2)}
              </span>
            </div>
            {matrix && (
              <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
                <StatTile label="TP" value={matrix.truePositive.toLocaleString()} />
                <StatTile label="FP" value={matrix.falsePositive.toLocaleString()} />
                <StatTile label="TN" value={matrix.trueNegative.toLocaleString()} />
                <StatTile label="FN" value={matrix.falseNegative.toLocaleString()} />
                <StatTile label="TPR" value={matrix.tpr.toFixed(3)} />
                <StatTile label="FPR" value={matrix.fpr.toFixed(3)} />
                <StatTile label="Precision" value={matrix.precision.toFixed(3)} />
              </div>
            )}
          </>
        )}
      </CardContent>
    </Card>
  );
}

export default function LiveTrafficPage() {
  // Feed state is hoisted to the page so both the feed table and the
  // taxonomy chart aggregate over the same single set of decisions
  // (initial batch + streamed entries), with exactly one SSE subscription.
  const [decisions, setDecisions] = useState<DecisionLogEntry[]>([]);
  const [hydrated, setHydrated] = useState(false);
  const {
    data: initialDecisions,
    isLoading,
    isError,
  } = useQuery({
    queryKey: ["recent-decisions"],
    queryFn: () => getRecentDecisions(50),
  });

  useEffect(() => {
    if (initialDecisions && !hydrated) {
      setDecisions(
        [...initialDecisions].sort((a, b) => b.timestamp - a.timestamp).slice(0, MAX_FEED_LENGTH)
      );
      setHydrated(true);
    }
  }, [initialDecisions, hydrated]);

  useEffect(() => {
    const unsubscribe = subscribeToDecisions((entry) => {
      setDecisions((current) => prependDecision(current, entry));
    });
    return unsubscribe;
  }, []);

  return (
    <div className="grid grid-cols-1 gap-4 p-4 lg:grid-cols-[2fr_1fr]">
      <div className="min-h-0">
        <DecisionFeed feed={decisions} isLoading={isLoading} isError={isError} />
      </div>
      <div className="flex flex-col gap-4">
        <TaxonomyBreakdown decisions={decisions} />
        <Separator />
        <ThresholdExplorer />
      </div>
    </div>
  );
}
