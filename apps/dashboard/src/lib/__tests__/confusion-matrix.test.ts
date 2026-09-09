import { describe, expect, it } from "vitest";
import { computeConfusionMatrix } from "../confusion-matrix";

describe("computeConfusionMatrix", () => {
  it("throws when labels and scores have different lengths", () => {
    expect(() => computeConfusionMatrix([0, 1], [0.1], 0.5)).toThrow(
      /same length/
    );
  });

  it("classifies every quadrant correctly at a fixed threshold", () => {
    // label=1 is "attack", label=0 is "benign" - matches the gateway's
    // own policy.py convention (ADR-0006/ADR-0009).
    const labels = [1, 1, 0, 0];
    const scores = [0.9, 0.3, 0.9, 0.1];
    const result = computeConfusionMatrix(labels, scores, 0.5);

    // row 0: attack, score>=0.5 -> true positive
    // row 1: attack, score<0.5  -> false negative
    // row 2: benign, score>=0.5 -> false positive
    // row 3: benign, score<0.5  -> true negative
    expect(result.truePositive).toBe(1);
    expect(result.falseNegative).toBe(1);
    expect(result.falsePositive).toBe(1);
    expect(result.trueNegative).toBe(1);
  });

  it("treats a score exactly at the threshold as predicted-attack (inclusive)", () => {
    // Matches the gateway's own boundary convention: a score exactly at
    // the threshold gets the stricter (attack) classification, not the
    // more lenient one - see core/policy/policy.py's decide().
    const result = computeConfusionMatrix([1], [0.5], 0.5);
    expect(result.truePositive).toBe(1);
    expect(result.falseNegative).toBe(0);
  });

  it("computes tpr as recall over actual positives", () => {
    const labels = [1, 1, 1, 1, 0];
    const scores = [0.9, 0.9, 0.9, 0.1, 0.1]; // 3 of 4 attacks caught
    const result = computeConfusionMatrix(labels, scores, 0.5);
    expect(result.tpr).toBeCloseTo(0.75);
  });

  it("computes fpr as the false-alarm rate over actual negatives", () => {
    const labels = [0, 0, 0, 0, 1];
    const scores = [0.9, 0.1, 0.1, 0.1, 0.9]; // 1 of 4 benign rows flagged
    const result = computeConfusionMatrix(labels, scores, 0.5);
    expect(result.fpr).toBeCloseTo(0.25);
  });

  it("computes precision as tp / (tp + fp)", () => {
    const labels = [1, 1, 0, 0];
    const scores = [0.9, 0.9, 0.9, 0.1]; // 2 true positives, 1 false positive
    const result = computeConfusionMatrix(labels, scores, 0.5);
    expect(result.precision).toBeCloseTo(2 / 3);
  });

  it("returns zero rates rather than NaN when there are no positives", () => {
    const result = computeConfusionMatrix([0, 0], [0.9, 0.1], 0.5);
    expect(result.tpr).toBe(0);
    expect(Number.isNaN(result.tpr)).toBe(false);
  });

  it("returns zero rates rather than NaN when there are no negatives", () => {
    const result = computeConfusionMatrix([1, 1], [0.9, 0.1], 0.5);
    expect(result.fpr).toBe(0);
    expect(Number.isNaN(result.fpr)).toBe(false);
  });

  it("returns zero precision rather than NaN when nothing is predicted positive", () => {
    const result = computeConfusionMatrix([1, 0], [0.1, 0.1], 0.5);
    expect(result.precision).toBe(0);
  });

  it("a threshold of 0 predicts every row as an attack", () => {
    const result = computeConfusionMatrix([1, 0], [0.5, 0.0], 0.0);
    expect(result.truePositive).toBe(1);
    expect(result.falsePositive).toBe(1);
    expect(result.trueNegative).toBe(0);
  });

  it("a threshold above every score predicts every row as benign", () => {
    const result = computeConfusionMatrix([1, 0], [0.5, 0.3], 1.0);
    expect(result.truePositive).toBe(0);
    expect(result.trueNegative).toBe(1);
    expect(result.falseNegative).toBe(1);
  });

  it("handles a realistic mixed batch consistently across all four counts", () => {
    const labels = [1, 0, 1, 0, 1, 0, 0, 1];
    const scores = [0.82, 0.15, 0.41, 0.63, 0.91, 0.05, 0.5, 0.2];
    const result = computeConfusionMatrix(labels, scores, 0.5);
    const total =
      result.truePositive +
      result.falsePositive +
      result.trueNegative +
      result.falseNegative;
    expect(total).toBe(labels.length);
  });
});
