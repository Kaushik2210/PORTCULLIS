/**
 * The threshold slider's core computation (ADR-0011, Decision 2 and 5):
 * given the eval set's real (label, score) pairs and a threshold, compute
 * a confusion matrix instantly, client-side. Pure, no I/O - this is the
 * kind of logic the working agreement's "test detection-adjacent logic"
 * instinct applies to on the TypeScript side too.
 */

export interface ConfusionMatrix {
  truePositive: number;
  falsePositive: number;
  trueNegative: number;
  falseNegative: number;
}

export interface ConfusionMatrixWithRates extends ConfusionMatrix {
  tpr: number;
  fpr: number;
  precision: number;
}

/**
 * A row is predicted "attack" when its score is >= threshold - matching
 * the gateway's own policy.py convention (ADR-0006/ADR-0009): boundaries
 * are inclusive on the stricter side.
 */
export function computeConfusionMatrix(
  labels: number[],
  scores: number[],
  threshold: number
): ConfusionMatrixWithRates {
  if (labels.length !== scores.length) {
    throw new Error(
      `labels (${labels.length}) and scores (${scores.length}) must be the same length`
    );
  }

  let truePositive = 0;
  let falsePositive = 0;
  let trueNegative = 0;
  let falseNegative = 0;

  for (let i = 0; i < labels.length; i++) {
    const predictedAttack = scores[i] >= threshold;
    const actualAttack = labels[i] === 1;
    if (predictedAttack && actualAttack) truePositive++;
    else if (predictedAttack && !actualAttack) falsePositive++;
    else if (!predictedAttack && actualAttack) falseNegative++;
    else trueNegative++;
  }

  const positives = truePositive + falseNegative;
  const negatives = trueNegative + falsePositive;
  const predictedPositives = truePositive + falsePositive;

  return {
    truePositive,
    falsePositive,
    trueNegative,
    falseNegative,
    tpr: positives > 0 ? truePositive / positives : 0,
    fpr: negatives > 0 ? falsePositive / negatives : 0,
    precision: predictedPositives > 0 ? truePositive / predictedPositives : 0,
  };
}
