# Method

[README](../README.md) · [Targets](targets.md) · [Integration](integration.md)

Diagnostic Loss (DLoss) adds a sequence-level term to the usual token-level
cross-entropy. For each training example it takes a set of **supported**
diagnostic concepts (stated in the reference finding) and **unsupported**
alternatives (other findings for the same organ), and pools their probability
over the report positions with two different operators:

- a supported concept should get enough probability **at some position**;
- an unsupported concept should stay unlikely **at every position**.

In the paper each example is one organ finding, so supervision for one organ is
never satisfied or contradicted by text about another.

## Probability mass of a concept

Let $\mathcal{P}_i$ be the report positions of example $i$ and
$\mathcal{V}_{i,d}$ the vocabulary IDs of concept $d$. With
$p_{i,t} = \mathrm{softmax}(z_{i,t})$,

$$
m_{i,d,t} = \sum_{v \in \mathcal{V}_{i,d}} p_{i,t}(v).
$$

$\mathcal{V}_{i,d}$ is a set of token IDs: a multi-token phrase contributes all
of its pieces, and the loss does not look for the exact phrase sequence.

## Pooling over positions

Supported concepts use noisy-OR, the probability that the concept appears at
least once:

$$
q^+_{i,d} = 1 - \prod_{t \in \mathcal{P}_i} \left(1 - m_{i,d,t}\right),
\qquad
L^+_{i,d} = -\log q^+_{i,d}.
$$

Unsupported concepts use a temperature-scaled log-sum-exp, which approaches the
maximum over positions as $\tau$ grows:

$$
L^-_{i,d} = \frac{1}{\tau} \log \sum_{t \in \mathcal{P}_i} \exp\left(\tau \log m_{i,d,t}\right),
\qquad \tau > 0.
$$

$L^-$ is not normalized by the number of positions and can be negative.

## Reduction

Each branch is a weighted mean over its concepts ($w$ are concept weights,
all 1 in the paper):

$$
L_i^\pm = \frac{\sum_{d \in \mathcal{D}_i^\pm} w_{i,d}^\pm L_{i,d}^\pm}{\sum_{d \in \mathcal{D}_i^\pm} w_{i,d}^\pm},
\qquad L_i^\pm = 0 \text{ if the branch is empty.}
$$

With sample weight $s_i$, the batch loss averages over the examples that have
any targets, $\mathcal{I} = \{i : \mathcal{D}_i^+ \cup \mathcal{D}_i^- \neq \varnothing\}$:

$$
\mathcal{L}_{\mathrm{diag}} = \frac{1}{|\mathcal{I}|} \sum_{i \in \mathcal{I}} s_i \left(\lambda_+ L_i^+ + \lambda_- L_i^-\right),
\qquad
\mathcal{L} = \mathcal{L}_{\mathrm{CE}} + \mathcal{L}_{\mathrm{diag}}.
$$

$\mathcal{L}_{\mathrm{diag}} = 0$ when no example in the batch has targets. An
example with $s_i = 0$ still counts in $|\mathcal{I}|$.

## In the code

| Paper | Code |
| --- | --- |
| $\lambda_+$, $\lambda_-$, $\tau$ | `DiagnosticLossConfig(lambda_pos, lambda_neg, tau)` |
| $\mathcal{P}_i$ | `report_mask[i]` |
| $\mathcal{D}_i^+$, $\mathcal{D}_i^-$ | `SampleTargets.positive`, `SampleTargets.negative` |
| $\mathcal{V}_{i,d}$, $w_{i,d}$ | `ConceptTarget.token_ids`, `ConceptTarget.weight` |
| $s_i$ | `SampleTargets.sample_weight` |
| $\lambda_+ \cdot$ mean $s_i L_i^+$, likewise for $-$ | `result.positive_term`, `result.negative_term` |

[`reference_diagnostic_loss`](../src/diagnostic_loss/loss.py) follows the
equations above with explicit loops. [`diagnostic_loss`](../src/diagnostic_loss/loss.py)
computes the same values in batched form and is the one to train with; the
tests check both against each other and against a direct transcription with
`torch.softmax` and `torch.prod`, including gradients.

### Numerics

- The softmax normalizer is computed once per position with `logsumexp`.
- Noisy-OR is evaluated as $-\log(-\mathrm{expm1}(\sum_t \mathrm{log1p}(-m)))$.
- Probabilities are clamped to $[\epsilon, 1 - \epsilon]$ before logarithms
  (`epsilon=1e-6` by default). `epsilon` must be at least the machine epsilon
  of the computation dtype.
- float16 and bfloat16 logits are evaluated in float32.
- Positions outside `report_mask` do not affect the value or the gradient, even
  if they hold NaN or infinity.

## Limitations

Token IDs are lexical evidence, not meaning. Subword pieces can be shared with
unrelated words, and negation ("no cyst") and uncertainty are not modelled. The
quality of the supervision depends on how the target concepts and their lexical
forms are chosen; see [Targets](targets.md).
