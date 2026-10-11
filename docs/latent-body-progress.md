# Latent scoring and useful code-body parallelism

8 October 2026. Exploratory experiments with frozen local LLaDA-8B Base.

## Hypotheses and controls

The revised objective separates useful completion benefit U, readiness risk M,
and within-batch interaction D. Extra commitments outside the decoded program
are weak evidence about U. The previous independent Fisher-dispersion count
matched Fisher singleton's final programs, so this study measures commitment
timing and correctness of earlier saved canvases as well as final pass rate.

One confidence-free readiness proposal uses the final block's observed-key value
contribution. Let h_i be the final normalized representation, w_y the frozen
decoder row for the current greedy proposal, and w_bar the decoder-row mean.
Differentiate the unit contrast d_y=(w_y-w_bar)/||w_y-w_bar|| with respect to the
post-attention residual u_i through the final tokenwise MLP and final RMS norm:

    g_i = derivative_(u_i) [d_y^T h_i].

Reconstruct native final attention and sum only value contributions from the
originally observed input keys, applying the actual attention output projection.
Call this contribution a_i^obs and the total attention contribution a_i^all.
The proposed support score is

    s_i = g_i^T a_i^obs / (||g_i|| ||a_i^all||),

with a small shared numerical floor and clipping. Normalize signed support over
remaining positions and subtract the existing uniform decoder-projection temporal
movement penalty. A cosine normalization against ||a_i^obs|| is an arithmetic
ablation. Neither uses a raw confidence or entropy term for ranking.

This is a local mechanistic proxy for observed-context support. Attention is held
fixed for the derivative; originally observed values already contain earlier-layer
context. It does not estimate whole-model causal prompt influence or true model
error. The loss value is not used for ranking; its derivative is used. Selecting
the proposal still uses the normal decoder, as every greedy scheduler does.

The computation requires one normal backbone forward plus one partial backward
through the final MLP and final norm per step. It performs no full-model backward,
weight update, auxiliary fitting, or counterfactual model forward. Hooks and frozen
parameter flags are restored. Tests check unchanged forwards, no parameter grads,
common decoder-row gauge invariance, and correct reveal/work accounting.

## Count and pool ablations

The independent allowance count uses cached temporal decoder-gradient displacement
as a readiness proxy and native attention as a pair-interaction proxy. It selects
the largest ranked prefix within the empirical allowance, with mandatory singleton
progress logged when no feasible nonempty prefix exists. The existing gradient-union
variant discards probability magnitudes and uses current/previous candidate decoder
rows. The Fisher-dispersion diagnostic instead uses model probabilities to weight
readout-gradient covariance. It is independent of EB's entropy count, but it is not
probability-free. These estimators are not certified KL or TC bounds.

Confidence/Fisher ranking is held fixed in count diagnostics. Small context-flow
perturbations of that ranking are also diagnostics; successful hybrids cannot be
claimed as confidence-free methods. Singleton controls isolate ranking quality.

Candidate-pool controls use either all remaining positions, only positions before
an actually committed EOS, or a fixed 32-position window starting at the first
remaining mask. EOS is a committed decoder value, not a prediction-confidence
threshold. The position window diagnoses the importance of output progress and
is not a wholly latent scheduler. Ranking normalizations remain over all remaining
positions, to avoid changing the score function when changing the pool.

No reference solutions, hidden tests, or final decoded boundaries enter generation.
Final boundaries are retrospective audit labels. Visible-body readiness can omit
mask holes and does not certify that later generation will preserve the decode
boundary. A safe EOS completion additionally requires a fully known prefix.

## Evaluation and artifacts

The completed pilot contains 12 valid HumanEval arms and 16 corrected arithmetic
trajectories. Both historical singleton controls reproduced all 20 programs.
Source snapshots, work counters, unique reveals and final-token accounting passed.

Independent Fisher-dispersion count retained 14/20 and all 20 final singleton
programs, but advanced mean visible-body commitment from step 48.1 to 38.0:
18 examples were earlier and two equal. At a replayed step 48 it passed 13/20
versus singleton's 9/20. It made 117 multi-token decisions inside the visible
body. This is useful progress under confidence ranking, not a final accuracy
gain or an automatic stopping mechanism. All primary runs still used 128 forwards.

The corrected context-flow scorer passed 0/20 with both singleton and the
probability-free adaptive allowance, versus the activation singleton's 7/20.
Adding a small flow perturbation to Fisher retained 14/20 with extra partial
backward work. EOS gating and frontier windows retained 14/20 under Fisher;
the frontier controls reached 14/20 already at replayed step 64. Their position
restriction is an explicit diagnostic, not a wholly latent method. Activation
allowance plus EOS gating passed 6/20. No new wholly confidence-free method
improved on the earlier activation-only control.

Arithmetic also rejects the standalone gradient scorer: about 47--49% answer
accuracy versus activation singleton's 58.85%/57.81%. The gradient hybrid's
larger seed-0 results did not survive seed 1. This supports dropping observed-key
support as the main readiness signal rather than scaling this scorer to 164 tasks.

Arithmetic uses 32 canvases, 384 masked answers per seed, seeds 0/1, and a 12-step
cap. HumanEval uses the first 20 problems, a 512-position canvas, a 128-step cap,
and matched standalone batch 20. This is exploratory tuning on previously exposed
problems. Every study saves source snapshots, geometry, reveals, final tokens,
work counters, and isolated-child correctness outcomes.

[HumanEval outcome and paired comparisons](/root/Strace/outputs/latent_body_progress_report/README.md).
[Body commitment audit](/root/Strace/outputs/latent_body_progress_audit/README.md).
[Earlier-step correctness replay](/root/Strace/outputs/latent_body_progress_budget_replay/README.md).
[Corrected arithmetic runs](/root/Strace/outputs/latent_context_flow_corrected_arithmetic).

Replay reconstructs earlier canvases from unique monotone commitments, and uses
exactly the existing decoder/stop-marker rules. It requires no new model passes.
Test outcomes evaluate those saved canvases and never determine a stopping policy.
One 128-step batch runtime measurement per arm is descriptive, not a replicated
latency benchmark. Partial backward work is accounted separately.

The initial gradient hook replaced the feed-forward input with a new detached
tensor, dropping the retained residual's identity derivative. That context-flow
run is excluded from conclusions and preserved with its source snapshot. The
corrected hook preserves tensor identity; an independent calculation checks the
complete final residual derivative. Corrected runs use a fresh output directory.

Run new generation directories using `schedulerlens.latent_allowance_bench`,
then `schedulerlens.latent_body_audit`, `schedulerlens.latent_body_report`, and
`schedulerlens.latent_budget_replay`. Generation source changes require a fresh
output directory. Snapshot/work/reveal checks must pass before interpreting results.
