# Directed latent flow for both scoring and count

The user's constraint is activation-only scheduling: no EB controller, token
confidence, entropy, probability margin, learned readout, or learned policy.
The frozen decoder supplies proposed token values. All scheduler weights and
count decisions are computed from continuous representations and their actual
responses to those values.

## Continuous quantities

For source i, reveal either of its two highest-logit proposed values in separate
temporary canvases. Logit magnitudes and probabilities are discarded; the two
branches have equal weights. Capture the final normalized state at target j:
h_j^(i,0) and h_j^(i,1). An unchanged canvas with the same probe batch shape
supplies h_j^control. Define

    C_ij = ¼ mean_d (h_j^(i,0) − h_j^(i,1))²
    B_ij = mean_d (½[h_j^(i,0)+h_j^(i,1)] − h_j^control)².

C measures sensitivity to the source value; B measures latent mean-update
inconsistency. Both are continuous activation diagnostics. They are not true MI,
calibrated errors, or estimates of a data-distribution joint. Set C_ii=B_ii=0:
the source's own activation change is not counted as external information.

Outgoing effect is o_i=Σ_j C_ij. Incoming effect is u_i=Σ_s C_si. The consistency
cost is d_i=Σ_j B_ij. For a restricted source pool, incoming totals are scaled by
the masked-target/source-pool ratio; this is an explicit representativeness
approximation, not measured influence from unprobed positions. The arithmetic
run probes every masked position, so no such extrapolation is needed there.

If the mask set exceeds the compute pool cap, build that pool by covering
distinct regularized spectral modes of centered hidden activations. This
candidate screen also uses no output probabilities. The initial arithmetic
experiment caps the pool at 12, covering all 12 answer positions.

## Joint objective

With a common scale s=mean(o_i), use

    F(v) = Σ_i v_i [(o_i − κu_i − δd_i)/s + τ]
           − Σ_ij v_i v_j C_ij/s.

The outgoing term rewards external latent influence; the incoming term delays
positions that respond to other source values. The pair term subtracts influence
delivered to positions that are selected together and therefore no longer remain
masked. τ rewards progress explicitly, including terminal actions with no
surviving targets. Initial κ=0.5, τ=0.05; δ=0 or 0.15 tests the inconsistency cost.
These are hand-selected objective coefficients, not trained weights.

Within the pool, let W=(C_pool+C_poolᵀ)/s and D=diag(W1). The signless Laplacian
Q=D+W is PSD. With

    g_i = (o_i−κu_i−δd_i)/s + τ + ½D_ii,

the exactly equivalent binary objective is

    F(v)=gᵀv−½vᵀQv.

The diagonal compensation is necessary because v_i²=v_i. Removing it changes
the objective. Decompose Q=U diag(λ) Uᵀ and evaluate Σ_m λ_m(Uᵀv)_m². Compare all
subsets (4096 at 12 sources), including the empty subset. The winner provides
both the selected positions and the count. If the empty subset wins, force the
best singleton and record the override. No count is taken from EB or an entropy
threshold; the cap is a model-probe compute limit.

This adapts the supplied memo's quadratic formulation using measured directed
latent responses. The original PDF and spectral implementation are unavailable
in this checkout. The recorded square-root ranking rule is not reused, and the
eigenvectors do not receive an unsupported optimality interpretation: the binary
surrogate is solved exactly within the pool.

## Difference from the unsuccessful curvature controller

The earlier activation-only prototype optimized contraction of late-layer
curvature. Its arithmetic tests were below EB, including variants that corrected
surviving-target bookkeeping and diversified the candidate pool. Smoother
representations did not establish better generation. This flow candidate instead
uses actual responses across token-value hypotheses to identify directed
dependencies, without assuming that small layer curvature means correctness.

Neither contrast energy nor consistency alone guarantees correct commitments.
In particular, equal-weight alternatives can include unlikely values. Completion
accuracy must decide whether this is useful. No reference tokens or tests enter
the scheduler. References are evaluated only after generation.

## Checks and experiment

Three targeted checks establish the exact PSD/direct-objective equivalence,
parallel upstream selection on two independent latent dependency chains, and
invariance when output confidence changes without changing activations or
proposed values. Integration checks preserve prompts and verify all branch and
control sequence work. The eight residual-controller checks cover shared
spectral optimization and pool helpers separately.

    python -m schedulerlens.latent_residual_bench --method flow --candidates 12 \
      --rewards 0,.15 --out outputs/latent_flow_arithmetic12

The runner's `--rewards` values select δ (consistency cost) in flow mode, not τ;
artifacts record the actual coefficients. Uses the existing two arithmetic
seeds, 32 canvases, 384 answer slots, and a 12-step cap. Native EB and the previous
EB+Fisher method are benchmark controls only. All model evaluations, including
unchanged controls and both source values, count toward reported work.

This is a new independent implementation and research candidate, not an
established paper-level novelty claim. Related work includes
[Info-Gain](https://arxiv.org/html/2602.18176v3),
[Stability-Weighted Decoding](https://arxiv.org/abs/2604.17068), and
[D5P4](https://arxiv.org/abs/2603.19146). A complete novelty argument needs a
broader comparison and positive mechanism/performance evidence.

## Completed pilot results

Raw activation flow with δ=0 scored **54.69% / 53.91%** on seeds 0/1, costing
122.00 / 122.125 forwards/canvas. With δ=0.15 it scored 54.43% / 54.43%, costing
132.00 / 134.25 forwards/canvas. Native EB scored 71.09% / 69.79% at 12 forwards.
Every answer slot was filled. Neither candidate was promoted to HumanEval.

A focused metric ablation derives a regularized PCA whitening basis from the
current masked source representations. It uses their activation covariance,
not output probabilities or a trained projection. Contrasts and mean-update
defects are measured in that basis. The projected variant scored
**55.47% / 55.21%**, costing 109.75 / 114.375 forwards/canvas. This also fails the
performance gate. An additional check verifies rotation invariance and rejection
of orthogonal nuisance directions for the derived metric.

Reproduce with `--geometry pca --rewards 0` and a separate output directory.
The report at `/root/Strace/outputs/latent_only_ablation/README.md` contains all
completed arms, work, and constraints. The implementation meets activation-only
scoring/count requirements, but the tested latent signal does not yet support
superior generation performance.

## Completed HumanEval-20 result

The PCA latent-flow policy was evaluated on the first 20 HumanEval problems with
all parameters frozen, a 512-position canvas and exactly 128 outer steps.
It scored **4/20 (20%)**, versus matched native EB **13/20** and temporal Fisher
**14/20**. It has zero wins and nine losses against EB. Ten completions have
invalid syntax; six have valid syntax but fail tests. It commits on average
290.8 positions, and forces singletons in 414 of 2560 sample-step decisions.
These results reject this candidate as a performance improvement.

Actual work is **3462.8 forward evaluations/problem**, including 3070.8 branch
evaluations and 264 unchanged controls, plus 128 normal evaluations. The first
11 completed steps used unchanged controls; their defect has zero objective
weight. After policy-equivalence checks, unused controls were removed and the
run resumed from its durable checkpoint. All earlier work remains in the total,
and both source-value branches remain. Removing unused controls does not change
the scorer or count rule. No training, backward passes or confidence scoring
were introduced. Completed generation took 2563 seconds; peak allocated memory
was 118.40 GiB. Interrupted in-flight/restart overhead is not included in these
completed-generation counters.

Artifacts: `/root/Strace/outputs/latent_pca_flow_humaneval20_lean/README.md`,
`diagnostic.json`, `comparison.json`, and the method's `results.json`. The manifest
records both source-code phases. Original controlled-phase checkpoints remain
in `/root/Strace/outputs/latent_pca_flow_humaneval20/`.

The separate paper-informed count proposal is in
`/root/Strace/docs/ugc-count-direction.md`. It was not applied to this run.
