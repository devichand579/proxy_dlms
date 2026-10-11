# Paper-informed direction for an activation-only count controller

Source: [Wainwright, The data geometry of masking diffusion](https://arxiv.org/html/2608.13520).
The paper connects reveal-path dependence geometry to random-subset sampling
error. Its strongest certification uses accurate posterior denoisers, clean
samples, and controlled KL-increment tails. It does not certify our activation
metric, position selector, or greedy code accuracy.

## Proposed experiment, not a result of the paper

The current latent-flow scheduler has a hand-selected progress reward and a
per-step subset optimizer. It does not explicitly allocate the 128-step budget
according to changing difficulty along the generation path. A separate latent
growth clock would test that missing component while retaining a fully internal
scorer and count rule.

1. At several generation states, construct nested tentative reveal sets of
   different sizes. Keep the surviving target positions fixed while comparing
   activation changes. Use frozen model proposals and no reference answers.
2. Record response spectra and continuous activation changes in an intrinsic
   metric. Normalize by the actual change in reveal fraction. This is a proposed
   activation-based geometry proxy, not a proven UGC estimator.
3. Build a nonnegative local profile over reveal progress. Test count allocation
   using approximately equal square-root-profile mass per remaining step.
   Convert that progress increment to a token count, with explicit integer and
   deadline handling. Do not borrow EB's entropy budget.
4. Compare constant count, the current latent subset controller, profile-derived
   count with the same latent scorer, and jointly adapted count/selection.
   Report full probe cost, forced pacing choices, and unfinished output.

The square-root-profile construction is inspired by the paper's scalar path
allocation result. It is not the historical square-root matrix ranking rule,
and it supplies no guarantee for our proxy or our chosen nonuniform path.

## First validation gate

Before LLaDA, use enumerable independent, repeated-bit, parity, and hierarchical
toy distributions. Compute their exact geometry for evaluation only. Check
whether the proposed latent proxy identifies the correct difficult phases and
whether its count schedule improves actual sampling error at a fixed step budget.
No oracle profile enters the deployed model scheduler.

The useful test is not simply whether an activation trace gets smaller: the
earlier curvature and raw-flow pilots already showed that continuous latent
measures can improve without improving answers. A proxy must match the location
of real difficulty, predict useful decisions, and survive incorrect-anchor cases.

This proposal is recorded separately from the current frozen HumanEval run.
Implementing it or tuning its parameters midway would change the experiment.
