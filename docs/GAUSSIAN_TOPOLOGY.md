# Experimental Learned Gaussian Topology

Status: implemented prototype, not a trained model or a paper reproduction.
Scope: the feature-color `PointForward` base class in this project only.
Existing configs remain unchanged; the module is disabled unless explicitly enabled.

## Recent Related Work

Search cutoff: 2026-09-22. Discovery used arXiv search pages for Gaussian
densification and recurrent fusion. The five selected papers' abstract pages and
v1 HTML methods were manually checked. Dates below are first arXiv submissions
in UTC, not formal publication dates. Treat these as preprints here; venue
acceptance was not independently verified. This is a targeted search, not an
exhaustive novelty claim.

| Paper / First Author | First Submitted | Mechanism and Relevance |
| --- | --- | --- |
| [L2D2-GS: Learning to Densify for Feedforward Dynamic Gaussian Scene Reconstruction](https://arxiv.org/abs/2606.29374), Zetian Song | 2026-06-28 | Learned candidate insertion and residual Gaussian updates using rendering-gradient evidence. Closest driving-scene allocation reference, but inference includes an iterative learned optimization loop, not one-shot decoding. No explicit unified split/clone/merge classifier. |
| [VS-Splat: Voxel-Selective feed-forward Gaussian Splatting for end-to-end 3D object reconstruction from sparse-views](https://arxiv.org/abs/2609.12343), Yunsu Jeong | 2026-09-11 | Learned voxel selection, fine Gaussian spawning and skeletal appearance adjustment. Relevant to feedforward one-to-many decoding; object-centric voxels are not PointForward's depth-aligned scene queries. |
| [RecurGS: Interactive Scene Modeling via Discrete-State Recurrent Gaussian Fusion](https://arxiv.org/abs/2512.18386), Wenhao Hu | 2025-12-20 | Per-state scene optimization with semantic/geometric association, visibility-based keep/update/insert and replay. Relevant to persistent observations, but not a learned many-to-one topology controller. |
| [MMGS: 10x Compressed 3DGS through Optimal Transport Aggregation based on Multi-view Ranking](https://arxiv.org/abs/2605.19304), Beizhen Zhao | 2026-05-19 | Multi-view importance ranking, spatial aggregation with moment matching and principal-axis splitting. Relevant merge operator reference; topology is analytic/rule-based and optimized per scene. |
| [AdpSplit: Error-Driven Adaptive Splitting for Faster Geometry Discovery in 3D Gaussian Splatting](https://arxiv.org/abs/2605.06876), Yongjae Lee | 2026-05-07 | Image-error regions guide adaptive children; inherited cloning and geometric/color rules merge sibling proposals within each parent. Not learned topology, and not global deduplication of arbitrary existing Gaussians. |

No inspected paper establishes a single learned five-action controller. AdpSplit
already combines several operations heuristically, so merely combining the
operations is not a novelty claim. No reported benchmark numbers are reproduced
or asserted here. No own-method code URL was found in the inspected pages.
The automated `verify_papers.py` helper was unavailable: the audit record at
`../.aris/verify-papers/verified_papers.json` marks automation as UNVERIFIED,
separately from the successful manual primary-source checks.

## Integration

```text
depth-aligned sampled queries
  -> multi-view feature fusion
  -> GaussianDecoder
  -> GaussianTopology (new, optional)
  -> pose dynamic instances
  -> gsplat feature rasterization
  -> RGB refiner
```

The policy sees fused query features, log scales, and opacity. It predicts
`keep/update/split/clone/merge` logits and a learned reliability score. A separate
symmetric pair head scores compatible merge proposals. No target-image label is
used at inference, and no hand-coded "pole implies split" semantic rule exists.

| Action | Hard Output | Implementation |
| --- | --- | --- |
| keep | one Gaussian | Preserve attributes. Also the fallback for unmatched or rejected merge intent. |
| update | one Gaussian | Bounded residuals for center, scale, rotation, opacity, color feature and lifespan, conditioned on fused evidence. |
| split | two children replacing parent | Move oppositely along the rotated longest axis. Reduce only that axis's scale; retain thin cross-section. Learn the displacement fraction. |
| clone | parent position plus displaced child | Retain parent geometry and add a learned local displacement/refinement. Redistribute opacity across the two, so parent opacity is not unchanged. |
| merge | one Gaussian replacing two | Both endpoints must choose merge and pass the pair score. Learn reliability-weighted attributes and approximate covariance moments. |

Pair proposals use a spatial voxel grouping, not an N-by-N distance matrix.
They require different source views, identical instance IDs and source times,
nearby centers, compatible scales/orientations/lifespans, and similar fused
features. Each group contributes at most one disjoint pair, so a point cannot
be consumed by multiple merges. Points across voxel boundaries are not paired.
These conservative guards reduce risk; they do not prove two points are the
same physical surface. Static and dynamic points, or different instances, never
merge with each other. Child metadata stays aligned with its parent instance.

Fusion is approximate: blend sign-aligned quaternions, compute weighted first
and second moments, then retain diagonal covariance in that blended frame.
Opacity is averaged for duplicate observations, not added. Residual temporal
envelope differences can remain even with equal source times. Rendering is not
mathematically preserved by merge, split, or clone; image supervision is needed.

## Trainability And Budget

Hard argmax, Top-K membership, and voxel matching are not differentiable.
Training therefore renders all keep/update/split/clone candidates plus proposed
merges, modulating opacity by soft gates. A merge gate is
`p_left(merge) * p_right(merge) * sigmoid(pair_score)`; unused merge probability
returns to keep. The branch logits, pair score, reliability and continuous
parameters receive rendering gradients; candidate matching remains detached.
Initial hard inference chooses keep, but training is NOT an identity operation.
The modest keep bias avoids starving initial merge candidates below gsplat's
alpha cutoff on the smoke fixture; low-opacity inputs can still be clipped.

The auxiliary cost is expected output count divided by original query count:

```text
expected_count = N + sum(p_split + p_clone) - sum(pair_merge_gate)
loss = L1 + lpips_weight * LPIPS + topology_weight * expected_count / N
```

This is a soft rendering surrogate, not an unbiased gradient of hard topology.
`model.eval()` uses hard actions, so quality must be evaluated in BOTH modes.
Inference count is `N + n_split + n_clone - n_accepted_merge_pairs`, at most `2N`.
Soft training materialises up to `6N + N/2` candidates. This increases memory,
even when most gates are small. A default 32,768 input-point guard fails loudly
instead of silently truncating points; do not simply enable this on million-query
configs. The experimental config uses at most 30,720 inputs for 12 views and
24 instances. This is a deliberately small-budget pilot, not a fair replacement
for the original million-point benchmark setting.

## Run

Use the existing environment with torch, gsplat, timm, LPIPS and InfiniDepth.
Commands below are examples to launch explicitly; no training was started.

```bash
python -m scripts.train --config configs/variants/01_base_topology.yaml \
  --init-from /path/to/compatible_base_checkpoint.pt

python -m scripts.eval --config configs/variants/01_base_topology.yaml \
  --ckpt /path/to/trained_topology_checkpoint.pt --max-clips 10
```

`--init-from` loads compatible weights only, resets training step/optimizer,
and permits missing topology and frozen-depth keys. Other missing, unexpected
or shape-incompatible keys are rejected. Fusion dimensions/view count must match
the base checkpoint. Use `--resume` only for the same trained architecture and
optimizer layout, not to add topology to an old optimizer. These flags are
mutually exclusive. Do not load new weights with old generic render scripts
that bypass the canonical builder; the canonical `scripts.eval` supports this
config. SH and Rec topology configs are explicitly rejected by the builder.

For diagnostics, `model(scene, targets, return_gaussians=True)` returns the new
Gaussians and `topology_cost`. Per-output `extras` include `topology_action`
(0=keep, 1=update, 2=split, 3=clone, 4=merge), `topology_parent`,
`topology_partner` (-1 unless merged), and `topology_gate`. Parent IDs refer to
the input Gaussian/query index for this forward call, not a persistent map ID.
In soft mode, these labels describe candidates, not committed decisions.

## Checks And Remaining Work

```bash
python -m pytest tests/test_gaussian_topology.py tests/test_long_context_model.py tests/test_variants.py -q
POINTFORWARD_TEST_CUDA=1 CUDA_VISIBLE_DEVICES=<idle_gpu> \
  python -m pytest tests/test_gaussian_topology.py -q
```

CPU tests cover counts, rotated splitting, clone displacement, bounded updates,
pair rejection, fallback, metadata, gradients, configuration, old defaults and
checkpoint initialization. The opt-in 32x32 gsplat test checks finite nonzero
image-loss gradients in action, merge, quality and update heads. It uses no
dataset, checkpoint or long-running job. Real-data quality and the full fixed
ten-scene video regression are not established by these tests.

Validation on 2026-09-22: 48 focused CPU/regression tests passed (the explicit
CUDA test was skipped in that run); all 17 topology tests passed when explicitly
enabled on the idle GPU, including real gsplat image-loss backpropagation.
The repository's baseline preflight was attempted but blocked by the missing
frozen checkpoint `clip20/runs/waymo_ds_paper/ckpt_020000.pt`. No substitute
baseline was selected, no dataset quality result is claimed, and no training
job was launched.

This implementation uses all supplied context views at once. It does NOT yet
maintain a map across successive calls, fuse different timestamps, provide
global deduplication, or choose variable child counts beyond two. A streaming
"next view updates the same map" model needs persistent IDs/state, association,
and an observation-conditioned update gate, preferably in the Rec variant.

Required experiments before claiming improvement: a topology-disabled baseline
with the same small query budget, separate update/split/clone/merge ablations,
hard-vs-soft render quality, pole/streetlight crop metrics, cross-view consistency,
action histograms, output count and peak memory. A thin structure absent from
the sampled depth queries cannot reliably be recovered by splitting alone;
depth errors may need updates, and missing coverage may need new query insertion.