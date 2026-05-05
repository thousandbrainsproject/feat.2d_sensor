# 2D Sensor Module Documentation Scratch Pad

## Context

Writing deep documentation for the final merged 2D Sensor Module implementation in
`tbp.monty` version `0.33.0`.

Primary output:
`2d_sm_docs/src/2dsm_docs.tex`.

Primary source files:
- `src/tbp/monty/frameworks/utils/edge_detection.py`
- `src/tbp/monty/frameworks/models/two_d_sensor_module.py`
- `src/tbp/monty/frameworks/utils/spatial_arithmetics.py`
- `src/tbp/monty/frameworks/utils/sensor_processing.py`

Skill status:
- The old skill files still exist at `/Users/hlee/.claude/skills/2d-sm-docs/`.
- The old reference inventory still exists at
  `/Users/hlee/.claude/skills/2d-sm-docs/references/module-overview.md`.
- The `2d-sm-docs` skill is not registered in the current Codex session's
  available skill list, so it cannot be invoked as an active skill here.

---

## What Is Done

### Code structure to document

The final merged code replaced the standalone `compute_edge_features` function
with the `EdgeDetector` class. `EdgeDetector.__call__` now owns the same dominant
edge-detection flow and returns `EdgeFeatures`. This establishes the expected
shape for future feature detectors: configurable class, reusable instance, and
callable extraction interface via `__call__`.

`TwoDSensorModule` accepts an optional `edge_detector`. If edge features are
requested and no detector is provided, it constructs `EdgeDetector()` and stores
it as `self.edge_detector`. `_extract_2d_edge` calls `self.edge_detector(observation)`.

### 2dsm_docs.tex — existing material

| Section | Status | Update Needed |
|---------|--------|---------------|
| Introduction | updated | now notes `tbp.monty` `0.33.0` |
| Part I intro (The Problem / The Solution) | partially updated | key function list and main algorithm now point at `EdgeDetector.__call__`; prose still needs deeper rewrite |
| `_compute_per_pixel_structure_tensors` | drafted | update as `EdgeDetector._compute_per_pixel_structure_tensors` |
| `_compute_sobel_gradients` | drafted | update as `EdgeDetector._compute_sobel_gradients` |
| Part II: `TangentFrame` / arc-length correction | drafted | keep, but later verify against final merged source |

Previous notes about code fixes committed to main are now obsolete for this
documentation pass. The relevant fixes are already in the merged code, or the
final code intentionally differs from the old plan.

---

## What Is Next

Update Part I around `EdgeDetector`, then work through the remaining methods in
call order, followed by the supporting dataclasses and geometry helpers.

### EdgeDetector flow

- [x] Confirm `compute_edge_features` became `EdgeDetector.__call__`
- [x] Confirm `tbp.monty` version is `0.33.0`
- [x] Confirm old `2d-sm-docs` files still exist but are not registered here
- [x] Update Introduction with `tbp.monty` version `0.33.0`
- [x] Update Part I key-function list to use `EdgeDetector`
- [ ] Deepen/rewrite Part I overview around `EdgeDetector.__call__`
- [ ] `EdgeDetector._compute_sobel_gradients(grayscale)`
- [ ] `EdgeDetector._compute_per_pixel_structure_tensors(Ix, Iy)`
- [ ] `EdgeDetector._compute_center_weights(shape, Ix, Iy)` — **next up**
- [ ] `EdgeDetector._aggregate_tensor(tensor_per_pixel, weights, total_weight)`
- [ ] `EdgeDetector._passes_center_check(weights, total_weight, gradient_theta)`
- [ ] `EdgeDetector._is_geometric_edge(depth_patch, edge_theta)`

### Supporting classes / functions used by EdgeDetector

- [ ] `StructureTensor` dataclass and properties:
  `eigenvalues`, `gradient_theta`, `edge_strength`, `coherence`, `edge_angle`
- [ ] `EdgeFeatures` dataclass: document the `angle is None` / `has_edge is False`
  contract
- [ ] `EdgeDetector.__init__`: document defaults and validation

### Other Part I functions (standalone, document after helpers)

- [ ] `_gradient_to_tangent_angle`
- [ ] `_angle_to_pose_2d`

---

## Key Decisions / Facts to Remember

- `strength_threshold=0.1` calibrated for float32 [0,1] input; perfect B&W step
  edge → edge_strength ≈ 4.0.
- `kernel_size=7` is a historical artifact from when the function read center pixel
  only; not analytically derived for the aggregated approach.
- `EdgeDetector` currently keeps `ksize=3` explicit in Sobel calls.
- Border reflection in `GaussianBlur` (3-pixel ring for kernel_size=7) is benign
  because center-weighting assigns near-zero weight to border pixels.
- `max_center_offset` was introduced to reduce visual clutter; default is None.
  A figure comparing None vs. small finite value is TODOed in the tex.
- Gabor filters were considered (biologically motivated) but rejected: they answer
  "does orientation X exist?" not "what is the dominant orientation?" and require
  many convolutions.
- `cv2.Canny` rejected: returns binary pixel map, no orientation output.
- Geometric edges are detected from the depth patch and filtered out in
  `TwoDSensorModule._extract_2d_edge` when `edge.has_edge` is true and
  `edge.is_geometric_edge` is true.
