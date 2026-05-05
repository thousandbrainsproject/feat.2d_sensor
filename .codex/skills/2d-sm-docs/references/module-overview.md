# 2D Sensor Module: Module Map and Knowledge Base

## Repository

- Repo: `thousandbrainsproject/feat.2d_sensor` (fork)
- Branch: `extract_edge` (PR #2)
- Working directory: `/Users/hlee/tbp/feat.2d_sensor`

## Module File Map

| File | Path | Role |
|------|------|------|
| `edge_detection.py` | `src/tbp/monty/frameworks/utils/edge_detection.py` | Structure-tensor edge detection; angle computation; edge-to-world-pose transform |
| `spatial_arithmetics.py` | `src/tbp/monty/frameworks/utils/spatial_arithmetics.py` | TangentFrame (parallel transport), vector math, curvature helpers |
| `sensor_processing.py` | `src/tbp/monty/frameworks/utils/sensor_processing.py` | Arc-length correction, directional curvature, surface normal estimation |
| `two_d_sensor_module.py` | `src/tbp/monty/frameworks/models/two_d_sensor_module.py` | TwoDSensorModule: orchestrates edge extraction + 2D position tracking |
| `transforms.py` | `src/tbp/monty/frameworks/utils/transforms.py` | Coordinate frame transforms (world/camera/image) |
| `sensor_modules.py` | `src/tbp/monty/frameworks/models/sensor_modules.py` | ObservationProcessor, FeatureChangeFilter, noise helpers |

## Key Constants

- `DIVISION_BY_ZERO_GUARD` (in `tbp/monty/math.py`): Small epsilon added to denominators to prevent divide-by-zero in coherence computation.
- `DEFAULT_TOLERANCE` (in `tbp/monty/math.py`): Tolerance for `normalize`, `is_parallel`.
- `FLAT_THRESHOLD = 0.001` (in `sensor_processing.py`): Below this value of `|k * p|`, arc-length correction is skipped.

---

## Known Gotchas

### edge_detection.py

- **RGB input contract not enforced**: `compute_edge_features` calls `cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)` assuming RGB. As of current implementation, Habitat-sim's `SensorType.COLOR` returns RGBA (R first), so slicing `[:,:,:3]` gives RGB — contract upheld. But OpenCV's own `cv2.imread` returns BGR; any caller constructing patches via OpenCV without `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` will silently pass BGR data, producing wrong luminance weights and subtly different gradient directions.

- **`orientation` can be None**: `EdgeFeatures.orientation` is `None` when total gradient weight < `DIVISION_BY_ZERO_GUARD` (uniform patch) or when `max_center_offset` rejects the edge. Both paths also set `strength=0.0`, but callers must guard against `None` before passing to `edge_angle_to_2d_pose`.

- **Angle range is (0, pi], not [0, pi]**: `gradient_to_tangent_angle` produces results in (0, pi] in practice. The formula `(gradient_angle + pi/2 + 2*pi) % (2*pi)` can in principle produce 0.0, but doing so requires `Jxy == -0.0` (IEEE 754 negative zero) so that `arctan2(-0.0, negative)` returns -pi. Because the structure tensor computes `Jxy = Ix * Iy`, the result is virtually always +0.0, so `arctan2` returns +pi, and the wrapping lands at pi (not 0). The lower bound 0 is excluded in practice, not by mathematical guarantee.

- **`gradient_to_tangent_angle` wraps to [0, 2*pi), caller uses (0, pi]**: The function wraps into [0, 2*pi) but the caller (`StructureTensor.edge_orientation` and `compute_edge_features`) expects (0, pi]. The (0, pi] range follows from the half-angle formula in `gradient_theta` (`0.5 * arctan2(...)` outputs (-pi/2, pi/2]), plus pi/2 offset giving (0, pi]. The wrapping formula `% (2*pi)` never changes this because the intermediate value is already in (0, pi].

### two_d_sensor_module.py

- **`curvature_pose_vectors` is saved before edge detection overwrites `pose_vectors`**: In `step()`, curvature-based pose vectors are captured into `curvature_pose_vectors` before `_extract_2d_edge` is called. Edge detection may overwrite `state.morphological_features["pose_vectors"]` with edge-derived pose. The saved copy is then passed to `_update_2d_position_and_displacement` for arc-length correction, which requires the original curvature directions. If this capture were omitted, arc-length correction would silently use the edge-based frame instead of the curvature frame, producing incorrect arc lengths.

- **`true_surface_normal` is also saved before potential overwrite**: Similarly, `get_surface_normal()` is called before `_extract_2d_edge` could modify pose-related state. This ensures the tangent frame transport uses the true geometric surface normal, not one derived from an edge angle.

---

## Known Subtle Details

### edge_detection.py

- **Gradient-magnitude factor in center weights is load-bearing**: `weights = w_r * (Ix**2 + Iy**2)` ensures zero-gradient pixels contribute nothing to the aggregate tensor. Without the gradient factor, uniform regions dilute the orientation estimate: a patch with one sharp edge and a large flat background would produce a biased (weakened) orientation. The gradient factor is a continuous weight, not a hard threshold — off-center pixels with strong gradients still contribute proportionally.

- **Structure tensor components squared before blurring**: `Jxx = Ix*Ix` then `GaussianBlur(Jxx)` computes E[Ix^2]. Blurring Ix first then squaring computes (E[Ix])^2, which underestimates variance and biases orientation. The order is not visible from the variable names alone.

- **`kernel_size=7` is a historical artifact**: Originally, the function detected an edge at the center pixel only (matching all other feature extractors in the codebase). The 7x7 Gaussian matched that context window. When the spec evolved to "dominant orientation of the whole patch," the kernel size was not revisited. It is a reasonable default but not analytically motivated for the current aggregated approach.

- **`world_angle = ref - theta` (minus is load-bearing)**: In `edge_angle_to_2d_pose`, the formula is `world_theta = ref_angle - theta`. The minus sign is required because image coordinates use a y-down convention (positive y points toward the bottom of the image), whereas mathematical/world coordinates use y-up. In image space, a counterclockwise angle increase goes toward the bottom-right, which maps to a clockwise rotation in world space. The minus sign corrects this flip. A plus sign would produce mirror-image edge orientations.

- **`ref_angle` projects image x-axis into world plane**: `ref_angle = arctan2(image_x_world[1], image_x_world[0])` extracts the angle of the camera's image x-axis as seen from above in world coordinates. This gives the "zero direction" for edge angles in world space. When the camera is tilted (rolled), this reference shifts accordingly, ensuring edge angles are always measured relative to the true image x-axis regardless of camera orientation.

### spatial_arithmetics.py

- **TangentFrame re-orthonormalizes in `transport()`**: After rotating the u-basis vector by the parallel-transport rotation, the code re-orthonormalizes: `self._u = normalize(project_onto_tangent_plane(self._u, new_normal))`. This prevents floating-point error from accumulating over many transport steps on a long trajectory. Without it, the basis would gradually drift off the tangent plane.

- **Fallback axis in TangentFrame.__init__**: The initial basis is computed using a cross product with `[0, 1, 0]`. If the surface normal is nearly parallel to `[0, 1, 0]` (|cos theta| > 0.95), the cross product becomes near-degenerate. The fallback to `[0, 0, 1]` prevents a zero-length basis vector. The choice of fallback axis is arbitrary — any non-parallel axis works.

### sensor_processing.py

- **Arc-length correction skips when `|k*p| >= 1.0`**: This guard means the chord length `p` exceeds the radius of curvature. Geometrically, the sensor has moved past the apex of the locally-circular surface, which means the chord-to-arc mapping is no longer monotonic. The code logs a warning and returns the chord length unchanged, which underestimates the true arc length. The correct policy-level fix is to take smaller 3D steps so this regime is never entered.

---

## Known Limitations

### two_d_sensor_module.py

- **z is forced to 0**: On first contact and on every subsequent step, `observed_state.location` is set to `[x, y, 0.0]`. The z component carries no information in the 2D model. If an LM were to use z for object recognition, it would always see zero, which would break any comparison against stored 3D observations.

- **Cannot learn from multiple 3D object rotations**: The 2D model accumulates displacements in the tangent plane, but the pose vectors sent to the LM are 3D vectors (edge tangent, normal, etc.). When the LM stores multiple observations across different object rotations, it uses `align_orthonormal_vectors` (or the Evidence LM equivalent) to reconcile them. Those functions operate in 3D and assume the object's 3D rotation is consistent. The 2D mapping has no mechanism to remap 3D pose vectors when the object appears at a different 3D orientation — seeing a cup rotated 90 degrees around its vertical axis would produce a shifted 2D position trace that cannot be aligned with the upright trace.

- **Arc-length correction assumes locally-constant curvature**: `arc_from_projection` uses `arcsin(k * p) / k`, which is exact only for a circular arc of constant curvature k. Real surfaces have varying curvature, so the correction is an approximation. The approximation degrades as the step size grows or as the surface curvature changes rapidly.

- **TangentFrame initial basis is arbitrary**: The initial orientation of `basis_u` and `basis_v` in the tangent plane depends on the arbitrary choice of `some_axis = [0, 1, 0]`. This means the 2D coordinate system of the first observation is not aligned to any world direction. Two episodes starting at different points on the same object will produce 2D coordinate systems that are rotated relative to each other. The LM must be able to handle this, or a canonical initial orientation must be enforced externally.

### edge_detection.py

- **`is_geometric_edge` uses only the center pixel of the depth patch**: The depth gradient is evaluated at `depth_patch[cy, cx]` — the single central pixel. On a noisy depth image this can be unreliable. A more robust approach would average the gradient over a small neighborhood.

---

## Known Alternatives Considered

### edge_detection.py

- **cv2.Canny rejected**: Returns uint8 single-channel image with edge pixels=255, others=0. No orientation, strength, or coherence output. Answers "where are edges?" not "what is the dominant orientation of this patch?" — wrong abstraction level.

- **Gabor filter bank considered but not implemented**: Would answer "is there an edge of orientation theta?" by running one convolution per candidate angle. Requires committing to a discrete orientation grid and many convolutions. Structure tensor answers the same question in one pass with continuous-valued orientation and no binning artifacts.

- **Simple Sobel vs. structure tensor for edge orientation**: A simple `arctan2(Gy, Gx)` on a single central pixel was an alternative for estimating edge orientation. The structure tensor was chosen because it aggregates gradient information over a neighborhood, producing more stable orientation estimates in the presence of texture noise and near-junctions. The center-weighting scheme was added to balance global context (the Gaussian blur) with local fidelity (the radial weight).

- **Global orientation vs. per-pixel then aggregate**: The structure tensor components (Jxx, Jyy, Jxy) are blurred per-pixel first, then aggregated with radial weights. An alternative is to compute the tensor directly on the center pixel without blurring. Blurring first was chosen because it smooths out pixel-level noise before aggregation.
