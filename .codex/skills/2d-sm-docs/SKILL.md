---
name: 2d-sm-docs
description: Write deep documentation for the 2D Sensor Module (2D_SM) codebase in this repository. Use when Codex is asked via /2d-sm-docs or $2d-sm-docs to document a function, class, module, or implementation decision in two_d_sensor_module.py, edge_detection.py, spatial_arithmetics.py, sensor_processing.py, transforms.py, or sensor_modules.py, especially when the documentation should capture gotchas, subtle implementation details, limitations, and rejected alternatives.
---

# 2D Sensor Module Documentation Writer

## Overview

Produce deep documentation for the 2D Sensor Module project. Do not merely restate algorithms and signatures; capture four layers that are hard to recover from code alone:

1. Gotchas: surprising behaviors that will trap future readers or maintainers
2. Subtle details: non-obvious choices where the code is load-bearing
3. Limitations: what the code cannot do and why those constraints exist
4. Alternatives: approaches considered but rejected, and why

Load `references/module-overview.md` at the start of each session for the module file map and the inventory of already-discovered items in each category.

## Workflow

### Step 1: Identify the target

Confirm which function, class, or module to document. If the user already stated the target, acknowledge it and proceed.

### Step 2: Read the source

Read the relevant source file(s) before analysis. Use the module file map in `references/module-overview.md` for canonical paths.

### Step 3: Analyze the four categories

Reason carefully about each category for the target before asking the user questions.

For gotchas, check for floating-point behavior, signed zero, `arctan2` edge cases, NaN propagation, equality versus near-equality, angle ranges, wrapping conventions, image y-down versus mathematical y-up coordinates, API contracts that are assumed but not enforced, and stateful first-step versus later-step behavior.

For subtle details, check for load-bearing arithmetic signs, mathematical identities, approximations, small-angle regimes, branching that is required for correctness, initialization order in stateful classes, re-orthonormalization that prevents accumulated error, and save-before-overwrite patterns.

For limitations, check for baked-in assumptions such as z=0 for 2D positions, single-rotation training, unit normals, flat approximations for arc length, default fallbacks rather than loud failures, lost 3D capabilities, and scale or step-size constraints.

For alternatives, identify the obvious alternative for each major algorithmic step, threshold, guard constant, data structure, and coordinate-frame convention. Explain why the current approach was chosen or what evidence is still missing.

### Step 4: Build a question task list

Before writing final documentation, present no more than five questions. For each question, first determine whether the codebase can answer it.

Use `[explored]` for questions answered by code exploration; state what you found and ask the user to confirm or correct it. Use `[needs your input]` for questions that require user rationale, history, or decision context; wait for the answer before writing final documentation.

Questions should surface decisions not visible in the code, confirm or refute inferred reasoning, elicit rejected alternatives, and identify missed gotchas or limitations.

Example:

```markdown
Before writing, I want to confirm a few things:

**Q1** [explored] For `cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)`: this assumes the patch
is RGB, not BGR. Habitat-sim's `SensorType.COLOR` returns RGBA (R first), and
`_extract_2d_edge` slices to `[:,:,:3]`, giving RGB, so the assumption is upheld.
Is there any other code path that could pass a BGR patch to `compute_edge_features`?

**Q2** [needs your input] For `world_angle = ref - theta`: I infer the minus is
load-bearing because image coordinates have y pointing down. Was there a version
with plus that produced mirrored edge orientations?
```

After the user responds, mark resolved questions as done and proceed to documentation.

### Step 5: Handle code issues separately

Do not mix documentation work with code fixes. If analysis uncovers a code issue:

- For a simple fix, commit or otherwise isolate the documentation work first, then make the code fix in a separate commit.
- For a larger fix, add a clear TODO only if the user agrees that editing source is in scope.

Never leave unrelated code changes tangled into the documentation output.

### Step 6: Write the documentation

Synthesize the user's answers with code analysis. Use this format for each item:

```markdown
## `function_name(signature)` / `ClassName`

**What it does** (1-2 sentences, present tense, no restating the name)

**Algorithm** (numbered steps for non-trivial procedures; omit for simple functions)

### Gotchas

- **[Short label]**: Precise description of the surprising behavior and why it happens.

### Subtle Details

- **[Short label]**: What the detail is and why it is load-bearing, including what breaks without it.

### Limitations

- **[Short label]**: What the code cannot do and what the violated assumption produces concretely.

### Alternatives Considered

- **[Short label]**: What was considered and why it was rejected.
```

Omit any category that has no items. Do not write "None." Keep each bullet to one paragraph maximum.

## Style Rules

- Use precise technical language.
- Do not hedge with "may" or "might" when something is definitively true.
- For angle and coordinate conventions, state the full convention explicitly, such as "measured from the image x-axis toward the image y-axis, i.e., downward in the image."
- When referencing IEEE 754 behavior, name the specific behavior, such as "`arctan2(+0.0, negative)` returns `pi`; `arctan2(-0.0, negative)` returns `-pi`."
- For limitations, state the consequence concretely.
- Do not write "this is important" or "note that."
- Do not pad with praise or affirmations; start directly with content.

## Resources

- `references/module-overview.md`: Module file map and running inventory of known gotchas, subtle details, limitations, and alternatives for the 2D_SM codebase.
