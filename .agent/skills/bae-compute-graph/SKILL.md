---
name: bae-compute-graph
description: Use when defining or modifying BAE compute graphs, sparse Jacobian traces, bundle adjustment or pose graph optimization problems, or any code using `TrackingTensor`, `pypose.Parameter`, `map_transform`, or `bae.autograd.graph.jacobian`.
---

# BAE Compute Graph

## Core mental model
- The forward pass records a lightweight operation trace on tensors.
- `TrackingTensor` preserves PyPose `LieTensor` type information, so tracked `pp.SE3` values stay LieTensor-aware through `nn.Parameter(...)`, tensor indexing, LieTensor operations, and concatenation, `torch.cat(..., dim=0)`.
- The sparse autograd logic classifies operations mainly by their effect on the Jacobian:
  - `index`: determines sparse block-column layout.
  - `map`: computes Jacobian block values.
- The current implementation also supports `torch.cat(..., dim=0)` as a structural operation that splits and routes upstream Jacobians.

## Backward Jacobian semantics
- `bae/autograd/graph.py` always propagates Jacobians backward from the final residual output to earlier inputs.
- For a forward op `y = op(x)`, the backward handler reads the upstream Jacobian `d residual_final / d y` and fills the downstream Jacobian `d residual_final / d x`.
- Internally, intermediate Jacobians may be stored as `(indices, values)` before they are materialized as sparse BSR tensors at the leaves. `indices=None` means the current trace still has identity column layout and only carries block values.

## Authoring recipe
1. Wrap each optimizable state as `nn.Parameter(TrackingTensor(data))`.
2. If `data` is already a true PyPose `LieTensor` such as `pp.SE3(nodes)`, keep it that way. The tracked parameter wrapped by `TrackingTensor` will stay LieTensor-aware, and its optimizer step shape is inferred automatically from `parameter_update_shape(...)`.
3. The usage of `param.trim_SE3_grad = True` is not recommended. It is only for mixed ambient tensor layouts, such as a stored 7D quaternion pose or a pose-plus-extra-parameters tensor whose SE(3) portion should optimize on a 6D tangent space. Consider this an escape hatch for legacy code or special cases, not a general pattern. When using `trim_SE3_grad`, the user must ensure the first 7 entries of the parameter tensor encode SE(3) to ensure compatability.
4. Define each custom per-factor residual block with `@map_transform`.
5. In `forward()`, gather participating states by tensor indexing such as `self.pose[camera_idx]` or `self.nodes[edges[..., 0]]`. Indexed tracked LieTensor values preserve their LieTensor behavior.
6. Combine factor groups or rebuilt state tables with `torch.cat(..., dim=0)` if needed. Other concatenation mode is only supported inside `@map_transform`.
7. Return the residual tensor. `LM.step()` will call `bae.autograd.graph.jacobian(...)` on it to automatically derive the sparse Jacobian.

## What each tracked op means

### `index`
- Use tensor indexing to say which parameter block each factor touches.
- This controls the sparse Jacobian layout, not the derivative values themselves.
- If the final residual is only an index result, the backward pass seeds identity blocks automatically.
- Repeated indices mean repeated block columns in the residual Jacobian.
- When the indexed source is a tracked PyPose `LieTensor`, the indexed result remains LieTensor-aware, so downstream code can keep using native LieTensor methods such as `.Inv()`, `.Log()`, or `.Act(...)`.

### `map`
- Use `@map_transform` for a vectorized residual function that maps indexed inputs to per-factor residuals.
- Simple tracked arithmetic such as `+`, `-`, and `*` is also recorded as a `map` op through `WHITELISTED_MAPS`, so expressions like `pred - obs` can stay inline.
- The backward pass computes local Jacobian blocks with `torch.vmap(jacrev(func, argnums=...))`.
- Those local blocks are then chained with any upstream Jacobian already attached to the output trace.
- Write the function in terms of trailing dimensions, not hard-coded batch positions. Use negative-dimension indexing such as `[..., :2]`, `[..., 2].unsqueeze(-1)`, `sum(dim=-1, keepdim=True)`, or `residual[..., None]` so the same function works both on a single factor and on a batch of factors, inside and outside the `vmap` wrapper.
- The function should still be factorwise: each row or batch element represents one factor, and the returned residual keeps that factor dimension in front when batched.
- Non-tracked arguments such as measurements or information matrices may be passed through the function; only tracked tensor arguments contribute Jacobian blocks.
- For true LieTensor inputs, prefer native LieTensor expressions directly. In PGO, write `poses.Inv() @ node1.Inv() @ node2` instead of repeatedly recasting tracked values with `pp.SE3(...)`.

### `cat(dim=0)`
- `torch.cat(..., dim=0)` is supported for stitching factor sets or state sets along the factor dimension.
- During backward, the upstream Jacobian is sliced by row-block ranges and routed into each input branch.
- This is the pattern used by the gauge-fixed and split-state graphs.
- If the concatenated inputs are tracked LieTensor values with the same `ltype`, the concatenated result remains LieTensor-aware.

## Hard constraints and gotchas
- The final residual trace must end in one of: `map`, `index`, or `cat(dim=0)`.
- Automatic indexing trace capture only happens when `TrackingTensor.__getitem__` receives a tensor index through PyTorch dispatch. Plain Python slicing is not the main supported sparse-layout path.
- `map_transform` functions must be compatible with `jacrev` and effectively batch-vectorized for `vmap`.
- Only `torch.cat(..., dim=0)` is supported.
- If a parameter never appears in observations, its block-columns will be empty. The authors explicitly treat this as a structural failure because it will cause the solver to fail.
- Jacobian column counts and optimizer step views follow `parameter_update_shape(param)`:
  - true `pp.SE3` parameters optimize in 6D tangent-space blocks automatically.
  - 7D stored pose tensors become 6 optimized columns only when `trim_SE3_grad = True`.
  - 10D camera tensors become 9 optimized columns only when the first 7 entries encode SE(3) and `trim_SE3_grad = True`.
- Use `trim_SE3_grad` for mixed ambient tensor layouts that pack SE(3) state into plain tensors, not for true PyPose `LieTensor` parameters.

## Canonical patterns in this repo
- Standard BAL: [references/bal.md](./references/bal.md)
- Gauge-fixed BAL and split-state BAL: [references/bal.md](./references/bal.md)
- Pose graph optimization: [references/pgo.md](./references/pgo.md) for the sparse pattern; when nodes and measurements are true `pp.SE3` values, prefer native LieTensor residual expressions and do not use `trim_SE3_grad`.

## Validation checklist
- Check residual shape first.
- Check each returned Jacobian is `torch.sparse_bsr`.
- Confirm `col_indices()` match the intended observation-to-state connectivity.
- Confirm each parameter block width matches `parameter_update_shape(...)`, especially when mixing true LieTensor parameters with ambient-tensor layouts.
- Confirm there are no empty parameter columns when the problem is expected to constrain every variable.
