---
name: bae-compute-graph
description: Use when defining or modifying BAE compute graphs, sparse Jacobian traces, BAL or PGO residual models, or any code using `TrackingTensor`, tensor indexing, `map_transform`, or `bae.autograd.graph.jacobian`.
---

# BAE Compute Graph

## Core mental model
- The forward pass records a lightweight operation trace on tensors.
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
2. Mark SE(3)-style parameters with `param.trim_SE3_grad = True` when the stored state is a 7D quaternion pose or a pose-plus-extra-parameters layout that should optimize on a 6D tangent space.
3. Define each custom per-factor residual block with `@map_transform`.
4. In `forward()`, gather participating states by tensor indexing such as `self.pose[camera_idx]` or `self.nodes[edges[..., 0]]`.
5. Combine factor groups with `torch.cat(..., dim=0)` if needed. Other concatenation mode is only supported inside `@map_transform`.
6. Return the residual tensor. `LM.step()` will call `bae.autograd.graph.jacobian(...)` on it to automatically derive the sparse Jacobian.

## What each tracked op means

### `index`
- Use tensor indexing to say which parameter block each factor touches.
- This controls the sparse Jacobian layout, not the derivative values themselves.
- If the final residual is only an index result, the backward pass seeds identity blocks automatically.
- Repeated indices mean repeated block columns in the residual Jacobian.

### `map`
- Use `@map_transform` for a vectorized residual function that maps indexed inputs to per-factor residuals.
- Simple tracked arithmetic such as `+`, `-`, and `*` is also recorded as a `map` op through `WHITELISTED_MAPS`, so expressions like `pred - obs` can stay inline.
- The backward pass computes local Jacobian blocks with `torch.vmap(jacrev(func, argnums=...))`.
- Those local blocks are then chained with any upstream Jacobian already attached to the output trace.
- Write the function in terms of trailing dimensions, not hard-coded batch positions. Use negative-dimension indexing such as `[..., :2]`, `[..., 2].unsqueeze(-1)`, `sum(dim=-1, keepdim=True)`, or `residual[..., None]` so the same function works both on a single factor and on a batch of factors, inside and outside the `vmap` wrapper.
- The function should still be factorwise: each row or batch element represents one factor, and the returned residual keeps that factor dimension in front when batched.
- Non-tracked arguments such as measurements or information matrices may be passed through the function; only tracked tensor arguments contribute Jacobian blocks.

### `cat(dim=0)`
- `torch.cat(..., dim=0)` is supported for stitching factor sets or state sets along the factor dimension.
- During backward, the upstream Jacobian is sliced by row-block ranges and routed into each input branch.
- This is the pattern used by the gauge-fixed and split-state graphs.

## Hard constraints and gotchas
- The final residual trace must end in one of: `map`, `index`, or `cat(dim=0)`.
- Automatic indexing trace capture only happens when `TrackingTensor.__getitem__` receives a tensor index through PyTorch dispatch. Plain Python slicing is not the main supported sparse-layout path.
- `map_transform` functions must be compatible with `jacrev` and effectively batch-vectorized for `vmap`.
- Only `torch.cat(..., dim=0)` is supported.
- If a parameter never appears in observations, its block-columns will be empty. The authors explicitly treat this as a structural failure because it will cause the solver to fail.
- `trim_SE3_grad` changes Jacobian column count. This is to prevent solver failure:
  - 7 stored pose parameters become 6 optimized columns, because the gradient of SE(3) has one less degree of freedom.
  - 10 stored camera parameters become 9 optimized columns when the first 7 entries are SE(3) and the remaining 3 are intrinsics.

## Canonical patterns in this repo
- Standard BAL: [references/bal.md](./references/bal.md)
- Gauge-fixed BAL and split-state BAL: [references/bal.md](./references/bal.md)
- Pose graph optimization: [references/pgo.md](./references/pgo.md)

## Validation checklist
- Check residual shape first.
- Check each returned Jacobian is `torch.sparse_bsr`.
- Confirm `col_indices()` match the intended observation-to-state connectivity.
- Confirm there are no empty parameter columns when the problem is expected to constrain every variable.
