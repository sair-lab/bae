# PGO Pattern

This file is intentionally self-contained. Use it as the canonical recipe for a pose-graph compute graph when the repo example is not available.

## Problem structure
- Each factor is an edge between two pose nodes.
- Every edge uses:
  - a measured relative pose `poses`
  - two node IDs `edges[..., 0]` and `edges[..., 1]`
  - an information or square-root information matrix `infos`
- The residual is usually 6D in the Lie algebra of SE(3).

## Parameter setup

```python
class PoseGraph(nn.Module):
    def __init__(self, nodes):
        super().__init__()
        self.nodes = nn.Parameter(TrackingTensor(nodes))
        self.nodes.trim_SE3_grad = True
```

- `self.nodes` is typically shape `(num_nodes, 7)` in quaternion SE(3) storage.
- `trim_SE3_grad = True` converts each stored pose block into a 6D optimized tangent-space block.

## Edge residual map

```python
@map_transform
def edge_residual(poses, node1, node2, infos):
    residual = (pp.SE3(poses).Inv() @ pp.SE3(node1).Inv() @ pp.SE3(node2)).Log().tensor()
    residual = infos @ residual[..., None]
    return residual[..., 0]
```

- This is written with trailing-dimension operations such as `residual[..., None]`, so it works both for a single edge and for a batch of edges under `vmap`.
- `edge_residual(...)` is the `map` op that provides Jacobian values.
- Left-multiplying by `infos` also left-multiplies the local Jacobian blocks through the chain rule.

## Forward graph

```python
def forward(self, edges, poses, infos):
    node1 = self.nodes[edges[..., 0]]
    node2 = self.nodes[edges[..., 1]]
    return edge_residual(poses, node1, node2, infos)
```

## Why this matches the sparse autograd design
- `self.nodes[edges[..., 0]]` defines the block-columns touched by the first endpoint of each edge.
- `self.nodes[edges[..., 1]]` defines the block-columns touched by the second endpoint of each edge.
- Both indexed tensors trace back to the same parameter `self.nodes`, so backward accumulation merges both endpoint contributions into a single Jacobian for `self.nodes`.
- The resulting Jacobian is sparse because each edge only touches two pose blocks.

## Expected Jacobian structure
- The returned Jacobian is `torch.sparse_bsr`.
- Its shape is `(num_edges * 6, num_nodes * 6)` after SE(3) trimming.
- The column pattern is the union of both endpoint index arrays.
- Repeated edges or repeated node IDs simply create repeated block-column entries in the sparse pattern.

## Variations
- To fix one or more nodes, split the state into fixed tensors and optimizable tensors, then rebuild the full node table with `torch.cat(..., dim=0)` inside `forward()`, just like the gauge-fixed BAL pattern.
- To add extra per-node attributes, store them in additional tracked parameters and index them with the same edge endpoint arrays.

## Practical checks
- Keep the residual function factorwise over edges.
- Use negative-dimension indexing inside the map function so the same code works inside and outside `vmap`.
- Confirm the Jacobian stays sparse and that all intended nodes actually appear in the edge set if the solver expects them to be constrained.
