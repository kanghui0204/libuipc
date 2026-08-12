# Fused PCG CUDA Graph

This document describes the CUDA Graph fast path in `LinearFusedPCG` and the
contracts that must remain true when it is changed.

## Scope

The optimization covers only the inner preconditioned conjugate-gradient
(PCG) solve. Matrix assembly, contact detection, Newton control, continuous
collision detection, and line search remain outside the Graph.

One Graph chunk contains `linear_system/check_interval` PCG iterations. The
default is ten. An even interval returns to the same ping-pong slot after every
chunk, so one Graph executable is sufficient. An odd interval swaps the slot
parity, so two mirror Graph executables are used alternately.

## One captured iteration

Each captured iteration performs:

```text
pipelined symmetric block SpMV
  -> Ap = A * p
  -> pAp = p^T * A * p
  -> clear the inactive Ap/pAp slot for the next iteration

prepare alpha
  -> alpha = rz_old / pAp
  -> detect non-finite values or non-positive curvature

fused local update and preconditioner apply
  -> x = x + alpha * p
  -> r = r - alpha * Ap
  -> z = P^-1 * r
  -> accumulate rz_new = r^T * z

update convergence
  -> set Running, Converged, NonFinite, or Breakdown
  -> record the first terminal iteration in fixed device state

update search direction and prepare the next slot
  -> p = z + (rz_new / rz_old) * p
  -> publish rz_new as the next iteration's rz_old
  -> clear the next rz_new slot
```

The scalar and vector ping-pong buffers remove the per-iteration clear and
copy operations that would otherwise appear between these stages.

## Dynamic matrix sizes

The Graph does not assume a fixed G2 matrix size or a fixed set of contact
blocks. It separates launch capacity from logical work:

- `triplet_bucket` determines the captured grid size;
- `FusedPcgDeviceParams::triplet_count` is the exact number of valid 3x3
  sparse blocks for the current Newton system;
- each extra thread checks the device count and performs no matrix access;
- matrix values and indices may be overwritten between Graph launches.

The launch bucket is rounded upward and only grows while the backing matrix
allocation is unchanged. Therefore a small count change does not require ten
`cudaGraphExecKernelNodeSetParams` calls for a ten-iteration Graph.

This does not make pointer changes safe automatically. The Graph signature
tracks the matrix arrays, PCG vectors and scalars, device metadata, local
preconditioner buffers and layout, scalar degree-of-freedom count, launch
bucket, and check interval. Any captured binding change invalidates the Graph
and triggers a slow-path rebuild before the next launch.

## Convergence and compatibility

The device checks every iteration. Once a terminal state is recorded, later
nodes in the static chunk return without changing `x`, `r`, `z`, or `p`.
The fixed `FusedPcgCheckState` retains the first terminal iteration and its
residual so the host does not have to infer which ping-pong slot is current.

The public iteration count retains the legacy host-check convention. For
example, if the numerical state converges at iteration 3 with a check interval
of 10, the terminal iteration is 3 and the reported iteration is 10. This
preserves existing timing and diagnostic semantics while exposing the more
precise device result internally.

Non-finite `rz` or `pAp`, non-positive `p^T A p`, and a zero `rz_old`
denominator are terminal states. They are reported after the Graph chunk and
must not be allowed to propagate through later numerical updates.

## Configuration

| Scene configuration key | Default | Effect |
|---|---:|---|
| `linear_system/check_interval` | `10` | Captured iterations and host-check interval. Valid range: 1 to 1024. |
| `linear_system/fused_pcg/graph_enable` | `1` | Enables Graph execution when the solver layout is supported. |
| `linear_system/fused_pcg/fused_preconditioner_enable` | `1` | Enables the fused local update/apply/dot kernels. Disabling it currently selects the legacy solver path. |

Unsupported global preconditioners, unsupported local preconditioners, or an
ambiguous local-preconditioner layout use the legacy path. This is a required
correctness fallback, not an error.

## Lifetime and ownership

Each `LinearFusedPCG` instance owns its Graph executables and private
non-blocking capture stream. Graph objects are destroyed by the solver
destructor. Different solver or World instances must not mutate and share the
same executable Graph.

The captured Graph is launched on the solver's execution stream after matrix
and preconditioner assembly. If assembly and solve are moved to different
streams in the future, an event dependency must be added before the Graph
launch.

## Required tests

Changes to this path must test:

- sequential iteration, Graph with interval 5, and Graph with interval 10 on
  the same fixed sparse system;
- one, five, and ten forced iterations;
- convergence at every position inside a static Graph chunk;
- non-finite and non-positive-curvature status propagation;
- logical block-count changes that remain inside one launch bucket;
- matrix, vector, preconditioner, and device-metadata pointer changes;
- bucket growth, Graph rebuild, destruction, and multiple independent Worlds;
- FEM-only, ABD-only, FEM-ABD contact, articulation constraints, plastic
  cloth, dynamic time step, and recovery integration paths.

Floating-point reductions use atomic additions, so bitwise equality is not a
portable contract. Fixed-system tests use strict absolute and relative error
bounds and require exact status and terminal-iteration results. End-to-end
contact trajectories must also be compared with the repeatability envelope of
the unchanged solver because the existing assembly and reduction path is not
bitwise deterministic.
