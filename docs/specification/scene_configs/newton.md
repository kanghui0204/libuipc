# Newton Solver Details

## Convergence Criteria

Each Newton step, the solver checks two conditions:

1. **Displacement tolerance** — maximum vertex displacement must satisfy:

$$\|\Delta x\|_\infty < \texttt{velocity\_tol} \times \texttt{dt}$$

2. **CCD tolerance** — the continuous collision detection step size $\alpha_{\text{ccd}}$ must satisfy:

$$\alpha_{\text{ccd}} \geq \texttt{ccd\_tol}$$

When `ccd_tol = 1.0` (default), the solver requires that a full Newton step is collision-free before converging. Lowering this value relaxes the requirement — useful when exact CCD convergence is unnecessary.

## Affine Body Rotation Tolerance

For affine bodies, convergence additionally requires:

$$\|\Delta q\|_\infty < \texttt{transrate\_tol} \times \texttt{dt}$$

where $\Delta q$ is the change in affine body degrees of freedom. The default `transrate_tol = 0.1 /s` with `dt = 0.01` gives an absolute tolerance of `0.001`.

## Iteration Bounds

- `max_iter` (default 1024): hard cap on Newton iterations. If reached, a warning is issued (or an exception if `extras/strict_mode/enable = 1`).
- `min_iter` (default 1): minimum iterations before early exit is allowed. Relevant for semi-implicit mode.

## Fused PCG Linear Solver

The CUDA backend uses the fused preconditioned conjugate-gradient (PCG)
solver when `linear_system/solver = "fused_pcg"`. The following scene
configuration values control its execution:

| Configuration key | Default | Meaning |
|---|---:|---|
| `linear_system/check_interval` | `5` | Number of device PCG iterations between host convergence checks. The device still checks convergence every iteration and freezes the numerical state after the first terminal iteration. A larger value reduces device-to-host synchronization but may schedule more guarded no-op iterations after early convergence. The CUDA Graph path requires a value from 1 to 1024; the legacy path retains its existing behavior for older configurations. |
| `linear_system/fused_pcg/graph_enable` | `0` | Enables the CUDA Graph PCG path when the active preconditioners support it. The default preserves the legacy sequential-launch path. |
| `linear_system/fused_pcg/fused_preconditioner_enable` | `1` | Enables the fused update/residual, local-preconditioner, and residual-dot kernels required by the CUDA Graph fast path. In the current implementation, setting this to `0` makes the solver use the legacy path even if `graph_enable` is `1`. |

CUDA Graph execution is opt-in. Applications that want a ten-iteration chunk
must set both `linear_system/check_interval = 10` and
`linear_system/fused_pcg/graph_enable = 1` before `World::init()`.

The Graph launch size is a capacity, not the exact number of matrix blocks.
The exact block count is read from a fixed-address device parameter. This lets
one instantiated Graph handle small matrix sparsity changes without patching
every Graph node. A Graph is rebuilt when a captured pointer or subsystem
layout changes, the required launch capacity grows, or the check interval
changes.

For compatibility with existing diagnostics, the solver reports the end of
the host check chunk as its public iteration count. Internally it also keeps
the first device iteration that reached convergence or detected an error.

## Semi-Implicit Mode

When `newton/semi_implicit/enable = 1`, the solver uses a $\beta$-schedule that blends implicit and explicit updates:

1. Run at least `min_iter` Newton iterations.
2. After `min_iter`, compute $\beta$ from the residual ratio.
3. Terminate when $\beta \leq \texttt{beta\_tol}$.

This trades strict convergence for speed — useful for real-time or interactive scenarios where frame budget matters more than per-step accuracy.
