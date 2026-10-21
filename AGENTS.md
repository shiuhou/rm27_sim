# RM UAV MuJoCo component instructions

## Common engineering standard

Apply `.agents/docs/engineering.md` for the shared engineering process. This
file is the local adapter for the independent simulation workspace.

## Shared physics ownership

`packages/physics` owns the reusable `rm27-contact-physics` package. It accepts explicit
configuration and must not import RL, Gymnasium, Torch, or parent workspace code.
RL owns training environments, rewards and policy evaluation orchestration.
`rm27_sim/` owns simulation scene composition, mission execution, SITL adapters
and recorded-state viewers. Parent `rm27/` retains compatibility entry points
and non-simulation cross-component work. Preserve existing M0–M3 and G1 behavior;
package relocation does not qualify a new flight or change those implementations.

## Component boundaries

- Preserve the separate MuJoCo simulator and Pygame controller processes and
their UDP protocol. Do not silently add keyboard callbacks to the viewer or
make the viewer an actuator.
- Keep M0-M3 benchmark contracts and their measured-state/truth separation.
A benchmark result is a software or simulation result; it is not a sensor,
flight-controller or real-aircraft qualification.
- Record MuJoCo/model versions, configuration and random seeds for comparable
runs. Classify headless and live-view runs separately and use bounded output
folders for generated reports.
- Simulation-only SITL adapters are owned here. Firmware source/builds, raw CAD,
field/MJCF assets and historical recordings are external inputs registered with
`python -m rm27_sim configure-assets`. Never import parent or RL implementation
from rm27_sim. Keep hardware operations and credentials outside this workflow.

## Verification

Use the repository environment and run `python -m pytest -q` plus the affected
M0-M3 benchmark when changing simulator, controller, protocol or estimator
behavior. Do not claim M4 or sim-to-real performance from the existing M0-M3
checks alone.

## Simulation workspace entry

Read README.md for setup and commands. `rm27_sim/` owns runtime code, `configs/`
owns reviewed simulation settings, and ignored `artifacts/` owns new results.
`.local/assets.json` is a machine-local path registry; it is not a portable
requirement or a source asset. A separate clone must configure its own asset paths.
The older root import/CLI adapters and config symlinks exist for historical
consumers; do not add a second implementation there. Remove adapters only after
the associated root consumers, tests and documented commands have migrated.
Preserve model hashes and recorded-state behavior during this migration.
