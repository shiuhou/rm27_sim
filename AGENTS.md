# RM UAV MuJoCo component instructions

## Common engineering standard

Apply `.agents/docs/engineering.md` for the shared engineering process. This
file is the local adapter for the standalone MuJoCo quadrotor baseline; it does
not inherit implementation ownership from the parent integration repository.

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
- Keep SITL, hardware, credentials, firmware and external workspaces outside
this repository unless the user explicitly authorizes a change.

## Verification

Use the repository environment and run `python -m pytest -q` plus the affected
M0-M3 benchmark when changing simulator, controller, protocol or estimator
behavior. Do not claim M4 or sim-to-real performance from the existing M0-M3
checks alone.
