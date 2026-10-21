# Geometry_Acoustic_Simu

Hybrid geometrical-acoustics room simulation (image sources + stochastic ray tracing + diffuse rain) that reads room geometry directly from Rhinoceros `.3dm` files and estimates octave-band reverberation time (125 Hz – 4 kHz).

> **Status.** This is the code of the 2023 master's thesis (tag `thesis-2023`), cleaned up so that it runs on macOS/Linux/Windows. The propagation physics is unchanged and has known errors (see below); a rewrite is in progress. Do not use the results of this version for design decisions.

## Setup

Python 3.9+ and five packages:

```bash
uv venv .venv && uv pip install -r requirements.txt
```

or `python -m venv .venv && .venv/bin/pip install -r requirements.txt`.

## Run

```bash
.venv/bin/python Simulation.py configs/test_room.py
```

Without an argument `configs/test_room.py` is used (9 × 7 × 5 m test room, 315 rays, about 30 s). Results go to the `output_folder` set in the config: decay plots, impulse responses (`.wav`) and the pickled receivers.

Sabine/Eyring reference for a model:

```bash
.venv/bin/python -m Acoustic.Eyring configs/test_room.py 315   # config, volume in m³
```

## Configs and models

| Config | Model | Notes |
|---|---|---|
| `configs/test_room.py` | `models/Simple_Test.3dm` | 9 × 7 × 5 m box, quick check |
| `configs/sweeps/absorption_*.py`, `scatter_*.py`, `rays_*xV.py` | `models/Simple_Test.3dm` | absorption, scattering and ray-count sweeps |
| `configs/theater_1075.py`, `theater_5057.py`, `l2b.py` | `models/*.3dm` | synthetic shape tests |
| `configs/classroom.py`, `lecture_hall_*.py`, `mf_hall.py` | `models/private/*.3dm` | real buildings — models are not published |

The configs were ported from the thesis runs; only file paths were changed. In a `.3dm` file every layer is a material, except `Source` and `Receiver` (points), and `Center_pt` / `Direc_line`, which fix the surface normal directions.

## Known issues in this version

- Ray hits and diffuse-rain contributions carry the full source power spread over the path length, W/(4πd²), instead of W/N per ray over the detector cross-section, so late energy is biased by 1/t² and results depend on the number of rays.
- The diffuse-rain solid angle uses r/d (sin γ) where 1 − cos γ is required; the Lambert angle for ray reflections uses the reflected direction instead of the surface normal.
- Specular rays are not reduced by (1 − s); direct sound and low-order paths are counted by both the image sources and the rays.
- Air absorption uses classical absorption only (19–280× too small).
- RT is read between −5 and −65 dB of a truncated response; the impedance fit (tolerance 0.20) does not preserve the input diffuse-field absorption.

In the 9 × 7 × 5 m test room this version gives RT 40–68 % below the Eyring estimate.
