# Geometry_Acoustic_Simu

Hybrid geometrical-acoustics room simulation (image sources + stochastic ray tracing + diffuse rain) that reads room geometry directly from Rhinoceros `.3dm` files and estimates octave-band reverberation time (125 Hz – 4 kHz).

> **Status.** The only engine so far is the one from the 2023 master's thesis (`gasim.legacy`, tag `thesis-2023`), cleaned up so that it runs on macOS/Linux/Windows. Its propagation physics has known errors (see below) and is being replaced by a new engine in `src/gasim/` — see [docs/rewrite-plan.md](docs/rewrite-plan.md). Do not use its results for design decisions.

## Layout

```
├── main.py              run from a source checkout: python main.py <command>
├── pyproject.toml       package metadata and dependencies (uv)
├── src/gasim/
│   ├── cli.py           command line interface (gasim run / eyring / export-paths)
│   └── legacy/          thesis-2023 engine, physics unchanged
├── configs/legacy/      run settings of the thesis (rooms and sweeps)
├── models/              Rhino models of the synthetic test rooms
│   └── private/         real buildings, git-ignored
├── tests/               pytest
├── docs/                design notes (rewrite plan)
└── references/          key literature (see references/README.md)
```

Measurement data and the paper draft are kept outside this repository.

## Setup

Python 3.10+ and [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

This creates `.venv/` with the package and pytest. Without uv: `python -m venv .venv && .venv/bin/pip install -e .`.

## Run

From the repository root (config paths are relative to it):

```bash
uv run gasim run configs/legacy/test_room.py
```

`python main.py run configs/legacy/test_room.py` does the same without installing. The test room (9 × 7 × 5 m, 315 rays) takes about 30 s. Results go to the `output_folder` set in the config: decay plots, impulse responses (`.wav`) and the pickled receivers.

Other commands:

```bash
uv run gasim eyring configs/legacy/test_room.py --volume 315        # Sabine/Eyring reference
uv run gasim export-paths image output/test_room/rec_0.pkl paths.3dm  # image-source paths to Rhino
uv run pytest                                                       # tests, about 5 s
```

## Configs and models

| Config | Model | Notes |
|---|---|---|
| `configs/legacy/test_room.py` | `models/Simple_Test.3dm` | 9 × 7 × 5 m box, quick check |
| `configs/legacy/sweeps/absorption_*.py`, `scatter_*.py`, `rays_*xV.py` | `models/Simple_Test.3dm` | absorption, scattering and ray-count sweeps |
| `configs/legacy/theater_1075.py`, `theater_5057.py`, `l2b.py` | `models/*.3dm` | synthetic shape tests |
| `configs/legacy/classroom.py`, `lecture_hall_*.py`, `mf_hall.py` | `models/private/*.3dm` | real buildings — models are not published |

The configs were ported from the thesis runs; only file paths were changed. In a `.3dm` file every layer is a material, except `Source` and `Receiver` (points), and `Center_pt` / `Direc_line`, which fix the surface normal directions.

## Known issues of the legacy engine

- Ray hits and diffuse-rain contributions carry the full source power spread over the path length, W/(4πd²), instead of W/N per ray over the detector cross-section, so late energy is biased by 1/t² and results depend on the number of rays.
- The diffuse-rain solid angle uses r/d (sin γ) where 1 − cos γ is required; the Lambert angle for ray reflections uses the reflected direction instead of the surface normal.
- Specular rays are not reduced by (1 − s); direct sound and low-order paths are counted by both the image sources and the rays.
- Air absorption uses classical absorption only (19–280× too small).
- RT is read between −5 and −65 dB of a truncated response; the impedance fit (tolerance 0.20) does not preserve the input diffuse-field absorption.

In the 9 × 7 × 5 m test room the legacy engine gives RT 40–68 % below the Eyring estimate.
