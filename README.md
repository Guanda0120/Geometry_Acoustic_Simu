# Geometry_Acoustic_Simu

Hybrid geometrical-acoustics room simulation (image sources + stochastic ray tracing + diffuse rain) that reads room geometry directly from Rhinoceros `.3dm` files and estimates octave-band reverberation time (125 Hz – 4 kHz).

> **Status.** The new engine in `src/gasim/` is being built step by step — see [docs/rewrite-plan.md](docs/rewrite-plan.md). It reads and checks room models (`gasim check`), computes the early specular reflections with image sources and everything else with energy-particle ray tracing and diffuse rain, evaluates the decay curves after ISO 3382 (`gasim run` with a TOML config), reports reflections and energy per surface (`gasim surfaces`) and exports the image-source paths to Rhino (`gasim paths`). It runs on all CPU cores; results do not depend on the number of threads. The engine of the 2023 master's thesis (`gasim.legacy`, tag `thesis-2023`) is kept for comparison; its propagation physics has known errors (see below), so do not use its results for design decisions.

## Layout

```
├── main.py              run from a source checkout: python main.py <command>
├── pyproject.toml       package metadata and dependencies (uv)
├── src/gasim/
│   ├── cli.py           command line interface (gasim check / run / surfaces / paths; eyring / export-paths for the thesis engine)
│   ├── config.py        TOML configs
│   ├── materials.py     absorption and scattering per octave band
│   ├── surfaces.py      named groups of polygons (walls, ceiling, ...) for the statistics per surface
│   ├── air.py           speed of sound, ISO 9613-1 air absorption
│   ├── geometry/        Room arrays, triangulation, inward normals, model checks
│   ├── io/              .3dm reader, result files (.npz)
│   ├── propagation/     image sources, ray tracing with diffuse rain, hybrid (Numba)
│   ├── simulate.py      config -> Result
│   ├── analysis/        Schroeder curves, EDT/T20/T30 (ISO 3382), Sabine/Eyring/Kuttruff, surface statistics
│   └── legacy/          thesis-2023 engine, physics unchanged
├── configs/             TOML configs of the rooms
│   └── legacy/          run settings of the thesis engine (rooms and sweeps)
├── models/              Rhino models of the synthetic test rooms
│   └── private/         real buildings, git-ignored
├── tests/               pytest
├── experiments/         scripts behind the paper's figures (need data outside the repo)
├── docs/                design notes (rewrite plan)
└── references/          key literature (see references/README.md)
```

Measurement data and the paper draft are kept outside this repository.

## Setup

Python 3.11+ and [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

This creates `.venv/` with the package and pytest. Without uv: `python -m venv .venv && .venv/bin/pip install -e .`.

## Run

From the repository root (config paths are relative to it). Check a room model and get its Sabine/Eyring reverberation time:

```bash
uv run gasim check configs/test_room.toml
```

This reports polygons, volume, surface per material, whether source and receivers are inside, rays leaking out of the model and overlapping coplanar surfaces, and exits non-zero if a check fails. Run a simulation:

```bash
uv run gasim run configs/test_room.toml
```

It prints EDT/T20/T30 per band (room mean and per receiver) next to Eyring and saves the energy histograms and image-source paths to `output/test_room/result.npz` (`--output` to change). The test room with 31 500 rays and third-order image sources takes under a second on a laptop (`--threads N` limits the CPU threads; the result is identical). `python main.py <command>` does the same without installing.

Print, for one band, the reflections and the energy per surface of a result: image-source reflection points, ray hits and incident energy per m² relative to the room mean (1 on every surface of a diffuse field), the share of the wall absorption each surface takes and the share of the received energy that arrives with it as the last reflection:

```bash
uv run gasim surfaces output/test_room/result.npz --band 1000
```

Write the image-source paths of a receiver to Rhino, one layer per reflection order, each curve named after the surfaces it reflects from:

```bash
uv run gasim paths output/test_room/result.npz paths.3dm --receiver 1 --max-order 2
```

Experiments for the paper live in `experiments/`. They need the private models and the measured data kept outside the repository (`../data`, see its README):

```bash
uv run python experiments/validation_rooms.py --seeds 5     # paper Fig. 4: three rooms vs measurement and thesis engine, ~4 min
uv run python experiments/wall_sweep.py absorption --seeds 5 # paper Fig. 6: wall absorption sweep in the test room vs CATT, ~30 s
uv run python experiments/wall_sweep.py scattering --seeds 5 # paper Fig. 7: wall scattering sweep, ~30 s
uv run python experiments/ray_count.py --seeds 10            # paper Fig. 5: T30 vs rays per m3 in the classroom, ~30 s
uv run python experiments/material_sources.py --seeds 3      # run-config vs paper-appendix materials, renovated hall
uv run python experiments/surface_statistics.py              # paper Fig. 9: reflections and energy per surface of the hall, ~1 min
```

Each writes CSV tables and a figure to `output/experiments/<name>/`. The sweeps and the ray-count script also run without the private data (test room; no CATT or thesis curves).

Thesis engine (Python configs): `uv run gasim run configs/legacy/test_room.py` writes decay plots, impulse responses (`.wav`) and pickled receivers to the config's `output_folder`; `gasim eyring` and `gasim export-paths` work on its configs and pickles.

Development:

```bash
uv run pytest              # tests, about 7 s
uv run pytest -m slow -s   # benchmarks against the performance targets (private models are skipped)
uv run ruff check          # lint (CI runs it and the tests on every PR)
```

## Configs and models

| Config | Model | Notes |
|---|---|---|
| `configs/test_room.toml` | `models/Simple_Test.3dm` | 9 × 7 × 5 m box, quick check |
| `configs/theater_1075.toml`, `theater_5057.toml`, `l2b.toml` | `models/*.3dm` | synthetic shape tests |
| `configs/classroom.toml`, `lecture_hall.toml`, `mf_hall.toml` | `models/private/*.3dm` | real buildings with the materials of the thesis run configs — models are not published |
| `configs/classroom_paper.toml`, `lecture_hall_paper.toml`, `mf_hall_paper.toml` | `models/private/*.3dm` | the same rooms with the materials of the paper's appendix (Tables A1–A3) |
| `configs/mf_hall_renovated.toml` | `models/private/Real_Work.3dm` | multifunction hall after renovation (paper materials + renovation materials) |
| `configs/legacy/test_room.py` | `models/Simple_Test.3dm` | thesis engine: 9 × 7 × 5 m box |
| `configs/legacy/sweeps/absorption_*.py`, `scatter_*.py`, `rays_*xV.py` | `models/Simple_Test.3dm` | absorption, scattering and ray-count sweeps |
| `configs/legacy/theater_1075.py`, `theater_5057.py`, `l2b.py` | `models/*.3dm` | synthetic shape tests |
| `configs/legacy/classroom.py`, `lecture_hall_*.py`, `mf_hall.py` | `models/private/*.3dm` | real buildings — models are not published |

The configs without `_paper` were ported from the thesis runs (materials and ray counts unchanged); the multifunction-hall configs define the walls A–D of the paper as surfaces. A TOML config looks like:

```toml
[room]
model = "models/Simple_Test.3dm"

[materials]   # layer name = absorption and scattering, one value or one per octave band 125 Hz - 4 kHz
Wood = { absorption = [0.10, 0.20, 0.10, 0.20, 0.30, 0.40], scattering = 0.10 }

[surfaces]    # optional: groups of polygons for the statistics per surface, first matching rule wins
floor   = { normal = [0, 0, 1] }                  # inward normal within max_angle (default 45 deg)
wall_x0 = { x = [0, 0] }                          # all vertices within the x (y, z) range
glazing = { layers = ["Glass"] }                  # layer names
rest    = {}                                      # everything else; without a catch-all it is "other"

[simulation]  # all optional
rays = 31500
ism_order = 3
receiver_radius = 0.2
time_step = 0.001
max_time = 3.0
seed = 1

[air]         # optional
temperature = 20.0
humidity = 50.0
```

Without `[surfaces]` every layer is one surface. In a `.3dm` file every layer with surfaces is a material; points on `Source` (exactly one) and `Receiver` are the source and receivers. Surfaces must be planar polygons without holes, in metres, centimetres or millimetres. The new engine orients normals itself; the legacy layers `Center_pt` / `Direc_line` are only read by the thesis engine.

## Known issues of the legacy engine

- Ray hits and diffuse-rain contributions carry the full source power spread over the path length, W/(4πd²), instead of W/N per ray over the detector cross-section, so late energy is biased by 1/t² and results depend on the number of rays.
- The diffuse-rain solid angle uses r/d (sin γ) where 1 − cos γ is required; the Lambert angle for ray reflections uses the reflected direction instead of the surface normal.
- Specular rays are not reduced by (1 − s); direct sound and low-order paths are counted by both the image sources and the rays.
- Air absorption uses classical absorption only (19–280× too small).
- RT is read between −5 and −65 dB of a truncated response; the impedance fit (tolerance 0.20) does not preserve the input diffuse-field absorption.

In the 9 × 7 × 5 m test room the legacy engine gives RT 40–68 % below the Eyring estimate.
