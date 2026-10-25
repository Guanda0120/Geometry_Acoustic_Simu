# Rewrite plan: a correct and fast engine for `gasim`

Status: proposed · Owner: Guanda Li · Last updated: 2026-10-22

This plan replaces the thesis engine (`gasim.legacy`) with a new engine in `src/gasim/`. The work lands as a series of small pull requests, each with its own tests, reviewed and merged by the owner.

## 1. Goals and non-goals

**Goals**

1. Physically consistent energy-based hybrid model: image sources (ISM) for early specular reflections, stochastic ray tracing with scattering for the rest, diffuse rain for variance reduction.
2. Octave-band (125 Hz – 4 kHz) energy–time curves and decay parameters EDT, T20, T30 per receiver, following ISO 3382.
3. Every physical claim checked by an automated test against an analytical or statistical reference.
4. Fast enough for design iterations: the 4389 m³ hall at 100 rays/m³ in minutes on a laptop, not days.
5. Reproducible: fixed random seeds, results saved with the config, code version and timing.
6. Same input as today: Rhinoceros `.3dm` models with one layer per material, plus `Source` and `Receiver` points.
7. A per-surface sensitivity ∂T/∂αᵢ as the new contribution for treatment planning (§7, PR 11).

**Non-goals for v1.0**

- Phase, complex reflection coefficients and auralisation (none of the target parameters needs them).
- Wave-based modelling below ~500 Hz.
- Curved (NURBS) surfaces; models must be planar polygons.
- Source directivity and GPU execution.

## 2. What is wrong with the legacy engine

Each problem below has a fix in the new design and a test that guards it.

| # | Legacy behaviour (file) | Effect | Fix in new engine | Guarded by |
|---|---|---|---|---|
| L1 | Every ray hit and diffuse-rain contribution carries the full source power over the path length, W/(4πd²) (`Acoustic/BaseAcoustic.py:98`, `:139`) | Extra factor N·r²/(4d²): early energy inflated, late energy suppressed (1/t²), RT too short and drifting with N | Energy particles: each ray carries 1/N of the source energy; a receiver records energy / (πr²) | T-diffuse, T-rays |
| L2 | Diffuse-rain solid angle uses r/d (= sin γ) instead of 1 − cos γ (`Graphic/BaseGeometry.py:151`) | Diffuse rain ~10³ times too strong | 1 − cos γ with sin γ = r/d | T-rain |
| L3 | Lambert angle of ray diffuse rain uses the reflected direction instead of the surface normal (`Acoustic/Receiver.py:333`) | Wrong angular weighting | cos θ from the surface normal | T-rain |
| L4 | Specular rays not reduced by (1 − s) while diffuse rain is added (`Acoustic/BaseAcoustic.py:91`) | Energy created at every reflection | Stochastic choice of specular/diffuse direction; ray energy × (1 − α) only | T-energy |
| L5 | Rays detect the direct sound and low orders already given by the ISM | Early energy counted several times | Order exclusion between ISM and rays (§4.4) | T-hybrid |
| L6 | First-order ISM diffuse rain ignores the absorption of the first wall (`Acoustic/Receiver.py:266`) | Too much early scattered energy at high α | No ISM diffuse rain: rays cover all paths ending in a diffuse reflection (§4.3) | T-rain, T-hybrid |
| L7 | Air absorption from classical absorption only, 19–280× too small (`Acoustic/CONST.py:6`) | HF reverberation too long in large rooms | ISO 9613-1 from temperature, humidity and pressure | T-air |
| L8 | Tracing stops when the loudest ray, including spherical spreading, is 65 dB down (`simulation.py:102`); RT read between −5 and −65 dB of the truncated curve (`AdaptiveDSP/RT_Process.py:126`) | Decay truncated at ~30 dB of real energy loss; RT evaluated in the truncation zone | Trace until energy is ≥ 70 dB down or `max_time`; ISO 3382 least-squares T20/T30/EDT on the Schroeder curve | T-decay |
| L9 | Impedance fit with tolerance 0.20 replaces the input absorption (`Acoustic/Material.py:46`) | Absorption used differs from input by up to 6× | Energy model uses the diffuse-field α directly | T-materials |
| L10 | Phase computed, then discarded by RMS summation (`Acoustic/Receiver.py:472`) | Complex reflection coefficients have no effect | Dropped (energy only) | — |
| L11 | No random seed (`Acoustic/RayGroup.py:33`) | Runs not reproducible | Seed in config and in results | T-repro |
| L12 | Normals fixed by hand-drawn `Center_pt`/`Direc_line` layers | Fragile model preparation | Normals oriented automatically towards the interior | T-geometry |
| L13 | One Python object per ray, Python loops, `np.linalg.det` point-in-polygon tests, O(n²) re-multiplication of reflection history, `deepcopy` per hit, brute-force ISM | 1.8–8.8 h for 31 500 rays in a 6-surface box; up to 600 MB per receiver | Arrays + Numba kernels, parallel over rays, histograms instead of stored paths | T-bench |

Symptom to investigate during PR 4: in the thesis classroom run each receiver recorded only ~30 ray hits where ~2800 were expected, which points to leaks or overlapping coplanar polygons in the model.

## 3. Architecture

```
src/gasim/
├── config.py            dataclasses + TOML loading/validation
├── bands.py             octave-band centres, band count
├── materials.py         material library (absorption, scattering per band)
├── surfaces.py          named groups of polygons ([surfaces] rules), independent of the materials
├── air.py               ISO 9613-1 air attenuation
├── geometry/
│   ├── room.py          Room: flat arrays of triangles, polygons, normals, materials
│   ├── build.py         triangulation, inward normals, volume, watertightness checks
│   └── intersect.py     Numba kernels: first hit, occlusion, sphere crossing
├── io/
│   ├── rhino.py         .3dm → polygons per layer, source, receivers
│   └── results.py       save/load results (.npz + JSON metadata), export paths to .3dm
├── propagation/
│   ├── image_sources.py ISM with visibility checks
│   ├── rays.py          energy-particle tracer + diffuse rain (Numba)
│   └── hybrid.py        combine ISM and rays into energy histograms
├── analysis/
│   ├── decay.py         Schroeder integration, EDT/T20/T30
│   ├── statistical.py   Sabine, Eyring (with air absorption)
│   ├── surfaces.py      reflections and energy per surface (PR 11a)
│   └── sensitivity.py   ∂T/∂αᵢ per surface (PR 11b)
├── simulate.py          one function: config → Result
├── cli.py               gasim check / run / eyring / export-paths (.py configs run the legacy engine)
└── legacy/              removed in PR 12
```

**Data model.** A `Room` holds everything as contiguous NumPy arrays so kernels can run without Python objects:

| Array | Shape | Content |
|---|---|---|
| `tri_vertices` | (T, 3, 3) | triangle corners |
| `tri_polygon` | (T,) | polygon index of each triangle |
| `poly_normal`, `poly_d` | (P, 3), (P,) | plane n·x + d = 0, n pointing into the room |
| `poly_material` | (P,) | index into the material table |
| `poly_area` | (P,) | m² |
| `absorption`, `scattering` | (M, 6) | per material and band |

**Units of the result.** Histograms store energy per unit area per unit emitted source energy (m⁻²), in bins of `time_step` (default 1 ms). For direct sound this is exp(−m·d)/(4πd²), which makes absolute levels (strength G) available later.

**Configs** become data, not code (TOML), e.g. `configs/test_room.toml`:

```toml
[room]
model = "models/Simple_Test.3dm"

[materials]                       # layer name → absorption / scattering per octave band
Wood    = { absorption = [0.10, 0.20, 0.10, 0.20, 0.30, 0.40], scattering = 0.10 }
Plaster = { absorption = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20], scattering = 0.10 }
Mat     = { absorption = [0.15, 0.30, 0.40, 0.75, 0.85, 0.70], scattering = 0.10 }

[simulation]
rays = 31500
ism_order = 3
receiver_radius = 0.2             # m
time_step = 0.001                 # s
max_time = 3.0                    # s
seed = 1

[air]
temperature = 20.0                # °C
humidity = 50.0                   # %
```

Legacy Python configs stay in `configs/legacy/` until PR 12.

## 4. Algorithms

### 4.1 Geometry

- Read planar faces of Breps (and meshes) from each material layer; reject non-planar faces.
- Triangulate polygons by ear clipping (handles concave polygons).
- Orient each normal towards the interior: test a point just off the polygon centroid with a parity ray cast.
- Room volume from the closed triangle mesh (divergence theorem).
- Checks reported on load: open edges, rays escaping the room from the source (leak test), overlapping coplanar polygons (e.g. a window drawn on top of a wall — resolved by a rule, not by chance).
- Intersection kernel: Möller–Trumbore ray–triangle test; brute force first (≤ a few hundred triangles), a BVH only if benchmarks need it.

### 4.2 Ray tracing (energy particles)

- N rays from an omnidirectional point source, directions uniform on the sphere (seeded), energy 1/N in every band.
- At each wall hit: energy × (1 − α) per band; continue in the specular direction with probability 1 − s, otherwise in a Lambert (cosine-weighted) direction. Bands whose scattering coefficients differ are traced as separate groups, each with its own rays. (Reweighting bands by s_b/p or (1 − s_b)/(1 − p) is unbiased, but the weights multiply over hundreds of reflections and degenerate: in PR 6 a band with s = 0 next to one with s = 1 received no energy at all.)
- Receiver detection: a segment that crosses a receiver sphere of radius r deposits energy · exp(−m·L) / (πr²) at time L/c, L being the path length to the crossing.
- Termination: energy below −70 dB in all bands (air included) or `max_time`.

### 4.3 Diffuse rain

At every wall hit with incident energy e and each receiver visible from the hit point (cos θ > 0 with the surface normal, no occlusion):

E_rain = e · (1 − α) · s · (1 − cos γ) · 2 cos θ · exp(−m·L) / (πr²),  sin γ = r / r_s

(Vorländer 2020, Eq. 11.7; Schröder 2011, Eq. 5.20), deposited at time (L_hit + r_s)/c. Because the rain is the expected direct contribution of the diffuse share, sphere detection is skipped on a segment that starts with a diffuse reflection; segments after a specular reflection are detected normally. Together this counts every path exactly once in expectation.

### 4.4 Image sources and the hybrid

- ISM up to `ism_order`: mirror recursively, prune when the parent lies behind the mirror plane (Borish 1984), validate each path by back-tracing (reflection points inside their polygons, no occlusion).
- Energy of a path: ∏(1 − αᵢ)(1 − sᵢ) · exp(−m·d) / (4πd²); with s = 0 this is the classical ISM.
- Order exclusion: rays whose reflections so far were all specular and whose order ≤ `ism_order` deposit nothing (the ISM already holds those paths, including the direct sound).
- Valid paths are kept (order, surfaces, points) for export to Rhino and for the reflection analysis in the paper.

### 4.5 Decay analysis

- Schroeder backward integration of each band histogram.
- EDT (0 to −10 dB), T20 (−5 to −25 dB), T30 (−5 to −35 dB) by least-squares fit, as in ISO 3382-1/-2, from the first bin with energy.
- A parameter is only reported when the fitted decay reaches 15 dB below its evaluation range by the end of the response (EDT 25 dB, T20 40 dB, T30 50 dB); otherwise it is NaN with the reason. 15 dB keeps the truncation bias of T30 below 0.3 % (10 dB: 0.85 %).
- Non-linearity ξ = 1000(1 − r²) and curvature C = 100(T30/T20 − 1) flag double-slope decays (ISO 3382-2 Annex B).
- Sabine and Eyring (both with 4mV air term) as references.

### 4.6 Performance design

- Kernels in Numba (`@njit(parallel=True)`); rays split into 32 fixed chunks, one histogram per chunk, summed at the end in a fixed order (no atomics), so results are bit-identical for any thread count. Image sources: one task per receiver and first reflecting wall on a thread pool (the search releases the GIL).
- Nothing per ray is stored after tracing except what the histograms and optional sensitivity counters need.
- Targets (8-core laptop):

| Case | Legacy | Target | Achieved (PR 8, 10-core MacBook Air) |
|---|---|---|---|
| Test room 315 m³, 31 500 rays, 4 receivers | 1.8–8.8 h | < 10 s | 0.4 s |
| Classroom 229 m³, 100 rays/m³ (22 900), ISM order 5 | 6–23 h | < 1 min | 0.9 s |
| Multifunction hall 4389 m³, 100 rays/m³ (438 900), 6 receivers | not feasible (ran 9/m³ in 35 h) | < 10 min | 33.6 s |

## 5. Validation suite

| Test | Setup | Pass criterion |
|---|---|---|
| T-geometry | Bundled models | Normals point inwards; triangulated area = polygon area; volume of the test room = 315 m³; no leaks |
| T-air | ISO 9613-1 reference values | Within 2 % at 20 °C / 50 % RH, 125 Hz – 4 kHz |
| T-decay | Synthetic exponential and double-slope decays | T30 within 1 % for single slope; fit range reported |
| T-energy | Closed box, α = 0, no air | Ray energy constant; absorbed + remaining = emitted for α > 0 |
| T-diffuse | Box with s = 1 on all walls, uniform α = 0.1 / 0.3 | T30 within 2 % of Eyring corrected for the free-path spread (Kuttruff Eq. 5.34, γ² from his Table 5.1); plain Eyring is 2–7 % short for such a box |
| T-rays | Same box, N and 4N rays, 5 seeds each | Mean T30 agrees within the combined standard error (no drift with N) |
| T-rain | Large single wall, s = 1 | Diffuse rain matches brute-force sphere detection with 10⁷ rays (statistical tolerance) |
| T-ism | Shoebox, orders ≤ 3 | Image positions, delays and energies equal the Allen–Berkley lattice |
| T-specular | 9 × 7 × 5 m shoebox, absorptive floor, hard walls, s = 0 (strongly non-diffuse) | T20/T30 of the tracer within 6 % of the exact image-lattice sum (8 M images); Eyring is 3× too short here |
| T-hybrid | Shoebox, ISM + rays | Direct sound energy equals exp(−m·d)/(4πd²) exactly; no path counted twice |
| T-materials | Material table round trip | Absorption used = absorption in config |
| T-repro | Same config, same seed | Identical histograms |
| T-bench | Cases of §4.6 (marked `slow`, not run on every push) | Meets targets |

Optional cross-check: pyroomacoustics (ISM + ray tracing) in a shoebox, as a development dependency.

## 6. Pull requests

| PR | Scope | Done when |
|---|---|---|
| 2 | This plan | Reviewed and merged |
| 3 | CI: GitHub Actions running `uv sync` + `pytest` (and `ruff` on new code) on every push/PR | Checks show on PRs |
| 4 | Config (TOML) + material library, Rhino import into `Room`, triangulation, inward normals, volume, leak/overlap checks, ISO 9613-1, Sabine/Eyring; TOML versions of the bundled configs | T-geometry, T-air, T-materials pass; all four bundled models load cleanly |
| 5 | Decay analysis and result files (`.npz` + JSON metadata) | T-decay passes |
| 6 | Ray tracer with diffuse rain (Numba, single thread is fine) | T-energy, T-diffuse, T-rays, T-rain, T-repro pass |
| 7 | ISM, hybrid combination, path export to Rhino (`gasim paths`); `gasim run` with TOML configs uses the hybrid | T-ism, T-hybrid pass; test room runs end to end |
| 8 | Performance: rays in 32 fixed parallel chunks (results independent of the thread count), image sources in parallel per receiver and first wall, benchmarks; no BVH needed (brute force meets the targets) | T-bench targets met |
| 9 | Real rooms: models load and pass the checks (done in PR 4: closed, no overlaps); materials stay as in the thesis run configs for now (decision of 2026-10-22); the paper's appendix materials are in `configs/*_paper.toml` and compared in `experiments/material_sources.py` (see §8) | Leak/overlap checks clean; runs complete |
| 10 | `experiments/`: scripts for every figure of the paper (validation rooms, absorption/scattering/ray-count sweeps with several seeds, CATT comparison), reading data from `../data` | One command per figure: `validation_rooms.py` (Fig. 4), `wall_sweep.py absorption|scattering` (Figs. 6, 7), `ray_count.py` (Fig. 5) |
| 11a | Surfaces (`[surfaces]` rules) and statistics per surface in every result: image-source reflection points, ray hits, incident and absorbed energy, received energy by last reflecting surface; `gasim surfaces`; `experiments/surface_statistics.py` (paper Fig. 9) | Hit counts and energies add up to the totals; hits per m² uniform in a diffuse box; hall walls A–D match the areas of Fig. 9 |
| 11b | Surface sensitivity ∂T/∂αᵢ per surface and uncertainty bands (§7) | Linear prediction agrees with full re-simulation for small Δα |
| 12 | Remove `gasim.legacy` and `configs/legacy/`, docs, v1.0 release with a Zenodo DOI | GitHub `main` is the complete, runnable version cited in the paper |

## 7. New contribution: surface sensitivity (PR 11)

The energy of a path is proportional to ∏ⱼ(1 − αⱼ)^nⱼ, nⱼ being the number of hits on surface j. Hence ∂E/∂αᵢ = −nᵢ/(1 − αᵢ) · E for each path. Accumulating nᵢ·E per receiver, band and time bin during the same run gives the derivative of the whole energy–time curve, and from it ∂T30/∂αᵢ, in one simulation. Uses:

- rank surfaces by RT reduction per m² of treatment (an energy- and time-weighted replacement for the hit-count density of the thesis);
- compare treatment layouts with the same added absorption area (the control the renovation case lacks);
- propagate absorption uncertainty to RT intervals.

Validation: compare the linear prediction T(α + Δα) ≈ T + ΣᵢΔαᵢ ∂T/∂αᵢ with full re-simulations.

Inputs for the multifunction hall (checked in the material-sources experiment):

- Surfaces: the model's polygons fall into the six surfaces of paper Fig. 9 by layer and position — ceiling (layer
  White, 594.4 m²), floor (555.7 m², Fig. 9: 552), wall A (y ≥ 16.6 m, 234.7 m², Fig. 9: 236), wall B (glazed,
  y ≤ 2.4 m, 273.6 m², Fig. 9: 274), wall C (behind the stage, LED + upholstery, 162.9 m², Fig. 9: 163) and wall D
  (rear, x ≥ 25.6 m, 160.2 m², Fig. 9: 160). Sensitivities are needed per surface, not per layer, so PR 11 adds surface
  groups (a rule or a group layer in the model) on top of the per-layer materials.
- Materials: paper Table A3 before and after (`configs/mf_hall_paper.toml`, `configs/mf_hall_renovated.toml`);
  scattering is not documented for the halls (0.10 assumed) and no test reports of the absorption data are on file, so
  the uncertainty ranges for the RT intervals have to come from the literature.
- Measurements: before per receiver; after only as a room mean whose source is unresolved (see data/README.md).

First result of PR 11a (paper materials, 100 rays/m³, 1 kHz): the ray hits per m² are within 1–5 % of the room mean on every surface of the hall, so the stochastic counts cannot rank surfaces, which is what a near-diffuse field predicts. Only the image-source points differ per m² (wall C 2.2×, D 1.7×, A 1.4× the mean; the source stands in front of wall C). Of the energy absorbed by the walls, wall C takes 36 %, A 26 %, D 16 % and the ceiling 10 % before renovation; after it the ceiling takes 32 %. These shares, and the sensitivities of PR 11b, replace the hit densities as the treatment indicator.

## 8. Risks

- **Real models may not be watertight** or may contain overlapping coplanar surfaces (see the classroom symptom in §2). The checks of PR 4 will show this; fixing models is PR 9.
- **Band-dependent scattering** (resolved in PR 6): bands with different scattering are traced separately, which costs one tracing pass per distinct scattering pattern (at most six).
- **Low-frequency bias** remains: geometrical acoustics ignores modes and diffraction. The new engine removes the implementation bias; it does not make 125–250 Hz wave-accurate.
- **CATT reference of the sweeps** (found in PR 10): with the documented inputs (thesis Tables 4.12/4.14) CATT's 125 Hz values lie far below Eyring even for fully diffuse walls (s = 0.99: 0.68 s vs 1.85 s), so CATT's model cannot have had those inputs at 125 Hz; at 1–4 kHz and s = 0.99 it agrees with Eyring. In the non-diffuse test room CATT stays near Eyring where the exact lattice (s = 0) gives 3× Eyring. The CATT comparison is therefore not a clean benchmark; T-specular is the exact one.
- **Materials of the real rooms** (found with `experiments/material_sources.py`, 100 rays/m³, 3 seeds): neither source reproduces the multifunction hall. The run-config materials give 0.85 at 500 Hz for both wall panels and a ceiling at 0.15–0.32 (the paper: plaster, 0.03–0.07) and miss the measurement by −60 % at 500 Hz; the paper's Table A3 gives +17 % to +107 % in every band, because `Real_Work.3dm` has no seats or tables (the thesis text says they were modelled as boxes). Rough Sabine estimate: ~120 m² of seating absorption would bring 1 kHz to ~2.2 s before (measured 2.50) and ~1.1 s after renovation (measured 1.16). With Table A3 the thesis engine's −18 % agreement can only come from its energy bias offsetting the missing absorption. The classroom's two sources are almost identical (new engine +5–17 % at 250 Hz–4 kHz); in the lecture hall the run-config set is closer at 250 Hz–2 kHz and the paper set at 4 kHz. Needed before the paper: a hall model with the seating, and one documented material set per room.
- **Measured data** need checking before the paper (duplicated position rows in the classroom sheet; logs that do not match the reported RT). This is outside the code but blocks PR 10's comparison with measurements.

## 9. Working agreement

- One branch per PR (`feature/…`, `fix/…`, `docs/…`); PRs stay small enough to review in one sitting.
- Every PR adds or updates tests; CI must pass before review.
- The owner reviews and merges; no force pushes to `main`.
- Measurement data, the paper draft and copyrighted references stay outside git.
