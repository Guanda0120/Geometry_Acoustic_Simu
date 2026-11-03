# Experiment plan: rerunning the paper with the verified engine

Status: draft for review (Guanda, Elsa). Nothing in this plan is run until the open decisions of §6 are settled.

The thesis results and the numbers in the current manuscript were computed with the original code or with a patched
version of it. This plan reruns every simulation of the paper with `gasim`, the rewritten and verified engine, in a
way that can be repeated with one command per figure or table.

## 1. Goals

1. Replace every simulated number in the paper by a `gasim` result with stated inputs, seeds and uncertainty.
2. Separate what the code gets right (verification) from what the input data allow (validation), and quantify the
   second instead of tuning inputs until they match.
3. Keep the original code as the object of the study: the counterexample (wrong code, good agreement) and the source
   of the reinserted faults.

Not a goal: making the predictions match the measurements. Where the documented inputs cannot reproduce a room, the
plan reports by how much and why.

## 2. Conventions for every run

| Item | Convention |
|---|---|
| Code | `gasim` at a tagged commit; the commit hash is stored with every result |
| Absorption | Diffuse-field αd from the paper's Appendix A, used directly (no impedance conversion) |
| Air | ISO 9613-1. Default 20 °C, 50 % RH, 101.325 kPa (not recorded during the measurements); range 10–30 °C, 40–80 % RH as a scenario |
| Rays | 100 rays/m³ unless the convergence study (E2) shows that more are needed |
| Seeds | 10 per base case, 5 per scenario, 20 per fault mutant |
| Descriptors | T30, T20, EDT per receiver and band; C50 and D50 for the fault reinsertion (to be added to `gasim.analysis`) |
| Room value | Median over receivers per run; mean, standard deviation and range over seeds |
| Baselines | Sabine and Eyring from the same model, materials and air; Kuttruff's corrected Eyring where the field is diffuse |
| Errors | ε = (T_sim − T_meas)/T_meas in %, and in JND (5 % for T30/T20/EDT, 1 dB for C50) |
| Measured value | Mean over positions with its 95 % confidence interval (Student t) |
| Output | One script per figure or table in `experiments/`; CSV, figure and a JSON file with commit, configs, seeds and run times in `output/experiments/<name>/` |

## 3. Input register

Every input is marked as documented (D), taken from the literature (L) or assumed (A). Only D inputs enter the base
cases; L and A inputs enter labelled scenarios.

| Room | Geometry | Materials | Scattering | Seats / furniture | Measured RT |
|---|---|---|---|---|---|
| Classroom | `Real_ClassRoom.3dm`, 221.8 m³ (paper: 229.15 m³, 3.0 vs 3.1 m height) (D) | Table A1 (D) | Table A1, after Hodgson (L) | Desks and chairs not modelled (A: none) | 4 positions; rows of positions 3/4 duplicate 1/2 in the summary sheet (flag) |
| Lecture hall | `JianZuTest.3dm`, 1089.3 m³ (D) | Table A2; both vertical absorber layers (`Absorber`, `Absorption_Plane`) = glass-fibre panel (A: to confirm) | Not documented; base 0.10, range 0.05–0.40 (A) | No seating layer; scenarios S1 (thesis seat coefficient) and S2 (upholstered seats) on an assumed 145 m² (A) | 8 positions |
| Multifunction hall, before | `Real_Work.3dm`, 4389.2 m³ (D) | Table A3 (D) | Not documented; base 0.10, range 0.01–0.40 (A) | 280 seats missing; scenario: seats from the literature on 168 m² (L) | 6 positions; values are 1/3-octave RT at the octave centres (flag) |
| Multifunction hall, after | As above | Table A3 with the renovation materials on walls A, C (except the LED screen) and D, and on the ceiling (D); treated areas by material and position (see E5) | As above | As above | Paper: room mean only (1.48 … 0.98 s). B&K export of 20 Dec 2022: 6 positions, mean 1.40 … 1.06 s (flag, source to be traced) |
| Test room 9 × 7 × 5 m | `Simple_Test.3dm`, 315 m³ (D) | Table A4 (D) | Swept | – | – (CATT values not used as reference) |

Wall groups A–D follow the material and position rules in `configs/mf_hall_*.toml` (areas within 1.5 m² of the
thesis Fig. 9), not the surface normal alone.

## 4. Experiments

| ID | Question | Cases | Output (paper) | Script | Run time* |
|---|---|---|---|---|---|
| E0 | Does the code reproduce problems with known answers? | Image lattice (counts, lengths, energies), lossless box, fully diffuse box vs Kuttruff, uniform box vs Eyring, specular box vs exact lattice, energy vs ray count, scattering-rain solid angle, synthetic decays | Table 2, Fig. 5 | `verification.py` (new; reuses the pytest cases) | ~5 min |
| E1 | How close are the documented inputs to the measurements? | Classroom, lecture hall, hall before; base inputs, 10 seeds; Sabine/Eyring; scattering range as a band | Table 3, Fig. 6, receiver tables (S4) | `validation_rooms.py` (rewrite) | ~20 min |
| E2 | How many rays are needed, and does T30 drift with N? | Three rooms, 1–100 rays/m³ (lecture hall up to 300), 10 seeds each | Convergence figure, Table S16 | `ray_count.py` (extend to all rooms) | ~30 min |
| E3 | Which inputs dominate the uncertainty (error budget)? | One-at-a-time perturbation of each material (±0.05 absolute or ±25 %, whichever is larger), scattering range, air range, volume; 5 seeds; combined by root sum of squares | New figure (tornado per room and band), prediction band in Fig. 6 | `input_sensitivity.py` (new); later checked against ∂T/∂α from PR 11b | ~1–2 h |
| E4 | How far do plausible missing inputs move the result? | Lecture hall S1/S2; hall with seats and low-frequency glazing absorption; s = 0.05–0.40 | Seating results (Table S22), Discussion | `scenarios.py` (new) | ~30 min |
| E5 | Can the model support the renovation? | Surface statistics before/after (done); calibration options for seats and glazing with an identifiability check (rank of the sensitivity matrix); blind prediction after renovation against both measured data sets | Fig. 8, Fig. 9, Table 4 | `surface_statistics.py` (exists), `renovation.py` (new) | ~30 min |
| E6 | Which checks reveal which faults? | Faults F1, F2, F3, F6, F9, F16, F17 as switches in `gasim`; known-answer tests and the classroom and lecture hall; 20 seeds per mutant; criteria fixed before the runs | Table 5, Fig. 10, Tables S17–S21 | `fault_reinsertion.py` (new, with Elsa's framework) | dev 1–2 days, run ~1 h |
| E7 | (Supplement) Absorption and scattering sweeps in the test room | Sweeps of Table A4 with 10 seeds; CATT and thesis values shown, not used as reference | Tables S6–S8 | `wall_sweep.py` (exists) | ~5 min |

\* On a 10-core laptop; one run takes about 1 s (classroom), a few seconds (lecture hall) and 0.5–1 min (hall).

## 5. Order of work

1. **Batch 1 (no open dependencies):** E0, E1, E2, plus C50/D50 in `gasim.analysis`. These give the backbone of the
   paper.
2. **Batch 2:** E3, E4 — the error budget and the scenarios, which answer the question of the material data.
3. **Batch 3:** E5 — after the after-renovation data are traced.
4. **Batch 4:** E6 — together with Elsa, porting the fault switches and criteria of her framework.
5. E7 at any time.

Each batch is one pull request: scripts and tests first, reviewed; the full runs only after the review. Results are
not committed (they go to `output/`), but every figure and table can be regenerated from the scripts.

## 6. Open decisions

1. **Code base:** the paper describes and releases `gasim` (to agree with Elsa).
2. **Lecture hall:** is `Absorption_Plane` (two full-height side walls) glass-fibre panel? Where does the audience
   seating lie (area for S1/S2)?
3. **Scattering:** base 0.10 and the ranges of §3 for the halls; keep Table A1 for the classroom?
4. **Perturbation sizes for E3:** ±0.05 absolute or ±25 % relative per material; ranges for scattering and air.
5. **Treated area of the renovation:** walls A, C (except LED) and D by material and position (459 m² of panels), or
   by layer as in the thesis (all panels, 483 m²)?
6. **Measured data:** which after-renovation values the paper uses until their source is traced; whether to convert
   the 1/3-octave values of the hall to octave bands.
7. **Seeds and ray density:** 10 seeds at 100 rays/m³ as the base, unless E2 shows otherwise.
8. **Original code in E0:** run the legacy engine on the known-answer tests for the "original code" column (slow, small
   ray counts), or quote the values from Elsa's runs.
