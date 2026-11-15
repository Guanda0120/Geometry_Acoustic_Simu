# Budgeted E3 / E4 cloud run

Target: Alibaba Cloud Shanghai, c9i.8xlarge, 32 vCPU / 64 GiB,
Ubuntu x86_64, 40 GiB system disk, pay-as-you-go. Observed on 2026-11-06:
6.239 CNY/hour including the disk, plus outgoing traffic at 0.8 CNY/GB.
Recheck the final quote before purchase. This is 16 cores with two threads each.

For a roughly 20 CNY budget, schedule cloud-side instance release 165 minutes
after creation (about 17.16 CNY fixed charges). Reserve the remaining budget
for traffic and billing variation. Keep setup within this same paid window.
Do not treat a process timeout or OS shutdown as stopping billing.

1. Set and verify the provider-side release time before starting work.
2. Upload only the prepared source/input bundle to the user's own instance.
3. Install Python 3.11 and uv from official sources. In the repository, run
   `uv sync --frozen --no-dev --python 3.11`.
4. Set `DEADLINE_EPOCH` to at least ten minutes before instance release, then
   run `bash scripts/cloud/run_e3_e4.sh` in a detached session.
5. Download completed results and logs periodically, and again before release.
   The system disk will be deleted with the instance; do not rely on recovering
   files after release. If setup or computation is slow, preserve partial work
   and respect the deadline rather than extending the paid window.

The runner keeps the existing scientific defaults (all three rooms, five seeds,
100 rays/m3), runs E4 before E3, and saves each room separately. This produces
per-room CSV/figures instead of the usual combined figure. It does not promise
completion within the budget. It stops on errors or timeout, with status.tsv
and logs identifying incomplete work. No instance is created by these scripts.

# E6 cloud run (fault reinsertion)

Same instance type and the same steps 1-5 as above, with `bash scripts/cloud/run_e6.sh` in step 4. The
runner executes the known-answer checks for the baseline and the seven faults first, then the classroom and
the lecture hall (20 seeds, 100 rays/m3, baseline and seven mutants each); `CASES="classroom lecture_hall
mf_hall"` adds the hall. Each step writes its own folder under `output/cloud/<run>/fault_reinsertion/`
(`checks`, then one per room), so the matrix is spread over the `matrix.csv` files of the steps. Expected
well under one hour on 32 vCPU without the hall; schedule the instance release accordingly (90 minutes
leaves room for the setup).
