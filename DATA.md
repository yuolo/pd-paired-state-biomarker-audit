# Data availability

The raw recordings are **not** included in this repository (they are large and publicly
hosted). The committed result tables in `figures/data/` are enough to reproduce every
figure without downloading anything. To re-run the analyses in `analysis/` from scratch,
obtain the two public datasets below.

## ds004998 — simultaneous MEG + STN-LFP (Rassoulou et al., 2024)
- OpenNeuro DOI: **10.18112/openneuro.ds004998**
- 18 PD patients, simultaneous 306-channel MEG and bipolar STN-LFP, MedOff/MedOn,
  Hold and Move tasks, UPDRS-III in both states.

## OXF — STN-LFP off/on + DBS (Wiest et al., 2022)
- Oxford data repository: `data.mrc.ox.ac.uk/stn-lfp-on-off-and-dbs`
- DOI: **10.5287/bodleian:mzJ7YwXvo** (CC BY-SA 4.0)
- 17 PD patients (30 complete off/on hemisphere pairs) across three centres, plus a
  separate set of 26 hemispheres with paired baseline and 130 Hz STN-DBS recordings.

## How the analysis scripts expect data
The scripts in `analysis/` were written against the original research-repository layout
(raw datasets under a local `data/` tree, intermediate results under `outputs/`). After
downloading the datasets, point the loaders in `src/data_loading/` at your local copies.
Each script writes its result CSVs, which are the inputs already provided in
`figures/data/` (so the figures can be reproduced independently of this step).

## Symptom-axis analysis plan
The Boundary-3 symptom-axis analysis followed a fixed testing procedure (targets,
components, FDR family, and an honest negative-result rule) defined before the
clinical-label modelling. The plan is written out in `analysis_plan/`.
