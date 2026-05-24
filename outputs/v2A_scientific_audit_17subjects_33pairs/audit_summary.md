# v2A Scientific Audit

Scope: frozen v2/v2A paired-state retrieval audit on verified ds004998 Hold/Move MedOff-MedOn pairs only.

## Cohort Checks
- complete_pairs: 33
- subjects_with_complete_pairs: 17
- downloaded_logical_holdmove_recordings: 68
- Rest excluded: True
- sub-BYJoWR excluded: True
- split files collapsed logically: True
- top_k: 5
- aperiodic_alpha: 0.5

## Observed Metrics
- v2_reference: top1=0.727, MRR=0.826, top3=0.909, top5=0.970, failures=9
- v2A_top5_aperiodic_rerank: top1=0.848, MRR=0.893, top3=0.909, top5=0.970, failures=5

## Warnings
- none
