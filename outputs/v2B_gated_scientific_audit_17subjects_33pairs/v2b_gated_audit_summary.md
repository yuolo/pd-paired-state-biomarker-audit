# v2B-Gated Scientific Audit

Scope: selected experimental v2B-gated variant audited against frozen v2/v2A on cached ds004998 Hold/Move pairs only.

## Cohort Checks
- complete_pairs: 33
- subjects_with_complete_pairs: 17
- Rest excluded: True
- sub-BYJoWR excluded: True
- split files collapsed logically: True
- top_k: 5
- aperiodic_alpha: 0.5

## Primary Observed Metrics
- v2_reference: top1=0.727, MRR=0.826, failures=9, subject_gap=0.000
- v2A_top5_aperiodic_rerank: top1=0.848, MRR=0.893, failures=5, subject_gap=0.000
- v2B_gated_aperiodic_top5: top1=0.848, MRR=0.893, failures=5, subject_gap=0.000

## Severity Ladder
- original_frozen_matched_pool / v2A_top5_aperiodic_rerank: top1=0.848, MRR=0.893, subject_gap=0.000
- original_frozen_matched_pool / v2B_gated_aperiodic_top5: top1=0.848, MRR=0.893, subject_gap=0.000
- task_side_quality_ignored / v2A_top5_aperiodic_rerank: top1=0.758, MRR=0.835, subject_gap=0.000
- task_side_quality_ignored / v2B_gated_aperiodic_top5: top1=0.758, MRR=0.835, subject_gap=0.000
- task_side_quality_strict / v2A_top5_aperiodic_rerank: top1=0.879, MRR=0.915, subject_gap=0.000
- task_side_quality_strict / v2B_gated_aperiodic_top5: top1=0.879, MRR=0.915, subject_gap=0.000
- task_family_side_quality_ignored / v2A_top5_aperiodic_rerank: top1=0.758, MRR=0.835, subject_gap=0.000
- task_family_side_quality_ignored / v2B_gated_aperiodic_top5: top1=0.758, MRR=0.835, subject_gap=0.000
- original_plus_same_subject_wrong_task_side / v2A_top5_aperiodic_rerank: top1=0.485, MRR=0.704, subject_gap=0.333
- original_plus_same_subject_wrong_task_side / v2B_gated_aperiodic_top5: top1=0.879, MRR=0.907, subject_gap=0.000
- all_medon_candidates / v2A_top5_aperiodic_rerank: top1=0.394, MRR=0.570, subject_gap=0.303
- all_medon_candidates / v2B_gated_aperiodic_top5: top1=0.667, MRR=0.741, subject_gap=0.000
- true_plus_all_other_subject_medon / v2A_top5_aperiodic_rerank: top1=0.576, MRR=0.676, subject_gap=0.000
- true_plus_all_other_subject_medon / v2B_gated_aperiodic_top5: top1=0.667, MRR=0.752, subject_gap=0.000

## Warnings
- Strict task/side/quality pool has queries below MIN_DISTRACTORS; interpret strict-pool metrics cautiously.
- No independent external validation cohort was used in this audit.
- v2B-gated is experimental; frozen v2A remains the locked reference pipeline.

Boundary statement: v2B-gated remains experimental; this is paired-state identifiability audit evidence, not clinical prediction, treatment recommendation, DBS optimization, or causal medication-effect estimation.
