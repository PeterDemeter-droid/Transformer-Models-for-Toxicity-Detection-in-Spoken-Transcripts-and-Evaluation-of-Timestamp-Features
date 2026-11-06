Transformer Models for Toxicity Detection in Spoken Transcripts and Evaluation of Timestamp Features

This release candidate contains the completed FLAN-T5 diagnostic evaluation and data-export code. It is not a new independent benchmark release.

Results
Text-only accuracy/macro-F1: 0.841584 / 0.808463.
GB-only: 0.707921 / 0.459133.
Corrected fusion with GB coefficient 0.22: 0.831683 / 0.794420.
All three use 202 fixed test rows (148 non-toxic; 54 toxic). Two text-only true positives become false negatives after fusion.
The exploratory paired macro-F1 interval is [-0.036230, 0.000000].
Reweighting the same saved scores with GB coefficients 0.02 or 0.05 leaves all text-only labels unchanged. FLANT5_weight_comparison.csv distinguishes this from separately supplied low-weight reports with unverified equivalence.

Data limitations
Original input counts are 11824/2602/2550 train/valid/test; complete-case filtering across ALL columns retains 930/207/202. Roughly 92.1% of rows are removed. Three retained test rows have normalized text also present in training. Recording-level independence and historical transformer training provenance are unverified. The evaluation status is diagnostic_only_independence_not_established.
Original benchmark resources: https://github.com/Sreyan88/Toxicity-Detection-in-Spoken-Utterances
Public access to source resources is not a blanket redistribution license for every component corpus.

What is included
results/flant5_paired_202_diagnostic/: JSON audits/configuration, checkpoint hashes, test manifest, per-row predictions/scores, metrics, bootstrap summary and classification reports.
Export_verified_data.ipynb and .py: exact input-hash checked preprocessing export, without model training.
figures/: simplified editable Transformer diagram and PNG.
FLANT5_weight_comparison.csv: paired score reweighting and clearly separated external reports.

How to export the local data
Open the notebook and run its single cell after setting DATA_DIR, DIAGNOSTIC_DIR and OUTPUT_DIR. The diagnostic directory must point to the completed run. If OUTPUT_DIR already exists, choose a new name; the script deliberately refuses overwrite. Alternatively run %run Export_verified_data.py in Jupyter.
The exporter requires the original local CSVs and pandas/numpy/scikit-learn. It writes UTF-8 exports, manifests, missing-column audit, current GB-refit fold memberships and a model-data registry. It does not recover historical transformer training folds or prove train/test independence. The export notebook was syntactically and synthetically tested; original input CSVs are not included in this release candidate.

Release completion
Add the actual evaluation notebook and associated training notebooks, retaining historical code-defect disclosures. Add generated data manifests and preprocessing audit from the author's DataLab export. Checkpoint files are not included here; identify any released weights using the provided hashes. Do not infer data identity from a common sample count or filename. Link supplementary weight/data archives with their actual contents and applicable licenses when released.
The evaluation environment JSON records the current evaluation environment, not all historical training environments.
