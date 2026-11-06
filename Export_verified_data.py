# Run this entire cell in Jupyter. No training, command-line arguments or GPU required.
from pathlib import Path
import pandas as pd
import numpy as np
import json, hashlib, zipfile, shutil, platform
from sklearn.model_selection import StratifiedKFold

DATA_DIR = Path("data")
DIAGNOSTIC_DIR = Path("flant5_paired_202_diagnostic")
OUTPUT_DIR = Path("repository_data_export")
ENCODING = "MacRoman"  # Reproduces the audited run; does not repair existing encoding artifacts.
# Only FLAN-T5's current diagnostic input files are verified. Do not assign these
# files to historical BERT/ModernBERT/DistilBERT/XLM-R runs without their evidence.
EXPECTED = {'train': {'raw_rows': 11824, 'retained_rows': 930, 'dropped_rows': 10894, 'class_counts': {'0': 696, '1': 234}, 'sha256': '2c69820ebf0afae1cf4ab30a049644b88d016c508de6b2b75af887f5c372af34'}, 'valid': {'raw_rows': 2602, 'retained_rows': 207, 'dropped_rows': 2395, 'class_counts': {'0': 156, '1': 51}, 'sha256': 'd20596f215f3fb1f9aa398217336a52feb1a936b945be01a6e1525976dd59cfc'}, 'test': {'raw_rows': 2550, 'retained_rows': 202, 'dropped_rows': 2348, 'class_counts': {'0': 148, '1': 54}, 'sha256': '5e70485f7167ff6a32aaaf2e3bf03c7861f54d03e2364a9cb0cdb158a550086c'}}

def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

def seconds(value):
    parts = str(value).strip().split(":")
    if len(parts) not in (1, 2, 3):
        raise ValueError(f"Invalid timestamp: {value!r}")
    result = sum(float(x) * 60**i for i, x in enumerate(reversed(parts)))
    if not np.isfinite(result):
        raise ValueError(f"Nonfinite timestamp: {value!r}")
    return result

# Validate ALL inputs before creating an export. Counts alone do not prove identity.
frames = {}; audits = {}; manifests = {}; raw_frames = {}
for split in ("train", "valid", "test"):
    src = DATA_DIR / f"{split}.csv"
    digest = sha(src)
    if digest != EXPECTED[split]["sha256"]:
        raise ValueError(f"{src} differs from the audited input. No export was created. "
                         "Use the original files; do not alter EXPECTED merely to bypass the check.")
    raw = pd.read_csv(src, encoding=ENCODING, encoding_errors="strict")
    required = ["text", "label2a", "Starting", "Ending"]
    if not set(required).issubset(raw.columns):
        raise ValueError(f"Missing columns in {src}")
    if {"csv_row", "sample_hash"} & set(raw.columns):
        raise ValueError("Input already contains audit columns; use the original CSV.")
    retained = raw.notna().all(axis=1)
    d = raw.loc[retained].copy()  # Exact historical complete-case policy, ALL columns.
    d["csv_row"] = d.index + 2  # pandas record index + header, not physical line for multiline CSV.
    d["label2a"] = pd.to_numeric(d["label2a"], errors="raise")
    if not d.label2a.isin([0, 1]).all():
        raise ValueError("Labels must be 0 or 1")
    d["label2a"] = d.label2a.astype(int)
    for col in ("Starting", "Ending"):
        d[col] = d[col].map(seconds)
    if (d.Ending < d.Starting).any():
        raise ValueError("Ending precedes Starting")
    d["sample_hash"] = [hashlib.sha256(json.dumps([str(t), float(a), float(b), int(y)],
        ensure_ascii=False).encode()).hexdigest()
        for t,a,b,y in zip(d.text,d.Starting,d.Ending,d.label2a)]
    d = d.reset_index(drop=True)
    if len(d) != EXPECTED[split]["retained_rows"]:
        raise ValueError(f"Unexpected retained count for {split}")
    frames[split] = d; raw_frames[split] = raw.loc[retained].copy()
    manifests[split] = d[["csv_row", "sample_hash", "label2a"]].copy()
    audits[split] = {"source_sha256":digest,"raw_rows":len(raw),"retained_rows":len(d),
        "excluded_rows":int((~retained).sum()),
        "class_counts":{str(k):int(v) for k,v in d.label2a.value_counts().items()},
        "missing_by_column":{str(k):int(v) for k,v in raw.isna().sum().items()}}
    excluded = pd.DataFrame({"csv_row":raw.index[~retained]+2,
        "missing_columns":[" | ".join(map(str,raw.columns[row]))
                           for row in raw.isna().loc[~retained].to_numpy()]})
    audits[split]["excluded_records"] = excluded.to_dict("records")

# Optional match against the exact diagnostic test manifest.
manifest_path = DIAGNOSTIC_DIR / "test_manifest.csv"
if manifest_path.exists():
    old = pd.read_csv(manifest_path)
    if not old[["csv_row","sample_hash","label2a"]].equals(manifests["test"]):
        raise ValueError("Test manifest mismatch; no export created.")

# Fail on existing output to avoid silently replacing a prior release.
OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
set_id = hashlib.sha256("".join(EXPECTED[s]["sha256"] for s in ("train","valid","test")).encode()).hexdigest()[:16]
set_dir = OUTPUT_DIR / "data" / set_id; set_dir.mkdir(parents=True)
for split,d in frames.items():
    # Ready-to-inspect data: timestamps are seconds, text decoding matches the audited run.
    d.to_csv(set_dir / f"{split}_processed.csv", index=False, encoding="utf-8")
    # These preserve the retained original timestamp strings. Both exports use UTF-8.
    raw_frames[split].to_csv(set_dir / f"{split}_retained_source.csv", index=False, encoding="utf-8")
    manifests[split].to_csv(set_dir / f"{split}_manifest.csv", index=False)
    pd.DataFrame(audits[split].pop("excluded_records")).to_csv(
        set_dir / f"{split}_excluded_rows.csv", index=False)

# Reconstruct ONLY the new GB refit's fold membership, not historical transformer training folds.
train = frames["train"]
fold_rows = []
for fold,(tr,va) in enumerate(StratifiedKFold(5,shuffle=True,random_state=42).split(train,train.label2a),1):
    for role,indices in (("gb_fit",tr),("gb_holdout_unused",va)):
        for i in indices:
            fold_rows.append({"fold":fold,"role":role,"csv_row":int(train.csv_row.iloc[i]),
                              "sample_hash":train.sample_hash.iloc[i]})
pd.DataFrame(fold_rows).to_csv(set_dir / "current_gb_refit_folds.csv",index=False)

normal=lambda d:d.text.astype(str).str.casefold().str.replace(r"\s+"," ",regex=True).str.strip()
overlap=[]
for split in ("train","valid"):
    other=set(normal(frames[split]))
    for i,text in enumerate(normal(frames["test"])):
        if text in other:
            overlap.append({"test_csv_row":int(frames["test"].csv_row.iloc[i]),
                            "overlaps_split":split,"sample_hash":frames["test"].sample_hash.iloc[i]})
pd.DataFrame(overlap,columns=["test_csv_row","overlaps_split","sample_hash"]).to_csv(
    set_dir / "text_overlap_manifest.csv",index=False)
dump(OUTPUT_DIR / "preprocessing_audit.json",audits)
dump(OUTPUT_DIR / "model_data_registry.json",{
    "flant5_current_diagnostic":{"dataset_id":set_id,"test_n":202,
        "status":"verified current diagnostic input; historical transformer-training provenance unverified",
        "variants":["text_only","gb_only","corrected_hybrid"],
        "shared_data_note":"All three use this one data directory; no duplicate dataset copies."},
    "historical_flant5_low_weight_runs":{"status":"unverified; do not infer identical rows from equal support"},
    "bert":{"status":"historical data identity unverified"},
    "modernbert":{"status":"historical data identity unverified"},
    "distilbert":{"status":"historical data identity unverified"},
    "xlmr":{"status":"historical data identity unverified"}})

# Optional evidence exports; no model weights or unsafe pickle deserialization.
evidence=OUTPUT_DIR/"evaluation";evidence.mkdir()
for name in ["metrics.json","metrics_summary.csv","paired_predictions.csv","paired_bootstrap.json",
             "checkpoint_hashes.json","environment.json","run_configuration.json","test_manifest.csv"]:
    src=DIAGNOSTIC_DIR/name
    if src.exists():shutil.copy2(src,evidence/name)
if (evidence/"paired_predictions.csv").exists():
    from sklearn.metrics import accuracy_score, f1_score
    p=pd.read_csv(evidence/"paired_predictions.csv")
    if not p[["csv_row","sample_hash","label2a"]].equals(manifests["test"]):
        raise ValueError("Copied predictions mismatch manifest; do not release this export.")
    z=p[["text_logit_0","text_logit_1"]].to_numpy()
    g=p[["gb_probability_0","gb_probability_1"]].to_numpy()
    rows=[]
    for w in [0.0,0.02,0.05,0.22]:
        pred=((1-w)*z+w*g).argmax(1)
        rows.append({"gb_coefficient":w,"accuracy":accuracy_score(p.label2a,pred),
                     "macro_f1":f1_score(p.label2a,pred,average="macro"),
                     "changed_vs_text":int((pred!=z.argmax(1)).sum())})
    pd.DataFrame(rows).to_csv(evidence/"same_prediction_weight_sensitivity.csv",index=False)

(OUTPUT_DIR/"README.txt").write_text(
    "Reproduces the audited FLAN-T5 diagnostic preprocessing, not a newly independent benchmark.\n"
    "Policy: MacRoman decode, dropna across ALL original columns, parse timestamps to seconds.\n"
    "Exported CSVs are UTF-8; read with encoding='utf-8', not MacRoman.\n"
    "The all-column filter removes about 92% of rows. Three test rows share normalized text with train.\n"
    "No resampling, silent overlap removal, relabeling or transformer retraining was performed.\n"
    "Registry maps verified variants to one dataset directory to avoid duplicate releases.\n"
    "Other models require their original files/manifests; checkpoint filenames alone do not recover them.\n"
    "GB fold export describes current refit only. Full training provenance remains unverified.\n"
    "Before distributing transcripts/audio/checkpoints, retain the original source attribution and applicable licenses.\n",
    encoding="utf-8")
files={str(p.relative_to(OUTPUT_DIR)):sha(p) for p in OUTPUT_DIR.rglob("*") if p.is_file()}
dump(OUTPUT_DIR/"SHA256.json",files)
archive=shutil.make_archive(str(OUTPUT_DIR),"zip",root_dir=OUTPUT_DIR)
print("HOTOVO:",archive)
print(pd.DataFrame([{**{"split":s},**{k:v for k,v in a.items() if k in ("raw_rows","retained_rows","excluded_rows")}}
                    for s,a in audits.items()]).to_string(index=False))
print("Splity sú uložené raz; registry ich priraďuje len k overenému FLAN-T5 diagnostickému behu.")
