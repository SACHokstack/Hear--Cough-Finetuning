import os
import site
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, recall_score
from xgboost import XGBClassifier
from joblib import dump
from tqdm import tqdm
import subprocess

# Ensure TF finds the venv CUDA libraries (cuDNN/cuBLAS)
_site = site.getsitepackages()[0]
_cudnn_lib = os.path.join(_site, "nvidia", "cudnn", "lib")
_cublas_lib = os.path.join(_site, "nvidia", "cublas", "lib")
os.environ["LD_LIBRARY_PATH"] = ":".join(
    [p for p in [_cudnn_lib, _cublas_lib, os.environ.get("LD_LIBRARY_PATH", "")] if p]
)

import tensorflow as tf

print("=== DOMAIN-AWARE DUAL HEADS + PATIENT AGGREGATION (Prize Strategy) ===")

# Force re-extract forced embeddings and save
FORCE_REEXTRACT_FORCED = True

def load_audio_ffmpeg(path, target_sr=16000):
    cmd = ["ffmpeg", "-i", path, "-f", "f32le", "-acodec", "pcm_f32le", "-ac", "1", "-ar", str(target_sr), "-"]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    return np.frombuffer(proc.stdout, dtype=np.float32)

def extract_forced_embeddings():
    forced_df = pd.read_csv("data/Forced_coughs/Forced_coughs.csv", low_memory=False)
    forced_df = forced_df[forced_df["Permission_sound"] == "Yes"].copy()
    forced_df["label"] = (forced_df["Label"] == "TB").astype(int)

    audio_folder = "data/Forced_coughs/Audio_files"
    hear_model = tf.saved_model.load("hear_model")
    serving_signature = hear_model.signatures["serving_default"]

    emb_list = []
    label_list = []
    subject_list = []

    for _, row in tqdm(forced_df.iterrows(), total=len(forced_df), desc="Extracting Forced"):
        path = os.path.join(audio_folder, row["path"] + ".wav")
        if not os.path.exists(path):
            continue
        y = load_audio_ffmpeg(path)
        if len(y) == 0:
            continue
        y = np.tile(y, int(np.ceil(32000 / len(y))))[:32000]
        emb = serving_signature(x=np.expand_dims(y.astype(np.float32), 0))["output_0"].numpy()[0]
        emb_list.append(emb)
        label_list.append(row["label"])
        subject_list.append(row["subject"])

    return np.array(emb_list), np.array(label_list), np.array(subject_list)

# Load Passive and Forced embeddings separately (you already have them)
passive_emb = np.load("passive_embeddings.npy")   # from your previous run
passive_lab = np.load("passive_labels.npy")
passive_sub = np.load("passive_subjects.npy")

if FORCE_REEXTRACT_FORCED or not (
    os.path.exists("forced_embeddings.npy")
    and os.path.exists("forced_labels.npy")
    and os.path.exists("forced_subjects.npy")
):
    forced_emb, forced_lab, forced_sub = extract_forced_embeddings()
    np.save("forced_embeddings.npy", forced_emb)
    np.save("forced_labels.npy", forced_lab)
    np.save("forced_subjects.npy", forced_sub)
else:
    forced_emb = np.load("forced_embeddings.npy")
    forced_lab = np.load("forced_labels.npy")
    forced_sub = np.load("forced_subjects.npy")

print(f"Passive: {len(passive_lab)} coughs | Forced: {len(forced_lab)} coughs")

# Patient-level 5-fold on combined subjects
all_subjects = np.concatenate([passive_sub, forced_sub])
unique_sub = np.unique(all_subjects)

# Patient-level labels for stratification
patient_labels = []
for sub in unique_sub:
    p_mask = passive_sub == sub
    f_mask = forced_sub == sub
    label = int(
        (p_mask.any() and np.any(passive_lab[p_mask] == 1)) or
        (f_mask.any() and np.any(forced_lab[f_mask] == 1))
    )
    patient_labels.append(label)
patient_labels = np.array(patient_labels)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
patient_aucs = []

for fold, (train_idx, val_idx) in enumerate(skf.split(range(len(unique_sub)), patient_labels)):
    # Train separate models
    passive_train_mask = np.isin(passive_sub, unique_sub[train_idx])
    forced_train_mask = np.isin(forced_sub, unique_sub[train_idx])

    # Passive model
    scaler_p = StandardScaler()
    Xp_train = scaler_p.fit_transform(passive_emb[passive_train_mask])
    yp_train = passive_lab[passive_train_mask]
    model_p = XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.03, eval_metric='auc', random_state=42, tree_method='hist')
    model_p.fit(Xp_train, yp_train)

    # Forced model
    scaler_f = StandardScaler()
    Xf_train = scaler_f.fit_transform(forced_emb[forced_train_mask])
    yf_train = forced_lab[forced_train_mask]
    model_f = XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.03, eval_metric='auc', random_state=42, tree_method='hist')
    model_f.fit(Xf_train, yf_train)

    # Patient-level evaluation
    val_subs = unique_sub[val_idx]
    probs = []
    y_true = []
    for sub in val_subs:
        # Passive prediction
        p_mask = passive_sub == sub
        if p_mask.sum() > 0:
            p = model_p.predict_proba(scaler_p.transform(passive_emb[p_mask]))[:,1].mean()
        else:
            p = 0.5

        # Forced prediction
        f_mask = forced_sub == sub
        if f_mask.sum() > 0:
            f = model_f.predict_proba(scaler_f.transform(forced_emb[f_mask]))[:,1].mean()
        else:
            f = 0.5

        prob = (p + f) / 2
        true = 1 if any(passive_lab[p_mask] == 1) or any(forced_lab[f_mask] == 1) else 0

        probs.append(prob)
        y_true.append(true)

    auc = roc_auc_score(y_true, probs)
    patient_aucs.append(auc)
    print(f"Fold {fold+1}: Patient AUC = {auc:.4f}")

print(f"\n🎯 FINAL Patient-level AUC with Domain-Aware Dual Heads: {np.mean(patient_aucs):.4f} ± {np.std(patient_aucs):.4f}")

# Retrain on ALL data for final models
scaler_p = StandardScaler()
Xp_all = scaler_p.fit_transform(passive_emb)
model_p = XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.03, eval_metric='auc', random_state=42, tree_method='hist')
model_p.fit(Xp_all, passive_lab)

scaler_f = StandardScaler()
Xf_all = scaler_f.fit_transform(forced_emb)
model_f = XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.03, eval_metric='auc', random_state=42, tree_method='hist')
model_f.fit(Xf_all, forced_lab)

# Save final models
dump({"model_p": model_p, "scaler_p": scaler_p, "model_f": model_f, "scaler_f": scaler_f}, "hear_tb_prize_domain_aware.joblib")
print("✅ Domain-Aware Prize model saved as hear_tb_prize_domain_aware.joblib")

print("\nReply with the new Patient AUC number. Then we immediately build Gradio demo + ablation table + full Novel Task Prize write-up.")
