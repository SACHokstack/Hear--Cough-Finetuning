import joblib
import numpy as np
import librosa

model = joblib.load("hear_tb_prize_domain_aware.joblib")

def predict_cough(audio_file):
    y, sr = librosa.load(audio_file, sr=16000)
    # ... get HeAR embedding here (your function)
    emb = get_hear_embedding(y)  # shape (1, 512)

    p_passive = model["model_p"].predict_proba(model["scaler_p"].transform(emb))[0,1]
    p_forced  = model["model_f"].predict_proba(model["scaler_f"].transform(emb))[0,1]
    
    return (p_passive + p_forced) / 2  # final TB probability
