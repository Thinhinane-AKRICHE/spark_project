import librosa
import numpy as np
import os, sys
import joblib

os.environ["PYSPARK_PYTHON"]        = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel

RF_MODEL_PATH  = os.path.join(os.path.dirname(__file__), "..", "scripts", "models", "rf_model")
XGB_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "scripts", "models", "xgb_model.joblib")

def get_spark():
    return (SparkSession.builder
            .master("local[*]")
            .config("spark.driver.memory", "4g")
            .config("spark.sql.shuffle.partitions", "4")
            .getOrCreate())

def get_rf_model(spark):
    return PipelineModel.load(RF_MODEL_PATH)

def extraire_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)
    mfccs_std = np.std(mfccs, axis=1)

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    chroma_std = np.std(chroma, axis=1)

    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_mean = np.mean(mel, axis=1)
    mel_std = np.std(mel, axis=1)

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    rolloff  = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr      = librosa.feature.zero_crossing_rate(y)
    rms      = librosa.feature.rms(y=y)

    features = {
        "spectral_centroid_mean": float(np.array(centroid.mean(), dtype=np.float64)),
        "spectral_centroid_std":  float(np.array(centroid.std(),  dtype=np.float64)),
        "spectral_rolloff_mean":  float(np.array(rolloff.mean(),  dtype=np.float64)),
        "spectral_rolloff_std":   float(np.array(rolloff.std(),   dtype=np.float64)),
        "zcr_mean": float(np.array(zcr.mean(), dtype=np.float64)),
        "zcr_std":  float(np.array(zcr.std(),  dtype=np.float64)),
        "rms_mean": float(np.array(rms.mean(), dtype=np.float64)),
        "rms_std":  float(np.array(rms.std(),  dtype=np.float64)),
    }
    array_cols = {
        "mfccs_mean": mfccs_mean, "mfccs_std": mfccs_std,
        "chroma_mean": chroma_mean, "chroma_std": chroma_std,
        "mel_spec_mean": mel_mean, "mel_spec_std": mel_std,
    }
    for col_name, values in array_cols.items():
        for i, v in enumerate(values):
            features[f"{col_name}_{i}"] = float(np.float64(v))

    return features

def predire_rf(file_path):
    spark    = get_spark()
    model    = get_rf_model(spark)
    features = extraire_features(file_path)
    df       = spark.createDataFrame([features])
    result   = model.transform(df)

    prediction    = result.select("prediction").collect()[0][0]
    label_indexer = model.stages[0]
    classe        = label_indexer.labels[int(prediction)]

    proba_vecteur = result.select("probability").collect()[0][0]
    probas = {label_indexer.labels[i]: float(proba_vecteur[i]) for i in range(len(label_indexer.labels))}

    return classe, probas

def predire_xgb(file_path):
    spark    = get_spark()
    rf_model = get_rf_model(spark)   # on rÃ©utilise les stages de preprocessing
    features = extraire_features(file_path)
    df       = spark.createDataFrame([features])

    # stages[1] = VectorAssembler, stages[2] = StandardScalerModel
    assembler = rf_model.stages[1]
    scaler    = rf_model.stages[2]
    df_scaled = scaler.transform(assembler.transform(df))

    X   = np.array(df_scaled.select(scaler.getOutputCol()).collect()[0][0]).reshape(1, -1)
    xgb = joblib.load(XGB_MODEL_PATH)

    proba     = xgb.predict_proba(X)[0]
    labels    = rf_model.stages[0].labels
    pred_idx  = int(np.argmax(proba))
    classe    = labels[pred_idx]
    probas    = {labels[i]: float(proba[i]) for i in range(len(labels))}

    return classe, probas