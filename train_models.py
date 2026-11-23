import os
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

FILE_DATASET = "dataset_kos_yogyakarta.csv"
FOLDER_MODELS = "models"


def hitung_jumlah_fasilitas(teks_fasilitas):
    if pd.isna(teks_fasilitas) or str(teks_fasilitas).strip() == "":
        return 1
    return len([f.strip() for f in str(teks_fasilitas).split(",") if f.strip()])


def latih_semua_model():
    if not os.path.exists(FILE_DATASET):
        raise FileNotFoundError(f"File {FILE_DATASET} tidak ditemukan! Jalankan crawl_jogja.py terlebih dahulu.")

    os.makedirs(FOLDER_MODELS, exist_ok=True)
    df = pd.read_csv(FILE_DATASET)
    jumlah_awal = len(df)

    # =========================================================================
    # PEMBERSIHAN DATA & PENGHAPUSAN DUPLIKAT SEBELUM TRAINING
    # =========================================================================
    df["nama_kos_clean"] = df["nama_kos"].astype(str).str.strip().str.lower()
    df = (
        df.dropna(subset=["nama_kos", "kecamatan", "tipe_kos", "harga_bulanan", "luas_kamar", "rating"])
        .query("harga_bulanan > 0 and luas_kamar > 0")
        .drop_duplicates(subset=["nama_kos_clean"], keep="last")
        .drop(columns=["nama_kos_clean"])
        .reset_index(drop=True)
    )
    print(f"Data awal: {jumlah_awal} baris -> Bersih & bebas duplikat: {len(df)} baris siap dilatih.")

    # Rekayasa fitur (Feature Engineering) dari teks fasilitas
    df["jumlah_fasilitas"] = df["fasilitas"].apply(hitung_jumlah_fasilitas)
    df["ada_ac"] = df["fasilitas"].str.contains("AC", case=False, na=False).astype(int)
    df["ada_km_dalam"] = df["fasilitas"].str.contains("K. Mandi Dalam", case=False, na=False).astype(int)
    df["ada_wifi"] = df["fasilitas"].str.contains("WiFi", case=False, na=False).astype(int)

    # =========================================================================
    # 1. MODEL REGRESI LINIER (PREDIKSI HARGA WAJAR BULANAN)
    # =========================================================================
    fitur_kategori = ["kecamatan", "tipe_kos"]
    fitur_numerik = ["luas_kamar", "rating", "jumlah_fasilitas", "ada_ac", "ada_km_dalam", "ada_wifi"]

    X_reg = df[fitur_kategori + fitur_numerik]
    y_reg = df["harga_bulanan"]

    X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

    preprocessor_reg = ColumnTransformer(
        transformers=[
            ("kat", OneHotEncoder(handle_unknown="ignore"), fitur_kategori),
            ("num", StandardScaler(), fitur_numerik),
        ]
    )

    pipeline_regresi = Pipeline(
        steps=[
            ("preprocessor", preprocessor_reg),
            ("regressor", LinearRegression()),
        ]
    )

    pipeline_regresi.fit(X_train, y_train)
    y_pred = pipeline_regresi.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    joblib.dump(pipeline_regresi, os.path.join(FOLDER_MODELS, "model_harga_regresi.pkl"))
    print(f"[OK] Model Regresi Linier tersimpan -> R2 Score: {r2:.3f} | MAE: Rp {mae:,.0f}")

    # =========================================================================
    # 2. MODEL K-MEANS CLUSTERING (SEGMENTASI KELAS KOS: EKONOMIS, MENENGAH, EKSKLUSIF)
    # =========================================================================
    fitur_cluster = ["harga_bulanan", "luas_kamar", "rating", "jumlah_fasilitas"]
    scaler_cluster = StandardScaler()
    X_cluster_scaled = scaler_cluster.fit_transform(df[fitur_cluster])

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_cluster_scaled)
    df["cluster_raw"] = clusters

    # Urutkan label klaster berdasarkan rata-rata harga (0 = Ekonomis, 1 = Menengah, 2 = Eksklusif)
    urutan_harga = df.groupby("cluster_raw")["harga_bulanan"].mean().sort_values().index.tolist()
    mapping_label = {
        urutan_harga[0]: "Ekonomis",
        urutan_harga[1]: "Menengah",
        urutan_harga[2]: "Eksklusif",
    }

    artefak_cluster = {
        "scaler": scaler_cluster,
        "kmeans": kmeans,
        "mapping_label": mapping_label,
        "fitur_cluster": fitur_cluster,
    }
    joblib.dump(artefak_cluster, os.path.join(FOLDER_MODELS, "model_klaster_kmeans.pkl"))
    print("[OK] Model K-Means Clustering tersimpan -> Label Klaster:", mapping_label)


if __name__ == "__main__":
    latih_semua_model()