import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

FILE_DATASET = "dataset_kos_yogyakarta.csv"
PATH_KLASTER = os.path.join("models", "model_klaster_kmeans.pkl")
PATH_ASISTEN = os.path.join("models", "model_asisten_kos.pkl")


def latih_asisten_kos():
    if not os.path.exists(FILE_DATASET):
        print(f"Error: File {FILE_DATASET} tidak ditemukan.")
        return

    df = pd.read_csv(FILE_DATASET)
    df["jumlah_fasilitas"] = df["fasilitas"].fillna("").apply(
        lambda x: len([f.strip() for f in str(x).split(",") if f.strip()]) if str(x).strip() else 1
    )

    # Gabungkan segmen klaster K-Means jika model sudah tersedia
    if os.path.exists(PATH_KLASTER):
        artefak_km = joblib.load(PATH_KLASTER)
        X_scaled = artefak_km["scaler"].transform(df[artefak_km["fitur_cluster"]])
        df["segmen_kos"] = pd.Series(artefak_km["kmeans"].predict(X_scaled)).map(artefak_km["mapping_label"])
    else:
        df["segmen_kos"] = "Umum"

    # Buat korpus profil teks kaya konteks untuk setiap kos
    def buat_dokumen(row):
        kategori_harga = (
            "murah terjangkau hemat ekonomis" if row["harga_bulanan"] <= 1000000
            else "menengah standar nyaman" if row["harga_bulanan"] <= 1800000
            else "eksklusif mewah lengkap premium"
        )
        status_teks = "tersedia kosong siap huni" if row["kamar_tersedia"] > 0 else "penuh habis"
        return (
            f"{row['nama_kos']} kos {row['tipe_kos']} di kemantren kecamatan {row['kecamatan']} yogyakarta jogja "
            f"segmen {row['segmen_kos']} {kategori_harga} {status_teks} "
            f"fasilitas {row['fasilitas']} ulasan {row['ulasan']}"
        ).lower()

    df["dokumen_pencarian"] = df.apply(buat_dokumen, axis=1)

    # Latih TF-IDF Vectorizer (unigram & bigram)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=3000)
    tfidf_matrix = vectorizer.fit_transform(df["dokumen_pencarian"])

    os.makedirs("models", exist_ok=True)
    artefak_asisten = {
        "vectorizer": vectorizer,
        "tfidf_matrix": tfidf_matrix,
        "data_kos": df.drop(columns=["dokumen_pencarian"])
    }
    joblib.dump(artefak_asisten, PATH_ASISTEN)

    print("=== PELATIHAN ASISTEN KOS SELESAI ===")
    print(f"Total profil kos terindeks : {len(df)} unit")
    print(f"Dimensi fitur kosakata TF-IDF: {tfidf_matrix.shape[1]} term")
    print(f"Model asisten disimpan ke  : {PATH_ASISTEN}")


if __name__ == "__main__":
    latih_asisten_kos()