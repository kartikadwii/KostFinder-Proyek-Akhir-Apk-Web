import os
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Prediksi Harga Kos - KosFinder", page_icon="💰", layout="wide")

FILE_DATASET = "dataset_kos_yogyakarta.csv"
PATH_REGRESI = os.path.join("models", "model_harga_regresi.pkl")
PATH_KLASTER = os.path.join("models", "model_klaster_kmeans.pkl")


@st.cache_data
def load_data():
    return pd.read_csv(FILE_DATASET) if os.path.exists(FILE_DATASET) else pd.DataFrame()


@st.cache_resource
def load_models():
    model_reg = joblib.load(PATH_REGRESI) if os.path.exists(PATH_REGRESI) else None
    artefak_km = joblib.load(PATH_KLASTER) if os.path.exists(PATH_KLASTER) else None
    return model_reg, artefak_km


st.title("💰 Kalkulator Prediksi Harga Sewa Kos Wajar")
st.markdown("Gunakan model **Linear Regression** untuk menghitung estimasi harga sewa bulanan yang wajar dan mengevaluasi apakah harga penawaran pemilik kos termasuk murah, wajar, atau kemahalan.")

df = load_data()
model_regresi, artefak_kmeans = load_models()

if df.empty or model_regresi is None:
    st.error("Dataset atau model regresi (`model_harga_regresi.pkl`) belum tersedia. Jalankan `crawl_jogja.py` dan `train_models.py` terlebih dahulu.")
    st.stop()

col_in1, col_in2 = st.columns(2)

with col_in1:
    st.subheader("1. Lokasi & Spesifikasi Kamar")
    pilih_kecamatan = st.selectbox("Pilih Kemantren (Kota Yogyakarta)", sorted(df["kecamatan"].unique()))
    pilih_tipe = st.selectbox("Tipe Penghuni Kos", ["Putri", "Putra", "Campur"])
    input_luas = st.slider("Perkiraan Luas Kamar (m²)", min_value=6.0, max_value=28.0, value=12.0, step=0.5)
    input_rating = st.slider("Target Rating Minimal", min_value=3.0, max_value=5.0, value=4.6, step=0.1)

with col_in2:
    st.subheader("2. Kelengkapan Fasilitas & Harga Penawaran")
    fas_wifi = st.checkbox("WiFi Kencang", value=True)
    fas_ac = st.checkbox("AC (Pendingin Ruangan)", value=True)
    fas_km_dalam = st.checkbox("Kamar Mandi Dalam", value=True)
    fas_kloset = st.checkbox("Kloset Duduk", value=False)
    fas_akses24 = st.checkbox("Akses 24 Jam", value=True)
    fas_parkir = st.checkbox("Parkir Motor Luas", value=True)

    harga_penawaran = st.number_input(
        "Harga Penawaran Pemilik Kos (Opsional - Rp/bulan)",
        min_value=0,
        max_value=15000000,
        value=1500000,
        step=50000,
        help="Masukkan harga yang ditawarkan pemilik kos untuk membandingkannya dengan estimasi harga wajar pasar."
    )

total_fasilitas = max(1, sum([fas_wifi, fas_ac, fas_km_dalam, fas_kloset, fas_akses24, fas_parkir]))

if st.button("Hitung Prediksi Harga Wajar", type="primary", use_container_width=True):
    data_uji = pd.DataFrame([{
        "kecamatan": pilih_kecamatan,
        "tipe_kos": pilih_tipe,
        "luas_kamar": input_luas,
        "rating": input_rating,
        "jumlah_fasilitas": total_fasilitas,
        "ada_ac": int(fas_ac),
        "ada_km_dalam": int(fas_km_dalam),
        "ada_wifi": int(fas_wifi),
    }])

    prediksi_harga = float(model_regresi.predict(data_uji)[0])
    prediksi_harga = max(450000, round(prediksi_harga / 25000) * 25000)

    # Tentukan segmen klaster jika model K-Means tersedia
    segmen_prediksi = "-"
    if artefak_kmeans is not None:
        fitur_uji_km = pd.DataFrame([{
            "harga_bulanan": prediksi_harga,
            "luas_kamar": input_luas,
            "rating": input_rating,
            "jumlah_fasilitas": total_fasilitas
        }])
        klaster_id = int(artefak_kmeans["kmeans"].predict(artefak_kmeans["scaler"].transform(fitur_uji_km))[0])
        segmen_prediksi = artefak_kmeans["mapping_label"][klaster_id]

    st.markdown("---")
    res_c1, res_c2, res_c3 = st.columns(3)
    res_c1.metric("Estimasi Harga Wajar Pasar", f"Rp {prediksi_harga:,.0f} / bln")
    res_c2.metric("Kategori Kelas Kos", f"Segmen {segmen_prediksi}")

    if harga_penawaran > 0:
        selisih = harga_penawaran - prediksi_harga
        persen = (selisih / prediksi_harga) * 100
        if persen <= -5:
            status_harga = "Sangat Terjangkau (Underpriced)"
        elif persen <= 8:
            status_harga = "Harga Wajar (Fair Price)"
        else:
            status_harga = "Di Atas Rata-Rata (Overpriced)"

        res_c3.metric(
            "Evaluasi Harga Penawaran",
            status_harga,
            delta=f"Rp {selisih:+,.0f} ({persen:+.1f}%)",
            delta_color="inverse"
        )

    # Tampilkan pembanding kos serupa di kemantren yang sama
    st.subheader(f"Referensi Kos Serupa di Kemantren {pilih_kecamatan}")
    pembanding = df[(df["kecamatan"] == pilih_kecamatan) & (df["kamar_tersedia"] > 0)].copy()
    if not pembanding.empty:
        pembanding["selisih_harga"] = (pembanding["harga_bulanan"] - prediksi_harga).abs()
        pembanding = pembanding.sort_values("selisih_harga").head(5)
        st.dataframe(
            pembanding[["nama_kos", "tipe_kos", "status_kamar", "harga_bulanan", "luas_kamar", "rating", "fasilitas"]],
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info(f"Belum ada data kos dengan kamar tersedia di Kemantren {pilih_kecamatan}.")