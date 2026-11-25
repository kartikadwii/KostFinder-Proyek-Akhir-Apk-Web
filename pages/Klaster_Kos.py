import os
import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Klaster Kos - KosFinder", page_icon="📊", layout="wide")

FILE_DATASET = "dataset_kos_yogyakarta.csv"
PATH_KLASTER = os.path.join("models", "model_klaster_kmeans.pkl")


@st.cache_data
def load_data():
    if not os.path.exists(FILE_DATASET):
        return pd.DataFrame()
    df = pd.read_csv(FILE_DATASET)
    df["jumlah_fasilitas"] = df["fasilitas"].fillna("").apply(
        lambda x: len([f.strip() for f in str(x).split(",") if f.strip()]) if str(x).strip() else 1
    )
    return df


@st.cache_resource
def load_klaster_model():
    return joblib.load(PATH_KLASTER) if os.path.exists(PATH_KLASTER) else None


st.title("📊 Segmentasi & Klaster Kos (K-Means Clustering)")
st.markdown("Pengelompokan unit kos di 14 Kemantren Kota Yogyakarta menjadi segmen **Ekonomis**, **Menengah**, dan **Eksklusif** berdasarkan harga, luas kamar, rating, dan kelengkapan fasilitas.")

df = load_data()
artefak_kmeans = load_klaster_model()

if df.empty or artefak_kmeans is None:
    st.error("Dataset atau model klaster (`model_klaster_kmeans.pkl`) belum tersedia. Jalankan `crawl_jogja.py` dan `train_models.py` terlebih dahulu.")
    st.stop()

scaler_km = artefak_kmeans["scaler"]
kmeans_model = artefak_kmeans["kmeans"]
mapping_klaster = artefak_kmeans["mapping_label"]
fitur_km = artefak_kmeans["fitur_cluster"]

X_scaled = scaler_km.transform(df[fitur_km])
df["klaster_id"] = kmeans_model.predict(X_scaled)
df["segmen_kos"] = df["klaster_id"].map(mapping_klaster)

# Ringkasan Metrik 3 Segmen
col_m1, col_m2, col_m3 = st.columns(3)
for col_ui, nama_segmen in zip([col_m1, col_m2, col_m3], ["Ekonomis", "Menengah", "Eksklusif"]):
    sub = df[df["segmen_kos"] == nama_segmen]
    rata_harga = sub["harga_bulanan"].mean() if not sub.empty else 0
    rata_luas = sub["luas_kamar"].mean() if not sub.empty else 0
    col_ui.metric(
        label=f"Klaster {nama_segmen} ({len(sub)} unit)",
        value=f"Rp {rata_harga:,.0f}/bln",
        delta=f"Rata-rata luas {rata_luas:.1f} m²"
    )

st.markdown("---")

# Filter Segmen & Kemantren
f1, f2 = st.columns(2)
pilih_segmen = f1.multiselect("Filter Segmen Klaster", ["Ekonomis", "Menengah", "Eksklusif"], default=["Ekonomis", "Menengah", "Eksklusif"])
pilih_kec = f2.multiselect("Filter Kemantren", sorted(df["kecamatan"].unique()), default=[])

df_tampil = df[df["segmen_kos"].isin(pilih_segmen)]
if pilih_kec:
    df_tampil = df_tampil[df_tampil["kecamatan"].isin(pilih_kec)]

# Visualisasi Scatter & Bar Chart
col_g1, col_g2 = st.columns([1.4, 1])

with col_g1:
    fig_scatter = px.scatter(
        df_tampil,
        x="luas_kamar",
        y="harga_bulanan",
        color="segmen_kos",
        size="jumlah_fasilitas",
        hover_data=["nama_kos", "kecamatan", "tipe_kos", "rating", "status_kamar"],
        color_discrete_map={"Ekonomis": "#2ecc71", "Menengah": "#3498db", "Eksklusif": "#9b59b6"},
        labels={"luas_kamar": "Luas Kamar (m²)", "harga_bulanan": "Harga Bulanan (Rp)", "segmen_kos": "Segmen"},
        title="Peta Persebaran Klaster (Luas Kamar vs Harga Bulanan)"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

with col_g2:
    ringkasan_kec = df_tampil.groupby(["kecamatan", "segmen_kos"]).size().reset_index(name="jumlah")
    fig_bar = px.bar(
        ringkasan_kec,
        x="jumlah",
        y="kecamatan",
        color="segmen_kos",
        orientation="h",
        color_discrete_map={"Ekonomis": "#2ecc71", "Menengah": "#3498db", "Eksklusif": "#9b59b6"},
        labels={"jumlah": "Jumlah Unit Kos", "kecamatan": "Kemantren", "segmen_kos": "Segmen"},
        title="Distribusi Segmen Kos per Kemantren"
    )
    st.plotly_chart(fig_bar, use_container_width=True)

st.subheader("Daftar Hasil Pengelompokan Kos")
st.dataframe(
    df_tampil[["nama_kos", "kecamatan", "tipe_kos", "segmen_kos", "status_kamar", "harga_bulanan", "luas_kamar", "rating", "fasilitas"]],
    use_container_width=True,
    hide_index=True
)