import os
import joblib
import pandas as pd
import plotly.express as px
import pydeck as pdk
import streamlit as st

st.set_page_config(
    page_title="KosFinder Yogyakarta - Pencarian & Analisis Kos Cerdas",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

FILE_DATASET = "dataset_kos_yogyakarta.csv"
PATH_KLASTER = os.path.join("models", "model_klaster_kmeans.pkl")


@st.cache_data
def load_dataset():
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


st.title("🏠 KosFinder Kota Yogyakarta")
st.markdown(
    "Platform **Pencarian, Pemetaan Spasial, dan Analisis Harga Kos Cerdas** berbasis *Machine Learning* "
    "yang mencakup seluruh **14 Kemantren di Kota Yogyakarta**."
)

df = load_dataset()
artefak_kmeans = load_klaster_model()

if df.empty:
    st.error("File `dataset_kos_yogyakarta.csv` tidak ditemukan. Pastikan dataset tersedia di folder proyek.")
    st.stop()

# Tambahkan label segmen klaster jika model K-Means tersedia
if artefak_kmeans is not None:
    scaler_km = artefak_kmeans["scaler"]
    kmeans_model = artefak_kmeans["kmeans"]
    mapping_klaster = artefak_kmeans["mapping_label"]
    fitur_km = artefak_kmeans["fitur_cluster"]
    X_scaled = scaler_km.transform(df[fitur_km])
    df["segmen_kos"] = pd.Series(kmeans_model.predict(X_scaled)).map(mapping_klaster)
else:
    df["segmen_kos"] = "Umum"

# =============================================================================
# SIDEBAR FILTER PENCARIAN KOS
# =============================================================================
st.sidebar.header("🔍 Filter Pencarian Kos")

daftar_kecamatan = sorted(df["kecamatan"].unique())
pilih_kecamatan = st.sidebar.multiselect("Kemantren (Kecamatan)", daftar_kecamatan, default=[])

pilih_tipe = st.sidebar.multiselect(
    "Tipe Penghuni",
    ["Putri", "Putra", "Campur"],
    default=["Putri", "Putra", "Campur"]
)

pilih_segmen = st.sidebar.multiselect(
    "Segmen Kelas (AI K-Means)",
    sorted(df["segmen_kos"].unique()),
    default=sorted(df["segmen_kos"].unique())
)

harga_min = int(df["harga_bulanan"].min())
harga_max = int(df["harga_bulanan"].max())
rentang_harga = st.sidebar.slider(
    "Rentang Harga Bulanan (Rp)",
    min_value=harga_min,
    max_value=harga_max,
    value=(harga_min, harga_max),
    step=50000
)

hanya_tersedia = st.sidebar.checkbox("Hanya Tampilkan Kamar Tersedia", value=False)

st.sidebar.markdown("**Filter Fasilitas Wajib:**")
wajib_wifi = st.sidebar.checkbox("WiFi", value=False)
wajib_ac = st.sidebar.checkbox("AC", value=False)
wajib_km_dalam = st.sidebar.checkbox("Kamar Mandi Dalam", value=False)

# Terapkan semua filter
df_filtered = df[
    (df["tipe_kos"].isin(pilih_tipe)) &
    (df["segmen_kos"].isin(pilih_segmen)) &
    (df["harga_bulanan"] >= rentang_harga[0]) &
    (df["harga_bulanan"] <= rentang_harga[1])
].copy()

if pilih_kecamatan:
    df_filtered = df_filtered[df_filtered["kecamatan"].isin(pilih_kecamatan)]

if hanya_tersedia:
    df_filtered = df_filtered[df_filtered["kamar_tersedia"] > 0]

if wajib_wifi:
    df_filtered = df_filtered[df_filtered["fasilitas"].str.contains("WiFi", case=False, na=False)]
if wajib_ac:
    df_filtered = df_filtered[df_filtered["fasilitas"].str.contains("AC", case=False, na=False)]
if wajib_km_dalam:
    df_filtered = df_filtered[df_filtered["fasilitas"].str.contains("K. Mandi Dalam", case=False, na=False)]

# =============================================================================
# KPI RINGKASAN EKSEKUTIF
# =============================================================================
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Kos Ditemukan", f"{len(df_filtered)} Unit", f"Dari {len(df)} total data")
k2.metric("Total Kamar Kosong", f"{int(df_filtered['kamar_tersedia'].sum())} Kamar", "Siap huni")
rata_harga_filter = df_filtered["harga_bulanan"].mean() if not df_filtered.empty else 0
k3.metric("Rata-Rata Harga Sewa", f"Rp {rata_harga_filter:,.0f}", "Per bulan")
rata_rating_filter = df_filtered["rating"].mean() if not df_filtered.empty else 0
k4.metric("Rata-Rata Rating", f"⭐ {rata_rating_filter:.2f} / 5.0", "Penilaian penghuni")

st.markdown("---")

if df_filtered.empty:
    st.warning("Tidak ada kos yang cocok dengan kombinasi filter saat ini. Coba longgarkan filter di sidebar sebelah kiri.")
    st.stop()

# =============================================================================
# PETA SPASIAL INTERAKTIF (PYDECK) & STATISTIK HARGA KEMANTREN
# =============================================================================
col_map, col_chart = st.columns([1.35, 1])

with col_map:
    st.subheader("🗺️ Peta Persebaran Lokasi Kos")

    warna_map = {
        "Ekonomis": [46, 204, 113, 210],    # Hijau
        "Menengah": [52, 152, 219, 210],    # Biru
        "Eksklusif": [155, 89, 182, 210],  # Ungu
        "Umum": [241, 196, 15, 210]
    }
    df_filtered["warna_titik"] = df_filtered["segmen_kos"].map(warna_map)
    df_filtered["harga_format"] = df_filtered["harga_bulanan"].apply(lambda x: f"Rp {x:,.0f}")

    view_state = pdk.ViewState(
        latitude=float(df_filtered["latitude"].mean()),
        longitude=float(df_filtered["longitude"].mean()),
        zoom=12.3,
        pitch=25
    )

    layer_kos = pdk.Layer(
        "ScatterplotLayer",
        data=df_filtered,
        get_position="[longitude, latitude]",
        get_fill_color="warna_titik",
        get_radius=95,
        pickable=True,
        auto_highlight=True
    )

    tooltip_peta = {
        "html": (
            "<b>{nama_kos}</b><br/>"
            "Kemantren: {kecamatan} ({tipe_kos})<br/>"
            "Segmen AI: <b>{segmen_kos}</b><br/>"
            "Harga: <b>{harga_format}/bln</b><br/>"
            "Status: {status_kamar} | Rating: ⭐ {rating}"
        ),
        "style": {"backgroundColor": "#1e293b", "color": "white", "fontSize": "12px"}
    }

    st.pydeck_chart(
        pdk.Deck(
            layers=[layer_kos],
            initial_view_state=view_state,
            tooltip=tooltip_peta,
            map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"
        ),
        use_container_width=True
    )
    st.caption("Keterangan warna titik: 🟢 **Ekonomis** | 🔵 **Menengah** | 🟣 **Eksklusif**")

with col_chart:
    st.subheader("📊 Rata-Rata Harga per Kemantren")
    avg_kec = (
        df_filtered.groupby("kecamatan")["harga_bulanan"]
        .mean()
        .sort_values(ascending=True)
        .reset_index()
    )
    fig_kec = px.bar(
        avg_kec,
        x="harga_bulanan",
        y="kecamatan",
        orientation="h",
        color="harga_bulanan",
        color_continuous_scale="Tealgrn",
        labels={"harga_bulanan": "Rata-Rata Harga (Rp)", "kecamatan": "Kemantren"},
        title="Perbandingan Tarif Bulanan Antar Wilayah"
    )
    fig_kec.update_layout(coloraxis_showscale=False, height=440)
    st.plotly_chart(fig_kec, use_container_width=True)

# =============================================================================
# REKOMENDASI TOP KOS & TABEL DATA LENGKAP
# =============================================================================
st.subheader("🏆 Rekomendasi Kos Terbaik Sesuai Filter")
urut_berdasarkan = st.selectbox(
    "Urutkan Daftar Berdasarkan:",
    ["Rating Tertinggi", "Harga Termurah", "Harga Termahal", "Kamar Tersedia Terbanyak"]
)

if urut_berdasarkan == "Rating Tertinggi":
    df_sorted = df_filtered.sort_values(by=["rating", "harga_bulanan"], ascending=[False, True])
elif urut_berdasarkan == "Harga Termurah":
    df_sorted = df_filtered.sort_values(by=["harga_bulanan", "rating"], ascending=[True, False])
elif urut_berdasarkan == "Harga Termahal":
    df_sorted = df_filtered.sort_values(by=["harga_bulanan", "rating"], ascending=[False, False])
else:
    df_sorted = df_filtered.sort_values(by=["kamar_tersedia", "rating"], ascending=[False, False])

st.dataframe(
    df_sorted[[
        "nama_kos", "kecamatan", "tipe_kos", "segmen_kos",
        "status_kamar", "harga_bulanan", "luas_kamar", "rating", "fasilitas", "ulasan"
    ]],
    use_container_width=True,
    hide_index=True
)