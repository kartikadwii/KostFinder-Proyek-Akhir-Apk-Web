import os
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Sentimen Ulasan - KosFinder", page_icon="💬", layout="wide")

FILE_DATASET = "dataset_kos_yogyakarta.csv"


@st.cache_data
def load_data():
    return pd.read_csv(FILE_DATASET) if os.path.exists(FILE_DATASET) else pd.DataFrame()


def analisis_sentimen_ulasan(teks):
    """Klasifikasi sentimen ulasan penghuni kos berbasis leksikon bahasa Indonesia."""
    if pd.isna(teks) or str(teks).strip() == "":
        return "Netral", 0

    teks_lower = str(teks).lower()
    kata_positif = [
        "bersih", "kencang", "stabil", "ramah", "strategis", "dekat", "murah",
        "aman", "tenang", "luas", "lancar", "dingin", "terawat", "nyaman",
        "kondusif", "membantu", "baru", "bagus", "recommended", "responsif", "terjaga"
    ]
    kata_negatif = [
        "putus", "susah", "kurang", "pengap", "terdengar", "sempit", "kehujanan",
        "kecil", "lambat", "melambat", "terbatas", "cepat habis", "ramai", "mahal", "kotor"
    ]

    skor_pos = sum(1 for k in kata_positif if k in teks_lower)
    skor_neg = sum(1 for k in kata_negatif if k in teks_lower)
    skor_akhir = skor_pos - skor_neg

    if skor_akhir > 0:
        return "Positif", skor_akhir
    elif skor_akhir < 0:
        return "Negatif", skor_akhir
    return "Netral", 0


st.title("💬 Analisis Sentimen Ulasan Penghuni Kos")
st.markdown("Pemetaan tingkat kepuasan penghuni kos di 14 Kemantren Kota Yogyakarta berdasarkan ulasan fasilitas, kenyamanan, dan lingkungan.")

df = load_data()
if df.empty:
    st.error("File `dataset_kos_yogyakarta.csv` tidak ditemukan. Jalankan `crawl_jogja.py` terlebih dahulu.")
    st.stop()

hasil = df["ulasan"].apply(analisis_sentimen_ulasan)
df["sentimen"] = [h[0] for h in hasil]
df["skor_sentimen"] = [h[1] for h in hasil]

# Metrik Ringkasan Sentimen
s_c1, s_c2, s_c3 = st.columns(3)
jml_pos = int((df["sentimen"] == "Positif").sum())
jml_net = int((df["sentimen"] == "Netral").sum())
jml_neg = int((df["sentimen"] == "Negatif").sum())

s_c1.metric("Ulasan Positif 😊", f"{jml_pos} Kos", f"{(jml_pos / len(df) * 100):.1f}%")
s_c2.metric("Ulasan Netral 😐", f"{jml_net} Kos", f"{(jml_net / len(df) * 100):.1f}%")
s_c3.metric("Ulasan Negatif 😟", f"{jml_neg} Kos", f"{(jml_neg / len(df) * 100):.1f}%")

st.markdown("---")

# Visualisasi Proporsi & Sebaran per Kemantren
col_s1, col_s2 = st.columns([1, 1.3])
with col_s1:
    fig_pie = px.pie(
        df,
        names="sentimen",
        color="sentimen",
        color_discrete_map={"Positif": "#2ecc71", "Netral": "#f1c40f", "Negatif": "#e74c3c"},
        hole=0.45,
        title="Proporsi Sentimen Ulasan Keseluruhan"
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with col_s2:
    sentimen_kec = df.groupby(["kecamatan", "sentimen"]).size().reset_index(name="jumlah")
    fig_sent_kec = px.bar(
        sentimen_kec,
        x="kecamatan",
        y="jumlah",
        color="sentimen",
        barmode="stack",
        color_discrete_map={"Positif": "#2ecc71", "Netral": "#f1c40f", "Negatif": "#e74c3c"},
        labels={"kecamatan": "Kemantren", "jumlah": "Jumlah Ulasan", "sentimen": "Sentimen"},
        title="Sebaran Sentimen Ulasan per Kemantren"
    )
    fig_sent_kec.update_layout(xaxis_tickangle=-40)
    st.plotly_chart(fig_sent_kec, use_container_width=True)

# Uji Analisis Sentimen Teks Bebas
with st.expander("🧪 Uji Analisis Sentimen Ulasan Baru"):
    teks_uji = st.text_input("Masukkan contoh kalimat ulasan kos:", placeholder="Contoh: Kamar bersih, WiFi kencang, tapi parkir motor agak sempit.")
    if teks_uji:
        label_uji, skor_uji = analisis_sentimen_ulasan(teks_uji)
        st.write(f"**Hasil Klasifikasi:** `{label_uji}` (Skor polaritas: `{skor_uji:+d}`)")

# Tabel Daftar Ulasan Terfilter
st.subheader("Daftar Ulasan Penghuni Kos")
f_col1, f_col2 = st.columns(2)
filter_sent = f_col1.multiselect("Filter Kategori Sentimen", ["Positif", "Netral", "Negatif"], default=["Positif", "Netral", "Negatif"])
filter_kec = f_col2.multiselect("Filter Kemantren", sorted(df["kecamatan"].unique()), default=[])

df_ulasan = df[df["sentimen"].isin(filter_sent)]
if filter_kec:
    df_ulasan = df_ulasan[df_ulasan["kecamatan"].isin(filter_kec)]

st.dataframe(
    df_ulasan[["nama_kos", "kecamatan", "tipe_kos", "status_kamar", "rating", "sentimen", "ulasan"]],
    use_container_width=True,
    hide_index=True
)