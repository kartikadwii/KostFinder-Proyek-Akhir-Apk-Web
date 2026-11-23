import os
import re
from difflib import SequenceMatcher
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Asisten Kos AI - KosFinder",
    page_icon="🤖",
    layout="wide"
)

FILE_INTENT = "intent_asisten_kos.csv"
FILE_DATASET = "dataset_kos_yogyakarta.csv"
PATH_REGRESI = os.path.join("models", "model_harga_regresi.pkl")
PATH_KLASTER = os.path.join("models", "model_klaster_kmeans.pkl")

THRESHOLD = 0.45  # Ambang batas minimal kemiripan intent

DAFTAR_KEMANTREN = [
    "Danurejan", "Gedongtengen", "Gondokusuman", "Gondomanan",
    "Jetis", "Kotagede", "Kraton", "Mantrijeron",
    "Mergangsan", "Ngampilan", "Pakualaman", "Tegalrejo",
    "Umbulharjo", "Wirobrajan"
]

NORMALISASI = {
    "gak": "tidak", "ga": "tidak", "nggak": "tidak", "tdk": "tidak",
    "gimana": "bagaimana", "gmn": "bagaimana", "brp": "berapa",
    "yg": "yang", "dgn": "dengan", "utk": "untuk", "sy": "saya",
    "aku": "saya", "kak": "", "min": "", "dong": "", "sih": "", "ya": "",
    "km": "kamar mandi", "ac": "ac", "wifi": "wifi", "bln": "bulan",
    "perbulan": "per bulan", "kec": "kecamatan", "jogja": "yogyakarta",
    "kost": "kos", "rb": "ribu", "jt": "juta", "sd": "sampai", "hingga": "sampai"
}

STOPWORDS = {
    "yang", "di", "ke", "dari", "dan", "atau", "itu", "ini", "untuk",
    "dengan", "adalah", "apa", "saya", "mau", "ingin", "tolong", "bisa",
    "ada", "tentang", "mohon", "info", "informasi", "apakah", "kah"
}


def preprocess(teks: str) -> list[str]:
    """Praproses teks: lowercase, hapus tanda baca, normalisasi kata, dan hapus stopword."""
    teks = re.sub(r"[^a-z0-9\s]", " ", str(teks).lower())
    tokens = []
    for kata in teks.split():
        kata_norm = NORMALISASI.get(kata, kata)
        for k in kata_norm.split():
            if k and k not in STOPWORDS:
                tokens.append(k)
    return tokens


def similarity(tokens_a: list[str], tokens_b: list[str]) -> float:
    """Menghitung skor kemiripan gabungan (0.4 Jaccard + 0.3 Fuzzy + 0.3 Cakupan)."""
    if not tokens_a or not tokens_b:
        return 0.0
    set_a, set_b = set(tokens_a), set(tokens_b)
    jaccard = len(set_a & set_b) / len(set_a | set_b)
    fuzzy = SequenceMatcher(None, " ".join(tokens_a), " ".join(tokens_b)).ratio()
    cakupan = len(set_a & set_b) / len(set_b)
    return round(0.4 * jaccard + 0.3 * fuzzy + 0.3 * cakupan, 4)


@st.cache_data
def load_resources():
    df_intent = pd.read_csv(FILE_INTENT) if os.path.exists(FILE_INTENT) else pd.DataFrame()
    df_kos = pd.read_csv(FILE_DATASET) if os.path.exists(FILE_DATASET) else pd.DataFrame()
    if not df_kos.empty:
        df_kos["jumlah_fasilitas"] = df_kos["fasilitas"].fillna("").apply(
            lambda x: len([f.strip() for f in str(x).split(",") if f.strip()]) if str(x).strip() else 1
        )
    return df_intent, df_kos


@st.cache_resource
def load_ml_models():
    reg = joblib.load(PATH_REGRESI) if os.path.exists(PATH_REGRESI) else None
    km = joblib.load(PATH_KLASTER) if os.path.exists(PATH_KLASTER) else None
    return reg, km


def ekstrak_kemantren(teks: str):
    teks_low = teks.lower()
    for k in DAFTAR_KEMANTREN:
        if k.lower() in teks_low:
            return k
    return None


def ekstrak_tipe_kos(teks: str):
    teks_low = teks.lower()
    if "putri" in teks_low or "cewek" in teks_low:
        return "Putri"
    if "putra" in teks_low or "cowok" in teks_low:
        return "Putra"
    if "campur" in teks_low:
        return "Campur"
    return None


def ekstrak_rentang_harga(teks: str):
    """
    Mendeteksi rentang harga (min_harga, max_harga) dari teks pengguna.
    Contoh: '500 ribu sampai 1 juta' -> (500000.0, 1000000.0)
            'di bawah 1.5 juta' -> (0.0, 1500000.0)
    """
    t = teks.lower().replace(",", ".")
    # Cari semua angka yang diikuti satuan jt/juta/rb/ribu/k
    matches = re.findall(r"(\d+(?:\.\d+)?)\s*(jt|juta|rb|ribu|k)\b", t)
    nilai_rupiah = []
    for angka_str, satuan in matches:
        angka = float(angka_str)
        if satuan in ("jt", "juta"):
            nilai_rupiah.append(angka * 1_000_000)
        elif satuan in ("rb", "ribu", "k"):
            nilai_rupiah.append(angka * 1_000)

    # Cek juga angka rupiah langsung seperti 500000 atau 1000000
    if not nilai_rupiah:
        angka_polos = re.findall(r"\b(\d{6,7})\b", t)
        nilai_rupiah = [float(x) for x in angka_polos]

    if len(nilai_rupiah) >= 2:
        return min(nilai_rupiah[0], nilai_rupiah[1]), max(nilai_rupiah[0], nilai_rupiah[1])
    elif len(nilai_rupiah) == 1:
        if any(kata in t for kata in ["di atas", "diatas", "minimal", "mulai"]):
            return nilai_rupiah[0], None
        return 0.0, nilai_rupiah[0]
    return None, None


def proses_dinamis(intent: str, pertanyaan: str, df_kos: pd.DataFrame, model_reg, artefak_km) -> str:
    kec = ekstrak_kemantren(pertanyaan)
    tipe = ekstrak_tipe_kos(pertanyaan)
    harga_min, harga_max = ekstrak_rentang_harga(pertanyaan)

    # 1. INTENT CEK KOS KOSONG
    if intent == "cek_kos_kosong":
        sub = df_kos[df_kos["kamar_tersedia"] > 0].copy()
        if kec:
            sub = sub[sub["kecamatan"].str.lower() == kec.lower()]
        if tipe:
            sub = sub[sub["tipe_kos"].str.lower() == tipe.lower()]
        if harga_min is not None:
            sub = sub[sub["harga_bulanan"] >= harga_min]
        if harga_max is not None:
            sub = sub[sub["harga_bulanan"] <= harga_max]

        if sub.empty:
            info_filter = f" di Kemantren {kec}" if kec else ""
            return f"Berdasarkan data crawling terbaru, belum ditemukan kos kosong{info_filter} yang sesuai kriteria tersebut."

        total_unit = len(sub)
        total_kamar = int(sub["kamar_tersedia"].sum())
        top3 = sub.sort_values(by=["rating", "harga_bulanan"], ascending=[False, True]).head(3)

        judul = f"Ditemukan **{total_unit} unit kos kosong** ({total_kamar} kamar siap huni)"
        if kec:
            judul += f" di **Kemantren {kec}**"
        lines = [judul + ":\n"]
        for i, r in enumerate(top3.itertuples(), 1):
            lines.append(
                f"{i}. **{r.nama_kos}** ({r.tipe_kos} • {r.kecamatan}) — "
                f"**{r.status_kamar}** | Rp {r.harga_bulanan:,.0f}/bln | Rating ⭐ {r.rating}"
            )
        return "\n".join(lines)

    # 2. INTENT PREDIKSI HARGA SESUAI FASILITAS
    if intent == "prediksi_harga":
        if model_reg is None:
            return "Model prediksi harga (`model_harga_regresi.pkl`) belum tersedia."

        t_low = pertanyaan.lower()
        ada_ac = int("ac" in t_low)
        ada_wifi = int("wifi" in t_low or "internet" in t_low)
        ada_km_dalam = int("mandi dalam" in t_low or "km dalam" in t_low)
        ada_kloset = int("kloset duduk" in t_low)
        ada_parkir = int("parkir" in t_low)
        ada_akses24 = int("24 jam" in t_low)

        jml_fas = max(2, ada_ac + ada_wifi + ada_km_dalam + ada_kloset + ada_parkir + ada_akses24)
        lokasi_pred = kec if kec else "Gondokusuman"
        tipe_pred = tipe if tipe else "Campur"
        luas_pred = 15.0 if ada_ac and ada_km_dalam else 11.0

        data_uji = pd.DataFrame([{
            "kecamatan": lokasi_pred,
            "tipe_kos": tipe_pred,
            "luas_kamar": luas_pred,
            "rating": 4.6,
            "jumlah_fasilitas": jml_fas,
            "ada_ac": ada_ac,
            "ada_km_dalam": ada_km_dalam,
            "ada_wifi": ada_wifi,
        }])

        harga_est = max(450000, round(float(model_reg.predict(data_uji)[0]) / 25000) * 25000)

        daftar_fas = []
        if ada_ac:
            daftar_fas.append("AC")
        if ada_wifi:
            daftar_fas.append("WiFi")
        if ada_km_dalam:
            daftar_fas.append("Kamar Mandi Dalam")
        if ada_kloset:
            daftar_fas.append("Kloset Duduk")
        if ada_parkir:
            daftar_fas.append("Parkir")
        if ada_akses24:
            daftar_fas.append("Akses 24 Jam")
        teks_fas = ", ".join(daftar_fas) if daftar_fas else "Fasilitas Standar (Kasur, Lemari, WiFi)"

        segmen_info = ""
        if artefak_km is not None:
            fitur_km = pd.DataFrame([{
                "harga_bulanan": harga_est,
                "luas_kamar": luas_pred,
                "rating": 4.6,
                "jumlah_fasilitas": jml_fas
            }])
            kid = int(artefak_km["kmeans"].predict(artefak_km["scaler"].transform(fitur_km))[0])
            segmen_info = f" (Masuk kategori **Klaster {artefak_km['mapping_label'][kid]}**)"

        return (
            f"💰 **Estimasi Prediksi Harga Sewa Kos:**\n"
            f"- **Tarif Wajar:** **Rp {harga_est:,.0f} / bulan**{segmen_info}\n"
            f"- **Parameter Simulasi:** Kemantren {lokasi_pred} | Tipe {tipe_pred} | Estimasi luas {luas_pred:.0f} m²\n"
            f"- **Fasilitas Terdeteksi:** {teks_fas}"
        )

    # 3. INTENT PENCARIAN KOS DI 14 KEMANTREN / RENTANG HARGA
    if intent == "cari_kos_kecamatan":
        sub = df_kos.copy()
        if kec:
            sub = sub[sub["kecamatan"].str.lower() == kec.lower()]
        if tipe:
            sub = sub[sub["tipe_kos"].str.lower() == tipe.lower()]
        if harga_min is not None:
            sub = sub[sub["harga_bulanan"] >= harga_min]
        if harga_max is not None:
            sub = sub[sub["harga_bulanan"] <= harga_max]
        if "ac" in pertanyaan.lower():
            sub = sub[sub["fasilitas"].str.contains("AC", case=False, na=False)]
        if "wifi" in pertanyaan.lower():
            sub = sub[sub["fasilitas"].str.contains("WiFi", case=False, na=False)]
        if "mandi dalam" in pertanyaan.lower():
            sub = sub[sub["fasilitas"].str.contains("K. Mandi Dalam", case=False, na=False)]

        if sub.empty:
            return (
                "Maaf, belum ditemukan kos yang cocok dengan filter rentang harga dan wilayah tersebut. "
                "Coba ubah kisaran anggaran atau pilih salah satu dari 14 Kemantren di Kota Yogyakarta."
            )

        top5 = sub.sort_values(by=["rating", "harga_bulanan"], ascending=[False, True]).head(5)

        ket_harga = ""
        if harga_min is not None and harga_max is not None:
            ket_harga = f" kisaran **Rp {harga_min:,.0f} – Rp {harga_max:,.0f}/bln**"
        elif harga_max is not None:
            ket_harga = f" maksimal **Rp {harga_max:,.0f}/bln**"

        header = (
            f"🏠 **Rekomendasi Kos di {kec if kec else 'Kota Yogyakarta'}**"
            f"{ket_harga} *(Ditemukan {len(sub)} unit cocok, menampilkan top {len(top5)})*:\n"
        )
        lines = [header]
        for i, r in enumerate(top5.itertuples(), 1):
            lines.append(
                f"**{i}. {r.nama_kos}** ({r.tipe_kos} • Kemantren {r.kecamatan})\n"
                f"   - 💰 Harga: **Rp {r.harga_bulanan:,.0f}/bln** | 🛏️ Status: {r.status_kamar} | ⭐ {r.rating}\n"
                f"   - ✨ Fasilitas: {r.fasilitas}"
            )
        return "\n".join(lines)

    return ""


def klasifikasi_dan_jawab(pesan: str, df_intent: pd.DataFrame, df_kos: pd.DataFrame, model_reg, artefak_km):
    tokens_pesan = preprocess(pesan)
    kec_terdeteksi = ekstrak_kemantren(pesan)
    h_min, h_max = ekstrak_rentang_harga(pesan)

    skor_terbaik = {}
    for row in df_intent.itertuples():
        if row.intent == "fallback":
            continue
        tokens_pola = preprocess(row.pattern)
        skor = similarity(tokens_pesan, tokens_pola)

        # Tambahan skor kontekstual bila user menyebut kemantren atau nominal budget pencarian
        if row.intent == "cari_kos_kecamatan":
            if kec_terdeteksi:
                skor = min(1.0, round(skor + 0.25, 4))
            if h_min is not None or h_max is not None:
                skor = min(1.0, round(skor + 0.25, 4))

        if row.intent not in skor_terbaik or skor > skor_terbaik[row.intent]["skor"]:
            skor_terbaik[row.intent] = {"skor": skor, "response": row.response, "pattern": row.pattern}

    kandidat = sorted(skor_terbaik.items(), key=lambda x: x[1]["skor"], reverse=True)
    fallback_resp = df_intent[df_intent["intent"] == "fallback"]["response"].iloc[0]

    # Aturan ambang batas minimal 0,45
    if not kandidat or kandidat[0][1]["skor"] < THRESHOLD:
        skor_maks = kandidat[0][1]["skor"] if kandidat else 0.0
        return fallback_resp, "fallback", skor_maks, kandidat[:3]

    intent_terpilih, detail = kandidat[0]
    template_resp = detail["response"]

    if template_resp.startswith("{DYNAMIC_"):
        jawaban = proses_dinamis(intent_terpilih, pesan, df_kos, model_reg, artefak_km)
    else:
        jawaban = template_resp

    return jawaban, intent_terpilih, detail["skor"], kandidat[:3]


# =============================================================================
# ANTARMUKA STREAMLIT CHATBOT ASISTEN KOS
# =============================================================================
st.title("🤖 Asisten Virtual Pencarian Kos Yogyakarta")
st.markdown(
    "Tanyakan **ketersediaan kos kosong**, **prediksi harga sewa bulanan sesuai fasilitas**, "
    "atau **pencarian kos berdasarkan rentang harga & 14 Kemantren Kota Yogyakarta** *(Threshold: `0.45`)*."
)

df_intent, df_kos = load_resources()
model_reg, artefak_km = load_ml_models()

if df_intent.empty or df_kos.empty:
    st.error("Pastikan `intent_asisten_kos.csv` dan `dataset_kos_yogyakarta.csv` tersedia di folder proyek.")
    st.stop()

with st.sidebar:
    st.header("⚙️ Pengaturan Asisten")
    mode_debug = st.checkbox("Tampilkan Skor Intent (Mode Belajar)", value=True)
    if st.button("🗑️ Hapus Riwayat Chat", use_container_width=True):
        st.session_state.chat_kos = [
            {
                "role": "assistant",
                "content": "Halo! Saya **Asisten KosFinder**. Silakan tanyakan apakah ada kos kosong, prediksi harga bulanan sesuai fasilitas, atau rekomendasi kos harga 500 ribu sampai 1 juta di 14 Kemantren Yogyakarta.",
                "meta": None
            }
        ]
        st.rerun()

# Tombol saran pertanyaan cepat
st.markdown("**💡 Saran Pertanyaan Cepat:**")
col_s1, col_s2, col_s3, col_s4 = st.columns(4)
saran_klik = None
if col_s1.button("Kos 500rb - 1 Juta", use_container_width=True):
    saran_klik = "Rekomendasi kos dengan harga 500 ribu sampai 1 juta"
if col_s2.button("Apakah ada kos kosong?", use_container_width=True):
    saran_klik = "Apakah ada kos kosong di Gondokusuman?"
if col_s3.button("Prediksi harga AC & WiFi", use_container_width=True):
    saran_klik = "Berapa prediksi harga kos per bulan kalau ada AC, WiFi, dan kamar mandi dalam?"
if col_s4.button("Cari kos di Umbulharjo", use_container_width=True):
    saran_klik = "Carikan kos putri di Umbulharjo"

if "chat_kos" not in st.session_state:
    st.session_state.chat_kos = [
        {
            "role": "assistant",
            "content": "Halo! Saya **Asisten KosFinder**. Silakan tanyakan apakah ada kos kosong, prediksi harga bulanan sesuai fasilitas, atau rekomendasi kos harga 500 ribu sampai 1 juta di 14 Kemantren Yogyakarta.",
            "meta": None
        }
    ]

for msg in st.session_state.chat_kos:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("meta") and mode_debug:
            st.caption(msg["meta"])

input_user = st.chat_input("Contoh: Rekomendasi kos dengan harga 500 ribu sampai 1 juta di Umbulharjo")
prompt = saran_klik if saran_klik else input_user

if prompt:
    st.session_state.chat_kos.append({"role": "user", "content": prompt, "meta": None})
    with st.chat_message("user"):
        st.markdown(prompt)

    jawaban, intent_hasil, skor_hasil, top3_kandidat = klasifikasi_dan_jawab(
        prompt, df_intent, df_kos, model_reg, artefak_km
    )
    info_meta = f"Intent: `{intent_hasil}` | Skor tertinggi: `{skor_hasil:.2f}` (Ambang batas: `{THRESHOLD}`)"

    st.session_state.chat_kos.append({"role": "assistant", "content": jawaban, "meta": info_meta})
    with st.chat_message("assistant"):
        st.markdown(jawaban)
        if mode_debug:
            st.caption(info_meta)