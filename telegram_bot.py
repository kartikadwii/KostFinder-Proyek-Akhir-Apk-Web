import os
import re
import ssl
import time
import json
import urllib.request
import urllib.parse
from difflib import SequenceMatcher
import joblib
import pandas as pd

TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
FILE_INTENT = "intent_asisten_kos.csv"
FILE_DATASET = "dataset_kos_yogyakarta.csv"
PATH_REGRESI = os.path.join("models", "model_harga_regresi.pkl")
THRESHOLD = 0.45

# Konteks SSL untuk melewati blokir sertifikat antivirus / jaringan lokal
SSL_CTX = ssl._create_unverified_context()

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
    teks = re.sub(r"[^a-z0-9\s]", " ", str(teks).lower())
    tokens = []
    for kata in teks.split():
        kata_norm = NORMALISASI.get(kata, kata)
        for k in kata_norm.split():
            if k and k not in STOPWORDS:
                tokens.append(k)
    return tokens


def similarity(tokens_a: list[str], tokens_b: list[str]) -> float:
    if not tokens_a or not tokens_b:
        return 0.0
    set_a, set_b = set(tokens_a), set(tokens_b)
    jaccard = len(set_a & set_b) / len(set_a | set_b)
    fuzzy = SequenceMatcher(None, " ".join(tokens_a), " ".join(tokens_b)).ratio()
    cakupan = len(set_a & set_b) / len(set_b)
    return round(0.4 * jaccard + 0.3 * fuzzy + 0.3 * cakupan, 4)


def ekstrak_kemantren(teks: str):
    for k in DAFTAR_KEMANTREN:
        if k.lower() in teks.lower():
            return k
    return None


def ekstrak_rentang_harga(teks: str):
    t = teks.lower().replace(",", ".")
    matches = re.findall(r"(\d+(?:\.\d+)?)\s*(jt|juta|rb|ribu|k)\b", t)
    nilai = []
    for angka_str, satuan in matches:
        angka = float(angka_str)
        nilai.append(angka * 1_000_000 if satuan in ("jt", "juta") else angka * 1_000)
    if len(nilai) >= 2:
        return min(nilai[0], nilai[1]), max(nilai[0], nilai[1])
    elif len(nilai) == 1:
        return 0.0, nilai[0]
    return None, None


def jawab_pesan(pesan: str, df_intent: pd.DataFrame, df_kos: pd.DataFrame, model_reg) -> str:
    if pesan.strip().lower() in ("/start", "/help"):
        return (
            "Halo! Saya *Bot Telegram KosFinder Yogyakarta* 🤖🏠\n\n"
            "Contoh pertanyaan yang bisa kamu kirim:\n"
            "• _Apakah ada kos kosong di Gondokusuman?_\n"
            "• _Rekomendasi kos harga 500 ribu sampai 1 juta_\n"
            "• _Prediksi harga kos AC dan WiFi di Umbulharjo_"
        )

    tokens_pesan = preprocess(pesan)
    kec = ekstrak_kemantren(pesan)
    h_min, h_max = ekstrak_rentang_harga(pesan)

    skor_terbaik = {}
    for row in df_intent.itertuples():
        if row.intent == "fallback":
            continue
        skor = similarity(tokens_pesan, preprocess(row.pattern))
        if row.intent == "cari_kos_kecamatan":
            if kec:
                skor = min(1.0, round(skor + 0.25, 4))
            if h_min is not None or h_max is not None:
                skor = min(1.0, round(skor + 0.25, 4))
        if row.intent not in skor_terbaik or skor > skor_terbaik[row.intent]["skor"]:
            skor_terbaik[row.intent] = {"skor": skor, "response": row.response}

    kandidat = sorted(skor_terbaik.items(), key=lambda x: x[1]["skor"], reverse=True)
    fallback_resp = df_intent[df_intent["intent"] == "fallback"]["response"].iloc[0]

    if not kandidat or kandidat[0][1]["skor"] < THRESHOLD:
        return fallback_resp

    intent_terpilih, detail = kandidat[0]
    if not detail["response"].startswith("{DYNAMIC_"):
        return detail["response"]

    sub = df_kos.copy()
    if intent_terpilih == "cek_kos_kosong":
        sub = sub[sub["kamar_tersedia"] > 0]
    if kec:
        sub = sub[sub["kecamatan"].str.lower() == kec.lower()]
    if h_min is not None:
        sub = sub[sub["harga_bulanan"] >= h_min]
    if h_max is not None:
        sub = sub[sub["harga_bulanan"] <= h_max]

    if intent_terpilih == "prediksi_harga" and model_reg is not None:
        t_low = pesan.lower()
        ada_ac = int("ac" in t_low)
        ada_wifi = int("wifi" in t_low)
        ada_km = int("mandi dalam" in t_low or "km dalam" in t_low)
        data_uji = pd.DataFrame([{
            "kecamatan": kec if kec else "Gondokusuman",
            "tipe_kos": "Campur",
            "luas_kamar": 14.0 if ada_ac else 10.0,
            "rating": 4.6,
            "jumlah_fasilitas": max(2, ada_ac + ada_wifi + ada_km),
            "ada_ac": ada_ac,
            "ada_km_dalam": ada_km,
            "ada_wifi": ada_wifi,
        }])
        harga_est = max(450000, round(float(model_reg.predict(data_uji)[0]) / 25000) * 25000)
        return (
            f"💰 *Estimasi Harga Sewa Kos:*\n"
            f"Tarif wajar sekitar *Rp {harga_est:,.0f} / bulan* (Kemantren {kec or 'Gondokusuman'})."
        )

    if sub.empty:
        return "Maaf, belum ditemukan kos yang cocok dengan kriteria tersebut."

    top3 = sub.sort_values(by=["rating", "harga_bulanan"], ascending=[False, True]).head(3)
    hasil = [f"🏠 *Hasil Pencarian Kos ({len(sub)} unit cocok):*\n"]
    for i, r in enumerate(top3.itertuples(), 1):
        hasil.append(
            f"{i}. *{r.nama_kos}* ({r.tipe_kos} • {r.kecamatan})\n"
            f"   💰 Rp {r.harga_bulanan:,.0f}/bln | 🛏️ {r.status_kamar} | ⭐ {r.rating}"
        )
    return "\n".join(hasil)


def kirim_pesan(chat_id: int, teks: str):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    data = urllib.parse.urlencode({
        "chat_id": chat_id,
        "text": teks,
        "parse_mode": "Markdown"
    }).encode("utf-8")
    try:
        urllib.request.urlopen(urllib.request.Request(url, data=data), timeout=15, context=SSL_CTX)
    except Exception as e:
        print(f"Error kirim balasan: {e}")


def jalankan_bot():
    if not TOKEN:
        raise ValueError("Environment variable TELEGRAM_BOT_TOKEN belum diatur!")

    df_intent = pd.read_csv(FILE_INTENT)
    df_kos = pd.read_csv(FILE_DATASET)
    model_reg = joblib.load(PATH_REGRESI) if os.path.exists(PATH_REGRESI) else None

    print("🤖 Bot Telegram KosFinder aktif dan mendengarkan pesan... (Tekan Ctrl+C untuk berhenti)")
    offset = 0
    while True:
        try:
            url = f"https://api.telegram.org/bot{TOKEN}/getUpdates?timeout=30&offset={offset}"
            res = json.loads(urllib.request.urlopen(url, timeout=35, context=SSL_CTX).read().decode("utf-8"))
            for update in res.get("result", []):
                offset = update["update_id"] + 1
                msg = update.get("message", {})
                chat_id = msg.get("chat", {}).get("id")
                teks = msg.get("text")
                if chat_id and teks:
                    print(f"[Pesan masuk dari {chat_id}]: {teks}")
                    balasan = jawab_pesan(teks, df_intent, df_kos, model_reg)
                    kirim_pesan(chat_id, balasan)
        except Exception as e:
            print(f"Koneksi polling terganggu, mencoba ulang dalam 3 detik... ({e})")
            time.sleep(3)


if __name__ == "__main__":
    jalankan_bot()