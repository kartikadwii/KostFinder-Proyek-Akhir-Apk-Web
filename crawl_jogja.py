from playwright.sync_api import sync_playwright
import pandas as pd
import numpy as np
import urllib.request
import urllib.parse
import random
import os
import re
import ssl

random.seed(42)
np.random.seed(42)

FILE_DATASET = "dataset_kos_yogyakarta.csv"
TARGET_MINIMAL_BARIS = 200

# Data lengkap 14 Kemantren resmi Kota Yogyakarta
KEMANTREAN_DATA = {
    "Gondokusuman": {"coords": (-7.7835, 110.3820), "bobot": 0.16, "faktor_harga": 1.15},
    "Umbulharjo":   {"coords": (-7.8120, 110.3880), "bobot": 0.16, "faktor_harga": 1.05},
    "Mergangsan":   {"coords": (-7.8135, 110.3735), "bobot": 0.09, "faktor_harga": 1.02},
    "Jetis":        {"coords": (-7.7820, 110.3635), "bobot": 0.08, "faktor_harga": 1.10},
    "Tegalrejo":    {"coords": (-7.7810, 110.3540), "bobot": 0.07, "faktor_harga": 0.98},
    "Danurejan":    {"coords": (-7.7915, 110.3715), "bobot": 0.06, "faktor_harga": 1.08},
    "Mantrijeron":  {"coords": (-7.8165, 110.3585), "bobot": 0.06, "faktor_harga": 0.95},
    "Kotagede":     {"coords": (-7.8265, 110.3985), "bobot": 0.06, "faktor_harga": 0.90},
    "Wirobrajan":   {"coords": (-7.8010, 110.3485), "bobot": 0.05, "faktor_harga": 0.92},
    "Gedongtengen": {"coords": (-7.7905, 110.3610), "bobot": 0.05, "faktor_harga": 1.05},
    "Pakualaman":   {"coords": (-7.8010, 110.3760), "bobot": 0.04, "faktor_harga": 1.03},
    "Gondomanan":   {"coords": (-7.8015, 110.3675), "bobot": 0.04, "faktor_harga": 1.06},
    "Ngampilan":    {"coords": (-7.8005, 110.3575), "bobot": 0.04, "faktor_harga": 0.94},
    "Kraton":       {"coords": (-7.8085, 110.3640), "bobot": 0.04, "faktor_harga": 0.93},
}

PREFIX_NAMA = [
    "Kost Griya", "Kost Omah", "Kost Wisma", "Kost Ndalem", "Kost Paviliun",
    "Kost Puri", "Kost Graha", "Kost Pondok", "Kost Asri", "Kost Residence",
    "Kost Dalem", "Kost Bale", "Kost Taman", "Kost Indah"
]
SUFFIX_TIPE = ["Tipe A", "Tipe B", "Tipe C", "Standard", "Exclusive", "Deluxe", "Reguler"]

PAKET_FASILITAS = [
    ("Kasur, WiFi, Parkir Motor", (7.5, 9.5), (600000, 950000)),
    ("Kasur, WiFi, K. Mandi Dalam, Parkir Motor", (9.0, 12.0), (950000, 1450000)),
    ("WiFi, AC, K. Mandi Dalam, Kasur, Akses 24 Jam", (11.0, 15.0), (1450000, 2100000)),
    ("WiFi, AC, K. Mandi Dalam, Kloset Duduk, Kasur, Akses 24 Jam, Parkir Motor", (14.5, 22.0), (2100000, 3400000)),
]

ULASAN_POSITIF = [
    "Kamar bersih, WiFi kencang stabil buat nugas dan kuliah online, ibu kos ramah banget.",
    "Lokasi sangat strategis di Kota Jogja, dekat halte Trans Jogja, minimarket, dan warung makan murah.",
    "Lingkungan aman dan tenang, parkir motor luas dan tertutup kanopi, air lancar terus.",
    "Fasilitas sesuai foto, AC dingin, kamar mandi dalam bersih dan terawat dengan baik.",
    "Nyaman dan kondusif buat istirahat, akses 24 jam sangat membantu kalau pulang kegiatan malam.",
    "Bangunan baru dan sirkulasi udara bagus, pencahayaan alami masuk kamar, sangat recommended.",
    "Penjaga kos responsif kalau ada kendala lampu atau air, kebersihan area bersama selalu terjaga."
]

ULASAN_NETRAL = [
    "Kamar cukup nyaman sesuai harga, cuma kalau jam malam WiFi kadang agak melambat.",
    "Lokasi strategis di tengah kota tapi gang masuknya agak sempit kalau papasan dua motor.",
    "Fasilitas standar oke dan air lancar, hanya saja area parkir motor agak terbatas kalau penuh.",
    "Kamar lumayan luas, namun token listrik cepat habis kalau menyalakan AC terus-menerus.",
    "Lingkungan sekitar ramai dan dekat kuliner, tapi suara kendaraan di pagi hari cukup terdengar."
]

ULASAN_NEGATIF = [
    "WiFi sering putus-putus saat malam hari dan sinyal seluler di dalam kamar agak susah.",
    "Kamar mandi kurang bersih saat awal masuk dan sirkulasi udara kamar terasa agak pengap.",
    "Suara kendaraan dari jalan raya dan kamar sebelah cukup terdengar, dinding kurang kedap suara.",
    "Parkiran motor sempit dan sering kehujanan karena belum ada atap penutup yang memadai.",
    "Air keran kadang kecil saat jam sibuk pagi hari dan respon pengurus agak lambat."
]


def parsing_kartu(cards, nama_kec_default, tipe_default):
    hasil = []
    for card in cards:
        teks = card.inner_text()
        if any(luar in teks for luar in ["Bintaro", "Pondok Aren", "Ciputat", "Depok", "Mlati", "Ngaglik", "Kasihan", "Banguntapan", "Sewon"]):
            continue

        baris = [b.strip() for b in teks.split("\n") if b.strip()]
        harga_match = re.search(r"Rp\s?([\d\.]+)", teks)
        harga = int(harga_match.group(1).replace(".", "")) if harga_match else 0
        if harga == 0:
            continue

        sisa_match = re.search(r"(Sisa\s+\d+\s+kamar)", teks, re.IGNORECASE)
        if "penuh" in teks.lower():
            status_kamar = "Penuh"
            jumlah_tersedia = 0
        elif sisa_match:
            status_kamar = sisa_match.group(1).capitalize()
            angka_sisa = re.search(r"\d+", status_kamar)
            jumlah_tersedia = int(angka_sisa.group(0)) if angka_sisa else 1
        else:
            status_kamar = "Tersedia"
            jumlah_tersedia = random.randint(2, 5)

        kecamatan = nama_kec_default
        for k in KEMANTREAN_DATA.keys():
            if k.lower() in teks.lower():
                kecamatan = k
                break

        tipe = tipe_default.capitalize() if tipe_default != "all" else "Campur"
        if "Putra" in teks:
            tipe = "Putra"
        elif "Putri" in teks:
            tipe = "Putri"
        elif "Campur" in teks:
            tipe = "Campur"

        rating_match = re.search(r"\b([345]\.\d)\b", teks)
        rating = float(rating_match.group(1)) if rating_match else 4.5

        nama_kos = next((b for b in baris if "Kost" in b or "Kos" in b), baris[0] if baris else f"Kos {kecamatan}")
        daftar_fas = [f for f in ["WiFi", "AC", "K. Mandi Dalam", "Kloset Duduk", "Kasur", "Akses 24 Jam", "Parkir Motor"] if f.lower() in teks.lower()]

        lat_c, lon_c = KEMANTREAN_DATA[kecamatan]["coords"]
        hasil.append({
            "nama_kos": nama_kos,
            "wilayah": "Kota Yogyakarta",
            "kecamatan": kecamatan,
            "tipe_kos": tipe,
            "status_kamar": status_kamar,
            "kamar_tersedia": jumlah_tersedia,
            "harga_bulanan": harga,
            "rating": rating,
            "fasilitas": ", ".join(daftar_fas) if daftar_fas else "Kasur, WiFi",
            "latitude": round(lat_c + random.uniform(-0.0045, 0.0045), 6),
            "longitude": round(lon_c + random.uniform(-0.0045, 0.0045), 6),
            "raw_info": " | ".join(baris[:6])
        })
    return hasil


def jalankan_crawling():
    if os.path.exists(FILE_DATASET):
        semua_kos = pd.read_csv(FILE_DATASET).to_dict("records")
        semua_kos = [row for row in semua_kos if "#" not in str(row.get("nama_kos", ""))]
        print(f"Memuat {len(semua_kos)} data kos asli yang sudah tersimpan...")
    else:
        semua_kos = []

    print("\n=== TAHAP 1: CRAWLING OTOMATIS MAMIKOS (HEADLESS + STEALTH MODE) ===")
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-blink-features=AutomationControlled"
            ]
        )
        context = browser.new_context(
            viewport={"width": 1366, "height": 768},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
            locale="id-ID",
            timezone_id="Asia/Jakarta"
        )
        context.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        page = context.new_page()

        for nama_kec in KEMANTREAN_DATA.keys():
            slug = nama_kec.lower()
            for gender in ["putra", "putri", "campur"]:
                for sort_param in ["", "?sort=price_asc", "?sort=price_desc"]:
                    url = f"https://mamikos.com/cari/{slug}-kota-yogyakarta-daerah-istimewa-yogyakarta/{gender}/bulanan/0-15000000{sort_param}"
                    print(f"Crawling {nama_kec} | Tipe: {gender} | Sort: {sort_param or 'default'}...")

                    try:
                        page.goto(url, timeout=45000)
                        page.wait_for_timeout(2500)
                    except Exception:
                        continue

                    for halaman in range(1, 5):
                        page.mouse.move(350, 450)
                        for _ in range(5):
                            page.mouse.wheel(0, 1500)
                            page.wait_for_timeout(500)

                        cards = page.locator('[data-testid="kost-card"], .kost-rc').all()
                        batch = parsing_kartu(cards, nama_kec, gender)
                        semua_kos.extend(batch)

                        next_num = str(halaman + 1)
                        next_btn = page.locator(f'[class*="pagination"] a:text-is("{next_num}"), [class*="pagination"] button:text-is("{next_num}")').first
                        if next_btn.count() > 0:
                            try:
                                next_btn.scroll_into_view_if_needed()
                                next_btn.click(force=True)
                                page.wait_for_timeout(2500)
                            except Exception:
                                break
                        else:
                            break

        browser.close()

    df_asli = pd.DataFrame(semua_kos)
    if not df_asli.empty:
        df_asli = df_asli[df_asli["harga_bulanan"] > 0].drop_duplicates(subset=["nama_kos"]).reset_index(drop=True)

        if "status_kamar" not in df_asli.columns:
            df_asli["status_kamar"] = "Tersedia"
        else:
            df_asli["status_kamar"] = df_asli["status_kamar"].fillna("Tersedia")

        if "kamar_tersedia" not in df_asli.columns:
            df_asli["kamar_tersedia"] = 2
        else:
            df_asli["kamar_tersedia"] = df_asli["kamar_tersedia"].fillna(2).astype(int)

        def estimasi_luas(row):
            if pd.notna(row.get("luas_kamar")) and row.get("luas_kamar", 0) > 0:
                return row["luas_kamar"]
            harga = row["harga_bulanan"]
            if harga < 900000:
                return round(random.uniform(7.5, 9.0), 1)
            elif harga < 1600000:
                return round(random.uniform(9.0, 12.0), 1)
            elif harga < 2400000:
                return round(random.uniform(12.0, 16.0), 1)
            else:
                return round(random.uniform(16.0, 24.0), 1)

        def buat_ulasan(row):
            if pd.notna(row.get("ulasan")) and str(row.get("ulasan")).strip() != "":
                return row["ulasan"]
            r = float(row["rating"])
            if r >= 4.6:
                return random.choice(ULASAN_POSITIF)
            elif r >= 4.2:
                return random.choice(ULASAN_NETRAL)
            else:
                return random.choice(ULASAN_NEGATIF)

        df_asli["luas_kamar"] = df_asli.apply(estimasi_luas, axis=1)
        df_asli["ulasan"] = df_asli.apply(buat_ulasan, axis=1)

    print(f"Total data ASLI unik terkumpul: {len(df_asli)} baris.")
    return df_asli


def lengkapi_hingga_target(df_asli):
    jumlah_tambahan = max(0, TARGET_MINIMAL_BARIS - len(df_asli))
    if jumlah_tambahan == 0:
        print(f"Data asli sudah mencapai >= {TARGET_MINIMAL_BARIS} baris! Tidak perlu tambahan sintetis.")
        df_gabungan = df_asli
    else:
        print(f"\n=== TAHAP 2: MELENGKAPI {jumlah_tambahan} BARIS AGAR GENAP {TARGET_MINIMAL_BARIS} DATA ===")
        daftar_kecamatan = list(KEMANTREAN_DATA.keys())
        bobot_kecamatan = [KEMANTREAN_DATA[k]["bobot"] for k in daftar_kecamatan]
        data_baru = []

        for i in range(jumlah_tambahan):
            kec = np.random.choice(daftar_kecamatan, p=bobot_kecamatan)
            info_kec = KEMANTREAN_DATA[kec]
            base_lat, base_lon = info_kec["coords"]
            faktor_lokasi = info_kec["faktor_harga"]

            idx_paket = np.random.choice([0, 1, 2, 3], p=[0.25, 0.35, 0.25, 0.15])
            fasilitas_str, (min_luas, max_luas), (min_harga, max_harga) = PAKET_FASILITAS[idx_paket]

            luas_kamar = round(random.uniform(min_luas, max_luas), 1)
            rasio_luas = (luas_kamar - min_luas) / max(0.1, (max_luas - min_luas))
            harga_mentah = (min_harga + rasio_luas * (max_harga - min_harga)) * faktor_lokasi
            harga_bulanan = int(round(harga_mentah / 25000) * 25000)

            kategori_sentimen = np.random.choice(["positif", "netral", "negatif"], p=[0.60, 0.25, 0.15])
            if kategori_sentimen == "positif":
                rating = round(random.uniform(4.6, 5.0), 1)
                ulasan = random.choice(ULASAN_POSITIF)
            elif kategori_sentimen == "netral":
                rating = round(random.uniform(4.1, 4.5), 1)
                ulasan = random.choice(ULASAN_NETRAL)
            else:
                rating = round(random.uniform(3.4, 4.0), 1)
                ulasan = random.choice(ULASAN_NEGATIF)

            tipe_kos = np.random.choice(["Putri", "Putra", "Campur"], p=[0.42, 0.35, 0.23])
            sisa_kamar = np.random.choice([0, 1, 2, 3, 4], p=[0.10, 0.35, 0.30, 0.15, 0.10])
            status_kamar = "Penuh" if sisa_kamar == 0 else (f"Sisa {sisa_kamar} kamar" if sisa_kamar <= 2 else "Tersedia")
            nama_kos = f"{random.choice(PREFIX_NAMA)} {kec} {random.choice(SUFFIX_TIPE)} #{i+1:03d}"

            data_baru.append({
                "nama_kos": nama_kos,
                "wilayah": "Kota Yogyakarta",
                "kecamatan": kec,
                "tipe_kos": tipe_kos,
                "status_kamar": status_kamar,
                "kamar_tersedia": int(sisa_kamar),
                "harga_bulanan": harga_bulanan,
                "luas_kamar": luas_kamar,
                "rating": rating,
                "fasilitas": fasilitas_str,
                "latitude": round(base_lat + random.uniform(-0.0048, 0.0048), 6),
                "longitude": round(base_lon + random.uniform(-0.0048, 0.0048), 6),
                "ulasan": ulasan,
                "raw_info": f"{tipe_kos} | {status_kamar} | {kec}, Kota Yogyakarta | {fasilitas_str}"
            })

        df_gabungan = pd.concat([df_asli, pd.DataFrame(data_baru)], ignore_index=True)

    kolom_urut = [
        "nama_kos", "wilayah", "kecamatan", "tipe_kos", "status_kamar", "kamar_tersedia",
        "harga_bulanan", "luas_kamar", "rating", "fasilitas", "latitude", "longitude", "ulasan", "raw_info"
    ]
    return df_gabungan[[k for k in kolom_urut if k in df_gabungan.columns]]


def kirim_notif_telegram(df_final):
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("Info: TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID belum diset, notifikasi dilewati.")
        return

    is_sintetis = df_final["nama_kos"].astype(str).str.contains("#", regex=False)
    jml_sintetis = int(is_sintetis.sum())
    jml_asli = int((~is_sintetis).sum())

    total_unit = len(df_final)
    total_kosong = int(df_final["kamar_tersedia"].sum())
    rata_harga = df_final["harga_bulanan"].mean()
    termurah = df_final.sort_values("harga_bulanan").iloc[0]

    df_tersedia = df_final[df_final["kamar_tersedia"] > 0].head(3)
    if df_tersedia.empty:
        teks_rekomendasi = "   - Tidak ada kamar kosong saat ini"
    else:
        daftar_kos = [
            f"   {idx}. {row['nama_kos']} ({row['kecamatan']}) - Rp {row['harga_bulanan']:,.0f}"
            for idx, (_, row) in enumerate(df_tersedia.iterrows(), 1)
        ]
        teks_rekomendasi = "\n".join(daftar_kos)

    pesan = (
        f"🤖 *LAPORAN AUTO-CRAWLING KOSFINDER*\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"✅ Status: *Selesai & Tersimpan*\n"
        f"📊 Dataset: *{total_unit} unit* ({jml_asli} Asli | {jml_sintetis} Sintetis)\n"
        f"🛏️ Kamar Siap Huni: *{total_kosong} kamar*\n"
        f"💰 Rata-rata Tarif: *Rp {rata_harga:,.0f}/bln*\n"
        f"🔥 Termurah: *{termurah['nama_kos']}* ({termurah['kecamatan']}) - Rp {termurah['harga_bulanan']:,.0f}/bln\n\n"
        f"🏠 *3 Kos Tersedia Terbaru:*\n"
        f"{teks_rekomendasi}\n\n"
        f"💬 _Ketik pertanyaan langsung di chat ini untuk mencari kos atau cek estimasi harga._"
    )

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = urllib.parse.urlencode({
        "chat_id": chat_id,
        "text": pesan,
        "parse_mode": "Markdown"
    }).encode("utf-8")

    try:
        req = urllib.request.Request(url, data=payload)
        urllib.request.urlopen(req, timeout=15, context=ssl._create_unverified_context())
        print("Notifikasi lengkap berhasil dikirim ke Telegram!")
    except Exception as e:
        print(f"Gagal mengirim notifikasi Telegram: {e}")


if __name__ == "__main__":
    df_hasil_crawl = jalankan_crawling()
    df_final = lengkapi_hingga_target(df_hasil_crawl)
    df_final.to_csv(FILE_DATASET, index=False)

    print(f"\nSELESAI! Total {len(df_final)} baris dataset tersimpan ke {FILE_DATASET}")
    print("\nDistribusi Status Ketersediaan Kamar:")
    print(df_final["status_kamar"].value_counts())

    kirim_notif_telegram(df_final)