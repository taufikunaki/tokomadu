from flask import Flask, render_template, request, jsonify
from sentence_transformers import SentenceTransformer, util
import torch
import re

app = Flask(__name__)

# =====================================================
# DATA PRODUK
# =====================================================
products = [
    {
        "nama": "Madu Hutan",
        "harga": "Rp100.000",
        "gambar": "maduhutan.jpg"
    },
    {
        "nama": "Madu Hitam",
        "harga": "Rp120.000",
        "gambar": "maduhitam.jpg"
    },
    {
        "nama": "Madu Akasia",
        "harga": "Rp130.000",
        "gambar": "maduakasia.jpg"
    }
]

# =====================================================
# FAQ
# =====================================================
faq_data = {
    # --- GRUP MENYAPA ---
    "halo": "Halo 😊 Selamat datang di Toko Madu Alami.",
    "hai": "Halo 😊 Ada yang bisa saya bantu?",
    "selamat pagi": "Selamat pagi 😊 Ada yang bisa saya bantu?",
    "selamat siang": "Selamat siang 😊 Ada yang bisa saya bantu?",
    "selamat malam": "Selamat malam 😊 Ada yang bisa saya bantu?",
    "siapa kamu": "Saya adalah chatbot Toko Madu Alami yang siap membantu informasi produk dan pemesanan.",
    
    # --- GRUP PRODUK ---
    "produk apa saja": "Kami menyediakan Madu Hutan, Madu Hitam, dan Madu Akasia.",
    "ada madu apa saja": "Kami menyediakan Madu Hutan, Madu Hitam, dan Madu Akasia.",
    "jual apa": "Kami menjual Madu Hutan, Madu Hitam, dan Madu Akasia.",
    "harga madu hutan": "Harga Madu Hutan adalah Rp100.000.",
    "harga madu hitam": "Harga Madu Hitam adalah Rp120.000.",
    "harga madu akasia": "Harga Madu Akasia adalah Rp130.000.",
    "harga": "Kami menyediakan beberapa varian madu:\n1. Madu Hutan: Rp100.000\n2. Madu Hitam: Rp120.000\n3. Madu Akasia: Rp130.000",
    "harga madu": "Kami menyediakan beberapa varian madu:\n1. Madu Hutan: Rp100.000\n2. Madu Hitam: Rp120.000\n3. Madu Akasia: Rp130.000",
    "berapa harga madu": "Kami menyediakan beberapa varian madu:\n1. Madu Hutan: Rp100.000\n2. Madu Hitam: Rp120.000\n3. Madu Akasia: Rp130.000",
    "cek harga": "Kami menyediakan beberapa varian madu:\n1. Madu Hutan: Rp100.000\n2. Madu Hitam: Rp120.000\n3. Madu Akasia: Rp130.000",
    "daftar harga": "Kami menyediakan beberapa varian madu:\n1. Madu Hutan: Rp100.000\n2. Madu Hitam: Rp120.000\n3. Madu Akasia: Rp130.000",
    
    # --- GRUP MANFAAT ---
    "apa manfaat madu": "Madu bermanfaat untuk meningkatkan daya tahan tubuh, sumber energi alami, membantu pencernaan dan membantu meredakan batuk.",
    "manfaat madu hutan": "Madu Hutan cocok untuk membantu meningkatkan stamina.",
    "manfaat madu hitam": "Kami merekomendasikan Madu Hitam untuk membantu menjaga kesehatan lambung.",
    "manfaat madu akasia": "Kami merekomendasikan Madu Akasia untuk membantu meredakan batuk.",
    "madu untuk batuk": "Kami merekomendasikan Madu Akasia untuk membantu meredakan batuk.",
    "madu untuk flu": "Madu Akasia dapat membantu membantu daya tahan tubuh saat flu.",
    "madu untuk stamina": "Madu Hutan cocok untuk membantu meningkatkan stamina.",
    "madu untuk maag": "Kami merekomendasikan Madu Hitam untuk membantu menjaga kesehatan lambung.",
    "manfaatnya apa": "Madu bermanfaat untuk meningkatkan daya tahan tubuh, sumber energi alami, membantu pencernaan dan membantu meredakan batuk.",
    
    # --- GRUP PEMESANAN (Ditambahkan Variasi Pendek) ---
    "cara membeli": "Silakan melakukan pemesanan melalui WhatsApp 085869047186.",
    "cara order": "Silakan melakukan pemesanan melalui WhatsApp 085869047186.",
    "cara pesan": "Silakan melakukan pemesanan melalui WhatsApp 085869047186.",
    "gimana cara pesan": "Silakan melakukan pemesanan melalui WhatsApp 085869047186.",
    "mau pesan": "Silakan melakukan pemesanan melalui WhatsApp 085869047186.",
    "pemesanan": "Silakan melakukan pemesanan melalui WhatsApp 085869047186.",
    "pesan": "Silakan melakukan pemesanan melalui WhatsApp 085869047186.",
    "order": "Silakan melakukan pemesanan melalui WhatsApp 085869047186.",
    "beli": "Silakan melakukan pemesanan melalui WhatsApp 085869047186.",
    
    # --- LAIN-LAIN ---
    "metode pembayaran": "Pembayaran dapat dilakukan melalui transfer bank dan e-wallet.",
    "pengiriman": "Kami melayani pengiriman melalui JNE, J&T dan SiCepat.",
    "stok": "Semua produk saat ini ready stock.",
    "lokasi": "Kami berlokasi di Yogyakarta.",
    "terima kasih": "Sama-sama 😊 Semoga sehat selalu."
}

# =====================================================
# LOAD MODEL
# =====================================================
print("Loading NLP Model...")
# Menggunakan device GPU jika tersedia, jika tidak gunakan CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", device=device)
print(f"Model loaded on {device}.")

# =====================================================
# EMBEDDINGS
# =====================================================
faq_questions = list(faq_data.keys())
faq_answers = list(faq_data.values())

faq_embeddings = model.encode(faq_questions, convert_to_tensor=True)
product_names = [p["nama"] for p in products]
product_embeddings = model.encode(product_names, convert_to_tensor=True)

# =====================================================
# TEXT PREPROCESS
# =====================================================
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# =====================================================
# INTENT ORDER
# =====================================================
def detect_order_intent(text):
    # Memisahkan kata per kata agar lebih akurat mendeteksi kata tunggal
    words = text.split()
    order_keywords = [
        "checkout", "beli", "pesan", "order", "pemesanan", 
        "pembelian", "pesanan", "transaksi"
    ]
    return any(k in words for k in order_keywords) or any(k in text for k in ["cara pesan", "cara order", "cara beli"])

# =====================================================
# EXACT PRODUCT MATCH
# =====================================================
def exact_product_match(text):
    for p in products:
        if p["nama"].lower() in text:
            return (
                f"🍯 {p['nama']}\n"
                f"Harga : {p['harga']}\n"
                f"Produk madu alami berkualitas."
            )
    return None

    # =====================================
    # KEYWORD BOOSTER (MANFAAT)
    # =====================================
    if "manfaat" in user_text or "khasiat" in user_text:
        return faq_data["apa manfaat madu"]

# =====================================================
# PRODUCT LIST
# =====================================================
def product_list_response():
    return (
        "Kami menyediakan produk madu berikut:\n\n"
        "1. Madu Hutan - Rp100.000\n"
        "2. Madu Hitam - Rp120.000\n"
        "3. Madu Akasia - Rp130.000"
    )

# =====================================================
# CHATBOT ENGINE
# =====================================================
def chatbot_response(user_text):
    user_text = clean_text(user_text)
    if not user_text:
        return "Halo, ada yang bisa saya bantu? Silakan ketik pertanyaan Anda."

    # 1. LIST PRODUK (Keyword)
    product_keywords = ["produk apa saja", "ada madu apa saja", "jual apa", "daftar produk", "list produk"]
    if any(k in user_text for k in product_keywords):
        return product_list_response()

    # 2. ORDER INTENT (Keyword)
    if detect_order_intent(user_text):
        return "Untuk pemesanan silakan hubungi:\n\n📱 WhatsApp 085869047186"

    # 3. EXACT FAQ MATCH (Pencocokan Persis)
    if user_text in faq_data:
        return faq_data[user_text]

    # 4. SEMANTIC FAQ MATCH (Biarkan AI Semantik menilai dulu!)
    # Di sini, "manfaat madu hutan" akan dicocokkan secara pintar oleh AI 
    # ke FAQ "madu untuk stamina" karena artinya mirip.
    user_embedding = model.encode(user_text, convert_to_tensor=True)
    faq_scores = util.cos_sim(user_embedding, faq_embeddings)[0]
    
    best_idx = torch.argmax(faq_scores).item()
    best_score = faq_scores[best_idx].item()

    print(f"\n[LOG] User: {user_text} | Best FAQ Match: {faq_questions[best_idx]} | Score: {best_score:.4f}")

    if best_score >= 0.65: # Threshold disesuaikan agar lebih fleksibel
        return faq_answers[best_idx]

    # 5. EXACT PRODUCT MATCH (Diturunkan ke sini sebagai Jaring Pengaman)
    # Jika AI bingung (skor di bawah 0.65), baru kita cek apakah user cuma sekadar menyebut nama produk.
    product = exact_product_match(user_text)
    if product:
        return product

    # 6. SEMANTIC PRODUCT MATCH
    product_scores = util.cos_sim(user_embedding, product_embeddings)[0]
    best_product_idx = torch.argmax(product_scores).item()
    best_product_score = product_scores[best_product_idx].item()

    if best_product_score >= 0.75:
        p = products[best_product_idx]
        return (
            f"🍯 {p['nama']}\n"
            f"Harga : {p['harga']}\n"
            f"Produk madu alami berkualitas."
        )

    # 7. FALLBACK
    return (
        "Maaf, saya belum memahami pertanyaan Anda.\n\n"
        "Silakan tanyakan tentang:\n"
        "• Produk madu\n"
        "• Harga\n"
        "• Manfaat madu\n"
        "• Stok & Pengiriman\n"
        "• Cara pemesanan"
    )

# ======================
# ROUTES
# ======================
@app.route("/")
def home():
    return render_template("index.html", products=products)

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json["message"]
    response = chatbot_response(user_input)
    return jsonify({"reply": response})

# ======================
# RUN
# ======================
if __name__ == "__main__":
    app.run(debug=True)