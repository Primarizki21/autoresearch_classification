# Autonomous ML Research Agent (Autoresearch) 🧠🤖

Repository ini dirancang sebagai template/framework agar AI Agent (contoh: Claude, GPT, Gemini) bisa melakukan eksperimen Machine Learning (AutoML) secara otonom! 

Inspirasi framework ini berangkat dari *loop* otomatisasi riset. Inti kerjanya sangat sederhana: kamu memberikan data + metrik evaluasinya, kemudian menyuruh AI Agent untuk berputar di dalam sebuah *loop* (membuat hipotesis model ➡️ mengedit kode ➡️ mengevaluasi ➡️ menimbang hasilnya). Jika skor naik, perubahan disimpan di Git; jika turun, langsung dibuang (*discard/reset*).

---

## ⚡ Prasyarat Penting: Instalasi `uv`
Repository ini dioptimasi dan dijalankan secara mutlak menggunakan manajer dependensi **[uv](https://github.com/astral-sh/uv)** dari Astral. Skrip utama pada agent loop memanggil instruksi `uv run` beserta `uv add`.
Karena itu, agar kelak gampang me-*manage* proyek atau dependensi:
- **Pastikan kamu sudah menginstal `uv`** di sistem lokalmu terlebih dahulu sebelum menjalankan siklus eksperimen!
- Pelajari cara kerjanya jika dirasa awam (*uv sangat ringan, super gesit, dan secara otomatis membungkus virtual environment Python-mu*).

---

## 📂 Arsitektur File

Setiap file di repository ini punya tugas terisolasi yang wajib dipahami:

- **`program.md`**: Otak instruksi (Prompt) bagi AI Agent. File ini mendikte apa yang boleh/tidak boleh diedit oleh AI, *goal* metrik (misal: naikkan `f1_macro`), dan SOP siklus eksperimentasinya.
- **`prepare.py`**: Skrip **HARAM DIUBAH** oleh agen! Disinilah dataset dimuat, dibagi (Train/Val split statis agar *fair*), dan fungsi baku `evaluate()` berada. Ini memastikan evaluasi agen selalu jujur.
- **`experiment.py`**: Inilah *canvas/bengkel* tempat AI akan bekerja. AI bebas mengubah model (XGBoost, Neural Network, Logistic Regression), melakukan *feature engineering*, dan *tuning hyperparameter* di sini.
- **`results.tsv`** *(Tracked)*: Rangkuman rekap komit-komit model yang pernah dibuat, lengkap dengan skor metriknya. Berguna untuk memilah mana eksperimen brilian, dan mana yang gagal.
- **`experiment_memory.md`** *(Gitignored)*: "Buku harian" sang agen. Tempat agen mencatat *insight* setelah sebuah eksperimen berjalan (contoh: *"LightGBM overfit jika leaves > 31"*). Wajib dijaga agar agen tak mengulangi eksperimen bodoh yang sama.
- **`evaluation_output/`** *(Gitignored)*: Gudang artefak visual. Agen diwajibkan menyimpan `history.png` (plot loss) dan `confusion_matrix.png` per kode hash commit Git.
- **`run.log`** *(Gitignored)*: Terminal output yang diamati AI untuk membaca metrik atau melihat *stack trace error*.

---

## 🛠️ Tutorial Memakai Repo ini untuk Project Lain

Kalau di masa depan kamu punya projek *tabular ML* klasifikasi/regresi lain yang mau kamu otomasi dengan AI, begini cara *re-use* repository ini:

### 1. Masukkan Dataset Baru
Ganti file `train.csv` dan `test.csv` dengan dataset dari tugas barumu.

### 2. Modifikasi Evaluator Baku (`prepare.py`)
Buka `prepare.py` dan ganti konstanta berikut sesuai studi kasus barumu:
- Ubah `TARGET_COL` ("price_range", "churn", dll).
- Sesuaikan metrik evaluasinya di fungsi `evaluate(model, X_val, y_val)`. (Jika klasifikasi ganti ke Accuracy/F1, kalau regresi ganti ke RMSE/MAE).

### 3. Reset Catatan AI
- Bersihkan isi `results.tsv` hingga menyisakan baris header saja (`commit | f1_macro | accuracy | status | description`).
- Hapus semua catatan di `experiment_memory.md` sehingga menjadi file kosong (biarkan AI belajar kembali dari awal!).
- Hapus isi folder `evaluation_output/`.

### 4. Perbarui Titah Sang Sistem (`program.md`)
Buka `program.md` dan ubah pada bagian **Dataset**:
- Tuliskan jumlah *features*, jumlah *samples*, dan deskripsi ringkas datasetnya.
- Ubah **Primary metric** (contoh: ganti ke `rmse` untuk Regresi).
- Opsional: Berikan saran ide baru pada section **Ideas to explore** (misal suruh pakai CatBoost).

### 5. Bangunkan sang AI Agent!
Setelah semua persiapan di atas bersih, buka *chat session* dengan AI favoritmu (lengkap dengan akses environment ini), lalu cukup lontarkan perintah sederhana berikut:

> *"Baca dengan seksama program.md dan mulai jalankan eksperimen loop-nya. Setup branch dan maksimalkan f1_score setinggi mungkin!"*

Biarkan agen menguasai terminal (`uv run`), mengeksekusi git branch/commit/reset, hingga menyajikan artefak grafis visual sendirian! 🚀
