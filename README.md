# Klasifikasi Tingkat Pendapatan dengan Decision Tree

Prototipe aplikasi web yang mengklasifikasikan apakah pendapatan seseorang di atas 50 ribu dolar per tahun
berdasarkan data sensus, lengkap dengan alur keputusan pohon. Dataset: Adult Census Income (Kaggle, uciml).

## Struktur
- `analisis_decision_tree.ipynb` : EDA, pra-pemrosesan, Model A (tanpa batas) vs Model B (pruned), interpretasi, simpan model
- `tree_utils.py` : daftar fitur, penjelasan jalur keputusan, ekstraksi aturan (dipakai notebook dan aplikasi)
- `app.py`, `templates/index.html` : aplikasi Flask
- `model/` : hasil notebook (`decision_tree_pruned.joblib`, `config.json`, `metrics.json`)
- `data/adult.csv` : dataset, `gambar/` : grafik hasil notebook

## Menjalankan (Windows / VS Code)
```
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
```
1. Buka `analisis_decision_tree.ipynb`, pilih kernel `venv`, lalu **Run All**.
2. Jalankan aplikasi: `python app.py`, lalu buka http://127.0.0.1:5000

## Deploy (Vercel / Render)
- Build command : `pip install -r requirements.txt`
- Start command (Render/Railway): `gunicorn app:app` (Vercel mendeteksi `app.py` otomatis)
- Samakan versi `scikit-learn`, `numpy`, dan `pandas` di `requirements.txt` dengan versi lokal
  (`pip show scikit-learn numpy pandas`) karena file `.joblib` sensitif terhadap versi.
- Folder `model/` harus ikut di-commit ke GitHub.
