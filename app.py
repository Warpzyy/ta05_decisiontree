import json
import os
import re

import joblib
import pandas as pd
from flask import Flask, render_template, request

from tree_utils import CAT_COLS, FEATURES, NUM_COLS, explain_path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

model = joblib.load(os.path.join(MODEL_DIR, "decision_tree_pruned.joblib"))
with open(os.path.join(MODEL_DIR, "config.json"), encoding="utf-8") as f:
    config = json.load(f)
with open(os.path.join(MODEL_DIR, "metrics.json"), encoding="utf-8") as f:
    metrics = json.load(f)

app = Flask(__name__)

# ---------------------------------------------------------------- label tampilan (bahasa Indonesia)
COL_LABEL = {
    "age": "usia", "education.num": "tingkat pendidikan (angka)", "capital.gain": "keuntungan modal",
    "capital.loss": "kerugian modal", "hours.per.week": "jam kerja per minggu",
    "workclass": "jenis pekerjaan (sektor)", "marital.status": "status pernikahan", "occupation": "jabatan",
}
WORKCLASS_ID = {
    "Federal-gov": "Pemerintah federal", "Local-gov": "Pemerintah daerah", "State-gov": "Pemerintah negara bagian",
    "Private": "Swasta", "Self-emp-inc": "Wiraswasta berbadan usaha", "Self-emp-not-inc": "Wiraswasta perorangan",
    "Without-pay": "Tanpa dibayar", "Never-worked": "Belum pernah bekerja", "Unknown": "Tidak diketahui",
}
MARITAL_ID = {
    "Divorced": "Cerai", "Married-AF-spouse": "Menikah (pasangan militer)", "Married-civ-spouse": "Menikah (pasangan sipil)",
    "Married-spouse-absent": "Menikah (pasangan tidak ada)", "Never-married": "Belum pernah menikah",
    "Separated": "Berpisah", "Widowed": "Janda/duda",
}
OCCUPATION_ID = {
    "Adm-clerical": "Administrasi", "Armed-Forces": "Angkatan bersenjata", "Craft-repair": "Kerajinan dan perbaikan",
    "Exec-managerial": "Eksekutif/manajerial", "Farming-fishing": "Pertanian dan perikanan",
    "Handlers-cleaners": "Pekerja kasar/kebersihan", "Machine-op-inspct": "Operator mesin/inspeksi",
    "Other-service": "Jasa lainnya", "Priv-house-serv": "Pekerja rumah tangga", "Prof-specialty": "Profesional/spesialis",
    "Protective-serv": "Layanan perlindungan", "Sales": "Penjualan", "Tech-support": "Dukungan teknis",
    "Transport-moving": "Transportasi", "Unknown": "Tidak diketahui",
}
EDU_ID = {
    "Preschool": "Prasekolah", "1st-4th": "Kelas 1-4", "5th-6th": "Kelas 5-6", "7th-8th": "Kelas 7-8", "9th": "Kelas 9",
    "10th": "Kelas 10", "11th": "Kelas 11", "12th": "Kelas 12", "HS-grad": "Lulus SMA", "Some-college": "Pernah kuliah (belum lulus)",
    "Assoc-voc": "Diploma (vokasi)", "Assoc-acdm": "Diploma (akademik)", "Bachelors": "Sarjana (S1)",
    "Masters": "Magister (S2)", "Prof-school": "Sekolah profesi", "Doctorate": "Doktor (S3)",
}
CAT_LABELS = {"workclass": WORKCLASS_ID, "marital.status": MARITAL_ID, "occupation": OCCUPATION_ID}

EDUCATION = sorted(
    [(int(num), EDU_ID.get(name, name)) for num, name in config["education_map"].items()], key=lambda t: t[0]
)
OPTIONS = {c: [(v, CAT_LABELS[c].get(v, v)) for v in config["categories"][c]] for c in CAT_COLS}
RANGES = config["ranges"]
NUM_LABEL = {"age": "Usia", "capital.gain": "Keuntungan modal", "capital.loss": "Kerugian modal", "hours.per.week": "Jam kerja per minggu"}

STEP = re.compile(r"^(?P<col>[\w.]+) (?P<op><=|>|=|bukan) (?P<val>.+)$")


def num_id(text):
    return text.replace(".", ",")


def friendly(step):
    """Ubah teks syarat dari tree_utils menjadi kalimat berbahasa Indonesia."""
    m = STEP.match(step)
    if not m:
        return step
    col, op, val = m.group("col"), m.group("op"), m.group("val")
    label = COL_LABEL.get(col, col)
    if col in CAT_LABELS:
        name = CAT_LABELS[col].get(val, val)
        return f"{label} adalah {name}" if op == "=" else f"{label} bukan {name}"
    sym = "kurang dari atau sama dengan" if op == "<=" else "lebih dari"
    text = f"{label} {sym} {num_id(val)}"
    if col == "education.num":
        text += " (13 = Sarjana, 9 = Lulus SMA)"
    return text


def validate(form):
    data, errors = {}, []
    for col in NUM_COLS:
        low, high = RANGES[col]
        label = NUM_LABEL.get(col, "Tingkat pendidikan")
        raw = (form.get(col) or "").strip().replace(",", ".")
        try:
            value = float(raw)
        except ValueError:
            errors.append(f"{label} harus berupa angka.")
            continue
        if col == "education.num":
            if int(value) not in dict(EDUCATION):
                errors.append("Pilihan pendidikan tidak valid.")
                continue
            value = int(value)
        elif not (low <= value <= high):
            errors.append(f"{label} harus antara {num_id(str(int(low)))} dan {num_id(str(int(high)))}.")
            continue
        data[col] = value
    for col in CAT_COLS:
        value = form.get(col)
        if value not in [v for v, _ in OPTIONS[col]]:
            errors.append(f"Pilihan {COL_LABEL[col]} tidak valid.")
            continue
        data[col] = value
    return data, errors


@app.route("/", methods=["GET", "POST"])
def index():
    result, errors = None, []
    values = request.form if request.method == "POST" else {}
    if request.method == "POST":
        data, errors = validate(request.form)
        if not errors:
            row = pd.DataFrame([data])[FEATURES]
            info = explain_path(model, row)
            high = info["prediction"] == 1
            result = {
                "high": high,
                "label": "Pendapatan di atas 50 ribu dolar per tahun" if high else "Pendapatan 50 ribu dolar per tahun atau kurang",
                "short": ">50K" if high else "<=50K",
                "p_high": round(info["p_high"] * 100, 1),
                "p_pred": round((info["p_high"] if high else 1 - info["p_high"]) * 100, 1),
                "samples": info["leaf_samples"],
                "steps": [friendly(s) for s in info["steps"]],
            }
    return render_template("index.html", options=OPTIONS, education=EDUCATION, values=values, errors=errors,
                           result=result, metrics=metrics, b_depth=config["b_depth"], ranges=RANGES)


@app.route("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    app.run(debug=True)
