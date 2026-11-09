"""Fungsi bersama untuk notebook dan aplikasi Flask: klasifikasi pendapatan dengan Decision Tree.

- NUM_COLS / CAT_COLS : fitur yang dipakai model
- explain_path        : jalur keputusan satu data (untuk ditampilkan di aplikasi)
- extract_leaf_rules  : aturan "jika-maka" dari daun-daun terbesar (untuk laporan)
"""
import numpy as np

NUM_COLS = ["age", "education.num", "capital.gain", "capital.loss", "hours.per.week"]
CAT_COLS = ["workclass", "marital.status", "occupation"]
FEATURES = NUM_COLS + CAT_COLS
CLASS_NAMES = ["<=50K", ">50K"]


def _split_name(name):
    """'num__age' -> ('age', None)  |  'cat__marital.status_Divorced' -> ('marital.status', 'Divorced')."""
    if name.startswith("num__"):
        return name[5:], None
    body = name[5:] if name.startswith("cat__") else name
    for col in CAT_COLS:
        if body.startswith(col + "_"):
            return col, body[len(col) + 1:]
    return body, None


def _fmt(v):
    return f"{v:.1f}".rstrip("0").rstrip(".") if abs(v - round(v)) > 1e-9 else str(int(round(v)))


def condition_text(feature_name, threshold, go_left):
    """Teks syarat pada satu cabang."""
    col, cat = _split_name(feature_name)
    if cat is None:
        return f"{col} <= {_fmt(threshold)}" if go_left else f"{col} > {_fmt(threshold)}"
    return f"{col} bukan {cat}" if go_left else f"{col} = {cat}"


def explain_path(pipeline, row_df):
    """Jalur dari akar ke daun untuk satu baris data (DataFrame 1 baris)."""
    prep, model = pipeline.named_steps["prep"], pipeline.named_steps["tree"]
    names = prep.get_feature_names_out()
    xt = prep.transform(row_df)
    tree = model.tree_
    indicator = model.decision_path(xt)
    nodes = indicator.indices[indicator.indptr[0]:indicator.indptr[1]]
    steps = []
    for node in nodes[:-1]:
        f, thr = tree.feature[node], tree.threshold[node]
        go_left = xt[0, f] <= thr
        steps.append(condition_text(names[f], thr, go_left))
    leaf = nodes[-1]
    counts = tree.value[leaf][0]
    total = counts.sum()
    share = counts / total if total else counts
    return {
        "steps": steps,
        "leaf_samples": int(tree.n_node_samples[leaf]),
        "p_high": float(share[1]),
        "prediction": int(np.argmax(counts)),
    }


def extract_leaf_rules(pipeline, top_n=3, target_class=None):
    """Aturan dari daun dengan sampel terbanyak (opsional dibatasi pada satu kelas prediksi)."""
    prep, model = pipeline.named_steps["prep"], pipeline.named_steps["tree"]
    names = prep.get_feature_names_out()
    tree = model.tree_
    rules = []

    def walk(node, conds):
        if tree.children_left[node] == tree.children_right[node]:      # daun
            counts = tree.value[node][0]
            pred = int(np.argmax(counts))
            if target_class is None or pred == target_class:
                rules.append({
                    "kelas": CLASS_NAMES[pred],
                    "sampel": int(tree.n_node_samples[node]),
                    "kemurnian": float(counts.max() / counts.sum()),
                    "syarat": list(conds),
                })
            return
        f, thr = tree.feature[node], tree.threshold[node]
        walk(tree.children_left[node], conds + [condition_text(names[f], thr, True)])
        walk(tree.children_right[node], conds + [condition_text(names[f], thr, False)])

    walk(0, [])
    rules.sort(key=lambda r: r["sampel"], reverse=True)
    return rules[:top_n]


def importance_by_variable(pipeline):
    """Jumlahkan feature importance fitur one-hot menjadi satu nilai per variabel asli."""
    prep, model = pipeline.named_steps["prep"], pipeline.named_steps["tree"]
    totals = {}
    for name, imp in zip(prep.get_feature_names_out(), model.feature_importances_):
        col, _ = _split_name(name)
        totals[col] = totals.get(col, 0.0) + float(imp)
    return dict(sorted(totals.items(), key=lambda kv: kv[1], reverse=True))
