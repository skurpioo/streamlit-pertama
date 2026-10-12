# =============================================================
# APLIKASI MACHINE LEARNING DENGAN STREAMLIT + MONGODB
# Regresi, Klasifikasi, CRUD data Iris di MongoDB, dan Retrain
# =============================================================

# --- Import library ---
import time
import warnings
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

warnings.filterwarnings("ignore")

# --- Konfigurasi halaman ---
st.set_page_config(
    page_title="ML Streamlit + MongoDB",
    page_icon="🤖",
    layout="wide",
)

DB_NAME = "test_db"
COLLECTION_NAME = "iris_db"
SPECIES = ["setosa", "versicolor", "virginica"]
IRIS_FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]


# =============================================================
# Fungsi bantu: dataset dan model
# =============================================================
@st.cache_data
def load_dataset(task: str, name: str):
    """Memuat toy dataset scikit-learn dan mengembalikan X, y, nama kelas."""
    loaders = {
        "Diabetes": datasets.load_diabetes,
        "Iris": datasets.load_iris,
        "Wine": datasets.load_wine,
        "Breast Cancer": datasets.load_breast_cancer,
    }
    data = loaders[name]()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")
    classes = list(data.target_names) if task == "Klasifikasi" else None
    return X, y, classes


def get_model(task: str, name: str, seed: int):
    """Membuat objek model sesuai pilihan di sidebar."""
    if task == "Regresi":
        models = {
            "Linear Regression": LinearRegression(),
            "Ridge": Ridge(),
            "Lasso": Lasso(),
            "Decision Tree": DecisionTreeRegressor(random_state=seed),
            "Random Forest": RandomForestRegressor(random_state=seed),
        }
    else:
        models = {
            "Logistic Regression": LogisticRegression(max_iter=5000),
            "KNN": KNeighborsClassifier(),
            "Decision Tree": DecisionTreeClassifier(random_state=seed),
            "Random Forest": RandomForestClassifier(random_state=seed),
            "SVM": SVC(),
        }
    return models[name]


# =============================================================
# Fungsi bantu: MongoDB
# =============================================================
@st.cache_resource
def get_collection():
    """Membuat koneksi MongoDB satu kali (di-cache). URI dibaca dari st.secrets."""
    from pymongo import MongoClient

    uri = st.secrets["MONGODB_URI"]
    client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    client.admin.command("ping")  # uji koneksi
    return client[DB_NAME][COLLECTION_NAME]


@st.cache_data(ttl=5)
def load_docs():
    """Membaca semua dokumen dari MongoDB (di-cache 5 detik)."""
    docs = list(get_collection().find())
    for d in docs:
        d["_id"] = str(d["_id"])
    return docs


def refresh_and_rerun():
    """Bersihkan cache query lalu muat ulang halaman."""
    load_docs.clear()
    st.rerun()


# =============================================================
# Sidebar
# =============================================================
st.sidebar.title("⚙️ Pengaturan Model")
st.sidebar.caption("Atur parameter di bawah, lalu klik **Train Model**.")

task = st.sidebar.radio("🎯 Pilih Jenis Model", ["Regresi", "Klasifikasi"], horizontal=True)

if task == "Regresi":
    dataset_options = ["Diabetes"]
    model_options = ["Linear Regression", "Ridge", "Lasso", "Decision Tree", "Random Forest"]
else:
    dataset_options = ["Iris", "Wine", "Breast Cancer"]
    model_options = ["Logistic Regression", "KNN", "Decision Tree", "Random Forest", "SVM"]

dataset_name = st.sidebar.selectbox("📊 Pilih Dataset", dataset_options)
model_name = st.sidebar.selectbox("🧠 Pilih Model", model_options)
test_size = st.sidebar.slider("📐 Test Size", 0.10, 0.50, 0.20, 0.05)
random_state = st.sidebar.number_input("🎲 Random State", min_value=0, value=42, step=1)
train_clicked = st.sidebar.button("🚀 Train Model", type="primary")

# =============================================================
# Header
# =============================================================
st.title("🤖 Regresi / Klasifikasi Machine Learning")
st.caption("Aplikasi demo ML + MongoDB CRUD + Retrain dengan scikit-learn")

# =============================================================
# Proses training (disimpan di session_state)
# =============================================================
if train_clicked:
    X, y, classes = load_dataset(task, dataset_name)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=int(random_state)
    )
    model = get_model(task, model_name, int(random_state))

    start = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - start

    st.session_state["result"] = {
        "task": task,
        "dataset": dataset_name,
        "model": model_name,
        "classes": classes,
        "y_test": np.asarray(y_test),
        "y_pred": model.predict(X_test),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "elapsed": elapsed,
    }

res = st.session_state.get("result")

tab1, tab2, tab3 = st.tabs(
    ["📊 Training & Visualisasi", "📈 Evaluation Metrics", "🗄️ Database & Retrain"]
)

# =============================================================
# Tab 1: Training & Visualisasi
# =============================================================
with tab1:
    if res is None:
        st.warning("Klik **Train Model** dulu di sidebar.")
    else:
        st.success(f"Model berhasil ditraining dalam {res['elapsed']:.3f} detik!")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🎯 Task", res["task"])
        c2.metric("📊 Dataset", res["dataset"])
        c3.metric("🧠 Model", res["model"])
        c4.metric("⏱️ Training Time", f"{res['elapsed']:.3f}s")

        st.info(f"Data Training: {res['n_train']} sampel | Data Testing: {res['n_test']} sampel")
        st.subheader("📊 Hasil Visualisasi")

        y_test, y_pred = res["y_test"], res["y_pred"]
        if res["task"] == "Regresi":
            left, right = st.columns(2)
            with left:
                st.markdown("**Actual vs Predicted (Scatter)**")
                fig, ax = plt.subplots(figsize=(5, 4))
                ax.scatter(y_test, y_pred, alpha=0.6, edgecolor="k")
                lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
                ax.plot(lims, lims, "r--", label="Perfect Prediction")
                ax.set_xlabel("Nilai Aktual")
                ax.set_ylabel("Nilai Prediksi")
                ax.legend()
                ax.grid(alpha=0.3)
                st.pyplot(fig)
            with right:
                st.markdown("**Perbandingan per Index (50 pertama)**")
                fig, ax = plt.subplots(figsize=(5, 4))
                n = min(50, len(y_test))
                ax.plot(range(n), y_test[:n], "o-", label="Actual")
                ax.plot(range(n), y_pred[:n], "x-", label="Predicted")
                ax.set_xlabel("Index")
                ax.set_ylabel("Nilai")
                ax.legend()
                ax.grid(alpha=0.3)
                st.pyplot(fig)
        else:
            st.markdown("**Confusion Matrix**")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(
                cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=res["classes"], yticklabels=res["classes"],
            )
            ax.set_xlabel("Prediksi")
            ax.set_ylabel("Aktual")
            st.pyplot(fig)

# =============================================================
# Tab 2: Evaluation Metrics
# =============================================================
with tab2:
    if res is None:
        st.warning("Klik **Train Model** dulu di sidebar.")
    else:
        y_test, y_pred = res["y_test"], res["y_pred"]
        if res["task"] == "Regresi":
            mse = mean_squared_error(y_test, y_pred)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("MAE", f"{mean_absolute_error(y_test, y_pred):.3f}")
            m2.metric("MSE", f"{mse:.3f}")
            m3.metric("RMSE", f"{np.sqrt(mse):.3f}")
            m4.metric("R²", f"{r2_score(y_test, y_pred):.3f}")
        else:
            avg = "binary" if len(res["classes"]) == 2 else "weighted"
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.3f}")
            m2.metric("Precision", f"{precision_score(y_test, y_pred, average=avg):.3f}")
            m3.metric("Recall", f"{recall_score(y_test, y_pred, average=avg):.3f}")
            m4.metric("F1-Score", f"{f1_score(y_test, y_pred, average=avg):.3f}")

            st.subheader("📋 Classification Report")
            report = classification_report(
                y_test, y_pred, target_names=res["classes"], output_dict=True
            )
            st.dataframe(pd.DataFrame(report).transpose().round(3))

# =============================================================
# Tab 3: Database & Retrain (MongoDB)
# =============================================================
with tab3:
    st.header("🗄️ MongoDB CRUD & Retrain Model")

    try:
        col = get_collection()
        docs = load_docs()
    except (KeyError, FileNotFoundError):
        st.error(
            "Secret `MONGODB_URI` belum diisi. Isi di `.streamlit/secrets.toml` (lokal) "
            "atau Settings → Secrets (Streamlit Cloud)."
        )
        st.stop()
    except Exception as e:  # koneksi gagal, password salah, IP belum diizinkan, dll.
        st.error(f"Gagal terhubung ke MongoDB: {e}")
        st.info("Periksa password di URI, dan Network Access di MongoDB Atlas.")
        st.stop()

    # --- Statistik collection ---
    st.subheader("📊 Statistik Collection")
    s1, s2, s3 = st.columns(3)
    s1.metric("Total Dokumen", len(docs))
    s2.metric("Database", DB_NAME)
    s3.metric("Collection", COLLECTION_NAME)

    st.divider()

    # --- Form tambah data ---
    st.subheader("➕ Tambah Data Iris")
    with st.form("form_iris", clear_on_submit=False):
        f1, f2 = st.columns(2)
        sepal_length = f1.number_input("Sepal Length (cm)", 0.0, 10.0, 5.1, 0.1)
        petal_length = f2.number_input("Petal Length (cm)", 0.0, 10.0, 1.4, 0.1)
        sepal_width = f1.number_input("Sepal Width (cm)", 0.0, 10.0, 3.5, 0.1)
        petal_width = f2.number_input("Petal Width (cm)", 0.0, 10.0, 0.2, 0.1)
        species = st.selectbox("Species", SPECIES)
        submitted = st.form_submit_button("💾 Simpan ke MongoDB")

    if submitted:
        col.insert_one(
            {
                "sepal_length": float(sepal_length),
                "sepal_width": float(sepal_width),
                "petal_length": float(petal_length),
                "petal_width": float(petal_width),
                "species": species,
                "timestamp": datetime.now(timezone.utc),
            }
        )
        st.success("Data berhasil disimpan!")
        refresh_and_rerun()

    st.divider()

    # --- Tabel data + hapus ---
    st.subheader("📋 Data di MongoDB")
    if docs:
        st.dataframe(pd.DataFrame(docs))
        d1, d2 = st.columns([3, 1])
        selected_id = d1.selectbox("Pilih _id yang akan dihapus", [d["_id"] for d in docs])
        if d2.button("🗑️ Hapus"):
            from bson import ObjectId

            col.delete_one({"_id": ObjectId(selected_id)})
            st.success("Data dihapus.")
            refresh_and_rerun()
    else:
        st.info("Belum ada data di collection. Tambahkan lewat form di atas.")

    st.divider()

    # --- Retrain ---
    st.subheader("🚀 Retrain Model")
    st.caption("Menggabungkan data Iris sklearn (150 sampel) dengan data dari MongoDB, lalu melatih Random Forest.")

    if st.button("🚀 Retrain Sekarang"):
        iris = datasets.load_iris()
        base = pd.DataFrame(iris.data, columns=IRIS_FEATURES)
        base["label"] = iris.target

        extra = pd.DataFrame(docs)
        label_map = {name: i for i, name in enumerate(SPECIES)}  # setosa=0, versicolor=1, virginica=2
        if not extra.empty:
            extra = extra[IRIS_FEATURES + ["species"]].copy()
            extra["label"] = extra["species"].map(label_map)
            extra = extra.drop(columns="species")
            full = pd.concat([base, extra], ignore_index=True)
        else:
            full = base

        X = full[IRIS_FEATURES]
        y = full["label"].astype(int)
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_tr, y_tr)
        pred = rf.predict(X_te)

        st.success(
            f"Retrain selesai dengan {len(full)} sampel "
            f"({len(base)} dari sklearn + {len(full) - len(base)} dari MongoDB)."
        )
        st.metric("Akurasi", f"{accuracy_score(y_te, pred):.3f}")

        rep = classification_report(y_te, pred, target_names=SPECIES, output_dict=True, zero_division=0)
        st.markdown("**Classification Report**")
        st.dataframe(pd.DataFrame(rep).transpose().round(3))

        st.markdown("**Feature Importance**")
        imp = pd.Series(rf.feature_importances_, index=IRIS_FEATURES).sort_values()
        fig, ax = plt.subplots(figsize=(6, 3))
        imp.plot(kind="barh", ax=ax, color="#0B6E63")
        ax.set_xlabel("Importance")
        st.pyplot(fig)
