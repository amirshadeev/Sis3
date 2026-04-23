"""
frontend/streamlit_app.py
Streamlit UI — полностью совместим с PT6 API (app.py).
Поля ответа: predicted_class_id, predicted_class_name, probabilities
"""

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os

API_URL    = os.getenv("API_URL", "http://api:8000")
MLFLOW_URL = os.getenv("MLFLOW_URL", "http://localhost:5000")

CLASS_NAMES  = ["setosa", "versicolor", "virginica"]
CLASS_COLORS = ["#4CAF50", "#2196F3", "#FF9800"]

st.set_page_config(
    page_title="Iris Classifier · SIS-3",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    h1, h2, h3 { font-family: 'Space Mono', monospace; }
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem; border-radius: 12px; margin-bottom: 2rem; text-align: center;
    }
    .main-header h1 { color: #e94560; margin: 0; font-size: 2.2rem; }
    .main-header p  { color: #a8b2d8; margin: 0.5rem 0 0; }
    .result-card {
        background: #1a1a2e; border: 2px solid #e94560; border-radius: 12px;
        padding: 1.5rem; text-align: center; color: white; margin-bottom: 1rem;
    }
    .result-card h2 { color: #e94560; font-family: 'Space Mono', monospace; margin: 0; }
    .result-card p  { color: #a8b2d8; margin: 0.4rem 0 0; }
    .info-box {
        background: #0f3460; border-radius: 8px; padding: 1rem;
        text-align: center; color: #a8b2d8;
    }
    .info-box .val { font-size: 1.8rem; font-weight: 700; color: #e94560; }
    .ok  { color: #4CAF50; font-weight: 600; }
    .err { color: #f44336; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>🌸 Iris Classifier — SIS-3</h1>
    <p>FastAPI (PT6) + MLflow Tracking &amp; Registry + Streamlit</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔧 System Status")
    try:
        r = requests.get(f"{API_URL}/", timeout=3)
        st.markdown('<p class="ok">✅ API: Online</p>', unsafe_allow_html=True)
    except Exception:
        st.markdown('<p class="err">❌ API: Offline</p>', unsafe_allow_html=True)

    try:
        info = requests.get(f"{API_URL}/model-info", timeout=3).json()
        st.markdown(f"**Model:** `{info.get('model_type','—')}`")
        st.markdown(f"**Accuracy:** `{info.get('test_accuracy','—')}`")
        st.markdown(f"**Classes:** {', '.join(info.get('classes', []))}")
    except Exception:
        pass

    st.markdown(f"[📊 MLflow UI ↗]({MLFLOW_URL})")
    st.markdown("---")
    st.markdown("""
### 📖 Feature Ranges
| Feature | Range (cm) |
|---------|-----------|
| Sepal Length | 4.3 – 7.9 |
| Sepal Width  | 2.0 – 4.4 |
| Petal Length | 1.0 – 6.9 |
| Petal Width  | 0.1 – 2.5 |
    """)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📂 Batch Predict", "ℹ️ About"])

# ── Tab 1: Single ─────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Single Sample Prediction")
    col_l, col_r = st.columns(2, gap="large")

    with col_l:
        st.markdown("#### Iris Measurements")
        sepal_length = st.slider("Sepal Length (cm)", 4.3, 7.9, 5.8, 0.1)
        sepal_width  = st.slider("Sepal Width (cm)",  2.0, 4.4, 3.0, 0.1)
        petal_length = st.slider("Petal Length (cm)", 1.0, 6.9, 3.8, 0.1)
        petal_width  = st.slider("Petal Width (cm)",  0.1, 2.5, 1.2, 0.1)
        go_btn = st.button("🔮 Predict Species", use_container_width=True, type="primary")

    with col_r:
        if go_btn:
            payload = {
                "sepal_length": sepal_length, "sepal_width":  sepal_width,
                "petal_length": petal_length, "petal_width":  petal_width,
            }
            try:
                resp = requests.post(f"{API_URL}/predict", json=payload, timeout=5)
                resp.raise_for_status()
                data = resp.json()

                label = data["predicted_class_name"].capitalize()
                cid   = data["predicted_class_id"]
                probs = data["probabilities"]   # {"setosa": 0.9, ...}

                st.markdown(f"""
                <div class="result-card">
                    <h2>🌸 {label}</h2>
                    <p>Class ID: {cid}</p>
                </div>""", unsafe_allow_html=True)

                prob_vals = [probs.get(c, 0.0) for c in CLASS_NAMES]
                fig = go.Figure(go.Bar(
                    x=[c.capitalize() for c in CLASS_NAMES],
                    y=prob_vals,
                    marker_color=CLASS_COLORS,
                    text=[f"{p:.1%}" for p in prob_vals],
                    textposition="outside",
                ))
                fig.update_layout(
                    plot_bgcolor="#1a1a2e", paper_bgcolor="#0f3460",
                    font_color="white",
                    yaxis=dict(range=[0, 1.2], gridcolor="#2a2a4a", tickformat=".0%"),
                    xaxis=dict(gridcolor="#2a2a4a"),
                    showlegend=False,
                    margin=dict(t=10, b=10, l=10, r=10), height=260,
                )
                st.plotly_chart(fig, use_container_width=True)

            except requests.exceptions.ConnectionError:
                st.error("❌ API unreachable. Make sure docker-compose is running.")
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.info("👈 Set measurements and click **Predict Species**")

# ── Tab 2: Batch ──────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Batch Prediction via CSV")
    st.caption("Columns required: `sepal_length, sepal_width, petal_length, petal_width`")

    sample = pd.DataFrame([
        [5.1, 3.5, 1.4, 0.2], [6.7, 3.0, 5.2, 2.3],
        [5.9, 3.0, 4.2, 1.5], [4.6, 3.1, 1.5, 0.2],
        [7.7, 2.6, 6.9, 2.3],
    ], columns=["sepal_length", "sepal_width", "petal_length", "petal_width"])

    st.download_button("⬇️ Sample CSV", sample.to_csv(index=False), "sample.csv", "text/csv")

    uploaded = st.file_uploader("Upload CSV", type="csv")
    if uploaded:
        df = pd.read_csv(uploaded)
        required = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            results, progress = [], st.progress(0, text="Predicting…")
            for i, (_, row) in enumerate(df[required].iterrows()):
                try:
                    r = requests.post(f"{API_URL}/predict", json={
                        "sepal_length": float(row["sepal_length"]),
                        "sepal_width":  float(row["sepal_width"]),
                        "petal_length": float(row["petal_length"]),
                        "petal_width":  float(row["petal_width"]),
                    }, timeout=5).json()
                    results.append({**row, "predicted_class": r["predicted_class_name"],
                                    "class_id": r["predicted_class_id"],
                                    "confidence": round(max(r["probabilities"].values()), 4)})
                except Exception:
                    results.append({**row, "predicted_class": "error", "class_id": -1, "confidence": 0.0})
                progress.progress((i + 1) / len(df), text=f"Row {i+1}/{len(df)}")

            rdf = pd.DataFrame(results)
            st.success(f"✅ {len(rdf)} predictions done")
            st.dataframe(rdf, use_container_width=True)

            c1, c2 = st.columns(2)
            with c1:
                dist = rdf["predicted_class"].value_counts()
                f1 = px.pie(values=dist.values, names=dist.index,
                            color_discrete_sequence=CLASS_COLORS, title="Distribution")
                f1.update_layout(paper_bgcolor="#0f3460", font_color="white")
                st.plotly_chart(f1, use_container_width=True)
            with c2:
                f2 = px.histogram(rdf, x="confidence", nbins=20, title="Confidence",
                                  color_discrete_sequence=["#e94560"])
                f2.update_layout(paper_bgcolor="#0f3460", plot_bgcolor="#1a1a2e", font_color="white")
                st.plotly_chart(f2, use_container_width=True)

            st.download_button("⬇️ Download Results", rdf.to_csv(index=False), "results.csv", "text/csv")

# ── Tab 3: About ──────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Architecture — SIS-3")
    for icon, title, desc in [
        ("🌐", "Streamlit",          "Этот UI. Вызывает /predict и /model-info из PT6 FastAPI."),
        ("⚡", "FastAPI  (PT6)",     "app.py из PT6 — без изменений. Загружает model.joblib + StandardScaler."),
        ("📊", "MLflow",             "Tracking + Registry. Experiment: iris_randomforest. Model: IrisClassifier."),
    ]:
        st.markdown(f"""<div class="info-box" style="margin-bottom:0.8rem">
            <span class="val">{icon}</span>
            <strong style="color:white;display:block">{title}</strong>
            <span style="font-size:0.85rem">{desc}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
**Что логируется в MLflow:**

- **Params:** `n_estimators=100`, `random_state=42`, `test_size=0.2`
- **Metrics:** `accuracy`, `f1_macro`, `precision`, `recall`
- **Artifacts:** `model.joblib`, `classification_report.txt`
- **Model Registry:** `IrisClassifier` (версия автоинкрементируется)

**Запуск:** `docker-compose up --build`
    """)
