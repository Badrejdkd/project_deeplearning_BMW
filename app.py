import streamlit as st
import os

st.set_page_config(
    page_title="BMW Deep Learning",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-title {
        font-size: 2.8rem; font-weight: 800; color: #1F4E79; text-align: center;
    }
    .sub-title {
        font-size: 1.05rem; color: #666; text-align: center; margin-bottom: 2rem;
    }
    .card {
        background: linear-gradient(135deg, #f8f9fa, #e9f0fb);
        border-left: 5px solid #2E75B6; border-radius: 12px;
        padding: 1.5rem; margin: 0.5rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    }
    .card-title { font-size: 1.2rem; font-weight: 700; color: #1F4E79; }
    .card-body  { color: #444; font-size: 0.93rem; margin-top: 0.4rem; }
    .badge {
        display: inline-block; background: #2E75B6; color: white;
        border-radius: 20px; padding: 2px 10px; font-size: 0.78rem;
        font-weight: 600; margin-right: 4px; margin-bottom: 4px;
    }
    .stat-box {
        background: white; border-radius: 10px; padding: 1.2rem;
        text-align: center; box-shadow: 0 2px 6px rgba(0,0,0,0.08);
        border-top: 3px solid #2E75B6;
    }
    .stat-number { font-size: 2rem; font-weight: 800; color: #2E75B6; }
    .stat-label  { font-size: 0.82rem; color: #666; margin-top: 2px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🚗 BMW Deep Learning Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Projet de Fin de Module — Deep Learning | EMSI Casablanca 2025-2026</div>', unsafe_allow_html=True)
st.divider()

c1,c2,c3,c4 = st.columns(4)
for col, num, label in zip(
    [c1,c2,c3,c4],
    ["3","31","10K+","PyTorch"],
    ["Architectures DL","Classes BMW (CNN)","Images entraînées","Framework"]
):
    col.markdown(f"""
    <div class="stat-box">
        <div class="stat-number">{num}</div>
        <div class="stat-label">{label}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

c1,c2,c3 = st.columns(3)
with c1:
    st.markdown("""<div class="card">
        <div class="card-title">📊 Partie I — MLP</div>
        <div class="card-body">
            <span class="badge">Tabulaire</span><span class="badge">bmw.csv</span><br><br>
            Prédiction du <b>prix</b> ou classification du <b>carburant</b>
            à partir des caractéristiques du véhicule.<br><br>
            Modèles : <b>MLP Sequential</b> & <b>MLP Custom</b>
        </div></div>""", unsafe_allow_html=True)
with c2:
    st.markdown("""<div class="card">
        <div class="card-title">🖼️ Partie II — CNN</div>
        <div class="card-body">
            <span class="badge">Images</span><span class="badge">31 classes</span><br><br>
            Identification du <b>modèle BMW</b> à partir d'une photo.
            Architecture <b>FastCNN</b> (Depthwise Separable).<br><br>
            Dataset : <b>10 707 images</b>
        </div></div>""", unsafe_allow_html=True)
with c3:
    st.markdown("""<div class="card">
        <div class="card-title">📝 Partie III — RNN</div>
        <div class="card-body">
            <span class="badge">Texte</span><span class="badge">Sentiment</span><br><br>
            Analyse de <b>sentiment</b> des reviews clients BMW.
            Comparaison <b>RNN / LSTM / GRU</b>.<br><br>
            Dataset : <b>Reviews BMW scrapées</b>
        </div></div>""", unsafe_allow_html=True)

st.divider()
st.markdown("### 🔍 Statut des fichiers")

models_required = {
    "models/best_mlp_custom.pth":         "📊 MLP Classification",
    "models/best_mlp_regression.pth":     "📊 MLP Régression",
    "models/best_fast_cnn_bmw_final.pth": "🖼️ CNN FastCNN",
    "models/best_rnn_simple.pth":         "📝 RNN Simple",
    "models/best_lstm.pth":               "📝 LSTM",
    "models/best_gru.pth":                "📝 GRU",
}
data_required = {
    "bmw.csv":                        "📊 bmw.csv",
    "Scrapped_Car_Reviews_BMW.csv":   "📝 Reviews BMW CSV",
    "bmw_image_dataset_v2/bmw_cars":  "🖼️ Images BMW",
}

col1, col2 = st.columns(2)
all_ok = True
with col1:
    st.markdown("**Modèles entraînés :**")
    for path, label in models_required.items():
        if os.path.exists(path):
            size = os.path.getsize(path)/1024
            st.success(f"✅ {label} ({size:.0f} KB)")
        else:
            st.error(f"❌ {label} — manquant")
            all_ok = False
with col2:
    st.markdown("**Fichiers de données :**")
    for path, label in data_required.items():
        if os.path.exists(path):
            st.success(f"✅ {label}")
        else:
            st.warning(f"⚠️ {label} — manquant")

if not all_ok:
    st.divider()
    st.warning("⚠️ Des modèles sont manquants. Lance le script d'entraînement :")
    st.code("python train_all_models.py", language="bash")
else:
    st.success("✅ Tout est prêt ! Utilise le menu latéral pour naviguer.")

st.divider()
st.markdown("### 🧭 Navigation")
st.info("👈 Utilise le **menu latéral gauche** pour naviguer entre les pages.")
col1, col2 = st.columns(2)
with col1:
    st.markdown("**Pages :**\n- 🏠 Accueil\n- 📊 MLP (prix / carburant)\n- 🖼️ CNN (image BMW)\n- 📝 RNN (sentiment reviews)")
with col2:
    st.code("""bmw_deeplearning_app/
├── app.py
├── train_all_models.py   ← lance en premier
├── bmw.csv
├── Scrapped_Car_Reviews_BMW.csv
├── bmw_image_dataset_v2/
└── models/  ← générés automatiquement""")

st.divider()
st.caption("EMSI Casablanca — Département Informatique — Deep Learning 2025-2026")
