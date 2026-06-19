import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os

st.set_page_config(page_title="MLP — BMW", page_icon="📊", layout="wide")

st.markdown("""
<style>
    .page-title { font-size:2rem; font-weight:800; color:#1F4E79; }
    .result-box {
        background: linear-gradient(135deg, #e8f4e8, #d4edda);
        border-left: 5px solid #28a745;
        border-radius: 10px;
        padding: 1.2rem;
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Modèles MLP ────────────────────────────────────────────────────
class MLPCustom(nn.Module):
    def __init__(self, n_in, hidden, n_classes, dropout=0.3):
        super().__init__()
        layers, prev = [], n_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h),
                       nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer  = nn.Linear(prev, n_classes)
    def forward(self, x):
        return self.output_layer(self.hidden_layers(x))

class MLPRegression(nn.Module):
    def __init__(self, n_in, dropout=0.3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_in, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128),  nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64),   nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 32),    nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.network(x).squeeze(1)

# ── En-tête ────────────────────────────────────────────────────────
st.markdown('<div class="page-title">📊 Partie I — MLP sur bmw.csv</div>',
            unsafe_allow_html=True)
st.markdown("**Tâche :** Prédiction du prix ou classification du type de carburant")
st.divider()

# ── Chargement dataset ─────────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('bmw.csv')
        for col in df.select_dtypes('object').columns:
            df[col] = df[col].str.strip()
        df.dropna(inplace=True)
        return df
    except:
        return None

df = load_data()

# ── Sidebar ────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Configuration")
task = st.sidebar.radio("Tâche :", ["🔢 Prédiction du prix", "⛽ Classification carburant"])

# ── Formulaire ────────────────────────────────────────────────────
st.markdown("### 🚗 Caractéristiques du véhicule")
if df is not None:
    models_list = sorted(df['model'].unique())
    trans_list  = sorted(df['transmission'].unique())
    year_min, year_max = int(df['year'].min()), int(df['year'].max())
else:
    models_list = ['1 Series','3 Series','5 Series','X1','X3','X5','M3','M5']
    trans_list  = ['Automatic','Manual','Semi-Auto']
    year_min, year_max = 2000, 2023

col1, col2, col3 = st.columns(3)
with col1:
    model_sel  = st.selectbox("Modèle BMW", models_list)
    year_sel   = st.slider("Année", year_min, year_max, 2018)
    trans_sel  = st.selectbox("Transmission", trans_list)
with col2:
    mileage_sel = st.number_input("Kilométrage (miles)", 0, 300000, 30000, 1000)
    tax_sel     = st.number_input("Taxe (£)", 0, 500, 145, 5)
with col3:
    mpg_sel    = st.number_input("MPG", 10.0, 200.0, 55.0, 0.5)
    engine_sel = st.selectbox("Taille moteur (L)", [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0])
    if '⛽' in task:
        price_sel = st.number_input("Prix (£)", 1000, 100000, 20000, 500)

# ── Encodage ──────────────────────────────────────────────────────
from sklearn.preprocessing import LabelEncoder, StandardScaler

def get_encoders(df):
    le_m = LabelEncoder().fit(df['model'])
    le_t = LabelEncoder().fit(df['transmission'])
    return le_m, le_t

# ── Prédiction ────────────────────────────────────────────────────
predict_btn = st.button("🔍 Lancer la prédiction", type="primary",
                        use_container_width=True)

if predict_btn:
    device = torch.device('cpu')

    if '🔢' in task:
        # ── Prix ──────────────────────────────────────────────────
        model_path = 'models/best_mlp_regression.pth'
        if not os.path.exists(model_path):
            st.error(f"❌ Modèle non trouvé : `{model_path}`")
        elif df is None:
            st.error("❌ bmw.csv non trouvé.")
        else:
            data = df.copy()
            le_m, le_t = get_encoders(data)
            data['model_enc']        = le_m.transform(data['model'])
            data['transmission_enc'] = le_t.transform(data['transmission'])
            FEAT = ['model_enc','year','transmission_enc','mileage','tax','mpg','engineSize']
            scaler_X = StandardScaler().fit(data[FEAT].values)
            scaler_y = StandardScaler().fit(data['price'].values.reshape(-1,1))

            m_enc = le_m.transform([model_sel])[0] if model_sel in le_m.classes_ else 0
            t_enc = le_t.transform([trans_sel])[0]  if trans_sel in le_t.classes_ else 0
            X_raw    = np.array([[m_enc, year_sel, t_enc, mileage_sel,
                                   tax_sel, mpg_sel, engine_sel]], dtype=np.float32)
            X_scaled = scaler_X.transform(X_raw).astype(np.float32)

            mlp = MLPRegression(n_in=7).to(device)
            mlp.load_state_dict(torch.load(model_path, map_location=device))
            mlp.eval()
            with torch.no_grad():
                pred_sc = mlp(torch.tensor(X_scaled)).numpy().reshape(-1,1)
            prix = scaler_y.inverse_transform(pred_sc)[0][0]

            st.markdown(f"""
            <div class="result-box">
                <b>💰 Prix estimé : £{prix:,.0f}</b><br>
                <small>BMW {model_sel} {year_sel} | {mileage_sel:,} miles | {engine_sel}L</small>
            </div>""", unsafe_allow_html=True)

            similar = df[(df['model']==model_sel) & (df['year']==year_sel)]
            if len(similar) > 0:
                st.info(f"Prix moyen dataset pour BMW {model_sel} {year_sel} : "
                        f"£{similar['price'].mean():,.0f}")

    else:
        # ── Carburant ──────────────────────────────────────────────
        model_path = 'models/best_mlp_custom.pth'
        if not os.path.exists(model_path):
            st.error(f"❌ Modèle non trouvé : `{model_path}`")
        elif df is None:
            st.error("❌ bmw.csv non trouvé.")
        else:
            data = df.copy()
            for col in data.select_dtypes('object').columns:
                data[col] = data[col].str.strip()
            data.dropna(inplace=True)

            le_m, le_t = get_encoders(data)
            le_f = LabelEncoder().fit(data['fuelType'])

            data['model_enc']        = le_m.transform(data['model'])
            data['transmission_enc'] = le_t.transform(data['transmission'])
            data['fuelType_enc']     = le_f.transform(data['fuelType'])

            # ✅ Supprime Electric exactement comme dans le notebook
            counts     = data['fuelType_enc'].value_counts()
            valid_idx  = counts[counts >= 10].index
            data_filt  = data[data['fuelType_enc'].isin(valid_idx)].copy()
            le_f2      = LabelEncoder().fit(data_filt['fuelType'])
            CLASS_NAMES = list(le_f2.classes_)

            FEAT = ['model_enc','year','price','transmission_enc',
                    'mileage','tax','mpg','engineSize']
            scaler = StandardScaler().fit(data_filt[FEAT].values)

            # ✅ N_CLASSES depuis le checkpoint (protection anti-mismatch)
            checkpoint = torch.load(model_path, map_location=device)
            last_key   = [k for k in checkpoint.keys()
                          if 'output_layer.weight' in k][0]
            N_CLASSES  = checkpoint[last_key].shape[0]
            CLASS_NAMES = CLASS_NAMES[:N_CLASSES]

            m_enc = le_m.transform([model_sel])[0] if model_sel in le_m.classes_ else 0
            t_enc = le_t.transform([trans_sel])[0]  if trans_sel in le_t.classes_ else 0
            try:
                p_sel = price_sel
            except NameError:
                p_sel = 20000

            X_raw = np.array([[m_enc, year_sel, p_sel, t_enc,
                                mileage_sel, tax_sel, mpg_sel, engine_sel]],
                             dtype=np.float32)
            X_sc  = scaler.transform(X_raw).astype(np.float32)

            mlp = MLPCustom(n_in=8, hidden=[128,64,32], n_classes=N_CLASSES)
            mlp.load_state_dict(checkpoint)
            mlp.eval()
            with torch.no_grad():
                logits = mlp(torch.tensor(X_sc))
                probs  = torch.softmax(logits, dim=1).numpy()[0]
                pred   = np.argmax(probs)

            fuel_icons = {'Diesel':'🛢️','Petrol':'⛽','Hybrid':'🔋',
                          'Electric':'⚡','Other':'🔧'}
            icon = fuel_icons.get(CLASS_NAMES[pred], '🚗')

            st.markdown(f"""
            <div class="result-box">
                <b>{icon} Carburant prédit : {CLASS_NAMES[pred]}</b>
                &nbsp;&nbsp; Confiance : {probs[pred]*100:.1f}%
            </div>""", unsafe_allow_html=True)

            prob_df = pd.DataFrame({'Classe': CLASS_NAMES, 'Probabilité': probs})
            prob_df = prob_df.sort_values('Probabilité', ascending=False)
            st.bar_chart(prob_df.set_index('Classe'))

# ── Exploration données ────────────────────────────────────────────
st.divider()
with st.expander("📂 Explorer le dataset bmw.csv"):
    if df is not None:
        st.write(f"**{len(df)} lignes | {len(df.columns)} colonnes**")
        st.dataframe(df.head(20), use_container_width=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Distribution fuelType**")
            st.bar_chart(df['fuelType'].value_counts())
        with c2:
            st.markdown("**Distribution modèles (top 10)**")
            st.bar_chart(df['model'].value_counts().head(10))
    else:
        st.error("bmw.csv non trouvé dans le dossier racine.")