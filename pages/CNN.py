import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import os
from PIL import Image
import pandas as pd

st.set_page_config(page_title="CNN — BMW", page_icon="🖼️", layout="wide")

st.markdown("""
<style>
    .page-title { font-size:2rem; font-weight:800; color:#1F4E79; }
    .pred-box {
        background: linear-gradient(135deg, #e8f4fd, #cce5ff);
        border-left: 5px solid #2E75B6;
        border-radius: 12px;
        padding: 1.5rem;
        font-size: 1.1rem;
        margin: 1rem 0;
    }
    .top5-item {
        background: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        margin: 0.3rem 0;
        display: flex;
        justify-content: space-between;
    }
</style>
""", unsafe_allow_html=True)

# ── Classes BMW (basé sur VOTRE structure réelle d'entraînement) ───
# Selon votre image : i3, i8, m, x, z4
CLASS_NAMES = ['i3', 'i8', 'm', 'x', 'z4']  # 5 classes seulement !
N_CLASSES = len(CLASS_NAMES)
IMG_SIZE = 128

# Mapping des catégories pour affichage plus clair
CATEGORY_INFO = {
    'i3': {'nom': 'i3', 'type': '🚗 Électrique', 'modeles': 'i3'},
    'i8': {'nom': 'i8', 'type': '🚀 Électrique', 'modeles': 'i8'},
    'm': {'nom': 'Série M', 'type': '🏎️ Sport', 'modeles': 'M2, M3, M4, M5, M6, M8'},
    'x': {'nom': 'Série X', 'type': '🚙 SUV', 'modeles': 'X1, X2, X3, X4, X5, X6, X7'},
    'z4': {'nom': 'Z4', 'type': '🌊 Roadster', 'modeles': 'Z4'}
}

# ── Modèle FastCNN ─────────────────────────────────────────────────
class FastCNN(nn.Module):
    def __init__(self, n_classes, dropout=0.6):
        super().__init__()
        def dw_block(in_ch, out_ch, stride=1):
            return nn.Sequential(
                nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1,
                          groups=in_ch, bias=False),
                nn.BatchNorm2d(in_ch), nn.ReLU6(),
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch), nn.ReLU6()
            )
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU6(),
            dw_block(32, 64, stride=1),
            dw_block(64, 128, stride=2),
            dw_block(128, 128, stride=1),
            dw_block(128, 256, stride=2),
            dw_block(256, 256, stride=1),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Dropout(dropout*0.5),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

def pil_to_tensor(img):
    """Convert PIL image to normalized tensor"""
    arr = np.array(img, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    arr = (arr - mean) / std
    return torch.tensor(arr).permute(2, 0, 1).float()

@st.cache_resource
def load_cnn_model():
    """Load the trained model"""
    model_path = 'models/best_fast_cnn_bmw_final.pth'
    
    if not os.path.exists(model_path):
        return None
    
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Vérifier le nombre de classes dans le modèle
        if 'classifier.5.weight' in checkpoint:
            n_classes = checkpoint['classifier.5.weight'].shape[0]
        else:
            last_key = [k for k in checkpoint.keys() if 'weight' in k][-1]
            n_classes = checkpoint[last_key].shape[0]
        
        st.info(f"📊 Modèle chargé : {n_classes} classes ({', '.join(CLASS_NAMES[:n_classes])})")
        
        model = FastCNN(n_classes, dropout=0.6)
        model.load_state_dict(checkpoint)
        model.eval()
        
        return model, n_classes
        
    except Exception as e:
        st.error(f"❌ Erreur : {str(e)}")
        return None

# ── En-tête ────────────────────────────────────────────────────────
st.markdown('<div class="page-title">🖼️ Partie II — CNN Classification BMW</div>', unsafe_allow_html=True)
st.markdown("**Tâche :** Identifier la catégorie BMW à partir d'une image (5 catégories)")
st.divider()

# Charger le modèle
model_data = load_cnn_model()

if model_data is None:
    st.error("⚠️ Modèle CNN non trouvé.")
    st.info("Chemin attendu : `models/best_fast_cnn_bmw_final.pth`")
    st.info("Assurez-vous que le modèle a été entraîné sur les dossiers : i3, i8, m, x, z4")
    st.stop()

model, n_classes = model_data

# Ajuster les classes si nécessaire
display_classes = CLASS_NAMES[:n_classes]
st.success(f"✅ Modèle FastCNN chargé avec succès ! Reconnaissance sur {n_classes} catégories BMW")

# ── Upload image ───────────────────────────────────────────────────
st.markdown("### 📸 Charger une image BMW")
uploaded = st.file_uploader(
    "Glisse une image BMW ici",
    type=['jpg', 'jpeg', 'png', 'webp'],
    help="Formats acceptés : JPG, JPEG, PNG, WEBP"
)

col_img, col_result = st.columns([1, 1])

if uploaded is not None:
    img = Image.open(uploaded).convert('RGB')

    with col_img:
        st.markdown("**Image chargée :**")
        st.image(img, use_column_width=True)
        st.caption(f"Taille originale : {img.size[0]}×{img.size[1]} px")

    with col_result:
        # Prétraitement
        img_resized = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        tensor = pil_to_tensor(img_resized).unsqueeze(0)

        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1).numpy()[0]
            pred = np.argmax(probs)

        top5_idx = np.argsort(probs)[::-1][:min(5, n_classes)]
        top5_probs = probs[top5_idx]
        top5_names = [display_classes[i] for i in top5_idx]

        # Résultat principal
        confidence = probs[pred] * 100
        predicted_class = display_classes[pred]
        info = CATEGORY_INFO.get(predicted_class, {'nom': predicted_class, 'type': 'BMW', 'modeles': ''})
        
        color = "#28a745" if confidence > 70 else "#ffc107" if confidence > 40 else "#dc3545"

        st.markdown(f"""
        <div class="pred-box">
            <b style="font-size:1.4rem;">{info['type']} BMW {info['nom']}</b><br>
            <span style="font-size:0.9rem; color:#666;">Modèles : {info['modeles']}</span><br>
            <span style="color:{color}; font-weight:600;">Confiance : {confidence:.1f}%</span>
        </div>
        """, unsafe_allow_html=True)

        # Top 5
        st.markdown("**Top prédictions :**")
        for i, (name, prob) in enumerate(zip(top5_names, top5_probs)):
            info_top = CATEGORY_INFO.get(name, {'type': 'BMW', 'nom': name})
            bar_width = int(prob * 100)
            color_bar = "#2E75B6" if i == 0 else "#ccc"
            st.markdown(f"""
            <div style="margin:4px 0;">
                <span style="font-weight:{'700' if i==0 else '400'}; width:120px; display:inline-block;">
                    {info_top['type']} {name.upper()}
                </span>
                <span style="
                    display:inline-block;
                    width:{max(bar_width*2, 5)}px;
                    height:16px;
                    background:{color_bar};
                    border-radius:3px;
                    vertical-align:middle;
                    margin: 0 8px;
                "></span>
                <span>{prob*100:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)

    # ── Graphique probabilités ─────────────────────────────────────
    st.divider()
    st.markdown("### 📊 Distribution des probabilités")
    
    top10_idx = np.argsort(probs)[::-1][:min(10, n_classes)]
    top10_probs = probs[top10_idx]
    top10_names = [f"{CATEGORY_INFO.get(display_classes[i], {'type': ''})['type']} {display_classes[i].upper()}" 
                   for i in top10_idx]

    prob_df = pd.DataFrame({'Catégorie': top10_names, 'Probabilité': top10_probs})
    st.bar_chart(prob_df.set_index('Catégorie'))

else:
    with col_img:
        st.info("👆 Charge une image BMW pour identifier sa catégorie")
        st.markdown("""
        **Catégories reconnues :**
        - 🚗 **i3** - Citadine électrique
        - 🚀 **i8** - Sportive hybride
        - 🏎️ **Série M** - M2, M3, M4, M5, M6, M8
        - 🚙 **Série X** - X1, X2, X3, X4, X5, X6, X7
        - 🌊 **Z4** - Roadster
        """)

    with col_result:
        st.markdown("### ℹ️ À propos du modèle")
        st.markdown(f"""
        | Paramètre | Valeur |
        |-----------|--------|
        | Architecture | FastCNN (Depthwise Separable) |
        | Classes | {n_classes} catégories BMW |
        | Types | i3, i8, Série M, Série X, Z4 |
        | Taille entrée | {IMG_SIZE}×{IMG_SIZE} px |
        """)

# ── Section infos ──────────────────────────────────────────────────
st.divider()
with st.expander("📚 Architecture FastCNN"):
    st.markdown("""
    Le **FastCNN** utilise des convolutions séparables en profondeur (Depthwise Separable) :
    
    - **Avantages :** 6-8x moins de calculs qu'un CNN classique
    - **Entrée :** 128×128 pixels
    - **Sortie :** 5 catégories BMW
    """)

with st.expander("📋 Détail des catégories"):
    categories_df = pd.DataFrame([
        {'Catégorie': 'i3', 'Type': 'Électrique', 'Modèles': 'i3'},
        {'Catégorie': 'i8', 'Type': 'Électrique/Sport', 'Modèles': 'i8'},
        {'Catégorie': 'Série M (m)', 'Type': 'Sport', 'Modèles': 'M2, M3, M4, M5, M6, M8'},
        {'Catégorie': 'Série X (x)', 'Type': 'SUV', 'Modèles': 'X1, X2, X3, X4, X5, X6, X7'},
        {'Catégorie': 'Z4 (z4)', 'Type': 'Roadster', 'Modèles': 'Z4'}
    ])
    st.dataframe(categories_df, use_container_width=True)