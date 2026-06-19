import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import re, os, math
from collections import Counter
import pickle

st.set_page_config(page_title="RNN — BMW", page_icon="📝", layout="wide")

st.markdown("""
<style>
    .page-title { font-size:2rem; font-weight:800; color:#1F4E79; }
    .pos-box {
        background: linear-gradient(135deg, #d4edda, #c3e6cb);
        border-left: 5px solid #28a745;
        border-radius: 12px;
        padding: 1.5rem;
        font-size: 1.1rem;
        margin: 1rem 0;
    }
    .neg-box {
        background: linear-gradient(135deg, #f8d7da, #f5c6cb);
        border-left: 5px solid #dc3545;
        border-radius: 12px;
        padding: 1.5rem;
        font-size: 1.1rem;
        margin: 1rem 0;
    }
    .neutral-box {
        background: linear-gradient(135deg, #fff3cd, #ffeeba);
        border-left: 5px solid #ffc107;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ── Constantes ─────────────────────────────────────────────────────
MAX_LEN  = 100
PAD_IDX  = 0
UNK_IDX  = 1
MIN_FREQ = 1  # Changé de 2 à 1 pour capturer tous les mots

# ── Modèles ────────────────────────────────────────────────────────
class RNNSimple(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_classes, n_layers=1, dropout=0.3, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.rnn       = nn.RNN(embed_dim, hidden_dim, num_layers=n_layers,
                                batch_first=True, nonlinearity='tanh',
                                dropout=dropout if n_layers>1 else 0)
        self.dropout   = nn.Dropout(dropout)
        self.fc        = nn.Linear(hidden_dim, n_classes)
    def forward(self, x):
        emb    = self.dropout(self.embedding(x))
        _, h_n = self.rnn(emb)
        return self.fc(self.dropout(h_n[-1]))


class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_classes, n_layers=1, dropout=0.3, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm      = nn.LSTM(embed_dim, hidden_dim, num_layers=n_layers,
                                 batch_first=True, dropout=dropout if n_layers>1 else 0)
        self.dropout   = nn.Dropout(dropout)
        self.fc        = nn.Linear(hidden_dim, n_classes)
    def forward(self, x):
        emb         = self.dropout(self.embedding(x))
        _, (h_n, _) = self.lstm(emb)
        return self.fc(self.dropout(h_n[-1]))


class GRUModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_classes, n_layers=1, dropout=0.3, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.gru       = nn.GRU(embed_dim, hidden_dim, num_layers=n_layers,
                                batch_first=True, dropout=dropout if n_layers>1 else 0)
        self.dropout   = nn.Dropout(dropout)
        self.fc        = nn.Linear(hidden_dim, n_classes)
    def forward(self, x):
        emb    = self.dropout(self.embedding(x))
        _, h_n = self.gru(emb)
        return self.fc(self.dropout(h_n[-1]))


@st.cache_data
def load_vocab():
    import pickle
    vocab_path = 'models/rnn_vocab.pkl'

    # ✅ Charge le vocabulaire exact utilisé à l'entraînement
    if os.path.exists(vocab_path):
        with open(vocab_path, 'rb') as f:
            vocab_data = pickle.load(f)
        word2idx = vocab_data['word2idx']
        vocab_size = vocab_data['vocab_size']
        print(f"✅ Vocab chargé depuis fichier : {vocab_size} tokens")
        return word2idx, vocab_size

    # Fallback : reconstruction depuis CSV (moins fiable)
    csv_path = 'Scrapped_Car_Reviews_BMW_final.csv'
    if not os.path.exists(csv_path):
        return None, None
    try:
        try:
            df = pd.read_csv(csv_path, encoding='latin-1', on_bad_lines='skip')
        except Exception:
            df = pd.read_csv(csv_path, encoding='utf-8', on_bad_lines='skip',
                             engine='python', quoting=3)
        df = df[['Review','Rating']].dropna()
        df['Review'] = df['Review'].astype(str)

        def tokenize(text):
            text = str(text).lower()
            text = re.sub(r'[^a-z0-9\s]', ' ', text)
            return text.split()

        all_tokens = []
        for review in df['Review']:
            all_tokens.extend(tokenize(review))

        counter  = Counter(all_tokens)
        vocab    = ['<PAD>','<UNK>','<BOS>','<EOS>'] + \
                   [w for w,c in counter.items() if c >= 2]
        word2idx = {w:i for i,w in enumerate(vocab)}
        return word2idx, len(vocab)
    except Exception:
        return None, None


@st.cache_resource
def load_rnn_models():
    """Charge les modèles RNN en utilisant la taille du vocabulaire des checkpoints"""
    models = {}
    configs = {
        'RNN':  ('models/best_rnn_simple.pth', RNNSimple),
        'LSTM': ('models/best_lstm.pth',        LSTMModel),
        'GRU':  ('models/best_gru.pth',         GRUModel),
    }
    
    for name, (path, ModelClass) in configs.items():
        if os.path.exists(path):
            try:
                # Charger le checkpoint
                checkpoint = torch.load(path, map_location='cpu')
                
                # Extraire les paramètres du checkpoint
                checkpoint_vocab_size = checkpoint['embedding.weight'].shape[0]
                checkpoint_hidden_dim = checkpoint['fc.weight'].shape[1]
                checkpoint_n_classes = checkpoint['fc.weight'].shape[0]
                
                # Créer le modèle avec les bons paramètres
                m = ModelClass(
                    vocab_size=checkpoint_vocab_size,
                    embed_dim=64,
                    hidden_dim=checkpoint_hidden_dim,
                    n_classes=checkpoint_n_classes,
                    n_layers=1,
                    dropout=0.3,
                    pad_idx=0
                )
                m.load_state_dict(checkpoint)
                m.eval()
                models[name] = m
                st.info(f"✅ {name} chargé (vocab: {checkpoint_vocab_size:,} mots)")
                
            except Exception as e:
                st.warning(f"⚠️ Erreur chargement {name}: {str(e)[:100]}")
    
    return models


def tokenize(text):
    """Tokenize le texte"""
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return text.split()


def encode_text(text, word2idx, max_len=MAX_LEN):
    """Convertit le texte en tenseur d'indices"""
    tokens = tokenize(text)[:max_len]
    ids = [word2idx.get(t, UNK_IDX) for t in tokens]
    padded = torch.zeros(1, max_len, dtype=torch.long)
    if ids:
        padded[0, :len(ids)] = torch.tensor(ids)
    return padded


def predict_with_model(model, text, word2idx):
    """Prédit le sentiment avec un modèle"""
    model.eval()
    src = encode_text(text, word2idx)
    with torch.no_grad():
        logits = model(src)
        probs = torch.softmax(logits, dim=1).numpy()[0]
        pred = np.argmax(probs)
    return pred, probs


# ── En-tête ────────────────────────────────────────────────────────
st.markdown('<div class="page-title">📝 Partie III — RNN / LSTM / GRU</div>', unsafe_allow_html=True)
st.markdown("**Tâche :** Analyse de sentiment des reviews BMW (Positif ✅ / Négatif ❌)")
st.divider()

# ── Chargement ─────────────────────────────────────────────────────
word2idx, vocab_size = load_vocab()

if word2idx is None:
    st.error("⚠️ Impossible de charger le vocabulaire.")
    st.info("Assurez-vous que le fichier `Scrapped_Car_Reviews_BMW_final.csv` existe dans le dossier racine.")
    st.stop()

models_loaded = load_rnn_models()

if not models_loaded:
    st.warning("⚠️ Aucun modèle RNN trouvé.")
    st.info("Chemins attendus :")
    st.code("""
    models/best_rnn_simple.pth
    models/best_lstm.pth
    models/best_gru.pth
    """)
    st.info("Entraînez d'abord les modèles RNN (Partie III Notebook).")
    st.stop()

loaded_names = list(models_loaded.keys())
st.success(f"✅ Modèles chargés : {', '.join(loaded_names)}")
st.caption(f"📚 Taille du vocabulaire du dataset actuel : {vocab_size:,} mots")

# ── Sidebar ────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Configuration")
model_choice = st.sidebar.radio("Modèle à utiliser :", loaded_names if loaded_names else ['LSTM'])
show_all = st.sidebar.checkbox("Comparer tous les modèles", value=True)

# ── Zone de saisie ─────────────────────────────────────────────────
st.markdown("### ✍️ Saisir une review BMW")

# Exemples prédéfinis
examples = {
    "Positif fort 🌟": "This BMW is absolutely amazing! Best car I have ever driven. The handling is superb, performance is incredible and the interior is luxurious.",
    "Négatif fort 😤": "Terrible car, always breaking down. Worst purchase I have ever made. Engine problems, electrical issues, dealer is useless. Never buying BMW again.",
    "Mitigé 😐": "The car is ok. Decent performance but too expensive for what it offers. Some quality issues with the interior plastics.",
    "Positif modéré 👍": "Great driving experience, love the sporty feel. A few minor issues but overall satisfied with my BMW purchase.",
}

col1, col2 = st.columns([2, 1])
with col1:
    review_input = st.text_area(
        "Review (en anglais) :",
        height=150,
        placeholder="Ex: This BMW is absolutely amazing, best car I have ever driven...",
        help="Saisie libre ou choisis un exemple à droite"
    )
with col2:
    st.markdown("**Exemples :**")
    for label, text in examples.items():
        if st.button(label, use_container_width=True):
            review_input = text
            st.rerun()

# ── Prédiction ─────────────────────────────────────────────────────
predict_btn = st.button("🔍 Analyser le sentiment", type="primary", use_container_width=True)

if predict_btn and review_input.strip():
    st.markdown("### 📊 Résultats")

    # Modèle principal
    chosen_model = models_loaded.get(model_choice)
    if chosen_model:
        pred, probs = predict_with_model(chosen_model, review_input, word2idx)
        confidence = probs[pred] * 100
        sentiment = "Positif ✅" if pred == 1 else "Négatif ❌"
        box_class = "pos-box" if pred == 1 else "neg-box"
        emoji = "😊" if pred == 1 else "😠"

        st.markdown(f"""
        <div class="{box_class}">
            <b style="font-size:1.5rem;">{emoji} {sentiment}</b><br>
            Confiance : <b>{confidence:.1f}%</b> &nbsp;|&nbsp;
            Modèle utilisé : <b>{model_choice}</b>
        </div>
        """, unsafe_allow_html=True)

        # Barres de probabilité
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Probabilité Négatif", f"{probs[0]*100:.1f}%")
            st.progress(float(probs[0]))
        with col2:
            st.metric("Probabilité Positif", f"{probs[1]*100:.1f}%")
            st.progress(float(probs[1]))

    # Comparaison tous modèles
    if show_all and len(models_loaded) > 1:
        st.divider()
        st.markdown("### 🔄 Comparaison RNN / LSTM / GRU")
        cols = st.columns(len(models_loaded))
        results = {}
        for i, (name, m) in enumerate(models_loaded.items()):
            pred_i, probs_i = predict_with_model(m, review_input, word2idx)
            results[name] = (pred_i, probs_i)
            with cols[i]:
                sent = "Positif ✅" if pred_i == 1 else "Négatif ❌"
                conf = probs_i[pred_i] * 100
                color = "#28a745" if pred_i == 1 else "#dc3545"
                st.markdown(f"""
                <div style="background:white; border-radius:10px; padding:1rem;
                            box-shadow:0 2px 6px rgba(0,0,0,0.1); text-align:center;
                            border-top: 4px solid {color};">
                    <b style="font-size:1.1rem;">{name}</b><br>
                    <span style="color:{color}; font-size:1.2rem; font-weight:700;">{sent}</span><br>
                    <small>Confiance : {conf:.1f}%</small>
                </div>
                """, unsafe_allow_html=True)

elif predict_btn and not review_input.strip():
    st.warning("⚠️ Saisis une review avant de lancer l'analyse.")

# ── Analyse du texte ───────────────────────────────────────────────
if review_input.strip():
    with st.expander("🔍 Analyse du texte saisi"):
        tokens = tokenize(review_input)
        known = [t for t in tokens if t in word2idx and word2idx[t] > 1]
        unk = [t for t in tokens if t not in word2idx or word2idx[t] == 1]

        col1, col2, col3 = st.columns(3)
        col1.metric("Tokens total", len(tokens))
        col2.metric("Tokens connus", len(known))
        col3.metric("Tokens inconnus (UNK)", len(unk))

        if unk:
            st.caption(f"Tokens inconnus : {', '.join(unk[:10])}" +
                      (" ..." if len(unk) > 10 else ""))

# ── Infos sur les modèles ──────────────────────────────────────────
st.divider()
with st.expander("ℹ️ Informations sur les modèles chargés"):
    for name, model in models_loaded.items():
        vocab_size_model = model.embedding.weight.shape[0]
        hidden_size = model.fc.weight.shape[1]
        n_classes = model.fc.weight.shape[0]
        st.markdown(f"""
        **{name}**
        - Vocabulaire : {vocab_size_model:,} mots
        - Dimension cachée : {hidden_size}
        - Classes : {n_classes} (0=Négatif, 1=Positif)
        """)

# ── Exemples du dataset ────────────────────────────────────────────
with st.expander("📂 Exemples du dataset BMW Reviews"):
    csv_path = 'Scrapped_Car_Reviews_BMW_final.csv'
    if os.path.exists(csv_path):
        try:
            try:
                df_show = pd.read_csv(csv_path, encoding='latin-1', on_bad_lines='skip')
            except Exception:
                df_show = pd.read_csv(csv_path, encoding='utf-8', on_bad_lines='skip',
                                      engine='python', quoting=3)
            
            if 'Review' in df_show.columns and 'Rating' in df_show.columns:
                df_show = df_show[['Review', 'Rating']].dropna()
                df_show['Rating'] = pd.to_numeric(df_show['Rating'], errors='coerce')
                df_show.dropna(inplace=True)
                df_show['Sentiment'] = df_show['Rating'].apply(
                    lambda x: '✅ Positif' if round(x) >= 4 else '❌ Négatif')
                st.write(f"**{len(df_show)} reviews disponibles**")
                st.dataframe(df_show[['Review', 'Rating', 'Sentiment']].head(20),
                             use_container_width=True)
                
                # Distribution des sentiments
                sentiment_counts = df_show['Sentiment'].value_counts()
                st.bar_chart(sentiment_counts)
            else:
                st.warning("Colonne 'Review' ou 'Rating' non trouvée")
        except Exception as e:
            st.error(f"Erreur lecture CSV : {e}")

# ── Infos architecture ─────────────────────────────────────────────
with st.expander("📚 Architectures RNN / LSTM / GRU"):
    st.markdown("""
    | Modèle | Portes | Avantage | Limite |
    |--------|--------|----------|--------|
    | **RNN** | Aucune | Simple, rapide | Gradient qui disparaît |
    | **LSTM** | 3 (forget, input, output) | Mémoire longue terme | Plus de paramètres |
    | **GRU** | 2 (reset, update) | Équilibre vitesse/précision | Légèrement moins expressif |

    **BPTT** : Backpropagation Through Time — le gradient est propagé à travers le temps  
    **Gradient Clipping** : `clip_grad_norm_(max=1.0)` évite l'explosion du gradient  
    **Perplexité** : PP = exp(Loss) — mesure la qualité du modèle
    """)