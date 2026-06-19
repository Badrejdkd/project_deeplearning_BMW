"""
train_all_models.py
====================
Script d'entraînement complet — génère tous les .pth nécessaires à l'application.

Usage :
    python train_all_models.py

Prérequis :
    - bmw.csv dans le même dossier
    - Scrapped_Car_Reviews_BMW.csv dans le même dossier
    - bmw_image_dataset_v2/bmw_cars/ dans le même dossier
    - pip install torch scikit-learn pandas pillow numpy
"""

import os, re, random, math, time, warnings
import numpy as np
import pandas as pd
from collections import Counter
from PIL import Image, ImageEnhance
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

os.makedirs('models', exist_ok=True)
SEED   = 42
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

print(f"\n{'='*60}")
print(f"  BMW Deep Learning — Entraînement de tous les modèles")
print(f"  Device : {device}")
print(f"{'='*60}\n")


# ══════════════════════════════════════════════════════════════════
# PARTIE I — MLP
# ══════════════════════════════════════════════════════════════════
print("📊 PARTIE I — MLP sur bmw.csv")
print("-"*50)

class MLPCustom(nn.Module):
    def __init__(self, n_in, hidden, n_classes, dropout=0.3):
        super().__init__()
        layers, prev = [], n_in
        for h in hidden:
            layers += [nn.Linear(prev,h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer  = nn.Linear(prev, n_classes)
    def forward(self, x):
        return self.output_layer(self.hidden_layers(x))

class MLPRegression(nn.Module):
    def __init__(self, n_in, dropout=0.3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_in,256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256,128),  nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128,64),   nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64,32),    nn.ReLU(),
            nn.Linear(32,1)
        )
    def forward(self, x):
        return self.network(x).squeeze(1)

def train_mlp(model, X_tr, y_tr, X_v, y_v, n_epochs, lr, save_path, regression=False):
    model = model.to(device)
    crit  = nn.MSELoss() if regression else nn.CrossEntropyLoss()
    opt   = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sch   = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
    best  = float('inf')
    bs    = 64

    Xtr_t = torch.tensor(X_tr); ytr_t = torch.tensor(y_tr)
    Xv_t  = torch.tensor(X_v);  yv_t  = torch.tensor(y_v)

    for ep in range(1, n_epochs+1):
        model.train()
        idx = torch.randperm(len(Xtr_t))
        for i in range(0, len(idx), bs):
            b = idx[i:i+bs]
            Xb, yb = Xtr_t[b].to(device), ytr_t[b].to(device)
            opt.zero_grad()
            out  = model(Xb)
            loss = crit(out, yb)
            loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            out_v  = model(Xv_t.to(device))
            loss_v = crit(out_v, yv_t.to(device)).item()
        sch.step(loss_v)
        if loss_v < best:
            best = loss_v
            torch.save(model.state_dict(), save_path)
        if ep % 20 == 0 or ep == 1:
            if regression:
                print(f"  Epoch {ep:3d} | Val MSE : {loss_v:.4f}")
            else:
                acc = (out_v.argmax(1).cpu() == yv_t).float().mean().item()
                print(f"  Epoch {ep:3d} | Val Loss : {loss_v:.4f} | Acc : {acc:.4f}")
    print(f"  ✅ Sauvegardé : {save_path}")
    return model

try:
    df = pd.read_csv('bmw.csv')
    for col in df.select_dtypes('object').columns:
        df[col] = df[col].str.strip()
    df.dropna(inplace=True)

    le_m  = LabelEncoder(); le_t = LabelEncoder(); le_f = LabelEncoder()
    df['model_enc']        = le_m.fit_transform(df['model'])
    df['transmission_enc'] = le_t.fit_transform(df['transmission'])
    df['fuelType_enc']     = le_f.fit_transform(df['fuelType'])

    # Outliers
    q1, q99 = df['price'].quantile(0.01), df['price'].quantile(0.99)
    df = df[(df['price']>=q1)&(df['price']<=q99)]

    # ── Classification carburant ──────────────────────────────────
    print("\n[1/2] MLP Classification (fuelType)...")
    FEAT_C = ['model_enc','year','price','transmission_enc','mileage','tax','mpg','engineSize']
    X_c = df[FEAT_C].values.astype(np.float32)
    y_c = df['fuelType_enc'].values.astype(np.int64)

    # Filtre classes rares
    counts = np.bincount(y_c)
    valid  = np.where(counts >= 10)[0]
    mask   = np.isin(y_c, valid)
    X_c, y_c = X_c[mask], y_c[mask]
    le_remap = LabelEncoder(); y_c = le_remap.fit_transform(y_c)
    N_CLS_C  = len(np.unique(y_c))

    sc_c = StandardScaler()
    Xtr,Xv,ytr,yv = train_test_split(X_c,y_c,test_size=0.2,random_state=SEED,stratify=y_c)
    Xtr = sc_c.fit_transform(Xtr).astype(np.float32)
    Xv  = sc_c.transform(Xv).astype(np.float32)

    mlp_cls = MLPCustom(8, [128,64,32], N_CLS_C)
    train_mlp(mlp_cls, Xtr, ytr, Xv, yv, n_epochs=60, lr=1e-3,
              save_path='models/best_mlp_custom.pth')

    # ── Régression prix ───────────────────────────────────────────
    print("\n[2/2] MLP Régression (price)...")
    FEAT_R = ['model_enc','year','transmission_enc','mileage','tax','mpg','engineSize']
    X_r = df[FEAT_R].values.astype(np.float32)
    y_r = df['price'].values.astype(np.float32)

    sc_rx = StandardScaler(); sc_ry = StandardScaler()
    Xtr,Xv,ytr,yv = train_test_split(X_r,y_r,test_size=0.2,random_state=SEED)
    Xtr = sc_rx.fit_transform(Xtr).astype(np.float32)
    Xv  = sc_rx.transform(Xv).astype(np.float32)
    ytr = sc_ry.fit_transform(ytr.reshape(-1,1)).flatten().astype(np.float32)
    yv  = sc_ry.transform(yv.reshape(-1,1)).flatten().astype(np.float32)

    mlp_reg = MLPRegression(7)
    train_mlp(mlp_reg, Xtr, ytr, Xv, yv, n_epochs=80, lr=1e-3,
              save_path='models/best_mlp_regression.pth', regression=True)

    print("\n✅ Partie I terminée !")
except FileNotFoundError:
    print("⚠️  bmw.csv non trouvé — Partie I ignorée")
except Exception as e:
    print(f"⚠️  Erreur Partie I : {e}")


# ══════════════════════════════════════════════════════════════════
# PARTIE II — CNN
# ══════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("🖼️  PARTIE II — CNN sur bmw_image_dataset_v2")
print("-"*50)

class FastCNN(nn.Module):
    def __init__(self, n_classes, dropout=0.6):
        super().__init__()
        def dw_block(in_ch, out_ch, stride=1):
            return nn.Sequential(
                nn.Conv2d(in_ch,in_ch,3,stride=stride,padding=1,groups=in_ch,bias=False),
                nn.BatchNorm2d(in_ch), nn.ReLU6(),
                nn.Conv2d(in_ch,out_ch,1,bias=False),
                nn.BatchNorm2d(out_ch), nn.ReLU6()
            )
        self.features = nn.Sequential(
            nn.Conv2d(3,32,3,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(32), nn.ReLU6(),
            dw_block(32,64,1), dw_block(64,128,2),
            dw_block(128,128,1), dw_block(128,256,2),
            dw_block(256,256,1), nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout),
            nn.Linear(256,128), nn.ReLU(),
            nn.Dropout(dropout*0.5), nn.Linear(128,n_classes)
        )
    def forward(self, x):
        return self.classifier(self.features(x))

IMG_SIZE = 128

def pil_to_tensor(img):
    arr  = np.array(img, dtype=np.float32)/255.0
    mean = np.array([0.485,0.456,0.406])
    std  = np.array([0.229,0.224,0.225])
    return torch.tensor((arr-mean)/std).permute(2,0,1).float()

def augment_pil(img, training=False):
    if training:
        if random.random()>0.5: img = img.transpose(Image.FLIP_LEFT_RIGHT)
        img = img.rotate(random.uniform(-20,20), fillcolor=(128,128,128))
        if random.random()>0.5:
            w,h  = img.size; scale=random.uniform(0.8,1.0); crop=int(w*scale)
            left = random.randint(0,w-crop); top=random.randint(0,h-crop)
            img  = img.crop((left,top,left+crop,top+crop)).resize((w,h),Image.BILINEAR)
        if random.random()>0.5:
            img = ImageEnhance.Brightness(img).enhance(random.uniform(0.7,1.3))
        if random.random()>0.5:
            img = ImageEnhance.Contrast(img).enhance(random.uniform(0.7,1.3))
    return img

class BMWDataset(Dataset):
    def __init__(self, samples, img_size=128, training=False):
        self.img_size = img_size; self.training = training
        print(f"  Cache RAM ({len(samples)} images)...")
        self.cache = []
        for i,(path,label) in enumerate(samples):
            try:
                img = Image.open(path).convert('RGB').resize((img_size,img_size),Image.BILINEAR)
                self.cache.append((img.copy(), label))
            except:
                self.cache.append((Image.new('RGB',(img_size,img_size)), label))
            if (i+1)%2000==0: print(f"    {i+1}/{len(samples)}...")
        print(f"  ✅ Cache prêt")
    def __len__(self): return len(self.cache)
    def __getitem__(self, idx):
        img, label = self.cache[idx]
        return pil_to_tensor(augment_pil(img, self.training)), label

DATA_DIR = 'bmw_image_dataset_v2/bmw_cars'
try:
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"Dossier {DATA_DIR} non trouvé")

    CLASS_NAMES  = [c for c in sorted(os.listdir(DATA_DIR))
                    if not c.startswith('.') and os.path.isdir(os.path.join(DATA_DIR,c))]
    N_CLASSES    = len(CLASS_NAMES)
    class_to_idx = {cls:i for i,cls in enumerate(CLASS_NAMES)}
    print(f"  Classes : {N_CLASSES}")

    all_samples = []
    for cls in CLASS_NAMES:
        cls_path = os.path.join(DATA_DIR, cls)
        for fname in os.listdir(cls_path):
            if fname.lower().endswith(('.jpg','.jpeg','.png','.webp')):
                fp = os.path.join(cls_path, fname)
                try:
                    with Image.open(fp) as img: img.verify()
                    all_samples.append((fp, class_to_idx[cls]))
                except: pass
    print(f"  Images valides : {len(all_samples)}")

    random.seed(SEED); random.shuffle(all_samples)
    n = len(all_samples)
    n_train = int(0.70*n); n_val = int(0.15*n)
    tr_s = all_samples[:n_train]; va_s = all_samples[n_train:n_train+n_val]

    # Class weights
    tr_labels = [l for _,l in tr_s]
    cnt = np.bincount(tr_labels, minlength=N_CLASSES).astype(np.float32)
    cw  = 1.0 / (np.sqrt(cnt)+1e-6); cw = cw/cw.sum()*N_CLASSES
    cw_t = torch.tensor(cw, dtype=torch.float32).to(device)

    train_ds = BMWDataset(tr_s, IMG_SIZE, training=True)
    val_ds   = BMWDataset(va_s, IMG_SIZE, training=False)

    BS = 64
    train_loader = DataLoader(train_ds, batch_size=BS, shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BS, shuffle=False, num_workers=0, pin_memory=True)

    model_cnn = FastCNN(N_CLASSES, dropout=0.6).to(device)
    crit_cnn  = nn.CrossEntropyLoss(weight=cw_t)
    opt_cnn   = optim.Adam(model_cnn.parameters(), lr=3e-4, weight_decay=1e-4)
    sch_cnn   = optim.lr_scheduler.ReduceLROnPlateau(opt_cnn, patience=5, factor=0.5)
    best_cnn  = float('inf')

    print(f"\n  Entraînement CNN ({60} époques)...")
    for ep in range(1, 61):
        t0 = time.time()
        model_cnn.train()
        tr_ok=tr_n=0
        for Xb,yb in train_loader:
            Xb,yb = Xb.to(device,non_blocking=True), yb.to(device,non_blocking=True)
            opt_cnn.zero_grad(set_to_none=True)
            if device.type=='cuda':
                with torch.autocast(device_type='cuda'):
                    out = model_cnn(Xb); loss = crit_cnn(out,yb)
            else:
                out = model_cnn(Xb); loss = crit_cnn(out,yb)
            loss.backward(); opt_cnn.step()
            tr_ok += (out.argmax(1)==yb).sum().item(); tr_n += Xb.size(0)

        model_cnn.eval()
        vl=vl_ok=vl_n=0.0
        with torch.no_grad():
            for Xb,yb in val_loader:
                Xb,yb = Xb.to(device,non_blocking=True), yb.to(device,non_blocking=True)
                if device.type=='cuda':
                    with torch.autocast(device_type='cuda'):
                        out=model_cnn(Xb); l=crit_cnn(out,yb)
                else:
                    out=model_cnn(Xb); l=crit_cnn(out,yb)
                vl+=l.item()*Xb.size(0); vl_ok+=(out.argmax(1)==yb).sum().item(); vl_n+=Xb.size(0)
        v_loss=vl/vl_n; v_acc=vl_ok/vl_n
        sch_cnn.step(v_loss)
        if v_loss<best_cnn:
            best_cnn=v_loss
            torch.save(model_cnn.state_dict(),'models/best_fast_cnn_bmw_final.pth')
        if ep%10==0 or ep==1:
            print(f"  Epoch {ep:3d}/60 | TrainAcc:{tr_ok/tr_n:.4f} | ValAcc:{v_acc:.4f} | {time.time()-t0:.1f}s")
    print("  ✅ CNN sauvegardé : models/best_fast_cnn_bmw_final.pth")
    print("\n✅ Partie II terminée !")
except FileNotFoundError as e:
    print(f"⚠️  {e} — Partie II ignorée")
except Exception as e:
    print(f"⚠️  Erreur Partie II : {e}")


# ══════════════════════════════════════════════════════════════════
# PARTIE III — RNN / LSTM / GRU
# ══════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("📝 PARTIE III — RNN / LSTM / GRU sur Reviews BMW")
print("-"*50)

MAX_LEN    = 100
BATCH_SIZE = 128
EMBED_DIM  = 64
HIDDEN_DIM = 128
N_LAYERS   = 1
DROPOUT    = 0.3
N_EPOCHS   = 10
MIN_FREQ   = 2
PAD_IDX    = 0
UNK_IDX    = 1

class RNNSimple(nn.Module):
    def __init__(self, vs, ed, hd, nc, nl=1, dr=0.3, pi=0):
        super().__init__()
        self.embedding = nn.Embedding(vs,ed,padding_idx=pi)
        self.rnn       = nn.RNN(ed,hd,nl,batch_first=True,nonlinearity='tanh',
                                dropout=dr if nl>1 else 0)
        self.dropout   = nn.Dropout(dr)
        self.fc        = nn.Linear(hd,nc)
    def forward(self,x):
        emb=self.dropout(self.embedding(x)); _,h=self.rnn(emb)
        return self.fc(self.dropout(h[-1]))

class LSTMModel(nn.Module):
    def __init__(self, vs, ed, hd, nc, nl=1, dr=0.3, pi=0):
        super().__init__()
        self.embedding = nn.Embedding(vs,ed,padding_idx=pi)
        self.lstm      = nn.LSTM(ed,hd,nl,batch_first=True,dropout=dr if nl>1 else 0)
        self.dropout   = nn.Dropout(dr)
        self.fc        = nn.Linear(hd,nc)
    def forward(self,x):
        emb=self.dropout(self.embedding(x)); _,(h,_)=self.lstm(emb)
        return self.fc(self.dropout(h[-1]))

class GRUModel(nn.Module):
    def __init__(self, vs, ed, hd, nc, nl=1, dr=0.3, pi=0):
        super().__init__()
        self.embedding = nn.Embedding(vs,ed,padding_idx=pi)
        self.gru       = nn.GRU(ed,hd,nl,batch_first=True,dropout=dr if nl>1 else 0)
        self.dropout   = nn.Dropout(dr)
        self.fc        = nn.Linear(hd,nc)
    def forward(self,x):
        emb=self.dropout(self.embedding(x)); _,h=self.gru(emb)
        return self.fc(self.dropout(h[-1]))

class ReviewDataset(Dataset):
    def __init__(self, texts, labels, word2idx, max_len=MAX_LEN):
        def tok(t):
            t=str(t).lower(); t=re.sub(r'[^a-z0-9\s]',' ',t); return t.split()
        def enc(tokens):
            return [word2idx.get(t,UNK_IDX) for t in tokens[:max_len]]
        self.encodings = [enc(tok(t)) for t in texts]
        self.labels    = labels
    def __len__(self): return len(self.labels)
    def __getitem__(self,idx):
        return torch.tensor(self.encodings[idx],dtype=torch.long), torch.tensor(self.labels[idx],dtype=torch.long)

def collate_fn(batch):
    seqs,labels = zip(*batch)
    padded = torch.zeros(len(seqs),MAX_LEN,dtype=torch.long)
    for i,seq in enumerate(seqs):
        l=min(len(seq),MAX_LEN)
        if l>0: padded[i,:l]=seq[:l]
    return padded, torch.stack(labels)

def train_rnn_model(model, tr_ld, va_ld, n_epochs, save_path, cw_t):
    model = model.to(device)
    crit  = nn.CrossEntropyLoss(weight=cw_t)
    opt   = optim.Adam(model.parameters(), lr=1e-3)
    sch   = optim.lr_scheduler.ReduceLROnPlateau(opt,patience=3,factor=0.5)
    best  = float('inf')
    for ep in range(1, n_epochs+1):
        model.train()
        tr_ok=tr_n=0
        for Xb,yb in tr_ld:
            Xb,yb=Xb.to(device,non_blocking=True),yb.to(device,non_blocking=True)
            opt.zero_grad(set_to_none=True)
            out=model(Xb); loss=crit(out,yb); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.0)
            opt.step()
            tr_ok+=(out.argmax(1)==yb).sum().item(); tr_n+=Xb.size(0)
        model.eval()
        vl=vl_ok=vl_n=0.0
        with torch.no_grad():
            for Xb,yb in va_ld:
                Xb,yb=Xb.to(device,non_blocking=True),yb.to(device,non_blocking=True)
                out=model(Xb); l=crit(out,yb)
                vl+=l.item()*Xb.size(0); vl_ok+=(out.argmax(1)==yb).sum().item(); vl_n+=Xb.size(0)
        v_loss=vl/vl_n; v_acc=vl_ok/vl_n
        sch.step(v_loss)
        if v_loss<best:
            best=v_loss; torch.save(model.state_dict(),save_path)
        if ep%5==0 or ep==1:
            print(f"  Epoch {ep:3d}/{n_epochs} | TrainAcc:{tr_ok/tr_n:.4f} | ValAcc:{v_acc:.4f}")
    print(f"  ✅ Sauvegardé : {save_path}")

try:
    csv_path = 'Scrapped_Car_Reviews_BMW.csv'
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} non trouvé")

    try:
        df_r = pd.read_csv(csv_path, encoding='latin-1', on_bad_lines='skip')
    except:
        df_r = pd.read_csv(csv_path, encoding='utf-8', on_bad_lines='skip', engine='python', quoting=3)

    df_r = df_r[['Review','Rating']].dropna()
    df_r['Review'] = df_r['Review'].astype(str)
    df_r['Rating'] = pd.to_numeric(df_r['Rating'], errors='coerce')
    df_r.dropna(inplace=True)
    df_r['Rating'] = df_r['Rating'].apply(lambda x: max(1,min(5,round(x))))
    df_r['label']  = df_r['Rating'].apply(lambda x: 1 if x>=4 else 0)
    print(f"  Reviews : {len(df_r)} | Positif : {df_r['label'].sum()} | Négatif : {(df_r['label']==0).sum()}")

    # Vocabulaire
    def tokenize(text):
        text=str(text).lower(); text=re.sub(r'[^a-z0-9\s]',' ',text); return text.split()
    all_tokens = []
    for rev in df_r['Review']: all_tokens.extend(tokenize(rev))
    counter  = Counter(all_tokens)
    vocab    = ['<PAD>','<UNK>','<BOS>','<EOS>'] + [w for w,c in counter.items() if c>=MIN_FREQ]
    word2idx = {w:i for i,w in enumerate(vocab)}
    VOCAB_SIZE = len(vocab)
    print(f"  Vocabulaire : {VOCAB_SIZE:,} tokens")

    # Split
    texts  = df_r['Review'].tolist(); labels = df_r['label'].tolist()
    Xtr,Xv,ytr,yv = train_test_split(texts,labels,test_size=0.2,random_state=SEED,stratify=labels)

    # Class weights
    neg_c = sum(1 for l in ytr if l==0); pos_c = sum(1 for l in ytr if l==1)
    tot   = neg_c+pos_c
    cw_rnn = torch.tensor([tot/(2*neg_c), tot/(2*pos_c)], dtype=torch.float32).to(device)

    # Datasets
    tr_ds = ReviewDataset(Xtr,ytr,word2idx); va_ds = ReviewDataset(Xv,yv,word2idx)
    tr_ld = DataLoader(tr_ds,batch_size=BATCH_SIZE,shuffle=True, collate_fn=collate_fn,num_workers=0,pin_memory=True)
    va_ld = DataLoader(va_ds,batch_size=BATCH_SIZE,shuffle=False,collate_fn=collate_fn,num_workers=0,pin_memory=True)

    args = (VOCAB_SIZE,EMBED_DIM,HIDDEN_DIM,2,N_LAYERS,DROPOUT,PAD_IDX)

    print("\n[1/3] RNN Simple...")
    rnn = RNNSimple(*args)
    train_rnn_model(rnn, tr_ld, va_ld, N_EPOCHS, 'models/best_rnn_simple.pth', cw_rnn)

    print("\n[2/3] LSTM...")
    lstm = LSTMModel(*args)
    train_rnn_model(lstm, tr_ld, va_ld, N_EPOCHS, 'models/best_lstm.pth', cw_rnn)

    print("\n[3/3] GRU...")
    gru = GRUModel(*args)
    train_rnn_model(gru, tr_ld, va_ld, N_EPOCHS, 'models/best_gru.pth', cw_rnn)

    print("\n✅ Partie III terminée !")

except FileNotFoundError as e:
    print(f"⚠️  {e} — Partie III ignorée")
except Exception as e:
    print(f"⚠️  Erreur Partie III : {e}")

print(f"\n{'='*60}")
print("  ✅ ENTRAÎNEMENT TERMINÉ")
print(f"{'='*60}")
print("\nModèles générés dans models/ :")
for f in os.listdir('models'):
    if f.endswith('.pth'):
        size = os.path.getsize(f'models/{f}') / 1024
        print(f"  ✅ {f:<45} ({size:.0f} KB)")
print("\nLance maintenant : streamlit run app.py")
