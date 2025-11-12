# app.py — FIXED VERSION (Weights-only loading)
import streamlit as st
st.set_page_config(page_title="🎬 Smart Ad Placement", layout="wide", initial_sidebar_state="expanded")

import pandas as pd
import numpy as np
import re, os, pickle, json, math
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dense, Dropout, Concatenate
from tensorflow.keras import regularizers
from datetime import timedelta
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============== CONFIG ==============
MODEL_PATH = r"bilstm_model_v3_balanced-20251112T130342Z-1-001\bilstm_model_v3_balanced\bilstm_ad_model.h5"
WEIGHTS_PATH = r"bilstm_model_v3_balanced-20251112T130342Z-1-001\bilstm_model_v3_balanced\bilstm_ad_model.weights.h5"
TOKENIZER_PATH = r"bilstm_model_v3_balanced-20251112T130342Z-1-001\bilstm_model_v3_balanced\tokenizer.pkl"
VOCAB_SIZE = 8000
MAX_LEN = 25

# Defaults
DEFAULT_THRESHOLD = 0.50
DEFAULT_MIN_SPACING = 180.0
INTRO_CUTOFF = 180.0
END_CUTOFF = 600.0
WINDOW_SECONDS = 3600
WINDOW_MAX = 3

# ============== BUILD MODEL ARCHITECTURE ==============
def build_model_architecture():
    """Recreate exact model architecture (same as training)"""
    available_numeric_cols = ['norm_gap', 'norm_duration', 'is_sentence_end', 'has_music_tag', 'is_shouting']
    
    # Input layers
    text_input = Input(shape=(MAX_LEN,), name="text_input")
    num_input = Input(shape=(len(available_numeric_cols),), name="num_input")
    
    # Text branch - BiLSTM
    x = Embedding(
        input_dim=VOCAB_SIZE,
        output_dim=128,
        mask_zero=True,
        name="embedding"
    )(text_input)
    
    x = Bidirectional(
        LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.3,
             kernel_regularizer=regularizers.l2(1e-4), name="bilstm_1")
    )(x)
    
    x = Bidirectional(
        LSTM(64, return_sequences=False, dropout=0.3, recurrent_dropout=0.3,
             kernel_regularizer=regularizers.l2(1e-4), name="bilstm_2")
    )(x)
    
    x = Dropout(0.4, name="dropout_text")(x)
    
    # Numeric branch
    y = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(1e-4),
              name="dense_num_1")(num_input)
    y = Dropout(0.3, name="dropout_num_1")(y)
    y = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(1e-4),
              name="dense_num_2")(y)
    y = Dropout(0.2, name="dropout_num_2")(y)
    
    # Merge
    combined = Concatenate(name="merge")([x, y])
    
    # Dense layers
    z = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4),
              name="dense_combined_1")(combined)
    z = Dropout(0.4, name="dropout_combined_1")(z)
    z = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(1e-4),
              name="dense_combined_2")(z)
    z = Dropout(0.3, name="dropout_combined_2")(z)
    
    # Output
    output = Dense(1, activation='sigmoid', name="output")(z)
    
    model = tf.keras.Model(inputs=[text_input, num_input], outputs=output)
    return model

# ============== LOAD MODEL + TOKENIZER ==============
@st.cache_resource(show_spinner=False)
def load_model_and_tokenizer():
    """Load model and tokenizer with fallback options"""
    
    # Check files exist
    if not os.path.exists(TOKENIZER_PATH):
        st.error(f"❌ Tokenizer not found: {TOKENIZER_PATH}")
        st.error(f"Current directory: {os.getcwd()}")
        st.error(f"Available files: {os.listdir('.')}")
        st.stop()
    
    # Load tokenizer
    try:
        with open(TOKENIZER_PATH, 'rb') as f:
            tokenizer = pickle.load(f)
        st.success("✅ Tokenizer loaded!")
    except Exception as e:
        st.error(f"❌ Failed to load tokenizer: {e}")
        st.stop()
    
    # Try loading model - multiple methods
    model = None
    
    # Method 1: Try loading full .h5 model
    if os.path.exists(MODEL_PATH):
        try:
            with tf.keras.utils.custom_object_scope({'NotEqual': tf.math.not_equal}):
                model = tf.keras.models.load_model(MODEL_PATH, compile=False)
            st.success("✅ Model loaded (Method 1: Full .h5)!")
            return model, tokenizer
        except Exception as e:
            st.warning(f"⚠️ Method 1 failed: {e}")
    
    # Method 2: Build architecture + load weights
    if os.path.exists(WEIGHTS_PATH):
        try:
            st.info("⏳ Loading model (Method 2: Architecture + Weights)...")
            model = build_model_architecture()
            model.load_weights(WEIGHTS_PATH)
            st.success("✅ Model loaded (Method 2: Weights)!")
            return model, tokenizer
        except Exception as e:
            st.warning(f"⚠️ Method 2 failed: {e}")
    
    # Method 3: Build fresh architecture (inference only, no weights)
    if model is None:
        try:
            st.warning("⚠️ No weights found. Using fresh model architecture (random weights).")
            st.info("⏳ Building model from architecture...")
            model = build_model_architecture()
            st.warning("⚠️ Using random weights - predictions may be inaccurate!")
            return model, tokenizer
        except Exception as e:
            st.error(f"❌ Failed to build model: {e}")
            st.stop()
    
    return model, tokenizer

# Try to load
model, tokenizer = load_model_and_tokenizer()

# ============== DARK THEME CSS ==============
st.markdown("""
    <style>
    body {background-color: #0e1117; color: #f8f9fa;}
    .stApp {background-color: #0e1117;}
    .stDataFrame {border-radius: 10px; overflow: hidden;}
    .stDownloadButton>button {
        background: linear-gradient(90deg, #0066ff, #00cc99) !important;
        color: white !important;
        border: none !important;
        padding: 10px 20px !important;
        border-radius: 6px !important;
    }
    .stDownloadButton>button:hover {
        background: linear-gradient(90deg, #0099ff, #00ffaa) !important;
    }
    .stButton>button {background-color: #222; color: #eee;}
    .stTextInput>div>div>input {background-color: #1e1e1e; color: #fff;}
    .stNumberInput>div>div>input {background-color: #1e1e1e; color: #fff;}
    .stSlider>div>div {color: #00ff99;}
    .block-container {padding-top: 1rem;}
    h1, h2, h3 {color: #00ffaa;}
    </style>
""", unsafe_allow_html=True)

# ============== HELPER FUNCTIONS ==============

def srt_time_to_sec(ts):
    """Convert SRT timestamp (HH:MM:SS,mmm) to seconds"""
    try:
        hh, mm, rest = ts.split(":")
        ss, ms = rest.split(",")
        return int(hh)*3600 + int(mm)*60 + int(ss) + int(ms)/1000.0
    except:
        return 0.0

def parse_srt(file):
    """Parse SRT subtitle file"""
    raw = file.read().decode("utf-8", errors="ignore")
    blocks = re.split(r'\n\s*\n', raw.strip())
    rows = []
    
    for b in blocks:
        lines = [l.strip() for l in b.splitlines() if l.strip()]
        if len(lines) >= 2 and "-->" in lines[1]:
            try:
                m = re.match(r'(.+?)\s*-->\s*(.+)', lines[1])
                if not m:
                    continue
                start = srt_time_to_sec(m.group(1).strip())
                end = srt_time_to_sec(m.group(2).strip())
                text = " ".join(lines[2:]) if len(lines) > 2 else ""
                rows.append({"start_time": start, "end_time": end, "text": text})
            except:
                continue
    
    return pd.DataFrame(rows)

def parse_txt(file):
    """Parse TXT subtitle file (format: [HH:MM:SS] text or [MM:SS] text)"""
    raw = file.read().decode("utf-8", errors="ignore")
    lines = raw.splitlines()
    rows = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        m = re.match(r'^\[?(\d{2}):(\d{2}):(\d{2})\]?\s*(.*)$', line)
        if m:
            hh, mm, ss, text = int(m.group(1)), int(m.group(2)), int(m.group(3)), m.group(4)
            sec = hh*3600 + mm*60 + ss
            rows.append({"start_time": float(sec), "text": text})
            continue
        
        m2 = re.match(r'^\[?(\d{2}):(\d{2})\]?\s*(.*)$', line)
        if m2:
            mm, ss, text = int(m2.group(1)), int(m2.group(2)), m2.group(3)
            sec = mm*60 + ss
            rows.append({"start_time": float(sec), "text": text})
    
    if not rows:
        return pd.DataFrame()
    
    df = pd.DataFrame(rows).sort_values("start_time").reset_index(drop=True)
    df['end_time'] = df['start_time'].shift(-1) - 1.0
    df['end_time'] = df['end_time'].fillna(df['start_time'] + 1.0)
    df['end_time'] = df.apply(lambda r: max(r['end_time'], r['start_time'] + 0.5), axis=1)
    
    return df

def compute_features(df):
    """Compute numeric features for model"""
    df = df.sort_values("start_time").reset_index(drop=True)
    
    df["duration"] = df["end_time"] - df["start_time"]
    df["gap"] = df["start_time"].shift(-1) - df["end_time"]
    df["gap"] = df["gap"].fillna(0.0)
    
    def ends_with_punct(t):
        return int(bool(re.search(r'[.!?…]$', str(t).strip())))
    
    def has_music_tag(t):
        return int(bool(re.search(r'\[(music|applause|door|sound)\]|\(music|applause|door|sound\)', str(t), re.IGNORECASE)))
    
    def is_shouting(t):
        words = str(t).split()
        if not words:
            return 0
        upper_ratio = sum(1 for w in words if w.isupper()) / len(words)
        return int('!' in str(t) or upper_ratio > 0.6)
    
    df["is_sentence_end"] = df["text"].apply(ends_with_punct)
    df["has_music_tag"] = df["text"].apply(has_music_tag)
    df["is_shouting"] = df["text"].apply(is_shouting)
    
    df["norm_gap"] = np.clip(df["gap"] / 6.0, 0, 1)
    df["norm_duration"] = np.clip(df["duration"] / 5.0, 0, 1)
    
    df["ad_score"] = (
        0.4 * df["norm_gap"] + 
        0.2 * df["is_sentence_end"] + 
        0.2 * df["has_music_tag"] - 
        0.2 * df["is_shouting"]
    ).clip(0, 1)
    
    return df

def select_ads(df, threshold=DEFAULT_THRESHOLD, min_spacing=DEFAULT_MIN_SPACING,
               intro_cut=INTRO_CUTOFF, end_cut=END_CUTOFF,
               window_seconds=WINDOW_SECONDS, window_max=WINDOW_MAX):
    """Select optimal ad placement positions"""
    
    movie_len = float(df['end_time'].max())
    min_ads = max(1, int(movie_len // 1800))
    max_ads = max(2, math.ceil(movie_len / 600.0))
    
    cands = df[(df['prob'] >= threshold) &
               (df['start_time'] > intro_cut) &
               (df['end_time'] < (movie_len - end_cut))].copy()
    cands = cands.sort_values(by='prob', ascending=False)
    
    selected = []
    
    for _, row in cands.iterrows():
        st = float(row['start_time'])
        
        if any(abs(st - s) < min_spacing for s in selected):
            continue
        
        window_count = sum(1 for s in selected if abs(s - st) < window_seconds)
        if window_count >= window_max:
            continue
        
        selected.append(st)
        if len(selected) >= max_ads:
            break
    
    if len(selected) < min_ads:
        fallback = df[(df['start_time'] > intro_cut) & 
                      (df['end_time'] < (movie_len - end_cut))].sort_values(by='ad_score', ascending=False)
        
        for _, row in fallback.iterrows():
            st = float(row['start_time'])
            if any(abs(st - s) < min_spacing for s in selected):
                continue
            selected.append(st)
            if len(selected) >= min_ads:
                break
    
    selected = sorted(selected)[:max_ads]
    return selected

def sec_to_time(sec):
    """Convert seconds to HH:MM:SS format"""
    return str(timedelta(seconds=int(round(sec))))

# ============== STREAMLIT UI ==============

st.title("🎬 AI-Powered Ad Placement Detector")
st.caption("Upload subtitle file (.srt, .csv, .txt) for intelligent ad break suggestions")

# ============== SIDEBAR CONTROLS ==============
st.sidebar.header("⚙️ Detection Settings")

threshold = st.sidebar.slider(
    "🎯 Decision Threshold",
    min_value=0.1,
    max_value=0.9,
    value=float(DEFAULT_THRESHOLD),
    step=0.05,
    help="Lowering increases ad detection sensitivity"
)

min_spacing = st.sidebar.slider(
    "📏 Minimum Spacing (seconds)",
    min_value=30,
    max_value=600,
    value=int(DEFAULT_MIN_SPACING),
    step=30,
    help="Minimum gap between consecutive ads"
)

intro_cut = st.sidebar.slider(
    "⏭️ Intro Skip (seconds)",
    min_value=0,
    max_value=600,
    value=int(INTRO_CUTOFF),
    step=30,
    help="Skip this duration from the start"
)

end_cut = st.sidebar.slider(
    "⏮️ End Skip (seconds)",
    min_value=0,
    max_value=1200,
    value=int(END_CUTOFF),
    step=30,
    help="Skip this duration from the end"
)

st.sidebar.markdown("---")
st.sidebar.write(f"**Window Settings:**")
st.sidebar.write(f"• Max {WINDOW_MAX} ads per {WINDOW_SECONDS//60} minutes")

# ============== FILE UPLOAD ==============

st.markdown("### 📁 Upload Subtitle File")
uploaded = st.file_uploader(
    "Choose a subtitle file",
    type=["srt", "csv", "txt"],
    help="Supported: SRT, CSV (with start_time, end_time, text), TXT"
)

if not uploaded:
    st.info("👆 **Upload a file to begin** (SRT / CSV / TXT)")
    st.markdown("""
    ---
    **Supported Formats:**
    - **SRT**: Standard subtitle format (00:00:00,000 --> 00:00:05,000)
    - **TXT**: Time-stamped format ([HH:MM:SS] text or [MM:SS] text)
    - **CSV**: Must have columns: start_time, end_time, text
    """)
else:
    filetype = uploaded.name.split(".")[-1].lower()
    
    try:
        if filetype == "srt":
            df = parse_srt(uploaded)
            st.success("✅ SRT file parsed")
        elif filetype == "txt":
            df = parse_txt(uploaded)
            st.success("✅ TXT file parsed")
        else:
            df = pd.read_csv(uploaded, low_memory=False)
            
            if 'text' not in df.columns:
                st.error("❌ CSV must contain 'text' column")
                st.stop()
            
            if 'start_time' not in df.columns:
                st.error("❌ CSV must contain 'start_time' column")
                st.stop()
            
            if 'end_time' not in df.columns:
                df = df.sort_values('start_time').reset_index(drop=True)
                df['end_time'] = df['start_time'].shift(-1) - 1.0
                df['end_time'] = df['end_time'].fillna(df['start_time'] + 1.0)
                df['end_time'] = df.apply(lambda r: max(r['end_time'], r['start_time'] + 0.5), axis=1)
            
            st.success("✅ CSV file parsed")
    
    except Exception as e:
        st.error(f"❌ Failed to parse file: {e}")
        st.stop()
    
    if df.empty or len(df) == 0:
        st.error("❌ Parsed file has zero rows. Check file format.")
        st.stop()
    
    df = compute_features(df)
    
    movie_length = float(df['end_time'].max())
    movie_length_min = movie_length / 60.0
    
    st.success(f"✅ Loaded {len(df)} subtitle rows | Movie: {sec_to_time(movie_length)} ({movie_length_min:.1f} mins)")
    
    # ============== MODEL PREDICTION ==============
    st.markdown("### 🧠 Running Model Prediction...")
    progress_bar = st.progress(0)
    
    seqs = tokenizer.texts_to_sequences(df['text'].astype(str).values)
    X_text = pad_sequences(seqs, maxlen=MAX_LEN, padding='post', truncating='post')
    
    X_num = df[['norm_gap', 'norm_duration', 'is_sentence_end', 'has_music_tag', 'is_shouting']].values.astype(np.float32)
    
    progress_bar.progress(50)
    df['prob'] = model.predict([X_text, X_num], batch_size=256, verbose=0).reshape(-1)
    progress_bar.progress(100)
    progress_bar.empty()
    
    st.success("✅ Predictions complete!")
    
    # ============== SIDEBAR INFO ==============
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Analysis Info")
    st.sidebar.metric("🎞️ Movie Length", sec_to_time(movie_length))
    st.sidebar.metric("📝 Subtitle Rows", len(df))
    
    raw_candidates = df[
        (df['prob'] >= threshold) & 
        (df['start_time'] > intro_cut) & 
        (df['end_time'] < (movie_length - end_cut))
    ]
    st.sidebar.metric("🎯 Raw Candidates (prob≥threshold)", len(raw_candidates))
    
    # ============== SELECT ADS ==============
    selected = select_ads(
        df,
        threshold=threshold,
        min_spacing=min_spacing,
        intro_cut=intro_cut,
        end_cut=end_cut,
        window_seconds=WINDOW_SECONDS,
        window_max=WINDOW_MAX
    )
    
    # Build output dataframe
    out = []
    for t in selected:
        row = df.iloc[(df['start_time'] - t).abs().argsort()[:1]].iloc[0]
        
        reasons = []
        if row['norm_gap'] > 0.5:
            reasons.append('long_gap')
        if row['is_sentence_end']:
            reasons.append('sentence_end')
        if row['has_music_tag']:
            reasons.append('music_tag')
        if not row['is_shouting']:
            reasons.append('not_shouting')
        
        out.append({
            'Ad Time': sec_to_time(t),
            'Seconds': round(float(t), 2),
            'Model Prob': f"{float(row['prob']):.3f}",
            'Ad Score': f"{float(row['ad_score']):.3f}",
            'Reasons': ", ".join(reasons) or "optimal_score",
            'Text': row['text'][:50] + "..." if len(str(row['text'])) > 50 else row['text']
        })
    
    df_out = pd.DataFrame(out)
    
    # ============== RESULTS ==============
    st.markdown("### 🎯 Suggested Ad Placements")
    
    if df_out.empty:
        st.warning("⚠️ No ad placements suggested. Try:")
        st.markdown("""
        - Lowering the decision threshold
        - Reducing intro/end skip durations
        - Checking if subtitle file has proper timing
        """)
    else:
        st.dataframe(df_out, use_container_width=True, height=400)
        
        st.sidebar.markdown("---")
        st.sidebar.subheader("📈 Summary")
        st.sidebar.metric("🎞️ Total Movie Length", sec_to_time(movie_length))
        st.sidebar.metric("📍 Ads Suggested", len(df_out))
        
        if len(df_out) > 0:
            avg_gap = int(movie_length / len(df_out))
            st.sidebar.metric("🕒 Avg Seconds/Ad", f"{avg_gap}s")
        
        st.markdown("### 🕒 Ad Placement Timeline")
        fig, ax = plt.subplots(figsize=(12, 2))
        ax.set_facecolor("#0e1117")
        fig.patch.set_facecolor("#0e1117")
        
        ax.set_xlim(0, movie_length if movie_length > 0 else 1)
        ax.set_ylim(0, 1)
        
        ax.barh(0.5, movie_length, height=0.1, color="#333333", alpha=0.5)
        
        for i, t in enumerate(selected):
            ax.scatter(t, 0.5, s=200, color="#00FFAA", marker='|', linewidth=3, zorder=10)
            ax.text(t, 0.7, f"Ad {i+1}", ha='center', fontsize=8, color="#00FFAA")
        
        ax.axvline(intro_cut, color="#FF6666", linestyle='--', linewidth=2, alpha=0.5, label="Intro cutoff")
        ax.axvline(movie_length - end_cut, color="#FF6666", linestyle='--', linewidth=2, alpha=0.5, label="End cutoff")
        
        ax.set_xlabel("Time (seconds)", color="white")
        ax.set_yticks([])
        ax.tick_params(colors="gray")
        ax.legend(loc='upper right', facecolor="#1e1e1e", edgecolor="#555")
        
        st.pyplot(fig, use_container_width=True)
        
        csv_data = df_out.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv_data,
            file_name=f"ad_suggestions_{uploaded.name.split('.')[0]}.csv",
            mime="text/csv"
        )

st.markdown("---")
st.caption("🚀 Built with BiLSTM + Streamlit | © 2025 Ad Placement AI")
