import streamlit as st
st.set_page_config(page_title="🎬 ADAPTOID - Adaptive AI Ad Placement Engine", layout="wide", initial_sidebar_state="expanded")

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

MODEL_PATH = r"bilstm_ad_model.h5"
WEIGHTS_PATH = r"bilstm_ad_model.weights.h5"
TOKENIZER_PATH = r"tokenizer.pkl"
VOCAB_SIZE = 8000
MAX_LEN = 25

# Defaults
DEFAULT_THRESHOLD = 0.50
DEFAULT_MIN_SPACING = 180.0
INTRO_CUTOFF = 180.0
END_CUTOFF = 600.0
WINDOW_SECONDS = 3600
WINDOW_MAX = 3

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
    
    if not os.path.exists(TOKENIZER_PATH):
        st.error(f"❌ Tokenizer not found: {TOKENIZER_PATH}")
        st.error(f"Current directory: {os.getcwd()}")
        st.stop()
    
    try:
        with open(TOKENIZER_PATH, 'rb') as f:
            tokenizer = pickle.load(f)
        st.success("✅ Tokenizer loaded!")
    except Exception as e:
        st.error(f"❌ Failed to load tokenizer: {e}")
        st.stop()
    
    model = None
    

    
    if os.path.exists(WEIGHTS_PATH):
        try:
            st.info("⏳ Loading model")
            model = build_model_architecture()
            model.load_weights(WEIGHTS_PATH)
            st.success("✅ Model loaded")
            return model, tokenizer
        except Exception as e:
            st.warning(f"⚠️ Method 2 failed: {e}")
    
    if model is None:
        try:
            st.warning("⚠️ No weights found. Using fresh model architecture.")
            model = build_model_architecture()
            return model, tokenizer
        except Exception as e:
            st.error(f"❌ Failed to build model: {e}")
            st.stop()
    
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

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


def extract_video_id(url):
    """Extract video ID from YouTube URL"""
    patterns = [
        r'(?:youtube\.com\/watch\?v=)([a-zA-Z0-9_-]{11})',
        r'(?:youtu\.be\/)([a-zA-Z0-9_-]{11})',
        r'(?:youtube\.com\/embed\/)([a-zA-Z0-9_-]{11})',
        r'^([a-zA-Z0-9_-]{11})$'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def fetch_youtube_subtitles_method1(video_id):
    """Method 1: youtube-transcript-api library - MOST RELIABLE"""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
        
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        
        # Try manual transcripts first (highest quality)
        for lang in ['en', 'en-US', 'en-GB']:
            try:
                transcript = transcript_list.find_manually_created_transcript([lang])
                data = transcript.fetch()
                if data and len(data) > 10:  # Validate substantial content
                    return data, "✅ Method 1 (Manual EN)"
            except:
                continue
        
        # Try auto-generated English
        for lang in ['en', 'en-US', 'en-GB']:
            try:
                transcript = transcript_list.find_generated_transcript([lang])
                data = transcript.fetch()
                if data and len(data) > 10:
                    return data, "✅ Method 1 (Auto EN)"
            except:
                continue
        
        # Try Hindi
        for lang in ['hi', 'hi-IN']:
            try:
                transcript = transcript_list.find_manually_created_transcript([lang])
                data = transcript.fetch()
                if data and len(data) > 10:
                    return data, "✅ Method 1 (Manual HI)"
            except:
                continue
        
        # Try any available transcript
        for transcript in transcript_list:
            try:
                data = transcript.fetch()
                if data and len(data) > 10:
                    return data, f"✅ Method 1 ({transcript.language_code})"
            except:
                continue
        
    except Exception as e:
        pass
    
    return None, None

def download_and_parse_subtitle_url(url):
    """Download subtitle file from URL and parse it"""
    try:
        import urllib.request
        import xml.etree.ElementTree as ET
        
        headers = {'User-Agent': 'Mozilla/5.0'}
        req = urllib.request.Request(url, headers=headers)
        response = urllib.request.urlopen(req, timeout=10)
        content = response.read().decode('utf-8', errors='ignore')
        
        rows = []
        
        # Try XML format (YouTube format)
        if '<text' in content or '<transcript>' in content:
            try:
                root = ET.fromstring(content)
                for text_elem in root.findall('.//text'):
                    start = float(text_elem.get('start', 0))
                    dur = float(text_elem.get('dur', 1))
                    text = text_elem.text or ''
                    text = text.replace('\n', ' ').strip()
                    if text:
                        rows.append({
                            'start': start,
                            'duration': dur,
                            'text': text
                        })
            except:
                pass
        
        # Try JSON format
        elif content.startswith('{') or content.startswith('['):
            try:
                data = json.loads(content)
                if isinstance(data, list):
                    for entry in data:
                        if 'start' in entry and 'text' in entry:
                            rows.append({
                                'start': float(entry.get('start', 0)),
                                'duration': float(entry.get('duration', entry.get('dur', 1))),
                                'text': str(entry.get('text', ''))
                            })
            except:
                pass
        
        # Try VTT/SRT format
        else:
            lines = content.split('\n')
            for line in lines:
                # Match timestamp patterns
                if '-->' in line:
                    continue
                # Extract text lines
                line = line.strip()
                if line and not line.isdigit() and 'WEBVTT' not in line:
                    rows.append({
                        'start': len(rows) * 2.0,  # Approximate timing
                        'duration': 2.0,
                        'text': line
                    })
        
        return rows if len(rows) > 10 else None
        
    except Exception as e:
        return None

def fetch_youtube_subtitles_method2(video_id):
    """Method 2: yt-dlp with proper subtitle download"""
    try:
        import yt_dlp
        
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'writesubtitles': True,
            'writeautomaticsub': True,
            'skip_download': True,
            'subtitleslangs': ['en', 'en-US', 'en-GB', 'hi'],
            'subtitlesformat': 'json3/srv3/srv2/srv1/ttml/vtt',
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
            
            # Check subtitles (manual)
            if 'subtitles' in info and info['subtitles']:
                for lang in ['en', 'en-US', 'en-GB', 'hi']:
                    if lang in info['subtitles']:
                        sub_list = info['subtitles'][lang]
                        for sub in sub_list:
                            if 'url' in sub:
                                rows = download_and_parse_subtitle_url(sub['url'])
                                if rows and len(rows) > 10:
                                    return rows, f"✅ Method 2 (Manual {lang})"
            
            # Check automatic captions
            if 'automatic_captions' in info and info['automatic_captions']:
                for lang in ['en', 'en-US', 'en-GB', 'hi']:
                    if lang in info['automatic_captions']:
                        sub_list = info['automatic_captions'][lang]
                        for sub in sub_list:
                            if 'url' in sub:
                                rows = download_and_parse_subtitle_url(sub['url'])
                                if rows and len(rows) > 10:
                                    return rows, f"✅ Method 2 (Auto {lang})"
        
    except Exception as e:
        pass
    
    return None, None

def fetch_youtube_subtitles_method3(video_id):
    """Method 3: Direct web scraping from YouTube page"""
    try:
        import urllib.request
        import xml.etree.ElementTree as ET
        
        url = f"https://www.youtube.com/watch?v={video_id}"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        
        req = urllib.request.Request(url, headers=headers)
        html = urllib.request.urlopen(req, timeout=10).read().decode('utf-8', errors='ignore')
        
        # Extract caption track URL
        if '"captions":' in html or '"captionTracks":' in html:
            # Find captionTracks JSON
            pattern = r'"captionTracks":\s*(\[.*?\])'
            match = re.search(pattern, html)
            
            if match:
                try:
                    tracks_json = match.group(1)
                    tracks = json.loads(tracks_json)
                    
                    # Try English first
                    for track in tracks:
                        if 'baseUrl' in track:
                            lang_code = track.get('languageCode', '')
                            if lang_code.startswith('en') or lang_code == 'a.en':
                                caption_url = track['baseUrl']
                                rows = download_and_parse_subtitle_url(caption_url)
                                if rows and len(rows) > 10:
                                    return rows, "✅ Method 3 (Web EN)"
                    
                    # Try any available
                    for track in tracks:
                        if 'baseUrl' in track:
                            caption_url = track['baseUrl']
                            rows = download_and_parse_subtitle_url(caption_url)
                            if rows and len(rows) > 10:
                                lang = track.get('languageCode', 'unknown')
                                return rows, f"✅ Method 3 (Web {lang})"
                except:
                    pass
        
    except Exception as e:
        pass
    
    return None, None

def fetch_youtube_subtitles(video_id):
    """Try all methods to fetch subtitles"""
    
    # Method 1 is most reliable
    subtitles, method = fetch_youtube_subtitles_method1(video_id)
    if subtitles and len(subtitles) > 10:
        return subtitles, method
    
    # Method 2 as backup
    subtitles, method = fetch_youtube_subtitles_method2(video_id)
    if subtitles and len(subtitles) > 10:
        return subtitles, method
    
    # Method 3 as last resort
    subtitles, method = fetch_youtube_subtitles_method3(video_id)
    if subtitles and len(subtitles) > 10:
        return subtitles, method
    
    return None, None

def youtube_to_dataframe(video_id):
    """Convert YouTube subtitles to DataFrame with validation"""
    
    subtitles, method = fetch_youtube_subtitles(video_id)
    
    if subtitles is None or len(subtitles) < 10:
        return None, "❌ No valid subtitles found (tried all 3 methods)"
    
    rows = []
    for entry in subtitles:
        start = float(entry.get('start', 0))
        duration = float(entry.get('duration', entry.get('dur', 1)))
        text = str(entry.get('text', '')).strip()
        
        if text and duration > 0:  # Validate entry
            rows.append({
                'start_time': start,
                'end_time': start + duration,
                'text': text
            })
    
    if len(rows) < 10:
        return None, f"❌ Insufficient subtitle data (only {len(rows)} entries found)"
    
    df = pd.DataFrame(rows)
    return df, method

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
    """Parse TXT subtitle file"""
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


st.title("🎬 ADAPTOID - Adaptive AI Ad Placement Engine")
st.caption("Upload subtitle file (.srt, .csv, .txt) or paste YouTube link for intelligent ad break suggestions")

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

# ============== INPUT SELECTION ==============

st.markdown("### 📥 Choose Input Method")

tab1, tab2 = st.tabs(["📁 Upload File", "🔗 YouTube Link"])

df = None
source_name = None

with tab1:
    st.markdown("**Upload a subtitle file**")
    uploaded = st.file_uploader(
        "Choose a subtitle file",
        type=["srt", "csv", "txt"],
        help="Supported: SRT, CSV (with start_time, end_time, text), TXT"
    )
    
    if uploaded:
        filetype = uploaded.name.split(".")[-1].lower()
        source_name = uploaded.name
        
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

with tab2:
    st.markdown("**Paste YouTube video URL or ID**")
    
    
    youtube_input = st.text_input(
        "YouTube URL or Video ID",
        placeholder="https://www.youtube.com/watch?v=VIDEO_ID or just VIDEO_ID",
        help="Paste full YouTube URL or just the 11-character video ID"
    )
    
    if youtube_input and youtube_input.strip():
        video_id = extract_video_id(youtube_input.strip())
        
        if not video_id:
            st.error("❌ Invalid YouTube URL or Video ID")
        else:
            st.info(f"🎥 Video ID: `{video_id}`")
            
            with st.spinner("⏳ Fetching subtitles (trying 3 methods sequentially)..."):
                df, method = youtube_to_dataframe(video_id)
            
            if df is None:
                st.error(f"{method}")
                st.markdown("""
                **Possible reasons:**
                - Video has no captions/subtitles enabled
                - Captions are disabled by the uploader
                - Video is private or age-restricted
                - Network connectivity issue
                
                **Try:**
                - Wait 1-2 minutes and try again
                - Try a different YouTube video
                - Use file upload method if subtitles are available for download
                """)
            else:
                st.success(f"✅ Subtitles fetched! {method} | {len(df)} entries")
                source_name = f"YouTube_{video_id}"

# ============== PROCESSING ==============

if df is None or df.empty:
    st.info("👆 **Choose an input method above** (Upload file or paste YouTube link)")
    st.markdown("""
    ---
    **Supported Formats:**
    - **YouTube**: Automatically fetches captions/subtitles
    - **SRT**: Standard subtitle format (00:00:00,000 --> 00:00:05,000)
    - **TXT**: Time-stamped format ([HH:MM:SS] text or [MM:SS] text)
    - **CSV**: Must have columns: start_time, end_time, text
    """)
else:
    if len(df) < 10:
        st.error(f"❌ Insufficient subtitle data: only {len(df)} rows found. Need at least 10 rows.")
        st.stop()
    
    df = compute_features(df)
    
    movie_length = float(df['end_time'].max())
    movie_length_min = movie_length / 60.0
    
    st.success(f"✅ Loaded {len(df)} subtitle rows | Video: {sec_to_time(movie_length)} ({movie_length_min:.1f} mins)")
    
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
    st.sidebar.metric("🎞️ Video Length", sec_to_time(movie_length))
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
        

        
        out.append({
            'Ad Time': sec_to_time(t),
            'Seconds': round(float(t), 2),
            'Model Prob': f"{float(row['prob']):.3f}",

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
        st.sidebar.metric("🎞️ Total Video Length", sec_to_time(movie_length))
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
            file_name=f"ad_suggestions_{source_name.split('.')[0] if source_name else 'results'}.csv",
            mime="text/csv"
        )

st.markdown("---")
st.caption("Made by Aryan & Harshit 😁")
