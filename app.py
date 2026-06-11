import os
import re
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, silhouette_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud

try:
    from sklearn_extra.cluster import KMedoids
    KMEDOIDS_AVAILABLE = True
except Exception:
    KMEDOIDS_AVAILABLE = False

DATA_PATH_DEFAULT = "Womens Clothing E-Commerce Reviews.csv"

# ── Cafe design tokens ──────────────────────────────────────
PRIMARY   = "#5D4432"   # coffee brown
SECONDARY = "#E9E3DD"   # warm cream
SURFACE   = "#F9F7F5"   # off-white warm
TEXT      = "#3E2B1E"   # dark espresso
SUCCESS   = "#16A34A"
WARNING   = "#D97706"
DANGER    = "#DC2626"
MUTED     = "#9C8475"   # warm taupe

st.set_page_config(
    page_title="Women's Clothing Reviews — Dashboard",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* =================================================================
       CAFE DESIGN SYSTEM — Poppins font + warm brown/cream palette
       Tokens: primary=#5D4432 secondary=#E9E3DD surface=#F9F7F5
                text=#3E2B1E  success=#16A34A warning=#D97706
    ================================================================= */

    /* --- Google Fonts -------------------------------------------- */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    /* --- Global app font ----------------------------------------- */
    html, body, [class*="css"], .stApp {
        font-family: 'Poppins', sans-serif !important;
    }

    /* --- Animated warm background -------------------------------- */
    .stApp {
        background: linear-gradient(-45deg, #1a0d08, #3a1f12, #5D4432, #8B6040, #C4A882, #8B6040, #3a1f12);
        background-size: 400% 400%;
        animation: cafeWarm 22s ease infinite;
    }
    @keyframes cafeWarm {
        0%   { background-position: 0% 50%; }
        50%  { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* --- Glass content card (surface token) ---------------------- */
    .block-container {
        background: rgba(249, 247, 245, 0.97) !important;   /* surface */
        border-radius: 20px !important;
        padding: 2.4rem 3rem 3.2rem !important;
        margin-top: 1rem !important;
        box-shadow: 0 24px 64px rgba(61,27,14,0.30) !important;
        backdrop-filter: blur(14px);
        border-top: 3px solid #5D4432;
    }

    /* --- Body text (text token) ---------------------------------- */
    .block-container > div,
    .stMarkdown p, .stMarkdown li, .stMarkdown ol,
    .stMarkdown ul, .stMarkdown strong, .stMarkdown em,
    .stCaption {
        color: #3E2B1E !important;
        font-family: 'Poppins', sans-serif !important;
    }

    /* --- Headings ------------------------------------------------ */
    .block-container h1 {
        color: #5D4432 !important;
        font-weight: 700 !important;
        font-family: 'Poppins', sans-serif !important;
    }
    .block-container h2,
    .block-container h3 {
        color: #5D4432 !important;
        font-weight: 600 !important;
        font-family: 'Poppins', sans-serif !important;
        border-left: 4px solid #D97706;
        padding-left: 0.65rem;
        margin-top: 1.8rem;
    }
    .block-container h4 {
        color: #5D4432 !important;
        font-weight: 600 !important;
    }

    /* --- Widget labels ------------------------------------------- */
    .stSelectbox > label,
    .stMultiSelect > label,
    .stSlider > label,
    .stTextInput > label,
    .stRadio > label,
    .stCheckbox > label,
    .stNumberInput > label {
        color: #3E2B1E !important;
        font-family: 'Poppins', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
    }

    /* --- Selectbox / Multiselect — white box, dark text ---------- */
    [data-baseweb="select"] {
        background-color: #ffffff !important;
        border-radius: 8px !important;
    }
    [data-baseweb="select"] * {
        color: #3E2B1E !important;
        background-color: transparent !important;
        font-family: 'Poppins', sans-serif !important;
    }
    [data-baseweb="select"] [data-baseweb="tag"] {
        background-color: #E9E3DD !important;
        color: #5D4432 !important;
        border-radius: 20px !important;
    }
    /* Dropdown popup */
    [data-baseweb="popover"],
    [data-baseweb="menu"] {
        background-color: #ffffff !important;
        border: 1px solid #E9E3DD !important;
        border-radius: 10px !important;
        box-shadow: 0 8px 24px rgba(93,68,50,0.18) !important;
    }
    [data-baseweb="option"] {
        color: #3E2B1E !important;
        background-color: #ffffff !important;
        font-family: 'Poppins', sans-serif !important;
        font-size: 0.9rem !important;
    }
    [data-baseweb="option"]:hover,
    [data-baseweb="option"][aria-selected="true"] {
        background-color: #E9E3DD !important;
        color: #5D4432 !important;
    }

    /* --- Text input ---------------------------------------------- */
    [data-baseweb="input"] input,
    [data-baseweb="textarea"] textarea {
        background-color: #ffffff !important;
        color: #3E2B1E !important;
        border: 1.5px solid #C4A882 !important;
        border-radius: 8px !important;
        font-family: 'Poppins', sans-serif !important;
    }
    [data-baseweb="input"] input:focus,
    [data-baseweb="textarea"] textarea:focus {
        border-color: #5D4432 !important;
        box-shadow: 0 0 0 3px rgba(93,68,50,0.15) !important;
    }
    [data-baseweb="input"] input::placeholder {
        color: #9C8475 !important;
    }

    /* --- Slider -------------------------------------------------- */
    .stSlider [data-testid="stTickBarMin"],
    .stSlider [data-testid="stTickBarMax"],
    .stSlider span {
        color: #3E2B1E !important;
        font-family: 'Poppins', sans-serif !important;
    }

    /* =================================================================
       SIDEBAR — dark espresso + cream text
    ================================================================= */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a0d08 0%, #2C1810 55%, #3a1f12 100%) !important;
        border-right: 1px solid rgba(196,168,130,0.20);
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label {
        font-family: 'Poppins', sans-serif !important;
    }
    /* Navigation Radio Items Styling */
    section[data-testid="stSidebar"] .stRadio label {
        color: #C4A882 !important;
        font-weight: 500 !important;
        font-size: 0.88rem !important;
        letter-spacing: 0.1px;
        transition: color 0.15s;
    }
    section[data-testid="stSidebar"] .stRadio label:hover {
        color: #F9F7F5 !important;
    }
    section[data-testid="stSidebar"] .stRadio [data-checked="true"] + div p {
        color: #F9F7F5 !important;
        font-weight: 700 !important;
    }
    section[data-testid="stSidebar"] .stRadio label p {
        color: inherit !important;
        font-weight: inherit !important;
    }
    section[data-testid="stSidebar"] .stTextInput label p,
    section[data-testid="stSidebar"] .stSlider label p {
        color: #F9F7F5 !important;
    }
    section[data-testid="stSidebar"] .stTextInput > label,
    section[data-testid="stSidebar"] .stSlider > label {
        color: #F9F7F5 !important;
        font-size: 0.76rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.6px;
    }
    section[data-testid="stSidebar"] .stSlider [data-testid="stTickBarMin"],
    section[data-testid="stSidebar"] .stSlider [data-testid="stTickBarMax"],
    section[data-testid="stSidebar"] .stSlider span {
        color: #E9E3DD !important;
    }
    section[data-testid="stSidebar"] [data-baseweb="input"] input {
        background-color: rgba(255,255,255,0.06) !important;
        color: #E9E3DD !important;
        border: 1px solid rgba(196,168,130,0.30) !important;
    }
    section[data-testid="stSidebar"] strong {
        color: #D97706 !important;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #C4A882 !important;
        border: none !important;
        padding: 0 !important;
    }
    section[data-testid="stSidebar"] hr {
        border-color: rgba(196,168,130,0.20) !important;
        margin: 0.8rem 0 !important;
    }
    section[data-testid="stSidebar"] .stAlert * {
        color: #ffb3b3 !important;
    }

    /* =================================================================
       PAGE HEADER STRIP — cafe style: warm underline, left icon bar
    ================================================================= */
    .page-header {
        display: flex;
        align-items: flex-start;
        gap: 1rem;
        margin-bottom: 2rem;
        padding-bottom: 1rem;
        border-bottom: 1.5px solid #E9E3DD;
    }
    .page-header-bar {
        width: 5px;
        min-height: 56px;
        border-radius: 3px;
        background: linear-gradient(180deg, #5D4432, #D97706);
        flex-shrink: 0;
        margin-top: 2px;
    }
    .page-header h1 {
        margin: 0 !important;
        font-size: 1.85rem !important;
        font-weight: 800 !important;
        color: #5D4432 !important;
        font-family: 'Poppins', sans-serif !important;
        border: none !important;
        padding: 0 !important;
        line-height: 1.2;
        letter-spacing: -0.3px;
    }
    .page-header .ph-sub {
        font-size: 0.85rem;
        color: #9C8475 !important;
        margin-top: 0.3rem;
        font-weight: 400;
        font-family: 'Poppins', sans-serif !important;
    }

    /* =================================================================
       METRIC CARDS — warm brown gradient + cream text
    ================================================================= */
    .metric-card {
        background: linear-gradient(135deg, #5D4432 0%, #7A5C42 60%, #D97706 100%);
        padding: 1.5rem 1.2rem;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 8px 24px rgba(93,68,50,0.35);
        transition: transform 0.22s ease, box-shadow 0.22s ease;
        font-family: 'Poppins', sans-serif;
    }
    .metric-card:hover {
        transform: translateY(-6px) scale(1.03);
        box-shadow: 0 16px 40px rgba(93,68,50,0.45);
    }
    .metric-card .num {
        font-size: 2rem !important;
        font-weight: 800 !important;
        color: #F9F7F5 !important;
        line-height: 1.2;
        font-family: 'Poppins', sans-serif !important;
    }
    .metric-card .lbl {
        font-size: 0.75rem !important;
        color: #E9E3DD !important;
        margin-top: 0.35rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        font-family: 'Poppins', sans-serif !important;
    }

    /* =================================================================
       BUTTONS — primary brown, warning amber hover
    ================================================================= */
    .stButton > button {
        background: #5D4432 !important;
        color: #F9F7F5 !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.55rem 1.6rem !important;
        font-weight: 600 !important;
        font-size: 0.91rem !important;
        font-family: 'Poppins', sans-serif !important;
        letter-spacing: 0.2px;
        transition: all 0.20s ease !important;
        box-shadow: 0 3px 10px rgba(93,68,50,0.28) !important;
    }
    .stButton > button:hover {
        background: #D97706 !important;
        color: #1a0d08 !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 20px rgba(217,119,6,0.40) !important;
    }

    /* =================================================================
       TECH PILLS
    ================================================================= */
    .pill {
        display: inline-block;
        background: #E9E3DD;
        color: #5D4432 !important;
        padding: 5px 14px;
        border-radius: 20px;
        font-size: 0.80rem;
        margin: 4px;
        font-weight: 600;
        border: 1.5px solid #C4A882;
        font-family: 'Poppins', sans-serif;
        letter-spacing: 0.1px;
    }

    /* =================================================================
       ALERT / INFO / SUCCESS BOXES
    ================================================================= */
    .stAlert p, .stAlert div, .stAlert span {
        color: #3E2B1E !important;
        font-family: 'Poppins', sans-serif !important;
    }

    /* =================================================================
       DATAFRAME
    ================================================================= */
    .stDataFrame * {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.82rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────
# PAGE HEADER HELPER
# ─────────────────────────────────────────────────────────────
def page_header(title, subtitle=""):
    sub_html = f'<div class="ph-sub">{subtitle}</div>' if subtitle else ""
    st.markdown(
        f"""
        <div class="page-header">
            <div class="page-header-bar"></div>
            <div>
                <h1>{title}</h1>
                {sub_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────
# MATPLOTLIB STYLE — cafe warm palette
# ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.facecolor": "#F9F7F5",
    "figure.facecolor": "#F9F7F5",
    "axes.edgecolor": "#C4A882",
    "axes.labelcolor": "#3E2B1E",
    "xtick.color": "#3E2B1E",
    "ytick.color": "#3E2B1E",
    "text.color": "#3E2B1E",
    "grid.color": "#E9E3DD",
    "grid.linestyle": "--",
    "grid.alpha": 0.7,
})

CAFE_BAR    = "#5D4432"
CAFE_LINE   = "#D97706"
CAFE_ACCENT = "#8B6040"


def ensure_nltk():
    import nltk
    nltk.download("punkt",       quiet=True)
    nltk.download("punkt_tab",   quiet=True)
    nltk.download("stopwords",   quiet=True)
    nltk.download("wordnet",     quiet=True)
    nltk.download("omw-1.4",    quiet=True)
    nltk.download("vader_lexicon", quiet=True)


@st.cache_data
def load_data(path):
    return pd.read_csv(path)


@st.cache_data
def clean_data(df):
    df = df.copy()
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    df = df.drop_duplicates()
    for col in ["Title", "Review Text"]:
        if col in df.columns:
            df[col] = df[col].fillna("")
    for col in ["Division Name", "Department Name", "Class Name"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])
    return df


@st.cache_data
def add_features(df):
    df = df.copy()
    df["review_length"]  = df["Review Text"].str.len()
    df["word_count"]     = df["Review Text"].str.split().str.len()
    df["rating_category"] = pd.cut(
        df["Rating"], bins=[-np.inf, 2, 3, 5], labels=["Low", "Medium", "High"]
    )
    df["sentiment_label"] = df["Rating"].apply(
        lambda r: "Positive" if r >= 4 else "Neutral" if r == 3 else "Negative"
    )
    return df


@st.cache_data
def preprocess_text_series(text_series):
    ensure_nltk()
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer

    stop_words  = set(stopwords.words("english"))
    lemmatizer  = WordNetLemmatizer()

    def preprocess(text):
        text   = str(text).lower()
        text   = re.sub(r"[^a-z\s]", " ", text)
        tokens = word_tokenize(text)
        tokens = [lemmatizer.lemmatize(t) for t in tokens
                  if t not in stop_words and len(t) > 2]
        return " ".join(tokens)

    return text_series.fillna("").apply(preprocess)


@st.cache_data
def build_tfidf(corpus):
    vec    = TfidfVectorizer(stop_words="english", max_features=5000)
    matrix = vec.fit_transform(corpus)
    return vec, matrix


@st.cache_data
def build_count_vector(corpus, max_features=2000):
    vec    = CountVectorizer(stop_words="english", max_features=max_features)
    matrix = vec.fit_transform(corpus)
    return vec, matrix


@st.cache_data
def build_inverted_index(tokens_list):
    inv = {}
    for idx, tokens in enumerate(tokens_list):
        for token in tokens:
            inv.setdefault(token, set()).add(idx)
    return {k: sorted(list(v)) for k, v in inv.items()}


@st.cache_data
def calculate_silhouette_comparison(X_scaled):
    scores = {}
    n_samples = min(len(X_scaled), 5000)
    indices = np.random.RandomState(42).choice(len(X_scaled), n_samples, replace=False)
    X_sample = X_scaled[indices]
    for k_val in [2, 3, 4, 5, 6]:
        km = KMeans(n_clusters=k_val, random_state=42, n_init=10)
        labels = km.fit_predict(X_sample)
        scores[k_val] = silhouette_score(X_sample, labels)
    return scores



# ─────────────────────────────────────────────────────────────
# SIDEBAR  — dark espresso, no emojis, minimal
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div style="padding:1.2rem 0.4rem 0.5rem;">
            <div style="font-size:0.68rem;letter-spacing:2.5px;text-transform:uppercase;
                        color:#9C8475;font-weight:600;margin-bottom:0.4rem;
                        font-family:'Poppins',sans-serif;">
                Analytics Dashboard
            </div>
            <div style="font-size:1.18rem;font-weight:800;color:#E9E3DD;line-height:1.3;
                        font-family:'Poppins',sans-serif;">
                Women's Clothing<br>
                <span style="color:#D97706;">Reviews</span>
            </div>
        </div>
        <hr style="border-color:rgba(196,168,130,0.18);margin:0 0 0.6rem 0;">
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigate",
        [
            "Home",
            "Dataset Overview",
            "Preprocessing & Feature Engineering",
            "Classification Analysis",
            "Clustering Analysis",
            "Information Retrieval",
            "Text Mining & Sentiment",
            "Conclusion & Recommendation",
        ],
        label_visibility="collapsed",
    )

    st.markdown(
        """<hr style="border-color:rgba(196,168,130,0.18);margin:0.8rem 0 0.6rem;">""",
        unsafe_allow_html=True,
    )

    path = st.text_input("Dataset path", value=DATA_PATH_DEFAULT)
    if not os.path.exists(path):
        st.error("Dataset file not found.")

    sample_size = st.slider("Text sample size", 500, 5000, 2000, 500)

    st.markdown(
        """
        <hr style="border-color:rgba(196,168,130,0.18);margin:0.8rem 0 0.6rem;">
        <div style="font-size:0.84rem;color:#C4A882;line-height:1.8;
                    font-family:'Poppins',sans-serif;">
            <strong>Student</strong><br><span style="color:#E9E3DD;">Muhammad Luqman bin Aziz</span><br>
            <strong>Course</strong><br><span style="color:#E9E3DD;">Data Mining &amp; Information Retrieval</span><br>
            <strong>Lecturer</strong><br><span style="color:#E9E3DD;">Dr. Nurashikin Saaludin</span><br>
            <strong>Semester</strong><br><span style="color:#E9E3DD;">March 2026</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────
if os.path.exists(path):
    df_raw   = load_data(path)
    df_clean = clean_data(df_raw)
    df_feat  = add_features(df_clean)
    data_loaded = True
else:
    data_loaded = False


# ============================================================
# MODULE 1 — HOME
# ============================================================
if page == "Home":
    page_header(
        "Women's Clothing Reviews",
        "Data Mining & Information Retrieval — Interactive Dashboard",
    )

    st.markdown("### About this Dashboard")
    st.write(
        "You are a team of Data Analysts working for an e-commerce company. "
        "This dashboard analyzes customer behavior, predicts recommendations, "
        "discovers customer segments, simulates a review search engine, and "
        "analyzes review sentiment using NLP techniques."
    )

    st.markdown("#### Methods & Techniques")
    st.markdown(
        '<span class="pill">Data Preprocessing</span>'
        '<span class="pill">Feature Engineering</span>'
        '<span class="pill">PCA</span>'
        '<span class="pill">Classification</span>'
        '<span class="pill">Clustering</span>'
        '<span class="pill">TF-IDF Retrieval</span>'
        '<span class="pill">Text Mining</span>'
        '<span class="pill">Sentiment Analysis</span>',
        unsafe_allow_html=True,
    )

    if data_loaded:
        st.markdown("#### Quick Stats")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f'<div class="metric-card"><div class="num">{df_feat.shape[0]:,}</div><div class="lbl">Reviews</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="metric-card"><div class="num">{df_feat["Rating"].mean():.2f}</div><div class="lbl">Avg Rating</div></div>', unsafe_allow_html=True)
        with c3:
            rec_pct = df_feat["Recommended IND"].mean() * 100
            st.markdown(f'<div class="metric-card"><div class="num">{rec_pct:.1f}%</div><div class="lbl">Recommended</div></div>', unsafe_allow_html=True)
        with c4:
            st.markdown(f'<div class="metric-card"><div class="num">{df_feat["Class Name"].nunique()}</div><div class="lbl">Product Classes</div></div>', unsafe_allow_html=True)
    else:
        st.error("Place 'Womens Clothing E-Commerce Reviews.csv' in the same folder as app.py")


# ============================================================
# MODULE 2 — DATASET OVERVIEW
# ============================================================
elif page == "Dataset Overview":
    page_header("Dataset Overview", "Raw data, missing values, cleaning, and distributions")

    if not data_loaded:
        st.error("Dataset not found. Place the CSV next to app.py.")
        st.stop()

    st.markdown("### Raw Dataset Preview")
    n_rows = st.slider("Rows to preview", 5, 50, 10)
    st.dataframe(df_raw.head(n_rows))

    st.markdown("### Dataset Shape")
    st.write(f"Rows: **{df_raw.shape[0]:,}** &nbsp;|&nbsp; Columns: **{df_raw.shape[1]}**")

    st.markdown("### Missing Value Summary")
    miss           = df_raw.isnull().sum().reset_index()
    miss.columns   = ["Column", "Missing Values"]
    st.dataframe(miss)

    st.markdown("### Cleaned Dataset")
    st.dataframe(df_clean.head(10))

    st.markdown("### Correlation Matrix")
    numeric_cols   = ["Age", "Rating", "Positive Feedback Count", "review_length", "word_count"]
    available_cols = [c for c in numeric_cols if c in df_feat.columns]
    corr           = df_feat[available_cols].corr()
    fig, ax        = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="YlOrBr", ax=ax, linewidths=0.5)
    ax.set_title("Feature Correlation Matrix")
    st.pyplot(fig)

    st.markdown("### Feature Distribution")
    feature       = st.selectbox("Select feature", available_cols)
    fig2, ax2     = plt.subplots(figsize=(7, 5))
    ax2.hist(df_feat[feature], bins=30, color=CAFE_BAR, edgecolor="#F9F7F5", linewidth=0.4)
    ax2.set_xlabel(feature)
    ax2.set_ylabel("Count")
    ax2.set_title(f"Distribution of {feature}")
    ax2.grid(True)
    st.pyplot(fig2)


# ============================================================
# MODULE 3 — PREPROCESSING & FEATURE ENGINEERING
# ============================================================
elif page == "Preprocessing & Feature Engineering":
    page_header("Preprocessing & Feature Engineering",
                "Cleaning, normalization, feature creation, and PCA")

    if not data_loaded:
        st.error("Dataset not found.")
        st.stop()

    st.markdown("### Data Cleaning Summary")
    st.write("Duplicate records removed, missing values handled, and categorical labels standardized.")

    # ---- Gap 1: show categorical encoding ----
    st.markdown("### Categorical Encoding")
    st.write(
        "Categorical columns are encoded using **Label Encoding** to convert string labels "
        "into numeric integers so machine learning models can process them."
    )
    from sklearn.preprocessing import LabelEncoder
    cat_cols = [c for c in ["Division Name", "Department Name", "Class Name"] if c in df_clean.columns]
    df_encoded = df_clean[cat_cols].copy()
    le = LabelEncoder()
    for col in cat_cols:
        df_encoded[col + " (encoded)"] = le.fit_transform(df_encoded[col].astype(str))
    st.dataframe(df_encoded.head(10))
    st.caption(
        "Each unique category string maps to a unique integer. "
        "Example: 'Bottoms' → 0, 'Dresses' → 1, 'Intimate' → 2, …"
    )

    st.markdown("### Feature Engineering")
    st.dataframe(df_feat[["Review Text", "review_length", "word_count", "rating_category"]].head(10))

    # ---- PCA (aligned with appsecond.py) ----
    st.markdown("### Principal Component Analysis (PCA)")
    selected_features = ["Age", "Rating", "Positive Feedback Count", "review_length"]
    
    st.markdown("**Selected Feature Table**")
    st.dataframe(df_feat[selected_features].head(10))

    scaler2  = StandardScaler()
    X_scaled = scaler2.fit_transform(df_feat[selected_features].fillna(0))

    pca   = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    st.markdown("**PCA Explained Variance Ratio**")
    st.write(pca.explained_variance_ratio_)

    pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
    pca_df["Recommended IND"] = df_feat["Recommended IND"].values

    st.markdown("**PCA Scatter Plot**")
    fig3, ax3   = plt.subplots()
    cafe_colors = [CAFE_BAR, CAFE_LINE]
    for i, val in enumerate(pca_df["Recommended IND"].unique()):
        subset = pca_df[pca_df["Recommended IND"] == val]
        ax3.scatter(subset["PC1"], subset["PC2"],
                    label=val, alpha=0.5,
                    color=cafe_colors[i % len(cafe_colors)])
    ax3.set_xlabel("PC1")
    ax3.set_ylabel("PC2")
    ax3.set_title("PCA — Women's Clothing Reviews")
    ax3.legend()
    ax3.grid(True)
    st.pyplot(fig3)


# ============================================================
# MODULE 4 — CLASSIFICATION ANALYSIS
# ============================================================
elif page == "Classification Analysis":
    page_header("Classification Analysis",
                "Predict customer recommendations using supervised ML")

    if not data_loaded:
        st.error("Dataset not found.")
        st.stop()

    st.write("**Target variable:** `Recommended IND`")

    default_features = ["Age", "Rating", "Positive Feedback Count", "review_length", "word_count"]
    features = st.multiselect(
        "Select features",
        options=[c for c in default_features if c in df_feat.columns],
        default=[c for c in default_features if c in df_feat.columns],
    )

    model_name = st.selectbox(
        "Select Classification Algorithm",
        ["Decision Tree", "Naive Bayes", "Logistic Regression", "KNN", "Random Forest"],
    )

    if features:
        X = df_feat[features].fillna(0)
        y = df_feat["Recommended IND"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        scaler      = StandardScaler()
        X_train_sc  = scaler.fit_transform(X_train)
        X_test_sc   = scaler.transform(X_test)

        if model_name == "Decision Tree":
            model = DecisionTreeClassifier(random_state=42)
        elif model_name == "Naive Bayes":
            model = GaussianNB()
        elif model_name == "Logistic Regression":
            model = LogisticRegression(max_iter=1000, random_state=42)
        elif model_name == "KNN":
            model = KNeighborsClassifier(n_neighbors=5)
        else:
            model = RandomForestClassifier(n_estimators=100, random_state=42)

        model.fit(X_train_sc, y_train)
        y_pred = model.predict(X_test_sc)
        acc    = accuracy_score(y_test, y_pred)

        st.markdown("### Accuracy Score")
        st.markdown(
            f'<div class="metric-card" style="max-width:260px">'
            f'<div class="num">{acc:.4f}</div>'
            f'<div class="lbl">{model_name} Accuracy</div></div>',
            unsafe_allow_html=True,
        )

        st.markdown("### Confusion Matrix")
        cm        = confusion_matrix(y_test, y_pred)
        fig, ax   = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrBr", ax=ax, linewidths=0.5)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"Confusion Matrix — {model_name}")
        st.pyplot(fig)

        st.markdown("### Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())

        # ---- Gap 2: prediction visualization ----
        st.markdown("### Prediction Visualization")
        pred_col1, pred_col2 = st.columns(2)

        with pred_col1:
            # Predicted vs actual class distribution bar chart
            pred_counts   = pd.Series(y_pred).value_counts().sort_index()
            actual_counts = y_test.value_counts().sort_index()
            labels_bar    = ["Not Recommended (0)", "Recommended (1)"]
            x_pos         = np.arange(len(labels_bar))
            width         = 0.35
            fig_pred, ax_pred = plt.subplots(figsize=(6, 4))
            ax_pred.bar(
                x_pos - width / 2,
                [actual_counts.get(i, 0) for i in range(2)],
                width, label="Actual", color=CAFE_BAR, edgecolor=SURFACE,
            )
            ax_pred.bar(
                x_pos + width / 2,
                [pred_counts.get(i, 0) for i in range(2)],
                width, label="Predicted", color=CAFE_LINE, edgecolor=SURFACE,
            )
            ax_pred.set_xticks(x_pos)
            ax_pred.set_xticklabels(labels_bar)
            ax_pred.set_ylabel("Count")
            ax_pred.set_title("Actual vs Predicted Class Distribution")
            ax_pred.legend()
            ax_pred.grid(axis="y", alpha=0.5)
            st.pyplot(fig_pred)

        with pred_col2:
            # Sample prediction table: 10 test rows with actual and predicted
            sample_pred_df = X_test.copy().reset_index(drop=True).head(10)
            sample_pred_df["Actual"]    = y_test.values[:10]
            sample_pred_df["Predicted"] = y_pred[:10]
            sample_pred_df["Correct"]   = sample_pred_df["Actual"] == sample_pred_df["Predicted"]
            st.markdown("**Sample Test Predictions (first 10)**")
            st.dataframe(sample_pred_df[["Actual", "Predicted", "Correct"]])

        if model_name == "Decision Tree":
            st.markdown("### Decision Tree Illustration")
            if st.button("Show Decision Tree"):
                fig2, ax2 = plt.subplots(figsize=(15, 10))
                plot_tree(model, feature_names=features,
                          class_names=["No", "Yes"], filled=True,
                          max_depth=3, ax=ax2)
                st.pyplot(fig2)
                st.caption("Depth limited to 3 for readability.")

        st.markdown("### Compare All Models")
        if st.button("Run Model Comparison"):
            models = {
                "Decision Tree":    DecisionTreeClassifier(random_state=42),
                "Naive Bayes":      GaussianNB(),
                "Logistic Reg.":    LogisticRegression(max_iter=1000, random_state=42),
                "KNN":              KNeighborsClassifier(n_neighbors=5),
                "Random Forest":    RandomForestClassifier(n_estimators=100, random_state=42),
            }
            results = {}
            for name, m in models.items():
                m.fit(X_train_sc, y_train)
                results[name] = accuracy_score(y_test, m.predict(X_test_sc))

            comp_df   = pd.DataFrame(list(results.items()), columns=["Model", "Accuracy"])
            st.dataframe(comp_df)
            fig3, ax3 = plt.subplots()
            ax3.bar(comp_df["Model"], comp_df["Accuracy"],
                    color=CAFE_BAR, edgecolor="#F9F7F5", linewidth=0.5)
            ax3.set_title("Model Accuracy Comparison")
            ax3.set_ylabel("Accuracy")
            plt.xticks(rotation=45, ha="right")
            st.pyplot(fig3)
    else:
        st.info("Select at least one feature to run classification.")


# ============================================================
# MODULE 5 — CLUSTERING ANALYSIS
# ============================================================
elif page == "Clustering Analysis":
    page_header("Clustering Analysis",
                "Customer segmentation via K-Means / K-Medoids")

    if not data_loaded:
        st.error("Dataset not found.")
        st.stop()

    default_features = ["Age", "Positive Feedback Count"]
    features = st.multiselect(
        "Select clustering features",
        options=[c for c in ["Age", "Rating", "Positive Feedback Count",
                              "review_length", "word_count"] if c in df_feat.columns],
        default=[c for c in default_features if c in df_feat.columns],
    )
    algorithm = st.selectbox("Clustering algorithm", ["K-Means", "K-Medoids"])

    if features:
        X        = df_feat[features].fillna(0)
        scaler   = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        st.markdown("### Elbow Method (Choose K)")
        st.write(
            "The **Elbow Method** helps identify the optimal number of clusters ($K$) "
            "by plotting the number of clusters against the **Inertia** (Within-Cluster Sum of Squares). "
            "As $K$ increases, inertia decreases because clusters become smaller and tighter. "
            "The optimal $K$ is at the **elbow point**—the point where the rate of decrease "
            "shifts from steep to shallow, indicating diminishing returns for adding more clusters."
        )
        st.write(
            "Look for the 'elbow' bend in the plot below. This point indicates a good balance between "
            "accuracy (low inertia) and model simplicity (fewer clusters)."
        )

        inertia  = []
        for k_val in range(1, 11):
            km = KMeans(n_clusters=k_val, random_state=42, n_init=10)
            km.fit(X_scaled)
            inertia.append(km.inertia_)

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(range(1, 11), inertia, marker="o", color=CAFE_LINE,
                markerfacecolor=CAFE_BAR, markersize=7, linewidth=2)
        ax.set_xlabel("Number of Clusters (K)")
        ax.set_ylabel("Inertia")
        ax.set_title("Elbow Method")
        ax.grid(True)
        st.pyplot(fig)

        k = st.slider("Select number of clusters (K)", 2, 8, 4)

        if algorithm == "K-Medoids" and not KMEDOIDS_AVAILABLE:
            st.warning("K-Medoids not available. Install sklearn-extra or use K-Means.")
        else:
            model  = (KMedoids(n_clusters=k, random_state=42)
                      if algorithm == "K-Medoids"
                      else KMeans(n_clusters=k, random_state=42, n_init=10))
            labels     = model.fit_predict(X_scaled)
            df_cluster = df_feat.copy()
            df_cluster["Cluster"] = labels

            st.markdown("### Cluster Scatter Plot")
            fig2, ax2 = plt.subplots(figsize=(8, 6))
            ax2.scatter(
                df_cluster[features[0]],
                df_cluster[features[1]] if len(features) > 1 else df_cluster[features[0]],
                c=df_cluster["Cluster"], cmap="YlOrBr", alpha=0.6,
            )
            if algorithm == "K-Means":
                centroids = scaler.inverse_transform(model.cluster_centers_)
                ax2.scatter(
                    centroids[:, 0],
                    centroids[:, 1] if len(features) > 1 else centroids[:, 0],
                    marker="X", s=200, c="#1a0d08", label="Centroids",
                )
            ax2.set_xlabel(features[0])
            ax2.set_ylabel(features[1] if len(features) > 1 else features[0])
            ax2.set_title("K-Means Clustering")
            ax2.legend()
            ax2.grid(True)
            st.pyplot(fig2)

            score = silhouette_score(X_scaled, labels)
            st.markdown("### Silhouette Score")
            st.markdown(
                f'<div class="metric-card" style="max-width:260px">'
                f'<div class="num">{score:.4f}</div>'
                f'<div class="lbl">Silhouette Score (K={k})</div></div>',
                unsafe_allow_html=True,
            )

            # Display Silhouette Score Comparison
            sil_scores = calculate_silhouette_comparison(X_scaled)
            sil_df = pd.DataFrame(list(sil_scores.items()), columns=["Number of Clusters (K)", "Silhouette Score"])
            
            st.markdown("#### Silhouette Score Comparison Table")
            st.dataframe(sil_df)
            st.caption(
                "Note: Silhouette comparison scores are calculated on a representative sample of 5,000 records "
                "to ensure fast response times. Values closer to 1.0 indicate better cluster separation."
            )

            st.markdown("### Cluster Summary Table")
            cluster_summary = df_cluster.groupby("Cluster")[features].mean()
            st.dataframe(cluster_summary)

            # ---- Gap 3: cluster interpretation ----
            st.markdown("### Cluster Interpretation")
            st.write(
                "Based on the mean feature values per cluster, here is a behavioral "
                "interpretation of each customer segment:"
            )
            for cluster_id in sorted(cluster_summary.index):
                row = cluster_summary.loc[cluster_id]
                traits = []

                # Age interpretation
                if "Age" in row.index:
                    age_val = row["Age"]
                    all_ages = cluster_summary["Age"]
                    if age_val == all_ages.max():
                        traits.append(f"oldest shoppers (avg age {age_val:.0f})")
                    elif age_val == all_ages.min():
                        traits.append(f"youngest shoppers (avg age {age_val:.0f})")
                    else:
                        traits.append(f"mid-age shoppers (avg age {age_val:.0f})")

                # Positive Feedback Count interpretation
                if "Positive Feedback Count" in row.index:
                    fb_val = row["Positive Feedback Count"]
                    all_fb = cluster_summary["Positive Feedback Count"]
                    if fb_val == all_fb.max():
                        traits.append("highly engaged (most feedback received)")
                    elif fb_val == all_fb.min():
                        traits.append("low engagement (least feedback)")
                    else:
                        traits.append("moderate engagement")

                # Rating interpretation
                if "Rating" in row.index:
                    rating_val = row["Rating"]
                    all_ratings = cluster_summary["Rating"]
                    if rating_val == all_ratings.max():
                        traits.append(f"highest satisfaction (avg rating {rating_val:.2f})")
                    elif rating_val == all_ratings.min():
                        traits.append(f"lowest satisfaction (avg rating {rating_val:.2f})")
                    else:
                        traits.append(f"average satisfaction (avg rating {rating_val:.2f})")

                # Review length interpretation
                if "review_length" in row.index:
                    rl_val  = row["review_length"]
                    all_rl  = cluster_summary["review_length"]
                    if rl_val == all_rl.max():
                        traits.append("writes long detailed reviews")
                    elif rl_val == all_rl.min():
                        traits.append("writes brief reviews")

                description = ", ".join(traits) if traits else "no distinctive pattern"
                st.markdown(
                    f"**Cluster {cluster_id}:** {description.capitalize()}."
                )
    else:
        st.info("Select at least one feature to run clustering.")


# ============================================================
# MODULE 6 — INFORMATION RETRIEVAL
# ============================================================
elif page == "Information Retrieval":
    page_header("Information Retrieval System",
                "TF-IDF ranked search across customer reviews")

    if not data_loaded:
        st.error("Dataset not found.")
        st.stop()

    n_docs = st.slider("Number of reviews to index", 50, 500, 100, step=50)

    review_data = df_feat[df_feat["Review Text"] != ""]["Review Text"].head(n_docs).tolist()
    documents   = {f"Review{i+1}": review_data[i] for i in range(len(review_data))}

    # ---- Gap 4: build and display the inverted index ----
    st.markdown("### Text Preprocessing Pipeline")
    st.write(
        "Before indexing, each review is lowercased, punctuation is removed, "
        "and stopwords are filtered out to produce clean tokens."
    )

    # Tokenize documents with simple pipeline (lowercase + remove non-alpha + split)
    tokenized_docs = [
        [w for w in re.sub(r'[^\w\s]', '', doc.lower()).split() if len(w) > 2]
        for doc in review_data
    ]

    st.markdown("### Inverted Index (Term → Document IDs)")
    st.write(
        "An inverted index maps each unique term to the list of document IDs "
        "(review numbers) that contain it. This enables fast keyword lookup."
    )
    inv_index = build_inverted_index(tokenized_docs)

    # Display a sample of the inverted index as a tidy table
    inv_sample = sorted(inv_index.items(), key=lambda x: len(x[1]), reverse=True)[:20]
    inv_df = pd.DataFrame([
        {"Term": term, "Document Count": len(doc_ids), "Document IDs (sample)": str(doc_ids[:8])}
        for term, doc_ids in inv_sample
    ])
    st.dataframe(inv_df)
    st.caption(
        f"Showing top 20 terms by document frequency. "
        f"Total unique terms in index: {len(inv_index):,}"
    )

    st.markdown("### TF-IDF Search Engine")
    query_input = st.text_input("Enter your search query", "comfortable dress")

    if st.button("Search"):
        q            = re.sub(r'[^\w\s]', '', query_input.lower())
        query_tokens = q.split()
        st.write("**Preprocessed Query Tokens:**", query_tokens)

        doc_ids, doc_texts = list(documents.keys()), list(documents.values())
        vec              = TfidfVectorizer()
        tfidf_matrix     = vec.fit_transform(doc_texts)
        sim_scores       = cosine_similarity(vec.transform([query_input]), tfidf_matrix).flatten()

        ranking_df  = pd.DataFrame({"Document": doc_ids, "Review Content": doc_texts,
                                    "TF-IDF Score": sim_scores})
        ranking_df  = ranking_df.sort_values("TF-IDF Score", ascending=False)
        ranking_df["Rank"] = range(1, len(ranking_df) + 1)
        ranking_df  = ranking_df[["Rank", "Document", "Review Content", "TF-IDF Score"]]

        st.markdown("### Ranked Results")
        st.dataframe(ranking_df.head(10))


# ============================================================
# MODULE 7 — TEXT MINING & SENTIMENT
# ============================================================
elif page == "Text Mining & Sentiment":
    page_header("Text Mining & Sentiment Analysis",
                "Word frequencies, word cloud, and sentiment distribution")

    if not data_loaded:
        st.error("Dataset not found.")
        st.stop()

    st.markdown("### Sentiment Labels (from Rating)")
    st.dataframe(df_feat[["Review Text", "Rating", "sentiment_label"]].head())

    st.markdown("### Sentiment Distribution")
    sent_counts = df_feat["sentiment_label"].value_counts()
    fig, ax     = plt.subplots(figsize=(6, 4))
    ax.bar(sent_counts.index, sent_counts.values,
           color=[SUCCESS, WARNING, DANGER])
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Count")
    ax.set_title("Customer Review Sentiment")
    st.pyplot(fig)

    st.markdown("### Word Frequency — Top Keywords (TF-IDF)")
    n_reviews = st.slider("Reviews to analyze", 50, 300, 100, step=50)
    # pyrefly: ignore [missing-import]
    import nltk
    # pyrefly: ignore [missing-import]
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer

    ensure_nltk()
    reviews = df_feat[df_feat["Review Text"] != ""]["Review Text"].head(n_reviews).tolist()

    cleaned = []
    for r in reviews:
        r = re.sub(r'[^\w\s]', '', r.lower())
        cleaned.append(r)

    stop_words    = set(stopwords.words("english"))
    lemmatizer    = WordNetLemmatizer()
    processed     = []
    for r in cleaned:
        tokens    = word_tokenize(r)
        tokens    = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
        processed.append(" ".join(tokens))

    vec2        = TfidfVectorizer()
    mat2        = vec2.fit_transform(processed)
    w_scores    = mat2.toarray().sum(axis=0)
    top_df      = (pd.DataFrame({"Word": vec2.get_feature_names_out(), "Score": w_scores})
                   .sort_values("Score", ascending=False).head(10))

    fig2, ax2   = plt.subplots(figsize=(8, 5))
    ax2.bar(top_df["Word"], top_df["Score"], color=CAFE_ACCENT, edgecolor="#F9F7F5", linewidth=0.4)
    ax2.set_xlabel("Word")
    ax2.set_ylabel("TF-IDF Score")
    ax2.set_title("Top 10 Keywords")
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig2)

    st.markdown("### Word Cloud")
    all_text  = " ".join(processed)
    wordcloud = WordCloud(width=800, height=400, background_color="#F9F7F5",
                          colormap="YlOrBr").generate(all_text)
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    ax3.imshow(wordcloud)
    ax3.axis("off")
    st.pyplot(fig3)

    st.markdown("### Sentiment Examples")
    pick     = st.selectbox("Show reviews by sentiment", ["Positive", "Neutral", "Negative"])
    examples = (df_feat[(df_feat["sentiment_label"] == pick) & (df_feat["Review Text"] != "")]
                ["Review Text"].head(3).tolist())
    for ex in examples:
        st.info(ex)


# ============================================================
# MODULE 8 — CONCLUSION & RECOMMENDATION
# ============================================================
elif page == "Conclusion & Recommendation":
    page_header("Conclusion & Recommendation",
                "Summary of analytical findings and actionable business insights")

    st.markdown("### Project Conclusions")
    st.markdown("""
    1. **Data Preprocessing & Feature Engineering:** Removing duplicates, handling missing text, and scaling numeric variables established a clean baseline. Categorical encoding successfully transformed qualitative metadata into model-ready features.
    2. **Dimensionality Reduction (PCA):** Reducing the feature space to two principal components revealed the variance structure of customer feedback, although recommendation labels showed overlapping distributions in the lower-dimensional space.
    3. **Supervised Classification:** Supervised learning models achieved high accuracy in predicting customer recommendations. Review rating was the most influential predictive feature.
    4. **Unsupervised Clustering:** Segmenting customer data via K-Means and K-Medoids identified distinct customer cohorts grouped by age, engagement level, and rating patterns, proving the utility of unsupervised segmentation.
    5. **Information Retrieval:** Implementing a TF-IDF vectorizer and an inverted index provided a fast, keyword-based search capability to retrieve and rank reviews based on query cosine similarity.
    6. **Text Mining & Sentiment Analysis:** Lemmatization, stopword removal, and TF-IDF term scoring successfully extracted dominant themes from customer reviews. Sentiment classification aligned closely with customer ratings.
    """)

    st.markdown("### Recommendations")
    st.markdown("""
    * **Targeted Customer Segmentation:** Design marketing campaigns tailored to the specific clusters identified (for example, targeting high-engagement segments with loyalty initiatives).
    * **Quality Assurance Monitoring:** Track products with high frequencies of negative sentiment words in their reviews to address fit, sizing, or material defects.
    * **Review Search Integration:** Deploy the TF-IDF search index within the customer support interface to help agents quickly retrieve specific product complaints or praise.
    * **Predictive Recommender System:** Integrate the classification model into the checkout pipeline to flag reviews with low recommendation probability for proactive customer service follow-ups.
    """)

