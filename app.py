import re
import joblib
import streamlit as st
import nltk

# Download required NLTK resources
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="CineSense",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM COLORS AND STYLE
# =========================================================

st.markdown("""
<style>

    /* ==============================
       MAIN WEBSITE
       ============================== */

    .stApp {
        background-color: #071A2B;
        color: #F8FAFC;
    }

    .block-container {
        max-width: 1150px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }


    /* ==============================
       SIDEBAR
       ============================== */

    [data-testid="stSidebar"] {
        background-color: #0D2B3E;
        border-right: 1px solid #1E475B;
    }


    /* ==============================
       HEADINGS
       ============================== */

    h1 {
        color: #F8FAFC !important;
        font-size: 46px !important;
        font-weight: 800 !important;
        letter-spacing: -1px;
    }

    h2 {
        color: #F8FAFC !important;
        font-weight: 750 !important;
    }

    h3 {
        color: #E2E8F0 !important;
    }


    /* ==============================
       NORMAL TEXT
       ============================== */

    p {
        color: #CBD5E1;
    }


    /* ==============================
       SIDEBAR TEXT
       ============================== */

    [data-testid="stSidebar"] p {
        color: #CBD5E1;
    }


    /* ==============================
       TEXT AREA
       ============================== */

    textarea {
        background-color: #0D2B3E !important;
        color: #F8FAFC !important;

        border: 1px solid #24566B !important;
        border-radius: 16px !important;

        font-size: 16px !important;

        padding: 15px !important;
    }

    textarea:focus {
        border: 1px solid #14B8A6 !important;

        box-shadow:
            0 0 0 1px #14B8A6 !important;
    }


    /* ==============================
       PRIMARY BUTTON
       ============================== */

    .stButton > button[kind="primary"] {

        background: linear-gradient(
            90deg,
            #0F766E,
            #14B8A6
        );

        color: #FFFFFF;

        border: none;

        border-radius: 12px;

        height: 50px;

        font-size: 16px;

        font-weight: 700;

        transition: all 0.2s ease;
    }

    .stButton > button[kind="primary"]:hover {

        background: linear-gradient(
            90deg,
            #14B8A6,
            #2DD4BF
        );

        transform: translateY(-2px);

        box-shadow:
            0 8px 20px
            rgba(20, 184, 166, 0.25);
    }


    /* ==============================
       NORMAL BUTTONS
       ============================== */

    .stButton > button {

        background-color: #12384B;

        color: #E2E8F0;

        border: 1px solid #24566B;

        border-radius: 12px;

        min-height: 45px;

        font-size: 15px;

        font-weight: 600;

        transition: all 0.2s ease;
    }

    .stButton > button:hover {

        background-color: #16465A;

        border-color: #14B8A6;

        color: #FFFFFF;

        transform: translateY(-1px);
    }


    /* ==============================
       METRIC CARDS
       ============================== */

    [data-testid="stMetric"] {

        background-color: #0D2B3E;

        border: 1px solid #1E475B;

        padding: 18px;

        border-radius: 16px;

        box-shadow:
            0 5px 20px
            rgba(0, 0, 0, 0.18);
    }

    [data-testid="stMetricLabel"] {
        color: #94A3B8 !important;
    }

    [data-testid="stMetricValue"] {
        color: #F8FAFC !important;
    }


    /* ==============================
       ALERTS
       ============================== */

    [data-testid="stAlert"] {

        border-radius: 15px;

        padding: 18px;

        font-size: 16px;
    }


    /* ==============================
       PROGRESS BAR
       ============================== */

    [data-testid="stProgress"] > div {

        background-color: #18384A;
    }

    [data-testid="stProgress"] > div > div {

        background: linear-gradient(
            90deg,
            #14B8A6,
            #F59E0B
        );
    }


    /* ==============================
       EXPANDER
       ============================== */

    [data-testid="stExpander"] {

        background-color: #0D2B3E;

        border: 1px solid #1E475B;

        border-radius: 14px;
    }


    /* ==============================
       DIVIDERS
       ============================== */

    hr {
        border-color: #1E475B !important;
    }


    /* ==============================
       CAPTIONS
       ============================== */

    .stCaption {
        color: #94A3B8 !important;
    }

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MODEL AND VECTORIZER
# =========================================================

@st.cache_resource
def load_model():

    model = joblib.load("best_model.pkl")

    vectorizer = joblib.load(
        "tfidf_vectorizer.pkl"
    )

    return model, vectorizer


model, vectorizer = load_model()


# =========================================================
# LOAD NLP TOOLS
# =========================================================

@st.cache_resource
def load_nlp_tools():

    stop_words = set(
        stopwords.words("english")
    )

    # Preserve important negative words
    stop_words.discard("no")
    stop_words.discard("not")
    stop_words.discard("nor")

    lemmatizer = WordNetLemmatizer()

    return stop_words, lemmatizer


stop_words, lemmatizer = load_nlp_tools()


# =========================================================
# TEXT PREPROCESSING
# =========================================================

def preprocess_text(text):

    # Convert to lowercase
    text = str(text).lower()

    # Remove HTML tags
    text = re.sub(
        r"<.*?>",
        " ",
        text
    )

    # Remove special characters and numbers
    text = re.sub(
        r"[^a-zA-Z\s]",
        " ",
        text
    )

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    # Tokenize
    words = text.split()

    cleaned_words = []

    for word in words:

        if word not in stop_words:

            cleaned_words.append(
                lemmatizer.lemmatize(word)
            )

    return " ".join(cleaned_words)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title("🎬 CineSense")

    st.caption(
        "Movie Review Sentiment Analysis"
    )

    st.divider()

    st.subheader("📌 How It Works")

    st.write(
        "① Enter your movie review"
    )

    st.write(
        "② Clean and preprocess the text"
    )

    st.write(
        "③ Convert text using TF-IDF"
    )

    st.write(
        "④ Analyze using the ML model"
    )

    st.write(
        "⑤ Display sentiment and confidence"
    )

    st.divider()

    st.subheader("💡 Try an Example")

    positive_example = (
        "This movie was absolutely fantastic! "
        "The story was brilliant and the acting "
        "was excellent."
    )

    negative_example = (
        "This movie was boring and disappointing. "
        "The story was weak and the acting "
        "was terrible."
    )

    if st.button(
        "😊 Positive Review",
        use_container_width=True
    ):

        st.session_state.review = (
            positive_example
        )

    if st.button(
        "😞 Negative Review",
        use_container_width=True
    ):

        st.session_state.review = (
            negative_example
        )

    st.divider()

    st.subheader("🤖 Model Information")

    st.write(
        "Machine Learning Classifier"
    )

    st.write(
        "TF-IDF Feature Extraction"
    )

    st.write(
        "Binary Sentiment Classification"
    )


# =========================================================
# MAIN HEADER
# =========================================================

st.title("🎬 CineSense")

st.subheader(
    "Movie Review Sentiment Analysis"
)

st.caption(
    "Understand the sentiment behind every movie review."
)

st.divider()


# =========================================================
# PROJECT OVERVIEW
# =========================================================

st.header("✨ Project Overview")

st.write(
    "CineSense uses Natural Language Processing "
    "and Machine Learning to analyze movie reviews "
    "and classify them as Positive or Negative."
)


# =========================================================
# INFORMATION CARDS
# =========================================================

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "🤖 Technology",
        "NLP + ML"
    )

with col2:

    st.metric(
        "🔍 Feature Extraction",
        "TF-IDF"
    )

with col3:

    st.metric(
        "🎯 Classification",
        "Binary"
    )


st.divider()


# =========================================================
# REVIEW INPUT
# =========================================================

st.header("📝 Analyze Your Review")

st.caption(
    "Write or paste a movie review below."
)

review = st.text_area(
    "Movie Review",
    value=st.session_state.get(
        "review",
        ""
    ),
    height=200,
    placeholder=(
        "Example: I really enjoyed this movie. "
        "The story was engaging and the actors "
        "gave amazing performances."
    ),
    label_visibility="collapsed"
)


# =========================================================
# ACTION BUTTONS
# =========================================================

button_col1, button_col2, button_col3 = st.columns(
    [1, 1, 1]
)

with button_col1:

    analyze = st.button(
        "🔍 Analyze Sentiment",
        type="primary",
        use_container_width=True
    )

with button_col2:

    clear = st.button(
        "↻ Clear Review",
        use_container_width=True
    )


# =========================================================
# CLEAR REVIEW
# =========================================================

if clear:

    st.session_state.review = ""

    st.rerun()


# =========================================================
# SENTIMENT PREDICTION
# =========================================================

if analyze:

    if review.strip():

        with st.spinner(
            "🔍 Analyzing your review..."
        ):

            # Preprocess review
            cleaned_review = preprocess_text(
                review
            )

            # Convert to TF-IDF
            review_vector = vectorizer.transform(
                [cleaned_review]
            )

            # Predict sentiment
            prediction = model.predict(
                review_vector
            )[0]

            # Calculate confidence
            if hasattr(
                model,
                "predict_proba"
            ):

                probabilities = model.predict_proba(
                    review_vector
                )[0]

                confidence = (
                    max(probabilities) * 100
                )

            else:

                confidence = 100.0


        st.divider()

        # =================================================
        # PREDICTION RESULT
        # =================================================

        st.header("🎯 Prediction Result")

        result_col, confidence_col = st.columns(
            2
        )


        # =================================================
        # SENTIMENT
        # =================================================

        with result_col:

            if prediction == 1:

                st.success(
                    "😊 POSITIVE REVIEW\n\n"
                    "The model detected a positive "
                    "sentiment in this review."
                )

            else:

                st.error(
                    "😞 NEGATIVE REVIEW\n\n"
                    "The model detected a negative "
                    "sentiment in this review."
                )


        # =================================================
        # CONFIDENCE
        # =================================================

        with confidence_col:

            st.metric(
                "Prediction Confidence",
                f"{confidence:.2f}%"
            )

            st.progress(
                int(
                    min(
                        confidence,
                        100
                    )
                )
            )


        # =================================================
        # REVIEW STATISTICS
        # =================================================

        st.subheader(
            "📊 Review Analysis"
        )

        stat1, stat2, stat3 = st.columns(
            3
        )

        with stat1:

            st.metric(
                "Characters",
                len(review)
            )

        with stat2:

            st.metric(
                "Original Words",
                len(review.split())
            )

        with stat3:

            st.metric(
                "Processed Words",
                len(cleaned_review.split())
            )


        # =================================================
        # PROCESSED TEXT
        # =================================================

        with st.expander(
            "🔎 View Processed Text"
        ):

            st.write(
                cleaned_review
            )


    else:

        st.warning(
            "⚠️ Please enter a movie review "
            "before analyzing."
        )


# =========================================================
# ANALYSIS PIPELINE
# =========================================================

st.divider()

st.header("⚙️ Analysis Pipeline")

step1, step2, step3, step4 = st.columns(
    4
)


with step1:

    st.write("### 01")

    st.write(
        "**Review Input**"
    )

    st.caption(
        "Movie review is provided."
    )


with step2:

    st.write("### 02")

    st.write(
        "**Preprocessing**"
    )

    st.caption(
        "Text is cleaned and normalized."
    )


with step3:

    st.write("### 03")

    st.write(
        "**TF-IDF**"
    )

    st.caption(
        "Text is converted into features."
    )


with step4:

    st.write("### 04")

    st.write(
        "**Prediction**"
    )

    st.caption(
        "Sentiment is classified."
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "🎬 CineSense  •  Movie Review Sentiment Analysis  •  "
    "Python | NLTK | TF-IDF | Machine Learning | Streamlit"
)
