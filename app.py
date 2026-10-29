
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import html

# ==================== CONFIGURATION ====================

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "clothing.csv"

st.set_page_config(
    page_title="StyleMuse | Fashion AI",
    page_icon="👗",
    layout="wide"
)

# ==================== PROFESSIONAL UI DESIGN ====================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');

:root {
    --primary: #ed8b4b;
    --primary-dark: #cf7135;
    --background: #fffaf6;
    --border: #f0e1d6;
    --text: #292522;
    --muted: #786e67;
}

.stApp, [data-testid="stAppViewContainer"] {
    background: var(--background);
    color: var(--text);
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stHeader"] { background: transparent; }

.block-container {
    max-width: 1180px;
    padding: 2rem 2rem 3rem;
}

h1, h2, h3, h4 {
    font-family: 'DM Sans', sans-serif !important;
    color: var(--text) !important;
    letter-spacing: -0.4px;
    line-height: 1.4 !important;
}

h1 { font-size: 2.35rem !important; font-weight: 700 !important; }
h2 { font-size: 1.65rem !important; font-weight: 700 !important; }
h3 { font-size: 1.15rem !important; font-weight: 650 !important; }

p {
    color: #514943;
    font-size: 0.94rem;
    line-height: 1.7;
}

label { color: #443a34; font-weight: 500; }

.hero {
    background: linear-gradient(120deg, #ffe6d5 0%, #ffbd91 100%);
    padding: 34px 36px;
    border: 1px solid #f7d0b7;
    border-radius: 24px;
    margin-bottom: 28px;
    box-shadow: 0 8px 24px rgba(125, 77, 44, 0.06);
}

.hero h1 {
    color: #30241e !important;
    font-size: 2.5rem !important;
    margin: 8px 0 10px !important;
}

.hero p { color: #634b3d; max-width: 680px; margin: 0; }

div[data-testid="stHorizontalBlock"] {
    gap: 1.1rem;
    align-items: stretch;
}

.item-card {
    background: #ffffff;
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 20px;
    min-height: 235px;
    height: 100%;
    box-shadow: 0 4px 16px rgba(69, 45, 29, 0.045);
    overflow-wrap: anywhere;
    transition: all 0.2s ease;
}

.item-card:hover {
    border-color: #edbd9c;
    box-shadow: 0 8px 24px rgba(69, 45, 29, 0.09);
    transform: translateY(-2px);
}

.item-visual {
    width: 100%;
    min-height: 125px;
    border-radius: 14px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 3.4rem;
    margin-bottom: 16px;
}

.item-card h3 { font-size: 1.08rem !important; margin: 8px 0 !important; }
.item-card p { font-size: 0.88rem; margin: 10px 0; }

.pill {
    display: inline-block;
    background: #fff5ed;
    color: #8c4d29;
    border: 1px solid #f1d0ba;
    border-radius: 999px;
    padding: 5px 11px;
    margin: 3px 5px 3px 0;
    font-size: 0.78rem;
}

.score {
    color: #b45f2b;
    font-size: 0.95rem;
    font-weight: 700;
    margin-top: 12px;
}

[data-testid="stSidebar"] {
    background: #fff3e9;
    border-right: 1px solid #f0e0d4;
}

[data-testid="stSidebar"] > div { padding-top: 1.5rem; }

div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    border-radius: 11px;
    border-color: #e8d8cd;
    background: #ffffff;
    min-height: 42px;
}

.stButton > button {
    background: var(--primary);
    color: #ffffff;
    border: 1px solid var(--primary);
    border-radius: 12px;
    min-height: 43px;
    padding: 0.55rem 1.1rem;
    font-family: 'DM Sans', sans-serif;
    font-weight: 650;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    background: var(--primary-dark);
    color: #ffffff;
    border-color: var(--primary-dark);
}

[data-testid="stFileUploader"] {
    background: #fffaf6;
    border: 1px dashed #e9b998;
    border-radius: 14px;
    padding: 12px;
}

[data-testid="stExpander"] {
    background: #ffffff;
    border: 1px solid var(--border);
    border-radius: 14px;
}

hr {
    border: none;
    border-top: 1px solid #eaded5;
    margin: 1.5rem 0;
}

small, [data-testid="stCaptionContainer"] {
    color: var(--muted);
    font-size: 0.82rem;
}

@media (max-width: 768px) {
    .block-container { padding: 1rem; }
    .hero { padding: 24px 20px; border-radius: 18px; }
    .hero h1 { font-size: 1.9rem !important; }
    .item-card { padding: 14px; min-height: 200px; }
    .item-visual { min-height: 100px; font-size: 2.8rem; }
}
</style>
""", unsafe_allow_html=True)


# ==================== LOAD DATA ====================

@st.cache_data
def load_data():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"clothing.csv file nahi mili: {DATA_FILE}"
        )

    df = pd.read_csv(DATA_FILE)

    required_columns = [
        "category", "color", "style", "occasion", "season",
        "item_id", "item_name", "color_hex", "description"
    ]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "clothing.csv mein ye columns missing hain: "
            + ", ".join(missing_columns)
        )

    df = df.dropna(subset=["category", "item_name"]).copy()

    for column in required_columns:
        df[column] = df[column].fillna("").astype(str)

    return df


try:
    catalog = load_data()
except Exception as e:
    st.error(f"Dataset load nahi ho saka: {e}")
    st.info(
        "Check karein ke clothing.csv aur app.py ek hi folder mein hain."
    )
    st.stop()


# ==================== RECOMMENDATION ENGINE ====================

def recommend(df, category, color, style, occasion,
              season="All Season", top_n=6):

    if df.empty:
        return df.copy()

    features = df[
        ["category", "color", "style", "occasion", "season"]
    ].fillna("").astype(str)

    feature_text = features.apply(
        lambda row: " ".join(row.values), axis=1
    ).tolist()

    query = f"{category} {color} {style} {occasion}"

    if season != "All Season":
        query += f" {season}"

    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(feature_text + [query])

    scores = cosine_similarity(
        matrix[-1], matrix[:-1]
    ).flatten()

    # Preferences ke mutabiq extra matching points
    for i, (_, row) in enumerate(df.iterrows()):

        if row["color"].lower() == color.lower():
            scores[i] += 0.18

        if row["style"].lower() == style.lower():
            scores[i] += 0.16

        if row["occasion"].lower() == occasion.lower():
            scores[i] += 0.12

        if (
            season != "All Season"
            and row["season"].lower() == season.lower()
        ):
            scores[i] += 0.10

        # Same clothing category ko kam priority dein
        if row["category"].lower() == category.lower():
            scores[i] -= 0.08

    # Result ko copy karein; original catalog change nahi hoga
    result = df.copy()
    result["match_score"] = (scores * 100).round().astype(int)

    # Selected item category ke bajaye complementary items dikhayein
    result = result[
        result["category"].str.lower() != category.lower()
    ]

    return result.sort_values(
        "match_score", ascending=False
    ).head(top_n)


# ==================== HERO SECTION ====================

st.markdown("""
<div class="hero">
  <div style="font-size:.85rem;letter-spacing:2px;
       text-transform:uppercase;font-weight:700;color:#8b4c27;">
       YOUR PERSONAL STYLE ASSISTANT
  </div>
  <h1>Find your perfect look ✨</h1>
  <p>
    Choose a clothing piece, set your style preferences,
    and discover combinations made for you.
  </p>
</div>
""", unsafe_allow_html=True)


# ==================== SIDEBAR ====================

with st.sidebar:
    st.markdown("## 👗 Style preferences")
    st.caption("Tell us what you're styling today.")

    categories = sorted(catalog["category"].unique().tolist())

    category = st.selectbox(
        "Your clothing item",
        categories,
        index=0
    )

    color = st.selectbox(
        "Main color",
        ["Black", "White", "Beige", "Blue", "Brown",
         "Pink", "Red", "Green", "Cream", "Orange"]
    )

    style = st.selectbox(
        "Your style",
        ["Casual", "Formal", "Party", "Traditional",
         "Streetwear", "Minimal"]
    )

    occasion = st.selectbox(
        "Occasion",
        ["Everyday", "Office", "Brunch", "Party",
         "Wedding", "Date Night"]
    )

    season = st.selectbox(
        "Season",
        ["All Season", "Summer", "Winter", "Spring", "Autumn"]
    )

    top_n = st.slider("Recommendations", 3, 9, 6)

    uploaded = st.file_uploader(
        "Optional: upload your clothing photo",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded:
        st.image(
            uploaded,
            caption="Your selected clothing item",
            use_container_width=True
        )
        st.caption(
            "Photo preview only. Automatic image recognition "
            "is not enabled in this version."
        )

    find_style = st.button(
        "✨ Find my outfits",
        use_container_width=True
    )


# ==================== MAIN CONTENT ====================

st.markdown("### Build your look")
st.write(
    "Choose a few details in the sidebar, then discover matching pieces."
)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="item-card">
      <div class="item-visual" style="background:#fbe4d3">👚</div>
      <b>Color harmony</b><br>
      <small>Coordinate complementary shades.</small>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="item-card">
      <div class="item-visual" style="background:#f8d7c0">🧥</div>
      <b>Style matching</b><br>
      <small>Keep your look cohesive.</small>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="item-card">
      <div class="item-visual" style="background:#fcebdc">👜</div>
      <b>Complete the outfit</b><br>
      <small>Add shoes and accessories.</small>
    </div>
    """, unsafe_allow_html=True)


# ==================== SHOW RECOMMENDATIONS ====================

if find_style:

    results = recommend(
        catalog,
        category,
        color,
        style,
        occasion,
        season,
        top_n
    )

    st.markdown("---")
    st.markdown("## Your recommended pieces")

    st.markdown(
        f'<span class="pill">{html.escape(color)}</span>'
        f'<span class="pill">{html.escape(style)}</span>'
        f'<span class="pill">{html.escape(occasion)}</span>'
        f'<span class="pill">{html.escape(season)}</span>',
        unsafe_allow_html=True
    )

    if results.empty:
        st.info(
            "No complementary items found. Try another clothing category "
            "or add more items to clothing.csv."
        )

    else:
        cols = st.columns(3)

        icons = {
            "Top": "👚",
            "Dress": "👗",
            "Jacket": "🧥",
            "Trousers": "👖",
            "Skirt": "🩱",
            "Shoes": "👟",
            "Bag": "👜",
            "Accessories": "💍",
            "Scarf": "🧣",
            "Jeans": "👖"
        }

        for idx, (_, item) in enumerate(results.iterrows()):

            with cols[idx % 3]:

                bg = str(item.get("color_hex", "#f8e4d7"))

                # Basic validation for CSS background color
                if not (
                    bg.startswith("#")
                    and len(bg) in (4, 7)
                    and all(c in "0123456789abcdefABCDEF"
                            for c in bg[1:])
                ):
                    bg = "#f8e4d7"

                icon = icons.get(item["category"], "✨")

                item_category = html.escape(item["category"])
                item_name = html.escape(item["item_name"])
                item_color = html.escape(item["color"])
                item_style = html.escape(item["style"])
                item_description = html.escape(item["description"])

                st.markdown(f"""
                <div class="item-card">
                  <div class="item-visual" style="background:{bg};">
                    {icon}
                  </div>
                  <div style="font-size:.75rem;color:#a45d32;
                       text-transform:uppercase;letter-spacing:1px;">
                    {item_category}
                  </div>
                  <h3>{item_name}</h3>
                  <span class="pill">{item_color}</span>
                  <span class="pill">{item_style}</span>
                  <p>{item_description}</p>
                  <div class="score">
                    ✦ {item['match_score']}% style match
                  </div>
                </div>
                """, unsafe_allow_html=True)

                if st.button(
                    "♡ Save to favorites",
                    key=f"fav_{item['item_id']}_{idx}"
                ):
                    saved = st.session_state.get("favorites", [])

                    if item["item_name"] not in saved:
                        saved.append(item["item_name"])

                    st.session_state["favorites"] = saved
                    st.success("Saved to favorites!")


    # ==================== FAVORITES ====================

    st.markdown("### Your saved favorites")

    favorites = st.session_state.get("favorites", [])

    if favorites:
        for favorite in favorites:
            st.markdown(f"- ♡ {favorite}")
    else:
        st.caption(
            "Save recommendations you like and they will appear here "
            "during this session."
        )


# ==================== HOW IT WORKS ====================

with st.expander("How does the recommendation work?"):
    st.write(
        "The app compares category, color, style, occasion, and season "
        "using TF-IDF text features and cosine similarity. Matching color, "
        "style, occasion, and season receive extra weight. This is a "
        "content-based recommendation demo, not a trained deep-learning "
        "vision model."
    )


# ==================== FOOTER ====================

st.markdown("---")

st.markdown("""
<div style="text-align:center;color:#8b7b70;
     font-size:.85rem;padding:8px">
    StyleMuse · Find combinations that feel like you.
</div>
""", unsafe_allow_html=True)
