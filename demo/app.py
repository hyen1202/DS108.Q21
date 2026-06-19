import os
import json
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ==========================================
# CONSTANTS & CONFIGURATIONS
# ==========================================

# List of 24 entity labels in ViEcomNER schema
VEC_LABELS = [
    "PRODUCT_TYPE", "BRAND", "MODEL", "QUANTITY", "ORIGIN", "COMPONENT",
    "MATERIAL", "SIZE", "COMPAT", "ATTRIBUTE", "OCCASION", "EFFECT",
    "COLOR", "STYLE", "TARGET_GROUP", "SPEC", "CONNECTIVITY", "POWER",
    "CAPACITY", "INGREDIENT", "SKIN_TYPE", "BODY_PART", "VOLUME_WEIGHT",
    "CONCENTRATION"
]

# Highlighting color scheme for 24 labels (Pastel palette for readability)
LABEL_COLORS = {
    "PRODUCT_TYPE": "#ffadad",
    "BRAND": "#ffd6a5",
    "MODEL": "#fdffb6",
    "QUANTITY": "#caffbf",
    "SIZE": "#9bf6ff",
    "COLOR": "#a0c4ff",
    "MATERIAL": "#bdb2ff",
    "STYLE": "#ffc6ff",
    "SPEC": "#ffadad",
    "CONNECTIVITY": "#ffd6a5",
    "POWER": "#fdffb6",
    "CAPACITY": "#caffbf",
    "INGREDIENT": "#9bf6ff",
    "SKIN_TYPE": "#a0c4ff",
    "BODY_PART": "#bdb2ff",
    "VOLUME_WEIGHT": "#ffc6ff",
    "CONCENTRATION": "#ffadad",
    "ATTRIBUTE": "#ffd6a5",
    "OCCASION": "#fdffb6",
    "EFFECT": "#caffbf",
    "ORIGIN": "#9bf6ff",
    "COMPONENT": "#a0c4ff",
    "COMPAT": "#bdb2ff",
    "TARGET_GROUP": "#ffc6ff"
}

# Set page configuration
st.set_page_config(
    page_title="ViEcomNER Dataset Explorer & Dashboard",
    page_icon="📁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for academic, clean, neutral styling
st.markdown("""
<style>
    /* Main body background & font */
    .stApp {
        background-color: #fafafa;
        color: #1e293b;
    }
    
    /* Justify align paragraphs and list items */
    .stApp p, .stApp li {
        text-align: justify;
    }
    
    /* Academic Headers */
    h1, h2, h3 {
        font-family: 'Outfit', 'Inter', sans-serif;
        font-weight: 700;
        color: #0f172a;
        letter-spacing: -0.025em;
    }
    
    h1 {
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 10px;
        margin-bottom: 25px;
    }
    
    /* Sleek metric container cards */
    div[data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }
    
    /* Custom tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #f1f5f9;
        padding: 6px 12px;
        border-radius: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 6px;
        color: #64748b;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        color: #0f172a;
        background-color: #e2e8f0;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #ffffff !important;
        color: #0f172a !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# DATA LOADING & GENERATING FUNCTIONS (CACHE)
# ==========================================

@st.cache_data
def load_cleaned_data():
    """Loads the preprocessed e-commerce titles dataset."""
    filepath = "data/processed/data_cleaned.csv"
    try:
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            # Ensure basic columns exist
            if "word_count" not in df.columns and "title_clean" in df.columns:
                df["word_count"] = df["title_clean"].astype(str).str.split().str.len()
            elif "word_count" not in df.columns and "title" in df.columns:
                df["word_count"] = df["title"].astype(str).str.split().str.len()
            return df, False
    except Exception as e:
        st.sidebar.warning(f"Error loading real cleaned dataset: {e}")
        
    # Falling back to high-fidelity Mock Data for Demo purposes
    np.random.seed(42)
    n = 2993
    platforms = np.random.choice(["lazada", "tiki"], p=[0.681, 0.319], size=n)
    categories = np.random.choice(
        ["home & living", "fashion & accessories", "electronics & tech", "beauty & health"],
        p=[0.303, 0.264, 0.280, 0.153],
        size=n
    )
    word_counts = np.random.negative_binomial(n=12, p=0.45, size=n) + 4
    brands = np.random.choice(["no brand", "samsung", "apple", "locknlock", "loreal", "adidas"], p=[0.55, 0.12, 0.08, 0.10, 0.07, 0.08], size=n)
    prices = np.random.randint(30000, 1500000, size=n)
    
    df_mock = pd.DataFrame({
        "id": [f"{platforms[i]}_{i:05d}" for i in range(n)],
        "platform": platforms,
        "industry_group": categories,
        "brand_clean": brands,
        "price": prices,
        "word_count": word_counts
    })
    return df_mock, True


@st.cache_data
def load_annotated_data():
    """Loads Gold Standard entities and JSONL records."""
    filepath = "data/annotated/full_dataset.jsonl"
    records = []
    try:
        if os.path.exists(filepath):
            with open(filepath, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        records.append(json.loads(line))
            if len(records) > 0:
                # Add default category and platform if missing
                for r in records:
                    if "platform" not in r:
                        r["platform"] = r.get("id", "tiki_").split("_")[0]
                    if "industry_group" not in r:
                        r["industry_group"] = "beauty & health" # default fallback
                return records, False
    except Exception as e:
        st.sidebar.warning(f"Error loading real annotated dataset: {e}")
        
    # Generate high-fidelity synthetic annotated titles for demo fallback
    mock_records = [
        {
            "id": "lazada_00001",
            "text": "Nồi cơm điện Lock&Lock EJR351BR 1.8L công suất 860W màu nâu kháng khuẩn",
            "entities": [
                {"start_char": 0, "end_char": 12, "label": "PRODUCT_TYPE"},
                {"start_char": 13, "end_char": 22, "label": "BRAND"},
                {"start_char": 23, "end_char": 32, "label": "MODEL"},
                {"start_char": 33, "end_char": 37, "label": "CAPACITY"},
                {"start_char": 47, "end_char": 51, "label": "POWER"},
                {"start_char": 52, "end_char": 59, "label": "COLOR"},
                {"start_char": 60, "end_char": 71, "label": "ATTRIBUTE"}
            ],
            "platform": "lazada",
            "industry_group": "home & living"
        },
        {
            "id": "tiki_00021",
            "text": "Serum Niacinamide 10% L'Oreal Paris Revitalift 30ml giảm thâm mụn cho da dầu",
            "entities": [
                {"start_char": 0, "end_char": 5, "label": "PRODUCT_TYPE"},
                {"start_char": 6, "end_char": 17, "label": "INGREDIENT"},
                {"start_char": 18, "end_char": 21, "label": "CONCENTRATION"},
                {"start_char": 22, "end_char": 35, "label": "BRAND"},
                {"start_char": 36, "end_char": 46, "label": "MODEL"},
                {"start_char": 47, "end_char": 51, "label": "VOLUME_WEIGHT"},
                {"start_char": 52, "end_char": 65, "label": "EFFECT"},
                {"start_char": 70, "end_char": 76, "label": "SKIN_TYPE"}
            ],
            "platform": "tiki",
            "industry_group": "beauty & health"
        },
        {
            "id": "lazada_00102",
            "text": "Điện thoại Samsung Galaxy S23 Ultra 8GB/256GB màn hình 120Hz sạc siêu nhanh",
            "entities": [
                {"start_char": 0, "end_char": 10, "label": "PRODUCT_TYPE"},
                {"start_char": 11, "end_char": 18, "label": "BRAND"},
                {"start_char": 19, "end_char": 35, "label": "MODEL"},
                {"start_char": 36, "end_char": 45, "label": "SPEC"},
                {"start_char": 55, "end_char": 60, "label": "SPEC"},
                {"start_char": 61, "end_char": 75, "label": "ATTRIBUTE"}
            ],
            "platform": "lazada",
            "industry_group": "electronics & tech"
        },
        {
            "id": "tiki_00455",
            "text": "Áo khoác gió thể thao nam Adidas màu xanh navy size L chống nước chống gió",
            "entities": [
                {"start_char": 0, "end_char": 12, "label": "PRODUCT_TYPE"},
                {"start_char": 13, "end_char": 21, "label": "STYLE"},
                {"start_char": 22, "end_char": 25, "label": "TARGET_GROUP"},
                {"start_char": 26, "end_char": 32, "label": "BRAND"},
                {"start_char": 37, "end_char": 46, "label": "COLOR"},
                {"start_char": 52, "end_char": 53, "label": "SIZE"},
                {"start_char": 54, "end_char": 64, "label": "ATTRIBUTE"},
                {"start_char": 65, "end_char": 74, "label": "ATTRIBUTE"}
            ],
            "platform": "tiki",
            "industry_group": "fashion & accessories"
        }
    ]
    
    # Programmatic Generation of 100 extra annotated records
    label_pool = {
        "home & living": [
            ("Nồi chiên không dầu", "PRODUCT_TYPE"), ("Philips", "BRAND"), ("HD9200", "MODEL"), 
            ("4.1L", "CAPACITY"), ("1400W", "POWER"), ("màu đen", "COLOR"), ("inox 304", "MATERIAL")
        ],
        "beauty & health": [
            ("Kem chống nắng", "PRODUCT_TYPE"), ("La Roche-Posay", "BRAND"), ("Anthelios", "MODEL"), 
            ("50ml", "VOLUME_WEIGHT"), ("kiềm dầu", "EFFECT"), ("da nhạy cảm", "SKIN_TYPE"), ("SPF50+", "CONCENTRATION")
        ],
        "electronics & tech": [
            ("Tai nghe không dây", "PRODUCT_TYPE"), ("Apple", "BRAND"), ("AirPods Pro 2", "MODEL"), 
            ("Bluetooth 5.3", "CONNECTIVITY"), ("chống ồn", "ATTRIBUTE"), ("1 cặp", "QUANTITY")
        ],
        "fashion & accessories": [
            ("Đầm voan hoa nhí", "PRODUCT_TYPE"), ("form xòe", "STYLE"), ("màu xanh", "COLOR"),
            ("size M", "SIZE"), ("cho nữ", "TARGET_GROUP"), ("chất mát", "ATTRIBUTE")
        ]
    }
    
    np.random.seed(108)
    for i in range(150):
        group = np.random.choice(list(label_pool.keys()))
        platform = np.random.choice(["tiki", "lazada"])
        parts = list(label_pool[group])
        
        # shuffle parts to simulate noisy titles
        np.random.shuffle(parts)
        text_parts = [p[0] for p in parts]
        title_text = " ".join(text_parts)
        
        # character offsets calculation
        entities = []
        current_pos = 0
        for val, lbl in parts:
            start = title_text.find(val, current_pos)
            if start != -1:
                end = start + len(val)
                entities.append({
                    "start_char": start,
                    "end_char": end,
                    "label": lbl
                })
                current_pos = end
        
        mock_records.append({
            "id": f"{platform}_{i+1000:05d}",
            "text": title_text,
            "entities": entities,
            "platform": platform,
            "industry_group": group
        })
        
    return mock_records, True


# ==========================================
# INITIALIZATION & LOAD DATA
# ==========================================

# Load datasets
df_cleaned, is_cleaned_mock = load_cleaned_data()
annotated_records, is_annotated_mock = load_annotated_data()

# Header title banner
st.title("📊 ViEcomNER: Dataset Analytics Dashboard")
st.caption("A Data-Centric AI Platform for Vietnamese E-commerce Named Entity Recognition (Class DS108 - Data Preprocessing)")

# Inform user of source
if is_cleaned_mock or is_annotated_mock:
    st.info("ℹ️ Running in **Demo Mode (Fallback Mock Data)**. Real dataset was not detected in `data/` directories, so synthetic records were dynamically simulated.")
else:
    st.success("⚡ Running in **Production Mode**. Successfully loaded active dataset from the project workspace.")

# Tab setup
tab_overview, tab_eda, tab_quality, tab_browser = st.tabs([
    "📂 Project Overview", 
    "📈 EDA Insights", 
    "🎯 Quality Assessment (IAA)", 
    "🔍 Data Browser & NER Highlighter"
])


# ==========================================
# TAB 1: OVERVIEW & DATA SHEET
# ==========================================
with tab_overview:
    st.subheader("1. General Project Information")
    
    # Metric cards layout
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.metric(label="Total Raw Crawled Records", value="19,361", delta="-16,368 after filtering")
    with m_col2:
        st.metric(label="Gold Standard Dataset Size", value="2,993 samples", delta="Train: 2,396 | Val: 298 | Test: 299")
    with m_col3:
        st.metric(label="Total Annotated Entities", value="18,837", delta="24 distinct labels")
    with m_col4:
        st.metric(label="Average Entities per Title", value="6.29", delta="Max density: 14")
        
    st.markdown("---")
    
    # Grid of details
    grid_col1, grid_col2 = st.columns([3, 2])
    
    with grid_col1:
        st.subheader("📚 Dataset Datasheet (Abstract & Motivation)")
        st.markdown("""
        **1. Why was ViEcomNER created?**
        Existing Vietnamese NER datasets (e.g. VLSP, PhoNER_COVID19) were built exclusively from formal texts (news articles, scientific writing). E-commerce product titles are highly informal, contain massive keyword stuffing, non-standard spelling, and complex English-Vietnamese code-mixing. ViEcomNER provides the first gold-standard, open-source dataset resolving this domain mismatch.
        
        **2. What is the architecture schema?**
        - **Tokenizer-Agnosticism:** Entities are stored as **character-level offsets** (`start_char`, `end_char`). This prevents the alignment shift errors introduced by Vietnamese word segmenters (which struggle on non-standard e-commerce vocabulary).
        - **Flat Schema Structure:** Designing 24 distinct labels (15 common across all domains, and 9 industry-specific) covering real-world business requirements.
        
        **3. Human-LLM Hybrid Annotation Strategy:**
        We initialized pre-annotations using Claude Sonnet 4.6 to boost labeling speed by 60%. To counter anchor bias (where human labelers blindly accept LLM outputs), we implemented a fallback workflow requiring double-blind review and manual label injection.
        """)
        
    with grid_col2:
        st.subheader("🏗️ Pipeline Workflow Diagram")
        # Mermaid-like simple visual layout
        st.info("""
        **Data-Centric AI Curation Pipeline:**
        1. **Raw Collection:** Tiki (31.9%) & Lazada (68.1%) crawling.
        2. **9-Step Automated Cleaning:** Deduplication, HTML stripping, NFC normalization, phone number masking (PII Protection), char TF-IDF near-deduplication (threshold >= 0.90).
        3. **Double-Blind Split:** 15% QC overlap rotation.
        4. **Annotation Quality Audit:** Kappa agreement calculation.
        5. **Stratified Sampling:** Balancing splits (Train, Val, Test).
        """)


# ==========================================
# LAYOUT & INTERACTIVE STYLING
# ==========================================

def make_gauge_chart(value, title, is_percent=False):
    max_val = 100.0 if is_percent else 1.0
    threshold_val = 80.0 if is_percent else 0.8
    steps = [
        {'range': [0, max_val * 0.4], 'color': '#fee2e2'},
        {'range': [max_val * 0.4, max_val * 0.6], 'color': '#fef3c7'},
        {'range': [max_val * 0.6, max_val * 0.8], 'color': '#d1fae5'},
        {'range': [max_val * 0.8, max_val], 'color': '#a7f3d0'}
    ]
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        number={
            'font': {'size': 26, 'color': '#0f172a', 'family': 'Outfit, Inter, sans-serif'},
            'valueformat': '.4f' if not is_percent else '.2f',
            'suffix': '%' if is_percent else ''
        },
        gauge={
            'axis': {'range': [0, max_val], 'tickwidth': 1, 'tickcolor': "#475569", 'tickfont': {'size': 9}},
            'bar': {'color': "#8b5cf6"},
            'bgcolor': "white",
            'borderwidth': 1,
            'bordercolor': "#cbd5e1",
            'steps': steps,
            'threshold': {
                'line': {'color': "#10b981", 'width': 3},
                'thickness': 0.75,
                'value': threshold_val
            }
        }
    ))
    fig.update_layout(
        height=130, 
        margin=dict(l=20, r=20, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={'color': "#1e293b", 'family': "sans-serif", 'size': 11}
    )
    return fig

with tab_eda:
    st.subheader("2. Exploratory Data Analysis & Statistics")
    
    # Compute label counts dynamically
    all_labels = []
    for rec in annotated_records:
        for ent in rec.get("entities", []):
            all_labels.append(ent["label"])
            
    df_labels = pd.Series(all_labels).value_counts().reset_index()
    df_labels.columns = ["Label", "Count"]
    
    # 1. Bar Chart: Label Distribution
    st.markdown("### 📊 Named Entity Class Imbalance Distribution")
    st.write(
        "*Methodological Rigor:* Plotting class frequency is vital to inspect model training challenges. "
        "ViEcomNER mimics actual e-commerce distributions, resulting in a severe class imbalance (up to 133:1 ratio "
        "between the common `PRODUCT_TYPE` label and rare tags like `SKIN_TYPE`)."
    )
    
    fig_labels = px.bar(
        df_labels, 
        x="Count", 
        y="Label", 
        orientation="h",
        color="Count",
        color_continuous_scale="Purples",
        height=600,
        text_auto=True
    )
    fig_labels.update_layout(
        yaxis={"categoryorder": "total ascending"},
        margin=dict(l=20, r=20, t=20, b=20),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig_labels, use_container_width=True)
    
    st.markdown("---")
    
    # 2. Columns layout for Platform, Length and Category Distributions
    col_dist1, col_dist2 = st.columns(2)
    
    with col_dist1:
        st.markdown("### 📁 Platform ➔ Category ➔ Label Hierarchy (Sunburst)")
        st.write(
            "*Methodological Rigor:* A Sunburst chart shows the hierarchical distribution of entities. "
            "Click on segments to drill down into platform (Tiki vs Lazada) and category splits."
        )
        
        # Build hierarchy dataframe dynamically
        hierarchy_rows = []
        for rec in annotated_records:
            plat = str(rec.get("platform", "Unknown")).upper()
            cat = str(rec.get("industry_group", "Other")).title()
            ents = rec.get("entities", [])
            if not ents:
                hierarchy_rows.append({
                    "Platform": plat,
                    "Industry Group": cat,
                    "Entity Label": "No Entity",
                    "Count": 1
                })
            else:
                for ent in ents:
                    hierarchy_rows.append({
                        "Platform": plat,
                        "Industry Group": cat,
                        "Entity Label": ent["label"],
                        "Count": 1
                    })
        df_hierarchy = pd.DataFrame(hierarchy_rows)
        
        fig_sunburst = px.sunburst(
            df_hierarchy, 
            path=["Platform", "Industry Group", "Entity Label"], 
            values="Count",
            color="Industry Group",
            color_discrete_sequence=px.colors.qualitative.Pastel,
            height=500
        )
        fig_sunburst.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_sunburst, use_container_width=True)
        
    with col_dist2:
        st.markdown("### 📝 Word Count Distribution in Titles")
        # Histogram of Word Counts
        fig_len = px.histogram(
            df_cleaned, 
            x="word_count", 
            nbins=30,
            color_discrete_sequence=["#a855f7"],
            labels={"word_count": "Number of Words"}
        )
        fig_len.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            yaxis_title="Title Frequency",
            xaxis_title="Word Count per Title",
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig_len, use_container_width=True)


# ==========================================
# TAB 3: QUALITY ASSESSMENT (IAA)
# ==========================================
with tab_quality:
    st.subheader("3. Inter-Annotator Agreement (IAA) & Error Analysis")
    
    col_iaa1, col_iaa2 = st.columns([1, 2])
    
    with col_iaa1:
        st.markdown("### 🎯 Key Agreement Metrics")
        
        st.markdown(
            "**Cohen's Kappa (Without O label) ℹ️**",
            help="Excluding 'O' tokens prevents inflated agreement scores due to the dominance of background text."
        )
        st.plotly_chart(make_gauge_chart(0.8207, "Kappa"), use_container_width=True)
        st.caption("Agreement interpretation: **Almost Perfect Agreement** (Landis & Koch)")
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown(
            "**Exact Span F1-Score ℹ️**",
            help="Measures percentage of entities where both start/end boundaries and labels match exactly."
        )
        st.plotly_chart(make_gauge_chart(81.67, "Exact F1", is_percent=True), use_container_width=True)
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown(
            "**Partial Match F1-Score ℹ️**",
            help="Measures percentage of overlapping boundaries with matching labels."
        )
        st.plotly_chart(make_gauge_chart(94.70, "Partial F1", is_percent=True), use_container_width=True)
        
        st.info(
            "💡 **Why the Gap between Exact and Partial F1?** "
            "The high Partial F1 (94.70%) compared to Exact F1 (81.67%) proves that annotators share a unified cognitive understanding "
            "of semantic labels. Mismatches are primarily boundary disputes caused by complex Vietnamese compound words."
        )
        
    with col_iaa2:
        st.markdown("### ⚠️ Primary Annotation Disagreements (Conflict Classification)")
        st.write(
            "*Methodological Rigor:* Analyzing conflict categories is crucial to audit guideline failures. "
            "A high rate of label errors indicates fuzzy tag boundaries, while boundary errors reveal word segmentation disputes."
        )
        
        conflict_data = {
            "Conflict Category": [
                "Label Error (Same bounds, different label)",
                "Boundary Error (Same label, overlapping bounds)",
                "Partial Overlap (Different bounds, different label)",
                "Missing Tag A (Annotator B labeled, A missed)",
                "Missing Tag B (Annotator A labeled, B missed)"
            ],
            "Frequency": [284, 452, 94, 381, 412],
            "Top Tag Pairs Involved": [
                "PRODUCT_TYPE ⟷ ATTRIBUTE, MODEL ⟷ BRAND",
                "SPEC, COMPAT, VOLUME_WEIGHT (Unit boundary)",
                "COMPONENT ⟷ PRODUCT_TYPE",
                "ATTRIBUTE, STYLE, TARGET_GROUP",
                "ATTRIBUTE, STYLE, TARGET_GROUP"
            ],
            "Resolution Rule Mapping": [
                "R01 (Drop-test rule), R10 (Brand vs Model rule)",
                "§5.2 (Numbers and Units must belong to one span)",
                "R09 (Check presence of anchor words like kèm/tặng)",
                "G14 (Fallback to Adjudication Queue if unsure)",
                "G14 (Fallback to Adjudication Queue if unsure)"
            ]
        }
        df_conflict = pd.DataFrame(conflict_data)
        st.dataframe(
            df_conflict,
            column_config={
                "Frequency": st.column_config.ProgressColumn(
                    "Frequency",
                    help="Number of conflict instances discovered",
                    format="%d",
                    min_value=0,
                    max_value=500
                )
            },
            use_container_width=True,
            hide_index=True
        )
        
    st.markdown("---")
    
    # IAA Growth Chart and Methodology
    st.markdown("### 📈 IAA Growth & Annotation History")
    st.write(
        "*Methodological Rigor:* The line chart below tracks the evolution of agreement metrics across annotation rounds. "
        "Notice the drop in Pilot Round 2 due to **Cognitive Overload** (when the schema was expanded to 24 labels), "
        "and the significant recovery in Pilot Round 3 and the Gold Set after introducing Guideline v3.2 (removing noisy categories like `USE_CASE`, `ROOM_USAGE`, and `SKIN_CONCERN`, and adding decision trees)."
    )
    
    rounds_data = pd.DataFrame({
        "Round": ["Pilot 1", "Pilot 2 (Overload)", "Pilot 3 (v3.2)", "Gold Test Set"],
        "Exact F1 (%)": [61.76, 53.76, 68.09, 81.67],
        "Boundary F1 (%)": [68.54, 62.53, 73.70, 86.84],
        "Partial F1 (%)": [84.61, 82.61, 88.02, 94.70],
        "Cohen's Kappa (No O)": [0.5709, 0.5313, 0.6518, 0.8207]
    })
    
    fig_iaa = go.Figure()
    fig_iaa.add_trace(go.Scatter(x=rounds_data["Round"], y=rounds_data["Exact F1 (%)"], name="Exact F1 (%)", mode="lines+markers", line=dict(color="#8b5cf6", width=3)))
    fig_iaa.add_trace(go.Scatter(x=rounds_data["Round"], y=rounds_data["Boundary F1 (%)"], name="Boundary F1 (%)", mode="lines+markers", line=dict(color="#3b82f6", width=2, dash="dash")))
    fig_iaa.add_trace(go.Scatter(x=rounds_data["Round"], y=rounds_data["Partial F1 (%)"], name="Partial F1 (%)", mode="lines+markers", line=dict(color="#10b981", width=2, dash="dot")))
    fig_iaa.add_trace(go.Scatter(x=rounds_data["Round"], y=rounds_data["Cohen's Kappa (No O)"] * 100, name="Cohen's Kappa (No O) x 100", mode="lines+markers", line=dict(color="#f59e0b", width=3)))
    
    fig_iaa.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        yaxis_title="Score (%) / Value",
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig_iaa, use_container_width=True)
    
    st.markdown("---")
    st.markdown("### 🛠️ Conflict Resolution Guideline Rules (R01 - R15)")
    
    exp1 = st.expander("🔍 Rule R01: PRODUCT_TYPE vs ATTRIBUTE (Drop-test rule)")
    exp1.write("Apply a **drop-test**: if you omit the modifier word, can you still understand what general product type is being sold? If YES, label the modifier as `ATTRIBUTE`. If NO (the word is required to identify the base product), label the entire span as `PRODUCT_TYPE` (e.g. *Nước rửa chén* vs *Áo thun [polo]_STYLE*).")
    
    exp2 = st.expander("🔍 Rule R09: COMPONENT vs PRODUCT_TYPE (Anchor word rule)")
    exp2.write("Check if there is an anchor word expressing inclusion (e.g. *kèm, tặng, +, đi kèm*). If YES, the words following the anchor are labeled as `COMPONENT`. If NO anchor word is present, label as `PRODUCT_TYPE` (e.g. *Nồi chiên tặng [vỉ nướng]_COMPONENT* vs *Nồi chiên vỉ nướng*).")
    
    exp3 = st.expander("🔍 Rule R10: MODEL vs BRAND (Organization Identifier rule)")
    exp3.write("Label corporate names/parent conglomerates as `BRAND`. Label specific product identifiers or version codes as `MODEL` (e.g. *[Samsung]_BRAND [Galaxy S23]_MODEL*).")


# ==========================================
# TAB 4: DATA BROWSER & HIGHLIGHTER
# ==========================================

def highlight_ner(text, entities):
    """Generates colored HTML tags enclosing NER spans in text."""
    sorted_entities = sorted(entities, key=lambda x: x["start_char"], reverse=True)
    html_text = text
    for ent in sorted_entities:
        start = ent["start_char"]
        end = ent["end_char"]
        label = ent["label"]
        span_text = text[start:end]
        
        color = LABEL_COLORS.get(label, "#e2e8f0")
        highlight = (
            f'<span style="background-color: {color}; padding: 2px 6px; margin: 0 2px; '
            f'border-radius: 4px; font-weight: 600; font-size: 0.9em; display: inline-block; '
            f'border: 1px solid rgba(0,0,0,0.15); color: #000000;">{span_text} '
            f'<span style="font-size: 0.7em; font-weight: 800; color: rgba(0,0,0,0.65); '
            f'vertical-align: middle; margin-left: 4px;">[{label}]</span></span>'
        )
        html_text = html_text[:start] + highlight + html_text[end:]
        
    return (
        f'<div style="line-height: 2.5; font-family: sans-serif; padding: 18px; '
        f'border-radius: 8px; border: 1px solid #cbd5e1; background-color: #f8fafc;">{html_text}</div>'
    )


with tab_browser:
    st.subheader("4. Gold Dataset Interactive Browser")
    st.write("Browse, query, and examine the character offsets and visual tag layouts of the final Gold dataset.")
    
    # Filter controllers
    f_col1, f_col2, f_col3, f_col4 = st.columns(4)
    
    with f_col1:
        f_industry = st.selectbox(
            "Filter by Industry Group",
            options=["All", "home & living", "beauty & health", "electronics & tech", "fashion & accessories"]
        )
    with f_col2:
        f_platform = st.selectbox(
            "Filter by E-commerce Platform",
            options=["All", "tiki", "lazada"]
        )
    with f_col3:
        # Dynamic label list from VEC_LABELS schema constant
        f_label = st.selectbox(
            "Filter by Entity Label",
            options=["All"] + VEC_LABELS
        )
    with f_col4:
        search_query = st.text_input("Search Titles", value="", placeholder="Enter keyword...")
        
    # Filtering logic
    filtered_records = annotated_records
    if f_industry != "All":
        filtered_records = [r for r in filtered_records if r.get("industry_group") == f_industry]
    if f_platform != "All":
        filtered_records = [r for r in filtered_records if r.get("platform") == f_platform]
    if f_label != "All":
        filtered_records = [r for r in filtered_records if any(e["label"] == f_label for e in r.get("entities", []))]
    if search_query.strip():
        q = search_query.lower()
        filtered_records = [r for r in filtered_records if q in r.get("text", "").lower()]
        
    # Push lazada_18477 to the end so it doesn't display at the top of the browser
    if len(filtered_records) > 0:
        target_id = "lazada_18477"
        filtered_records = [r for r in filtered_records if r.get("id") != target_id] + [r for r in filtered_records if r.get("id") == target_id]
        
    st.markdown(f"Found **{len(filtered_records)}** matching titles.")
    
    # Display titles in a clean list
    for idx, rec in enumerate(filtered_records[:30]):  # display maximum 30 to keep rendering speedy
        with st.container():
            st.markdown(f"**ID:** `{rec.get('id', 'N/A')}` | **Platform:** `{rec.get('platform', 'N/A').upper()}` | **Category:** `{rec.get('industry_group', 'N/A')}`")
            # Highlight NER
            html_block = highlight_ner(rec["text"], rec.get("entities", []))
            st.markdown(html_block, unsafe_allow_html=True)
            
            # Show offsets table in an expander for inspection
            with st.expander("🔍 Show Character Offset Coordinates"):
                df_offsets = pd.DataFrame(rec.get("entities", []))
                if not df_offsets.empty:
                    st.dataframe(
                        df_offsets[["start_char", "end_char", "label"]],
                        use_container_width=True,
                        hide_index=True
                    )
                else:
                    st.write("No entities annotated for this sample.")
            st.markdown("<br>", unsafe_allow_html=True)
            
    if len(filtered_records) > 30:
        st.warning(f"⚠️ Truncated output. Showing first 30 of {len(filtered_records)} matching records to optimize loading times.")
