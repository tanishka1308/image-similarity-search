import os
import json
import shutil
import base64
import numpy as np
import streamlit as st
import torch
from PIL import Image
import torchvision.transforms.v2 as T
import plotly.express as px

from src.model import ImageEmbeddingNet

# -------------------------------------------------------------
# 1. Page Configuration & Multi-Theme Engine
# -------------------------------------------------------------
st.set_page_config(
    page_title="AI Visual Intelligence & Clustering Hub",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Persistent UI State
if "theme_mode" not in st.session_state:
    st.session_state.theme_mode = "Cyber Dark"
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "selected_query_fname" not in st.session_state:
    st.session_state.selected_query_fname = None

THEMES = {
    "Cyber Dark": {
        "bg": "#0B1120",
        "sidebar_bg": "#070D18",
        "card_bg": "rgba(15, 23, 42, 0.85)",
        "card_border": "rgba(56, 189, 248, 0.25)",
        "card_border_hover": "#38BDF8",
        "text": "#F8FAFC",
        "subtext": "#94A3B8",
        "widget_bg": "#1E293B",
        "widget_text": "#F8FAFC",
        "widget_border": "#334155",
        "primary_grad": "linear-gradient(135deg, #0284C7, #6366F1)",
        "hero_grad": "linear-gradient(135deg, #38BDF8 0%, #818CF8 50%, #C084FC 100%)",
        "glow": "0 8px 24px rgba(14, 165, 233, 0.35)",
        "scrollbar": "#0284C7"
    },
    "Midnight Purple": {
        "bg": "#0F0728",
        "sidebar_bg": "#0A031E",
        "card_bg": "rgba(26, 16, 60, 0.85)",
        "card_border": "rgba(192, 132, 252, 0.3)",
        "card_border_hover": "#C084FC",
        "text": "#FAF5FF",
        "subtext": "#D8B4FE",
        "widget_bg": "#281A4E",
        "widget_text": "#FAF5FF",
        "widget_border": "#4C2882",
        "primary_grad": "linear-gradient(135deg, #8B5CF6, #EC4899)",
        "hero_grad": "linear-gradient(135deg, #C084FC 0%, #F472B6 50%, #FB7185 100%)",
        "glow": "0 8px 24px rgba(192, 132, 252, 0.4)",
        "scrollbar": "#8B5CF6"
    },
    "Clean Light Glass": {
        "bg": "#F8FAFC",
        "sidebar_bg": "#F1F5F9",
        "card_bg": "rgba(255, 255, 255, 0.9)",
        "card_border": "rgba(203, 213, 225, 0.8)",
        "card_border_hover": "#0284C7",
        "text": "#0F172A",
        "subtext": "#475569",
        "widget_bg": "#FFFFFF",
        "widget_text": "#0F172A",
        "widget_border": "#CBD5E1",
        "primary_grad": "linear-gradient(135deg, #0284C7, #2563EB)",
        "hero_grad": "linear-gradient(135deg, #0284C7 0%, #4F46E5 100%)",
        "glow": "0 8px 20px rgba(2, 132, 199, 0.2)",
        "scrollbar": "#0284C7"
    }
}

t = THEMES[st.session_state.theme_mode]

css_styles = f"""
<style>
    /* Main Background & Text */
    .stApp {{
        background-color: {t['bg']} !important;
        color: {t['text']} !important;
    }}
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {{
        background-color: {t['sidebar_bg']} !important;
        border-right: 1px solid {t['card_border']} !important;
    }}
    section[data-testid="stSidebar"] * {{
        color: {t['text']} !important;
    }}

    /* Global Typography */
    h1, h2, h3, h4, h5, h6, p, label, span, div {{
        color: {t['text']};
    }}
    
    .hero-title {{
        font-size: 2.3rem;
        font-weight: 800;
        background: {t['hero_grad']};
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2px;
        letter-spacing: -0.5px;
    }}
    .hero-tagline {{
        font-size: 1rem;
        color: {t['subtext']} !important;
        margin-bottom: 20px;
    }}

    /* Selectbox Container & Background Override */
    div[data-baseweb="select"] {{
        background-color: transparent !important;
    }}
    div[data-baseweb="select"] > div {{
        background-color: {t['widget_bg']} !important;
        border: 1px solid {t['widget_border']} !important;
        border-radius: 8px !important;
    }}
    div[data-baseweb="select"] div,
    div[data-baseweb="select"] span,
    div[data-baseweb="select"] input,
    div[data-baseweb="select"] svg {{
        color: {t['widget_text']} !important;
        fill: {t['widget_text']} !important;
    }}

    /* Dropdown Popover Menus */
    ul[data-testid="stVirtualDropdown"],
    div[data-baseweb="popover"] ul,
    div[data-baseweb="menu"] {{
        background-color: {t['widget_bg']} !important;
        border: 1px solid {t['widget_border']} !important;
    }}
    ul[data-testid="stVirtualDropdown"] li,
    div[data-baseweb="popover"] li,
    div[data-baseweb="menu"] [role="option"] {{
        background-color: {t['widget_bg']} !important;
        color: {t['widget_text']} !important;
    }}
    ul[data-testid="stVirtualDropdown"] li:hover,
    div[data-baseweb="popover"] li:hover,
    div[data-baseweb="menu"] [role="option"]:hover {{
        background-color: {t['card_border_hover']} !important;
        color: #FFFFFF !important;
    }}

    /* Radio Controls */
    div[data-testid="stRadio"] label span {{
        color: {t['text']} !important;
        font-weight: 500;
    }}

    /* Expander Container */
    div[data-testid="stExpander"],
    details[data-testid="stExpander"] {{
        background-color: {t['card_bg']} !important;
        border: 1px solid {t['card_border']} !important;
        border-radius: 10px !important;
        overflow: hidden;
    }}

    /* Expander Header Summary */
    div[data-testid="stExpander"] summary,
    details[data-testid="stExpander"] summary {{
        background-color: {t['widget_bg']} !important;
        color: {t['text']} !important;
        border-radius: 8px !important;
        padding: 8px 14px !important;
    }}
    div[data-testid="stExpander"] summary *,
    details[data-testid="stExpander"] summary * {{
        color: {t['text']} !important;
        fill: {t['text']} !important;
    }}
    div[data-testid="stExpander"] div[data-testid="stExpanderDetails"] {{
        background-color: transparent !important;
        color: {t['text']} !important;
        padding: 12px !important;
    }}

    /* Toast Notification Fix */
    div[data-testid="stToast"] {{
        background-color: {t['widget_bg']} !important;
        border: 1px solid {t['card_border']} !important;
        color: {t['text']} !important;
        box-shadow: {t['glow']} !important;
    }}
    div[data-testid="stToast"] * {{
        color: {t['text']} !important;
    }}

    /* Carousel Elements */
    .carousel-container {{
        display: flex;
        gap: 16px;
        overflow-x: auto;
        padding: 12px 6px 18px 6px;
        scroll-snap-type: x mandatory;
        scrollbar-width: thin;
        scrollbar-color: {t['scrollbar']} transparent;
    }}
    .carousel-container::-webkit-scrollbar {{
        height: 6px;
    }}
    .carousel-container::-webkit-scrollbar-thumb {{
        background-color: {t['scrollbar']};
        border-radius: 6px;
    }}

    .carousel-card {{
        flex: 0 0 190px;
        scroll-snap-align: start;
        background: {t['card_bg']};
        backdrop-filter: blur(10px);
        border: 1px solid {t['card_border']};
        border-radius: 12px;
        padding: 10px;
        box-shadow: {t['glow']};
        transition: transform 0.25s ease, border-color 0.25s ease;
        text-align: center;
    }}
    .carousel-card:hover {{
        transform: translateY(-5px) scale(1.02);
        border-color: {t['card_border_hover']};
    }}
    .carousel-img {{
        width: 100%;
        height: 140px;
        object-fit: cover;
        border-radius: 8px;
        margin-bottom: 8px;
    }}
    .carousel-title {{
        font-size: 0.8rem;
        font-weight: 600;
        color: {t['text']};
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }}
    .carousel-badge {{
        font-size: 0.75rem;
        padding: 2px 8px;
        border-radius: 10px;
        background: {t['primary_grad']};
        color: #ffffff !important;
        font-weight: 600;
        display: inline-block;
        margin-top: 4px;
    }}

    /* Buttons */
    div.stButton > button[kind="primary"] {{
        background: {t['primary_grad']} !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease;
    }}
    div.stButton > button[kind="primary"]:hover {{
        transform: translateY(-2px);
        box-shadow: {t['glow']};
    }}
    div.stButton > button[kind="secondary"] {{
        background: {t['widget_bg']} !important;
        color: {t['text']} !important;
        border: 1px solid {t['widget_border']} !important;
        border-radius: 8px;
    }}
</style>
"""
st.markdown(css_styles, unsafe_allow_html=True)

# -------------------------------------------------------------
# 2. Carousel HTML Helper
# -------------------------------------------------------------
def image_to_base64(path):
    with open(path, "rb") as img_f:
        return base64.b64encode(img_f.read()).decode()

def render_carousel(items):
    cards_html = ""
    for fname, p, tag in items:
        if not os.path.exists(p):
            continue
        try:
            b64 = image_to_base64(p)
            tag_badge = f'<div class="carousel-badge">{tag}</div>' if tag else ''
            card = (
                f'<div class="carousel-card">'
                f'<img class="carousel-img" src="data:image/jpeg;base64,{b64}" alt="{fname}"/>'
                f'<div class="carousel-title" title="{fname}">{fname}</div>'
                f'{tag_badge}'
                f'</div>'
            )
            cards_html += card
        except Exception:
            continue

    html = f'<div class="carousel-container">{cards_html}</div>'
    st.markdown(html, unsafe_allow_html=True)

# -------------------------------------------------------------
# 3. Model & Pipeline Loader
# -------------------------------------------------------------
@st.cache_resource
def load_inference_pipeline(weights_path="model_weights.pth"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ImageEmbeddingNet(embedding_dim=256)
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return model, transform, device

model, transform, device = load_inference_pipeline()

def extract_vector(img_path):
    img = Image.open(img_path).convert('RGB')
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        vec = model(tensor)
    return vec.squeeze(0).cpu().numpy().astype('float32')

# -------------------------------------------------------------
# 4. Registry & State
# -------------------------------------------------------------
ORIGIN_DIR = "dataset/images"
BASE_OUTPUT = "sorted_folders"
REGISTRY_FILE = "registry.json"
VALID_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

def load_or_build_registry():
    if os.path.exists(REGISTRY_FILE):
        try:
            with open(REGISTRY_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass

    registry = {"images": {}, "next_group_id": 1}
    if os.path.exists(ORIGIN_DIR):
        for fname in os.listdir(ORIGIN_DIR):
            if fname.lower().endswith(VALID_EXTS):
                registry["images"][fname] = {
                    "current_folder": "origin",
                    "path": os.path.join(ORIGIN_DIR, fname)
                }
    save_registry(registry)
    return registry

def save_registry(reg):
    with open(REGISTRY_FILE, "w") as f:
        json.dump(reg, f, indent=2)

def reset_all_data():
    os.makedirs(ORIGIN_DIR, exist_ok=True)
    if os.path.exists(BASE_OUTPUT):
        for root, _, files in os.walk(BASE_OUTPUT):
            for file in files:
                if file.lower().endswith(VALID_EXTS):
                    src_f = os.path.join(root, file)
                    dest_f = os.path.join(ORIGIN_DIR, file)
                    if not os.path.exists(dest_f):
                        shutil.move(src_f, dest_f)
        shutil.rmtree(BASE_OUTPUT, ignore_errors=True)
    
    if os.path.exists(REGISTRY_FILE):
        os.remove(REGISTRY_FILE)

registry = load_or_build_registry()

# -------------------------------------------------------------
# 5. Sidebar Controls & Theme Selector
# -------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🎨 Visual Theme")
    selected_theme = st.radio(
        "Choose Theme", 
        list(THEMES.keys()), 
        index=list(THEMES.keys()).index(st.session_state.theme_mode),
        label_visibility="collapsed"
    )
    if selected_theme != st.session_state.theme_mode:
        st.session_state.theme_mode = selected_theme
        st.rerun()
        
    st.markdown("---")
    st.markdown("### 🎛️ Retrieval Settings")
    threshold = st.slider(
        "Similarity Cut-off (Cosine)",
        min_value=0.30,
        max_value=0.95,
        value=0.55,
        step=0.01
    )

    origin_count = sum(1 for info in registry["images"].values() if info["current_folder"] == "origin" and os.path.exists(info["path"]))
    grouped_count = sum(1 for info in registry["images"].values() if info["current_folder"] != "origin" and os.path.exists(info["path"]))
    total_images = origin_count + grouped_count
    total_groups = max(0, registry["next_group_id"] - 1)

    if total_images > 0:
        fig_pie = px.pie(
            names=["Origin Pool", "Grouped"],
            values=[origin_count, grouped_count],
            hole=0.65,
            color=["Origin Pool", "Grouped"],
            color_discrete_map={"Origin Pool": "#38BDF8", "Grouped": "#818CF8"}
        )
        fig_pie.update_traces(textinfo="none", hoverinfo="label+value")
        fig_pie.update_layout(
            showlegend=False,
            margin=dict(t=0, b=0, l=0, r=0),
            height=130,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    c1, c2 = st.columns(2)
    c1.metric("Unsorted", origin_count)
    c2.metric("Clusters", total_groups)

    st.markdown("---")
    if st.button("🔄 Reset Entire Pipeline", use_container_width=True, type="secondary"):
        reset_all_data()
        st.session_state.last_result = None
        st.session_state.selected_query_fname = None
        st.toast("Restored all images to origin!", icon="🧼")
        st.rerun()

# -------------------------------------------------------------
# 6. Main Dashboard Interface
# -------------------------------------------------------------
st.markdown('<div class="hero-title">Visual Intelligence & Dynamic Grouping</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-tagline">Metric Learning Vector Search • Stateful Dynamic Depletion • Horizontal Carousel Navigation</div>', unsafe_allow_html=True)

tab_names = [
    "⚡ Search & Grouping", 
    "📂 Active Clusters (Carousels)", 
    "📦 Origin Pool Inventory"
]

tabs = st.tabs(tab_names)

# =============================================================
# TAB 1: Search & Dynamic Grouping
# =============================================================
with tabs[0]:
    col_l, col_r = st.columns([1, 2], gap="large")

    with col_l:
        st.markdown("#### 1. Select Target")
        query_mode = st.radio("Input Mode", ["Select Existing File", "Upload Direct Image"], horizontal=True)

        selected_image_path = None
        if query_mode == "Select Existing File":
            available_files = sorted([
                fname for fname, info in registry["images"].items() 
                if os.path.exists(info["path"])
            ])
            if not available_files:
                st.warning("No files found. Click Reset in the sidebar.")
            else:
                default_idx = 0
                if st.session_state.selected_query_fname in available_files:
                    default_idx = available_files.index(st.session_state.selected_query_fname)
                
                chosen_fname = st.selectbox("Choose file:", available_files, index=default_idx)
                selected_image_path = registry["images"][chosen_fname]["path"]
        else:
            uploaded = st.file_uploader("Upload image:", type=["jpg", "jpeg", "png", "webp"])
            if uploaded is not None:
                os.makedirs("temp_uploads", exist_ok=True)
                temp_path = os.path.join("temp_uploads", uploaded.name)
                with open(temp_path, "wb") as f:
                    f.write(uploaded.getbuffer())
                selected_image_path = temp_path

        if selected_image_path and os.path.exists(selected_image_path):
            st.image(selected_image_path, caption=f"Selected: {os.path.basename(selected_image_path)}", use_container_width=True)
            execute_btn = st.button("🚀 Run Dynamic Search & Grouping", type="primary", use_container_width=True)
        else:
            execute_btn = False

    with col_r:
        st.markdown("#### 2. Execution Results")

        if execute_btn and selected_image_path:
            query_fname = os.path.basename(selected_image_path)
            img_info = registry["images"].get(query_fname)
            current_folder = img_info["current_folder"] if img_info else "unknown"

            with st.spinner("Computing 256-D normalized vector & scanning similarities..."):
                query_vec = extract_vector(selected_image_path)

            if current_folder not in ["origin", "unknown"]:
                # --- SCENARIO B: Existing Group ---
                existing_members = [
                    (fname, info["path"], f"Sim: {float(np.dot(query_vec, extract_vector(info['path']))):.3f}")
                    for fname, info in registry["images"].items()
                    if info["current_folder"] == current_folder and os.path.exists(info["path"])
                ]

                origin_candidates = [
                    (fname, info["path"])
                    for fname, info in registry["images"].items()
                    if info["current_folder"] == "origin" and os.path.exists(info["path"])
                ]

                new_matches = []
                for fname, p in origin_candidates:
                    sim = float(np.dot(query_vec, extract_vector(p)))
                    if sim >= threshold:
                        new_matches.append((fname, p, sim))

                if new_matches:
                    target_dir = os.path.join(BASE_OUTPUT, current_folder)
                    os.makedirs(target_dir, exist_ok=True)
                    for fname, old_p, sim in new_matches:
                        new_p = os.path.join(target_dir, fname)
                        shutil.move(old_p, new_p)
                        registry["images"][fname] = {
                            "current_folder": current_folder,
                            "path": new_p
                        }
                    save_registry(registry)

                st.session_state.last_result = {
                    "type": "existing",
                    "folder": current_folder,
                    "existing_members": existing_members,
                    "new_matches": [(fname, p, f"Added: {sim:.3f}") for fname, p, sim in new_matches],
                    "threshold": threshold
                }

            else:
                # --- SCENARIO A: Origin Image -> New Group ---
                origin_candidates = [
                    (fname, info["path"])
                    for fname, info in registry["images"].items()
                    if info["current_folder"] == "origin" and os.path.exists(info["path"])
                ]

                matched_files = []
                scores_plot = []

                for fname, p in origin_candidates:
                    sim = float(np.dot(query_vec, extract_vector(p)))
                    scores_plot.append({"File": fname, "Similarity": sim})
                    if sim >= threshold:
                        matched_files.append((fname, p, sim))

                if matched_files:
                    matched_files.sort(key=lambda x: x[2], reverse=True)
                    group_id = registry["next_group_id"]
                    group_name = f"group_{group_id:03d}"
                    target_dir = os.path.join(BASE_OUTPUT, group_name)
                    os.makedirs(target_dir, exist_ok=True)

                    for fname, old_p, sim in matched_files:
                        new_p = os.path.join(target_dir, fname)
                        shutil.move(old_p, new_p)
                        registry["images"][fname] = {
                            "current_folder": group_name,
                            "path": new_p
                        }

                    registry["next_group_id"] += 1
                    save_registry(registry)

                st.session_state.last_result = {
                    "type": "new",
                    "matched_files": [(fname, p, f"Sim: {sim:.3f}") for fname, p, sim in matched_files],
                    "scores_plot": scores_plot,
                    "group_name": group_name if matched_files else None,
                    "threshold": threshold
                }

        # Render Results
        res = st.session_state.last_result
        if not res:
            st.info("Pick an image on the left and click **Run Dynamic Search & Grouping**.")
        else:
            if res["type"] == "existing":
                st.info(f"📍 **Existing Folder Detected:** Belongs to **`{res['folder']}`**. No duplicate folder created.")
                
                st.markdown(f"##### 🎠 Current Members Carousel ({len(res['existing_members'])})")
                render_carousel(res["existing_members"])

                if not res["new_matches"]:
                    st.write(f"No additional matching images found in origin (Threshold: ≥ {res['threshold']:.2f}).")
                else:
                    st.success(f"Added **{len(res['new_matches'])}** new images from origin into `{res['folder']}`:")
                    st.markdown("##### 🎠 Newly Back-Filled Images")
                    render_carousel(res["new_matches"])

            elif res["type"] == "new":
                if res["scores_plot"]:
                    fig_bar = px.histogram(
                        res["scores_plot"], 
                        x="Similarity", 
                        nbins=20, 
                        title="Candidate Similarity Spectrum",
                        color_discrete_sequence=["#0EA5E9"]
                    )
                    fig_bar.add_vline(x=res["threshold"], line_dash="dash", line_color="#EF4444", annotation_text=f"Threshold ({res['threshold']:.2f})")
                    fig_bar.update_layout(height=180, margin=dict(t=30, b=10, l=10, r=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig_bar, use_container_width=True)

                if not res["matched_files"]:
                    st.error(f"No origin images reached threshold ≥ {res['threshold']:.2f}.")
                else:
                    st.success(f"Formed **`{res['group_name']}`** with **{len(res['matched_files'])}** images!")
                    st.markdown(f"##### 🎠 Grouped Images Carousel (`{res['group_name']}`)")
                    render_carousel(res["matched_files"])

# =============================================================
# TAB 2: Active Clusters (Carousels)
# =============================================================
with tabs[1]:
    st.markdown("### 🗃️ Active Groups & Cluster Carousels")
    if not os.path.exists(BASE_OUTPUT) or not os.listdir(BASE_OUTPUT):
        st.info("No clusters formed yet. Run a query in the first tab to begin clustering.")
    else:
        group_folders = sorted([f for f in os.listdir(BASE_OUTPUT) if os.path.isdir(os.path.join(BASE_OUTPUT, f))])
        for g_folder in group_folders:
            folder_path = os.path.join(BASE_OUTPUT, g_folder)
            files = [f for f in os.listdir(folder_path) if f.lower().endswith(VALID_EXTS)]
            
            with st.expander(f"📁 **{g_folder}** — ({len(files)} items)", expanded=True):
                items = [(fname, os.path.join(folder_path, fname), g_folder) for fname in files]
                render_carousel(items)

# =============================================================
# TAB 3: Origin Pool Inventory (Interactive Selection)
# =============================================================
with tabs[2]:
    st.markdown(f"### 📦 Unsorted Origin Pool (`dataset/images`) — [{origin_count} Remaining]")
    st.markdown("*Click **'⚡ Query This'** under any image below to immediately load it into Search & Grouping.*")

    origin_items = [
        (os.path.basename(info["path"]), info["path"])
        for info in registry["images"].values() 
        if info["current_folder"] == "origin" and os.path.exists(info["path"])
    ]
    
    if not origin_items:
        st.success("All images have been sorted out of origin into groups!")
    else:
        cols_per_row = 5
        rows = [origin_items[i:i + cols_per_row] for i in range(0, len(origin_items), cols_per_row)]

        for row in rows:
            grid_cols = st.columns(cols_per_row)
            for idx, (fname, p) in enumerate(row):
                with grid_cols[idx]:
                    st.image(p, use_container_width=True)
                    st.caption(f"**{fname}**")
                    if st.button("⚡ Query This", key=f"btn_origin_{fname}", use_container_width=True):
                        st.session_state.selected_query_fname = fname
                        st.toast(f"Selected {fname}! Head over to the Search & Grouping tab.", icon="🎯")
                        st.rerun()