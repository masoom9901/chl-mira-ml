import os
import time
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Import custom modules
from src.ingestion import BhoonidhiClient, generate_mock_liss_image
from src.preprocessing import RemoteSensingPreprocessor
from src.annotation_parser import VIAAnnotationParser, generate_mock_via_annotations
from src.augmentation import RemoteSensingAugmenter
from src.synthetic_generator import GenerativeCropSynthesizer
from src.models_train import UnsupervisedLULCClassifier, train_supervised_model, run_semi_supervised_training
from src.yield_prediction import YieldPredictor

# Setup paths
DATA_DIR = "data"
MODELS_DIR = "models"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Page Configuration
st.set_page_config(
    page_title="AgriSense: Remote Crop Health & Yield Analytics",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Theme Selector in Sidebar
st.sidebar.markdown("### 🎨 Appearance Settings")
theme_mode = st.sidebar.radio("Select Theme", ["Dark Space Mode", "Light Mode"])

# Theme variables assignment
if theme_mode == "Dark Space Mode":
    bg_color = "#0b0f19"
    text_color = "#e2e8f0"
    card_bg = "rgba(22, 28, 45, 0.65)"
    card_border = "rgba(255, 255, 255, 0.08)"
    primary_glow = "#00ffb7"
    secondary_glow = "#0099ff"
    metric_box_bg = "rgba(255, 255, 255, 0.03)"
    metric_box_border = "rgba(255, 255, 255, 0.05)"
    text_muted = "#94a3b8"
    highlight_color = "#00ffb7"
else:
    bg_color = "#f1f5f9"
    text_color = "#0f172a"
    card_bg = "rgba(255, 255, 255, 0.85)"
    card_border = "rgba(0, 0, 0, 0.08)"
    primary_glow = "#10b981" # Emerald Green
    secondary_glow = "#0284c7" # Sky Blue
    metric_box_bg = "rgba(0, 0, 0, 0.02)"
    metric_box_border = "rgba(0, 0, 0, 0.04)"
    text_muted = "#475569"
    highlight_color = "#10b981"

# Custom Premium CSS Injection
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');

/* Dynamic Theme Variables */
:root {{
    --bg-color: {bg_color};
    --text-color: {text_color};
    --primary-glow: {primary_glow};
    --secondary-glow: {secondary_glow};
    --background-card: {card_bg};
    --border-card: {card_border};
}}

.stApp {{
    background-color: var(--bg-color) !important;
}}

/* Custom Typography */
body, [class*="st-"] {{
    font-family: 'Inter', sans-serif;
    color: var(--text-color);
}}

h1, h2, h3, .title-text {{
    font-family: 'Space Grotesk', sans-serif;
    background: linear-gradient(135deg, var(--primary-glow) 0%, var(--secondary-glow) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700;
}}

/* Dashboard Cards (Glassmorphism) */
.glass-card {{
    background: var(--background-card);
    backdrop-filter: blur(12px);
    border: 1px solid var(--border-card);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.15);
    transition: all 0.3s ease;
}}

.glass-card:hover {{
    border-color: rgba(0, 255, 183, 0.3);
    box-shadow: 0 8px 32px 0 rgba(0, 255, 183, 0.08);
}}

/* Status Badges */
.status-healthy {{
    background: rgba(0, 255, 183, 0.15);
    color: var(--primary-glow);
    border: 1px solid rgba(0, 255, 183, 0.3);
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 0.85em;
    font-weight: 600;
}}

.status-stressed {{
    background: rgba(255, 165, 0, 0.15);
    color: #ffa500;
    border: 1px solid rgba(255, 165, 0, 0.3);
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 0.85em;
    font-weight: 600;
}}

.status-soil {{
    background: rgba(139, 69, 19, 0.15);
    color: #d2b48c;
    border: 1px solid rgba(139, 69, 19, 0.3);
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 0.85em;
    font-weight: 600;
}}

.status-pending {{
    background: rgba(128, 128, 128, 0.15);
    color: #888888;
    border: 1px solid rgba(128, 128, 128, 0.3);
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 0.85em;
    font-weight: 600;
}}

/* Metric Display CSS */
.metric-container {{
    display: flex;
    justify-content: space-around;
    gap: 15px;
    margin-bottom: 15px;
}}

.metric-box {{
    flex: 1;
    background: {metric_box_bg};
    border: 1px solid {metric_box_border};
    border-radius: 12px;
    padding: 15px;
    text-align: center;
}}

.metric-val {{
    font-size: 1.8rem;
    font-weight: 800;
    color: {highlight_color};
}}

.metric-lbl {{
    font-size: 0.9rem;
    color: {text_muted};
    margin-top: 5px;
}}
</style>
""", unsafe_allow_html=True)

# Title Banner
st.markdown(f"""
<div class='glass-card' style='text-align: center; padding: 25px; margin-bottom: 25px; background: linear-gradient(135deg, rgba(13, 27, 42, 0.85) 0%, rgba(22, 48, 65, 0.85) 100%);'>
    <h1 style='font-size: 2.8rem; margin: 0;'>AgriSense Platform</h1>
    <p style='font-size: 1.1rem; color: #a0aec0; margin-top: 8px; font-weight: 300;'>
        Crop Health Monitoring & Yield Prediction using LISS-3/LISS-4 Remote Sensing & Machine Learning
    </p>
</div>
""", unsafe_allow_html=True)

# Initialize Session States
if 'current_scene_path' not in st.session_state:
    st.session_state['current_scene_path'] = None
if 'preprocessed_data' not in st.session_state:
    st.session_state['preprocessed_data'] = None
if 'annotations_loaded' not in st.session_state:
    st.session_state['annotations_loaded'] = False
if 'trained_model' not in st.session_state:
    st.session_state['trained_model'] = None
if 'active_field_metrics' not in st.session_state:
    st.session_state['active_field_metrics'] = None

# Sidebar Controls
st.sidebar.markdown("### 🛰️ Sensor Configuration")
satellite_type = st.sidebar.selectbox("Satellite", ["RESOURCESAT-2", "RESOURCESAT-2A"])
sensor_type = st.sidebar.selectbox("Sensor Type", ["LISS-3 (24m Res, 4 Bands)", "LISS-4 (5.8m Res, 3 Bands)"])
resolution_val = 24.0 if "LISS-3" in sensor_type else 5.8
is_liss4_val = "LISS-4" in sensor_type

# Tab Setup
tabs = st.tabs([
    "📊 Overview Dashboard",
    "📥 Step 1: Acquisition",
    "⚙️ Step 2: Preprocessing",
    "🏷️ Steps 3 & 4: Annotations & Augmentation",
    "🧠 Step 5: Model Training",
    "📈 Step 6: Inference & Yield Prediction"
])

# ==========================================
# TAB 0: OVERVIEW DASHBOARD
# ==========================================
with tabs[0]:
    st.markdown("### 📊 Region & Analytics Overview")
    
    # 1. Pipeline Status Tracker
    st.markdown("#### 🔄 Pipeline Execution Status")
    status_cols = st.columns(6)
    
    # Scene check
    scene_status = "<span class='status-healthy'>Active</span>" if st.session_state['current_scene_path'] else "<span class='status-pending'>Pending</span>"
    status_cols[0].markdown(f"**1. Ingestion**<br>{scene_status}", unsafe_allow_html=True)
    
    # Preprocess check
    preprocess_status = "<span class='status-healthy'>Completed</span>" if st.session_state['preprocessed_data'] else "<span class='status-pending'>Pending</span>"
    status_cols[1].markdown(f"**2. Preprocessing**<br>{preprocess_status}", unsafe_allow_html=True)
    
    # Annotations check
    ann_status = "<span class='status-healthy'>Loaded</span>" if st.session_state['annotations_loaded'] else "<span class='status-pending'>Pending</span>"
    status_cols[2].markdown(f"**3. Annotation**<br>{ann_status}", unsafe_allow_html=True)
    
    # Augmentations (derived if we have an image and mask)
    aug_status = "<span class='status-healthy'>Ready</span>" if st.session_state['annotations_loaded'] else "<span class='status-pending'>Pending</span>"
    status_cols[3].markdown(f"**4. Augmentation**<br>{aug_status}", unsafe_allow_html=True)
    
    # Model check
    model_status = f"<span class='status-healthy'>{st.session_state['trained_model']['type'].capitalize()}</span>" if st.session_state['trained_model'] else "<span class='status-pending'>Pending</span>"
    status_cols[4].markdown(f"**5. Model Training**<br>{model_status}", unsafe_allow_html=True)
    
    # Inference metrics check
    inf_status = "<span class='status-healthy'>Generated</span>" if st.session_state['active_field_metrics'] else "<span class='status-pending'>Pending</span>"
    status_cols[5].markdown(f"**6. Predictions**<br>{inf_status}", unsafe_allow_html=True)
    
    st.markdown("<hr style='border-color: rgba(255,255,255,0.05); margin-top: 15px; margin-bottom: 25px;'>", unsafe_allow_html=True)
    
    # 2. Main Dashboard Panel
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.write("#### 🛰️ Active Satellite Analysis Scene")
        
        if st.session_state['preprocessed_data'] is not None:
            # We display the preprocessed False Color Composite
            proc_data = st.session_state['preprocessed_data']
            st.image(proc_data['fcc'], caption=f"Active Scene FCC Preview ({os.path.basename(st.session_state['current_scene_path'])})", use_container_width=True)
        else:
            # Welcome/Simulated Map Preview
            st.info("No active satellite scene loaded. Displaying default regional workspace preview (Guntur Region).")
            # Draw a simulated agricultural regional pattern
            np.random.seed(42)
            grid = np.zeros((100, 100, 3), dtype=np.uint8)
            grid[10:45, 10:90, 0] = 180 # simulated crop fields (NIR high)
            grid[50:85, 15:80, 0] = 140
            grid[10:85, :, 1] = 60 # Red band
            grid[10:85, :, 2] = 80 # Green band
            # Show a styled color composite of the simulated landscape
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.imshow(grid)
            ax.set_title("Simulated Regional Workspace")
            ax.axis('off')
            st.pyplot(fig)
            st.caption("Awaiting Step 1 & 2 to render LISS-3 / LISS-4 False Color Composite (FCC).")
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col2:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.write("#### 📊 Crop Yield & Health Statistics")
        
        # Load active or mock stats
        if st.session_state['active_field_metrics'] is not None:
            fields = st.session_state['active_field_metrics']
            df_fields = pd.DataFrame(fields)
            
            # Show active yield metrics
            total_yield = df_fields["yield_tons"].sum()
            total_area = df_fields["area_hectares"].sum()
            avg_ndvi = df_fields["mean_ndvi"].mean()
            
            st.markdown(f"""
            <div class='metric-container'>
                <div class='metric-box'>
                    <div class='metric-val'>{len(df_fields)}</div>
                    <div class='metric-lbl'>Analyzed Fields</div>
                </div>
                <div class='metric-box'>
                    <div class='metric-val'>{total_area:.1f} ha</div>
                    <div class='metric-lbl'>Total Cropland</div>
                </div>
                <div class='metric-box'>
                    <div class='metric-val'>{total_yield:.1f} t</div>
                    <div class='metric-lbl'>Total Yield Forecast</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Group by Crop Type
            df_yield_crop = df_fields.groupby("crop_type")["yield_tons"].sum().reset_index()
            # Group by Health Status
            df_health_count = df_fields.groupby("health")["field_id"].count().reset_index()
            
            chart_cols = st.columns(2)
            # Chart 1: Yield by Crop Type
            with chart_cols[0]:
                st.write("**Yield by Crop (Tons)**")
                st.bar_chart(data=df_yield_crop, x="crop_type", y="yield_tons", color="#00ffb7", use_container_width=True)
                
            # Chart 2: Health status count
            with chart_cols[1]:
                st.write("**Field Health Distribution**")
                st.bar_chart(data=df_health_count, x="health", y="field_id", color="#0099ff", use_container_width=True)
        else:
            # Display simulated regional historical statistics
            st.write("Historical Crop Distribution & District Yield Projection:")
            
            # Mock Data
            mock_yields = pd.DataFrame({
                "Crop Type": ["Rice", "Sugarcane", "Cotton", "Wheat", "Maize"],
                "Expected Yield (Tons/ha)": [4.2, 75.0, 2.1, 3.9, 5.2]
            })
            mock_health = pd.DataFrame({
                "Health Grade": ["Healthy Crop", "Moderate Stress", "Severe Stress", "Bare Soil"],
                "Acreage %": [45, 35, 12, 8]
            })
            
            st.markdown(f"""
            <div class='metric-container'>
                <div class='metric-box'>
                    <div class='metric-val'>16</div>
                    <div class='metric-lbl'>Default Fields</div>
                </div>
                <div class='metric-box'>
                    <div class='metric-val'>104.5 ha</div>
                    <div class='metric-lbl'>Avg Area</div>
                </div>
                <div class='metric-box'>
                    <div class='metric-val'>4.5 t/ha</div>
                    <div class='metric-lbl'>Average Yield Multiplier</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            chart_cols = st.columns(2)
            with chart_cols[0]:
                st.write("**Yield Benchmark by Crop Type**")
                st.bar_chart(data=mock_yields, x="Crop Type", y="Expected Yield (Tons/ha)", color="#00ffb7", use_container_width=True)
            with chart_cols[1]:
                st.write("**Typical Seasonal Crop Health Distribution**")
                st.bar_chart(data=mock_health, x="Health Grade", y="Acreage %", color="#0099ff", use_container_width=True)
                
        st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# TAB 1: DATA ACQUISITION
# ==========================================
with tabs[1]:
    st.markdown("### 🌐 Acquire LULC Satellite Imagery")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.write("#### Bhoonidhi STAC Portal Query")
        
        # Region Selection
        region = st.selectbox("Area of Interest (AOI)", [
            "Guntur (Andhra Pradesh, India)",
            "Hyderabad Agriland (Telangana, India)",
            "Bathinda Croplands (Punjab, India)",
            "Nashik Vineyards (Maharashtra, India)"
        ])
        
        date_range = st.date_input("Acquisition Dates", [pd.to_datetime("2026-01-01"), pd.to_datetime("2026-03-31")])
        cloud_thresh = st.slider("Max Cloud Cover (%)", 0, 100, 15)
        
        # Credentials (Optional)
        st.info("Bhoonidhi STAC search is initialized with public indices. To authenticate for raw full-scene NRSC downloads, request SIS credentials.")
        username = st.text_input("Bhoonidhi Username", value="", placeholder="Enter NRSC username")
        password = st.text_input("Bhoonidhi Password", type="password", placeholder="Enter NRSC password")
        
        btn_search = st.button("Search Bhoonidhi Catalog", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col2:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.write("#### Catalog Search Results")
        
        if btn_search:
            client = BhoonidhiClient(username, password)
            if username:
                success, msg = client.authenticate()
                st.success(msg)
            
            bbox_coords = [78.2, 17.1, 78.6, 17.6] # Default bounding box approx
            start_d = str(date_range[0])
            end_d = str(date_range[1])
            
            with st.spinner("Querying Bhoonidhi STAC API..."):
                time.sleep(1.0)
                results = client.search_scenes(bbox_coords, start_d, end_d, satellite_type, "LISS-4" if is_liss4_val else "LISS-3")
                
                df_results = pd.DataFrame(results)
                st.dataframe(df_results.drop("download_url", axis=1), use_container_width=True)
                
                st.write("#### Download & Scene Extraction")
                selected_product = df_results.iloc[0]["product_id"]
                st.write(f"Product Selected: **{selected_product}**")
                
                btn_download = st.button("Download & Load Scene", type="primary")
                if btn_download:
                    with st.spinner("Downloading GeoTIFF bands from Bhoonidhi Open Archive..."):
                        time.sleep(1.5)
                        
                        target_name = "sample_liss4.tif" if is_liss4_val else "sample_liss3.tif"
                        dest_path = os.path.join(DATA_DIR, target_name)
                        
                        # Generate simulated image file
                        generate_mock_liss_image(dest_path, is_liss4=is_liss4_val)
                        
                        st.session_state['current_scene_path'] = dest_path
                        st.session_state['preprocessed_data'] = None # Reset preprocessed image cache
                        st.session_state['active_field_metrics'] = None
                        st.success(f"Success! Multispectral scene downloaded and loaded to {dest_path}.")
                        st.rerun()
        else:
            if st.session_state['current_scene_path']:
                st.success(f"Active scene loaded: **{st.session_state['current_scene_path']}**")
            else:
                st.warning("Please run catalog query and click 'Download & Load Scene' to begin.")
                
            # Quick Load options
            st.write("Or instantly load standard sample dataset:")
            quick_cols = st.columns(2)
            if quick_cols[0].button("Load Sample LISS-3 Image (4 Bands)"):
                dest_path = os.path.join(DATA_DIR, "sample_liss3.tif")
                generate_mock_liss_image(dest_path, is_liss4=False)
                st.session_state['current_scene_path'] = dest_path
                st.session_state['preprocessed_data'] = None
                st.session_state['active_field_metrics'] = None
                st.rerun()
            if quick_cols[1].button("Load Sample LISS-4 Image (3 Bands)"):
                dest_path = os.path.join(DATA_DIR, "sample_liss4.tif")
                generate_mock_liss_image(dest_path, is_liss4=True)
                st.session_state['current_scene_path'] = dest_path
                st.session_state['preprocessed_data'] = None
                st.session_state['active_field_metrics'] = None
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# TAB 2: PREPROCESSING
# ==========================================
with tabs[2]:
    st.markdown("### ⚙️ Image Enhancements & Artifact Removal")
    
    if not st.session_state['current_scene_path']:
        st.warning("Please acquire or load a satellite scene in Tab 1 first.")
    else:
        # Load preprocessor
        proc = RemoteSensingPreprocessor(st.session_state['current_scene_path'])
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.write("#### Enhancement Parameters")
            
            # CLAHE toggle
            apply_clahe_flag = st.checkbox("Apply Adaptive CLAHE (Contrast Enhancement)", value=True)
            
            # Denoising filter
            filter_opt = st.selectbox("Artifact Denoising Filter", ["Bilateral Filter (Edge Preserving)", "Gaussian Filter (Blur)", "None"])
            filter_map = {"Bilateral Filter (Edge Preserving)": "bilateral", "Gaussian Filter (Blur)": "gaussian", "None": "none"}
            
            # Indices Toggle
            display_mode = st.radio("Spectral Layer Map", [
                "False Color Composite (FCC: NIR-Red-Green)",
                "NDVI Map (Normalized Difference Vegetation Index)",
                "NDWI Map (Normalized Difference Water Index)",
                "Individual Band (Gray Scale)"
            ])
            
            selected_band = "NIR"
            if display_mode == "Individual Band (Gray Scale)":
                bands_avail = ["Green", "Red", "NIR"] + ([] if proc.is_liss4 else ["SWIR"])
                selected_band = st.selectbox("Select Band", bands_avail)
                
            btn_process = st.button("Apply Preprocessing", type="primary", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col2:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.write("#### Preview Panel")
            
            # Default process logic
            with st.spinner("Processing bands..."):
                fcc_img = proc.generate_fcc()
                ndvi = proc.calculate_ndvi()
                ndwi = proc.calculate_ndwi()
                
                # Apply filter
                filter_type = filter_map[filter_opt]
                if filter_type != "none":
                    fcc_img = proc.remove_artifacts(fcc_img, filter_type)
                    
                # Apply CLAHE
                if apply_clahe_flag:
                    fcc_img = proc.apply_clahe(fcc_img)
                
                # Render logic
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.axis('off')
                
                if "False Color Composite" in display_mode:
                    ax.imshow(fcc_img)
                    st.pyplot(fig)
                    st.caption("Vegetation appears in bright red. Water body/Canal appears dark blue/black. Soil appears cyan/gray.")
                elif "NDVI" in display_mode:
                    im = ax.imshow(ndvi, cmap='RdYlGn', vmin=-0.1, vmax=0.9)
                    fig.colorbar(im, ax=ax, label="NDVI Index")
                    st.pyplot(fig)
                    st.caption("High NDVI (Green) represents dense healthy crops. Moderate NDVI (Yellow/Orange) represents stressed crops or sparse vegetation. Negative/Zero NDVI represents soil or water.")
                elif "NDWI" in display_mode:
                    im = ax.imshow(ndwi, cmap='Blues', vmin=-0.5, vmax=0.8)
                    fig.colorbar(im, ax=ax, label="NDWI Index")
                    st.pyplot(fig)
                    st.caption("High NDWI (Deep Blue) marks the irrigation water canal cutting through the fields.")
                else:
                    band_data = proc.get_band(selected_band)
                    im = ax.imshow(band_data, cmap='gray')
                    fig.colorbar(im, ax=ax, label="Reflectance")
                    st.pyplot(fig)
                    
                # Cache preprocessed results for ML tabs
                st.session_state['preprocessed_data'] = {
                    'fcc': fcc_img,
                    'ndvi': ndvi,
                    'ndwi': ndwi,
                    'preprocessor': proc
                }
            st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# TAB 3: ANNOTATIONS & AUGMENTATION
# ==========================================
with tabs[3]:
    st.markdown("### 🏷️ Step 3 & 4: Demographic Annotations & Augmentation")
    
    subtabs = st.tabs(["🖌️ VIA Demarcation & Parsing", "➕ Data Augmentation (Albumentations)", "🧠 Generative AI Synthesis"])
    
    # Subtab 3.1: VIA Parsing
    with subtabs[0]:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.write("#### Load VGG Annotator Data")
            st.write("Demarcate boundaries (polygons) around crop fields using VIA. Export annotations as JSON and load them here.")
            
            # Setup standard annotation path
            ann_path = os.path.join(DATA_DIR, "annotations", "sample_annotations.json")
            
            if st.button("Generate Simulated VIA Annotations for active scene", use_container_width=True):
                fname = os.path.basename(st.session_state['current_scene_path']) if st.session_state['current_scene_path'] else "sample_liss3.tif"
                generate_mock_via_annotations(ann_path, filename=fname)
                st.session_state['annotations_loaded'] = True
                st.success("Generated and loaded annotations!")
                st.rerun()
                
            uploaded_file = st.file_uploader("Upload custom VIA Annotations (JSON)", type=["json"])
            if uploaded_file:
                with open(ann_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())
                st.session_state['annotations_loaded'] = True
                st.success("Uploaded annotations loaded successfully!")
                st.rerun()
                
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col2:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.write("#### Demarcated Visualizer")
            if os.path.exists(ann_path):
                # Load FCC
                if st.session_state['preprocessed_data'] is not None:
                    fcc_img = st.session_state['preprocessed_data']['fcc']
                    fname = os.path.basename(st.session_state['current_scene_path'])
                else:
                    # Fallback standard
                    p = RemoteSensingPreprocessor(os.path.join(DATA_DIR, "sample_liss3.tif"))
                    fcc_img = p.generate_fcc()
                    fname = "sample_liss3.tif"
                    
                parser = VIAAnnotationParser(ann_path)
                regions = parser.get_regions_for_image(fname)
                
                if regions:
                    # Draw regions on FCC image
                    fcc_overlay = fcc_img.copy()
                    import cv2
                    for r_idx, reg in enumerate(regions):
                        pts = reg["points"]
                        lbl = reg["label"]
                        
                        # Draw boundaries
                        color = (0, 255, 0) if "healthy" in lbl else ((255, 165, 0) if "stressed" in lbl else (139, 69, 19))
                        cv2.polylines(fcc_overlay, [pts], True, color, 2)
                        # Add text label
                        centroid = np.mean(pts, axis=0).astype(int)
                        cv2.putText(fcc_overlay, f"F-{r_idx+1}:{lbl[:4]}", tuple(centroid), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                        
                    fig, ax = plt.subplots(figsize=(6,6))
                    ax.imshow(fcc_overlay)
                    ax.axis('off')
                    st.pyplot(fig)
                    st.write(f"Detected **{len(regions)}** crop field boundaries in VGG file.")
                else:
                    st.warning(f"No matching regions found in VIA file for image: {fname}")
            else:
                st.info("No annotations generated yet. Click 'Generate Simulated VIA Annotations' to populate.")
            st.markdown("</div>", unsafe_allow_html=True)
            
    # Subtab 3.2: Albumentations
    with subtabs[1]:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.write("#### Augment Image and Mask Simultaneously")
        
        col1_aug, col2_aug = st.columns([1, 2])
        with col1_aug:
            st.write("Parameters")
            aug_rot = st.checkbox("Random 90 degree Rotations", value=True)
            aug_flip = st.checkbox("Flips (Horizontal & Vertical)", value=True)
            aug_noise = st.checkbox("Radiometric Sensor Noise", value=True)
            
            btn_run_aug = st.button("Generate Augmented Sample")
            
        with col2_aug:
            if btn_run_aug:
                # Load preprocessor & annotations
                fname = os.path.basename(st.session_state['current_scene_path']) if st.session_state['current_scene_path'] else "sample_liss3.tif"
                img_path = st.session_state['current_scene_path'] if st.session_state['current_scene_path'] else os.path.join(DATA_DIR, "sample_liss3.tif")
                
                p = RemoteSensingPreprocessor(img_path)
                g = p.get_band("green")
                r = p.get_band("red")
                nir = p.get_band("nir")
                raw_image = np.stack([g, r, nir], axis=0) # bands first
                
                # Load mask
                parser = VIAAnnotationParser(ann_path)
                raw_mask = parser.create_mask_for_image(fname)
                
                # Augment
                augmenter = RemoteSensingAugmenter(use_albumentations=True)
                aug_image_chw, aug_mask = augmenter.augment(raw_image, raw_mask)
                
                # Convert back to HWC for showing
                # Scaling back to FCC displayable format
                fcc_stack = np.stack([aug_image_chw[2], aug_image_chw[1], aug_image_chw[0]], axis=-1)
                fcc_disp = (p.contrast_stretch(fcc_stack) * 255).astype(np.uint8)
                
                col_i1, col_i2 = st.columns(2)
                
                fig1, ax1 = plt.subplots()
                ax1.imshow(fcc_disp)
                ax1.set_title("Augmented Image (FCC)")
                ax1.axis('off')
                col_i1.pyplot(fig1)
                
                fig2, ax2 = plt.subplots()
                ax2.imshow(aug_mask, cmap='viridis', vmin=0, vmax=3)
                ax2.set_title("Coordinated Augmented Mask")
                ax2.axis('off')
                col_i2.pyplot(fig2)
                st.success("Successfully applied coordinated augmentations to both the multi-band image and the mask!")
            else:
                st.info("Click 'Generate Augmented Sample' to run the Albumentations transformation.")
        st.markdown("</div>", unsafe_allow_html=True)
        
    # Subtab 3.3: Generative AI Data Synthesizer
    with subtabs[2]:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.write("#### Generative AI Crop Field Synthesizer")
        st.write("Generate high-fidelity synthetic satellite imagery with matching segmentation masks and labels using the Generative AI simulation engine.")
        
        col1_gen, col2_gen = st.columns([1, 2])
        with col1_gen:
            gen_fields = st.slider("Fields Count", 5, 20, 10)
            gen_size = st.selectbox("Resolution Size", [256, 512], index=1)
            gen_seed = st.number_input("Random Seed", value=101)
            
            btn_generate_ai = st.button("Synthesize Data with Generative AI", type="primary", use_container_width=True)
            
        with col2_gen:
            if btn_generate_ai:
                with st.spinner("Executing Generative Cropland Synthesizer model..."):
                    synthesizer = GenerativeCropSynthesizer(size=gen_size, seed=gen_seed)
                    
                    synth_img_path = os.path.join(DATA_DIR, "synthetic", f"gen_scene_{gen_seed}.tif")
                    synth_ann_path = os.path.join(DATA_DIR, "synthetic", f"gen_scene_{gen_seed}_ann.json")
                    
                    os.makedirs(os.path.dirname(synth_img_path), exist_ok=True)
                    num_fields = synthesizer.generate_synthetic_scene(synth_img_path, synth_ann_path, is_liss4=is_liss4_val)
                    
                    # Display results
                    p = RemoteSensingPreprocessor(synth_img_path)
                    fcc = p.generate_fcc()
                    mask = np.load(synth_img_path.replace(".tif", "_mask.npy"))
                    
                    col_show1, col_show2 = st.columns(2)
                    
                    fig1, ax1 = plt.subplots()
                    ax1.imshow(fcc)
                    ax1.set_title("Generated Image (FCC)")
                    ax1.axis('off')
                    col_show1.pyplot(fig1)
                    
                    fig2, ax2 = plt.subplots()
                    ax2.imshow(mask, cmap='viridis', vmin=0, vmax=3)
                    ax2.set_title("Generated Ground Truth Mask")
                    ax2.axis('off')
                    col_show2.pyplot(fig2)
                    
                    st.success(f"Generated **{num_fields}** synthetic crop fields successfully!")
                    st.write(f"Image Path: `{synth_img_path}`")
                    st.write(f"Annotation Path: `{synth_ann_path}`")
            else:
                st.info("Click the button to run the Generative AI simulation.")
        st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# TAB 4: MODEL TRAINING
# ==========================================
with tabs[4]:
    st.markdown("### 🧠 Step 5: Model Training Panel")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.write("#### Select Model Paradigm")
        model_paradigm = st.radio("Training Type", [
            "Unsupervised (K-Means Spectral Clustering)",
            "Supervised (Mask R-CNN / PyTorch)",
            "Semi-Supervised (Self-Training / Pseudo-Labeling)"
        ])
        
        epochs = st.slider("Training Epochs", 1, 10, 3)
        batch_size = st.selectbox("Batch Size", [1, 2, 4, 8])
        
        btn_train = st.button("Start Model Training", type="primary", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col2:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.write("#### Training Progress & Loss Charts")
        
        if btn_train:
            if "Unsupervised" in model_paradigm:
                if st.session_state['preprocessed_data'] is None:
                    st.warning("Please run Step 2 Preprocessing on the image first.")
                else:
                    with st.spinner("Fitting K-Means model on spectral signatures..."):
                        proc = st.session_state['preprocessed_data']['preprocessor']
                        classifier = UnsupervisedLULCClassifier()
                        mask = classifier.fit_predict(proc)
                        
                        st.success("K-Means fitted successfully!")
                        st.session_state['trained_model'] = {
                            "type": "unsupervised",
                            "model": classifier,
                            "mask": mask
                        }
                        
                        # Show mapping
                        st.write("#### Cluster ID to LULC Class Mapping:")
                        df_map = pd.DataFrame(list(classifier.cluster_labels_map.items()), columns=["Cluster ID", "Assigned Class"])
                        st.dataframe(df_map, hide_index=True)
            
            elif "Supervised" in model_paradigm:
                img_path = st.session_state['current_scene_path'] if st.session_state['current_scene_path'] else os.path.join(DATA_DIR, "sample_liss3.tif")
                ann_path = os.path.join(DATA_DIR, "annotations", "sample_annotations.json")
                
                if not os.path.exists(ann_path):
                    st.warning("Please generate VIA annotations in Tab 3 before supervised training.")
                else:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Set up placeholder lists for loss graphing
                    epoch_losses = []
                    chart_placeholder = st.empty()
                    
                    def callback(epoch, total_epochs, loss):
                        pct = epoch / total_epochs
                        progress_bar.progress(pct)
                        status_text.write(f"Epoch {epoch}/{total_epochs} completed. Loss: **{loss:.4f}**")
                        epoch_losses.append(loss)
                        
                        # Update live chart
                        df_loss = pd.DataFrame(epoch_losses, columns=["Training Loss"])
                        chart_placeholder.line_chart(df_loss)
                        
                    with st.spinner("Training Supervised Mask R-CNN model..."):
                        res = train_supervised_model(
                            image_dir=DATA_DIR,
                            annotation_path=ann_path,
                            epochs=epochs,
                            progress_callback=callback
                        )
                        st.success(f"Mask R-CNN model trained successfully! Weights saved at {res['checkpoint_path']}.")
                        st.session_state['trained_model'] = {
                            "type": "supervised",
                            "checkpoint_path": res['checkpoint_path']
                        }
                        
            elif "Semi-Supervised" in model_paradigm:
                img_path = st.session_state['current_scene_path'] if st.session_state['current_scene_path'] else os.path.join(DATA_DIR, "sample_liss3.tif")
                ann_path = os.path.join(DATA_DIR, "annotations", "sample_annotations.json")
                
                # We need some unlabeled images: generate a synthetic one if not present
                unlabeled_path = os.path.join(DATA_DIR, "synthetic", "gen_scene_101.tif")
                unlabeled_ann = os.path.join(DATA_DIR, "synthetic", "gen_scene_101_ann.json")
                if not os.path.exists(unlabeled_path):
                    synthesizer = GenerativeCropSynthesizer(seed=101)
                    synthesizer.generate_synthetic_scene(unlabeled_path, unlabeled_ann)
                    
                if not os.path.exists(ann_path):
                    st.warning("Please generate VIA annotations in Tab 3 before training.")
                else:
                    status_text = st.empty()
                    progress_bar = st.progress(0)
                    
                    def callback(stage_text, val):
                        status_text.write(stage_text)
                        progress_bar.progress(val)
                        
                    with st.spinner("Running Semi-Supervised training..."):
                        res = run_semi_supervised_training(
                            image_dir=DATA_DIR,
                            labeled_ann_path=ann_path,
                            unlabeled_img_paths=[unlabeled_path],
                            epochs=epochs,
                            progress_callback=callback
                        )
                        st.success("Semi-Supervised model pipeline finished. Weights updated.")
                        st.session_state['trained_model'] = {
                            "type": "semi_supervised",
                            "checkpoint_path": res['checkpoint_path']
                        }
        else:
            if st.session_state['trained_model']:
                st.success(f"Active model loaded in session: **{st.session_state['trained_model']['type']}**")
            else:
                st.info("Select a model paradigm and click 'Start Model Training'.")
        st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# TAB 5: INFERENCE & YIELD PREDICTION
# ==========================================
with tabs[5]:
    st.markdown("### 📈 Automated Field Detection & Crop Yield Prediction")
    
    if st.session_state['preprocessed_data'] is None:
        st.warning("Please process an image in Tab 2 to obtain the preprocessed data and NDVI metrics first.")
    else:
        proc_data = st.session_state['preprocessed_data']
        ndvi = proc_data['ndvi']
        fcc = proc_data['fcc']
        preprocessor = proc_data['preprocessor']
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.write("#### Settings")
            
            # Select target crop configuration
            crop_mode = st.selectbox("Select Crop Type Mapping", ["Varied Crops (Rice, Sugarcane, Cotton)", "Sugarcane Only", "Rice Only"])
            crop_mapping = {
                "Varied Crops (Rice, Sugarcane, Cotton)": ["rice", "sugarcane", "cotton", "wheat", "maize"],
                "Sugarcane Only": ["sugarcane"],
                "Rice Only": ["rice"]
            }
            
            btn_run_inference = st.button("Run Crop Analysis & Localization", type="primary", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col2:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.write("#### Results visualization")
            
            if btn_run_inference:
                # We extract field masks
                with st.spinner("Analyzing field patterns and computing yield..."):
                    if st.session_state['trained_model'] and st.session_state['trained_model']['type'] == "unsupervised":
                        lulc_mask = st.session_state['trained_model']['mask']
                        lbl_map = st.session_state['trained_model']['model'].cluster_labels_map
                        field_mask = np.zeros_like(lulc_mask)
                        for c_id, name in lbl_map.items():
                            if name == "Healthy Crop":
                                field_mask[lulc_mask == c_id] = 1
                            elif name == "Stressed Crop":
                                field_mask[lulc_mask == c_id] = 2
                    else:
                        # Fallback/Default: Run Unsupervised K-Means on the fly to demarcate fields
                        classifier = UnsupervisedLULCClassifier()
                        lulc_mask = classifier.fit_predict(preprocessor)
                        lbl_map = classifier.cluster_labels_map
                        
                        field_mask = np.zeros_like(lulc_mask)
                        for c_id, name in lbl_map.items():
                            if name == "Healthy Crop":
                                field_mask[lulc_mask == c_id] = 1
                            elif name == "Stressed Crop":
                                field_mask[lulc_mask == c_id] = 2
                                
                    # Predict yield
                    predictor = YieldPredictor(pixel_resolution=resolution_val)
                    fields_data = predictor.predict_fields_metrics(field_mask, ndvi, crop_types_list=crop_mapping[crop_mode])
                    
                    if not fields_data:
                        st.warning("No crop fields detected in this scene. Try adjusting preprocessing parameters.")
                    else:
                        # Cache metrics for Tab 0 Overview Dashboard
                        st.session_state['active_field_metrics'] = fields_data
                        
                        # Draw localized contours on FCC image
                        fcc_overlay = fcc.copy()
                        import cv2
                        
                        for fd in fields_data:
                            pts = np.array(fd["polygon"])
                            if len(pts) > 0:
                                pts = pts.astype(np.int32)
                                color = (0, 255, 100) if fd["health"] == "Healthy" else (255, 160, 0)
                                cv2.polylines(fcc_overlay, [pts], True, color, 2)
                                # Centroid
                                centroid = (int(fd["centroid"][1]), int(fd["centroid"][0]))
                                cv2.putText(fcc_overlay, f"ID {fd['field_id']}", centroid, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                                
                        fig, ax = plt.subplots(figsize=(6, 6))
                        ax.imshow(fcc_overlay)
                        ax.axis('off')
                        st.pyplot(fig)
                        
                        # Total Summary Dashboard
                        st.write("#### Total Region Summary:")
                        total_area = sum([f["area_hectares"] for f in fields_data])
                        total_yield = sum([f["yield_tons"] for f in fields_data])
                        mean_health_val = np.mean([f["mean_ndvi"] for f in fields_data])
                        
                        st.markdown(f"""
                        <div class='metric-container'>
                            <div class='metric-box'>
                                <div class='metric-val'>{len(fields_data)}</div>
                                <div class='metric-lbl'>Detected Fields</div>
                            </div>
                            <div class='metric-box'>
                                <div class='metric-val'>{total_area:.2f} ha</div>
                                <div class='metric-lbl'>Total Cropland Area</div>
                            </div>
                            <div class='metric-box'>
                                <div class='metric-val'>{total_yield:.1f} tons</div>
                                <div class='metric-lbl'>Predicted Crop Yield</div>
                            </div>
                            <div class='metric-box'>
                                <div class='metric-val'>{mean_health_val:.3f}</div>
                                <div class='metric-lbl'>Average NDVI Health Index</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Dataframe of fields
                        st.write("#### Detailed Field Classification Table:")
                        df_fields = pd.DataFrame(fields_data)
                        
                        # Custom render formatting
                        df_fields_disp = df_fields[["field_id", "area_hectares", "mean_ndvi", "health", "crop_type", "yield_tons"]].copy()
                        df_fields_disp.columns = ["Field ID", "Area (Hectares)", "NDVI Mean", "Health Status", "Crop Type", "Predicted Yield (Tons)"]
                        st.dataframe(df_fields_disp, use_container_width=True, hide_index=True)
                        
                        # Dynamic dashboard update trigger
                        st.info("📊 Metrics successfully loaded into the main Overview Dashboard! Go to Tab 1 to see the updated visual projections.")
            else:
                st.info("Click 'Run Crop Analysis & Localization' to see detection results.")
            st.markdown("</div>", unsafe_allow_html=True)
