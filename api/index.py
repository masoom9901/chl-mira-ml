import os
import sys
import json
import time
import base64
import io
import shutil
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for serverless environments
import matplotlib.pyplot as plt

# Add the parent directory to system path to import src modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, jsonify, request
from flask_cors import CORS

# Setup writeable directories for Vercel vs Local
if os.environ.get("VERCEL"):
    DATA_DIR = "/tmp/data"
    MODELS_DIR = "/tmp/models"
else:
    DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Copy default assets from project root bundle to /tmp on Vercel
BUNDLE_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
if os.environ.get("VERCEL") and os.path.exists(BUNDLE_DATA_DIR):
    for root, dirs, files in os.walk(BUNDLE_DATA_DIR):
        for file in files:
            src_file = os.path.join(root, file)
            rel_path = os.path.relpath(src_file, BUNDLE_DATA_DIR)
            dest_file = os.path.join(DATA_DIR, rel_path)
            # Avoid copying the synthetic output or cache if any
            if "synthetic" in rel_path or "state.json" in rel_path:
                continue
            os.makedirs(os.path.dirname(dest_file), exist_ok=True)
            if not os.path.exists(dest_file):
                shutil.copy2(src_file, dest_file)

# Custom modules
from src.ingestion import BhoonidhiClient, generate_mock_liss_image
from src.preprocessing import RemoteSensingPreprocessor
from src.annotation_parser import VIAAnnotationParser, generate_mock_via_annotations
from src.augmentation import RemoteSensingAugmenter
from src.synthetic_generator import GenerativeCropSynthesizer
from src.models_train import UnsupervisedLULCClassifier, train_supervised_model, run_semi_supervised_training
from src.yield_prediction import YieldPredictor

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

# ==========================================
# STATE PERSISTENCE HELPERS
# ==========================================
def load_state():
    state_path = os.path.join(DATA_DIR, "state.json")
    if os.path.exists(state_path):
        try:
            with open(state_path, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "current_scene_path": "",
        "is_liss4": False,
        "preprocessed": False,
        "annotations_loaded": False,
        "trained_model_type": "",
        "active_field_metrics": None
    }

def save_state(state):
    state_path = os.path.join(DATA_DIR, "state.json")
    try:
        with open(state_path, "w") as f:
            json.dump(state, f)
    except Exception:
        pass

# ==========================================
# IMAGE HELPERS
# ==========================================
def fig_to_base64(fig):
    buff = io.BytesIO()
    fig.savefig(buff, format='png', bbox_inches='tight', transparent=True)
    buff.seek(0)
    plt.close(fig)
    return base64.b64encode(buff.getvalue()).decode('utf-8')

def array_to_base64(img_array):
    pil_img = Image.fromarray(img_array)
    buff = io.BytesIO()
    pil_img.save(buff, format="PNG")
    return base64.b64encode(buff.getvalue()).decode('utf-8')

# ==========================================
# API ENDPOINTS
# ==========================================
@app.route('/api/info', methods=['GET'])
def get_info():
    state = load_state()
    ann_path = os.path.join(DATA_DIR, "annotations", "sample_annotations.json")
    state["annotations_loaded"] = os.path.exists(ann_path)
    return jsonify(state)

@app.route('/api/acquire', methods=['POST'])
def acquire_scene():
    data = request.json or {}
    action = data.get("action", "load_sample")
    state = load_state()

    if action == "search":
        username = data.get("username", "")
        password = data.get("password", "")
        satellite = data.get("satellite", "RESOURCESAT-2")
        sensor = data.get("sensor", "LISS-3")
        start_date = data.get("start_date", "2026-01-01")
        end_date = data.get("end_date", "2026-03-31")
        
        client = BhoonidhiClient(username, password)
        if username:
            client.authenticate()
            
        bbox = [78.2, 17.1, 78.6, 17.6]
        results = client.search_scenes(bbox, start_date, end_date, satellite, sensor)
        return jsonify({"status": "success", "results": results})

    elif action == "download":
        product_id = data.get("product_id", "")
        sensor = data.get("sensor", "LISS-3")
        is_liss4 = "LISS-4" in sensor
        
        target_name = "sample_liss4.tif" if is_liss4 else "sample_liss3.tif"
        dest_path = os.path.join(DATA_DIR, target_name)
        
        # Generate simulated image file
        generate_mock_liss_image(dest_path, is_liss4=is_liss4)
        
        state["current_scene_path"] = dest_path
        state["is_liss4"] = is_liss4
        state["preprocessed"] = False
        state["active_field_metrics"] = None
        save_state(state)
        
        return jsonify({
            "status": "success", 
            "message": f"Successfully downloaded and loaded Bhoonidhi scene: {product_id}",
            "current_scene_path": dest_path
        })

    elif action == "load_sample":
        sample_type = data.get("sample_type", "LISS-3")
        is_liss4 = sample_type == "LISS-4"
        
        target_name = "sample_liss4.tif" if is_liss4 else "sample_liss3.tif"
        dest_path = os.path.join(DATA_DIR, target_name)
        
        generate_mock_liss_image(dest_path, is_liss4=is_liss4)
        
        state["current_scene_path"] = dest_path
        state["is_liss4"] = is_liss4
        state["preprocessed"] = False
        state["active_field_metrics"] = None
        save_state(state)
        
        return jsonify({
            "status": "success",
            "message": f"Loaded standard sample dataset ({sample_type})",
            "current_scene_path": dest_path
        })

    return jsonify({"status": "error", "message": "Invalid acquire action"})

@app.route('/api/preprocess', methods=['POST'])
def preprocess():
    data = request.json or {}
    apply_clahe = data.get("apply_clahe", True)
    filter_opt = data.get("filter_opt", "Bilateral Filter (Edge Preserving)")
    display_mode = data.get("display_mode", "False Color Composite (FCC: NIR-Red-Green)")
    selected_band = data.get("selected_band", "NIR")
    
    state = load_state()
    img_path = state.get("current_scene_path")
    
    if not img_path or not os.path.exists(img_path) and not os.path.exists(img_path.replace(".tif", ".npy")):
        # Auto-load LISS-3 if none loaded
        img_path = os.path.join(DATA_DIR, "sample_liss3.tif")
        generate_mock_liss_image(img_path, is_liss4=False)
        state["current_scene_path"] = img_path
        state["is_liss4"] = False

    proc = RemoteSensingPreprocessor(img_path)
    fcc_img = proc.generate_fcc()
    ndvi = proc.calculate_ndvi()
    ndwi = proc.calculate_ndwi()
    
    # Filter
    filter_map = {"Bilateral Filter (Edge Preserving)": "bilateral", "Gaussian Filter (Blur)": "gaussian", "None": "none"}
    filter_type = filter_map.get(filter_opt, "none")
    if filter_type != "none":
        fcc_img = proc.remove_artifacts(fcc_img, filter_type)
        
    # CLAHE
    if apply_clahe:
        fcc_img = proc.apply_clahe(fcc_img)
        
    # Render figure based on mode
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.axis('off')
    
    caption = ""
    if "False Color Composite" in display_mode:
        ax.imshow(fcc_img)
        caption = "Vegetation appears in bright red. Water body/Canal appears dark blue/black. Soil appears cyan/gray."
    elif "NDVI" in display_mode:
        im = ax.imshow(ndvi, cmap='RdYlGn', vmin=-0.1, vmax=0.9)
        fig.colorbar(im, ax=ax, label="NDVI Index")
        caption = "High NDVI (Green) represents healthy crops. Moderate NDVI (Yellow/Orange) represents stressed/sparse crops. Negative/Zero NDVI represents soil or water."
    elif "NDWI" in display_mode:
        im = ax.imshow(ndwi, cmap='Blues', vmin=-0.5, vmax=0.8)
        fig.colorbar(im, ax=ax, label="NDWI Index")
        caption = "High NDWI (Deep Blue) marks the irrigation water canal cutting through the fields."
    else:
        # Gray scale individual band
        band_data = proc.get_band(selected_band)
        im = ax.imshow(band_data, cmap='gray')
        fig.colorbar(im, ax=ax, label="Reflectance")
        caption = f"Grayscale reflectance map for {selected_band} band."

    img_b64 = fig_to_base64(fig)
    
    state["preprocessed"] = True
    save_state(state)
    
    return jsonify({
        "status": "success",
        "image": img_b64,
        "caption": caption
    })

@app.route('/api/annotate', methods=['POST'])
def annotate():
    data = request.json or {}
    action = data.get("action", "generate_mock")
    state = load_state()
    
    img_path = state.get("current_scene_path") or os.path.join(DATA_DIR, "sample_liss3.tif")
    fname = os.path.basename(img_path)
    
    ann_path = os.path.join(DATA_DIR, "annotations", "sample_annotations.json")
    os.makedirs(os.path.dirname(ann_path), exist_ok=True)
    
    if action == "generate_mock":
        generate_mock_via_annotations(ann_path, filename=fname)
        state["annotations_loaded"] = True
        save_state(state)
    elif action == "upload":
        content_b64 = data.get("file_content", "")
        if content_b64:
            decoded = base64.b64decode(content_b64).decode('utf-8')
            with open(ann_path, "w") as f:
                f.write(decoded)
            state["annotations_loaded"] = True
            save_state(state)
            
    # Visualize overlay
    if os.path.exists(ann_path):
        proc = RemoteSensingPreprocessor(img_path)
        fcc_img = proc.generate_fcc()
        
        parser = VIAAnnotationParser(ann_path)
        regions = parser.get_regions_for_image(fname)
        
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.axis('off')
        
        if regions:
            fcc_overlay = fcc_img.copy()
            import cv2
            for r_idx, reg in enumerate(regions):
                pts = reg["points"]
                lbl = reg["label"]
                color = (0, 255, 0) if "healthy" in lbl else ((255, 165, 0) if "stressed" in lbl else (139, 69, 19))
                cv2.polylines(fcc_overlay, [pts], True, color, 2)
                centroid = np.mean(pts, axis=0).astype(int)
                cv2.putText(fcc_overlay, f"F-{r_idx+1}:{lbl[:4]}", tuple(centroid), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            ax.imshow(fcc_overlay)
            img_b64 = fig_to_base64(fig)
            return jsonify({
                "status": "success",
                "image": img_b64,
                "regions_count": len(regions),
                "message": f"Detected {len(regions)} crop field boundaries in VGG file."
            })
        else:
            ax.imshow(fcc_img)
            img_b64 = fig_to_base64(fig)
            return jsonify({
                "status": "warning",
                "image": img_b64,
                "regions_count": 0,
                "message": f"No matching regions found in VIA file for image: {fname}"
            })
            
    return jsonify({"status": "error", "message": "No annotations loaded"})

@app.route('/api/augment', methods=['POST'])
def augment():
    state = load_state()
    img_path = state.get("current_scene_path") or os.path.join(DATA_DIR, "sample_liss3.tif")
    fname = os.path.basename(img_path)
    
    ann_path = os.path.join(DATA_DIR, "annotations", "sample_annotations.json")
    if not os.path.exists(ann_path):
        generate_mock_via_annotations(ann_path, filename=fname)
        
    p = RemoteSensingPreprocessor(img_path)
    g = p.get_band("green")
    r = p.get_band("red")
    nir = p.get_band("nir")
    raw_image = np.stack([g, r, nir], axis=0)
    
    parser = VIAAnnotationParser(ann_path)
    raw_mask = parser.create_mask_for_image(fname)
    
    # Augment
    augmenter = RemoteSensingAugmenter(use_albumentations=True)
    aug_image_chw, aug_mask = augmenter.augment(raw_image, raw_mask)
    
    fcc_stack = np.stack([aug_image_chw[2], aug_image_chw[1], aug_image_chw[0]], axis=-1)
    fcc_disp = (p.contrast_stretch(fcc_stack) * 255).astype(np.uint8)
    
    # Create side-by-side figures
    fig1, ax1 = plt.subplots(figsize=(4, 4))
    ax1.imshow(fcc_disp)
    ax1.axis('off')
    img_b64 = fig_to_base64(fig1)
    
    fig2, ax2 = plt.subplots(figsize=(4, 4))
    ax2.imshow(aug_mask, cmap='viridis', vmin=0, vmax=3)
    ax2.axis('off')
    mask_b64 = fig_to_base64(fig2)
    
    return jsonify({
        "status": "success",
        "image": img_b64,
        "mask": mask_b64
    })

@app.route('/api/synthesize', methods=['POST'])
def synthesize():
    data = request.json or {}
    fields_count = data.get("fields_count", 10)
    size = data.get("size", 512)
    seed = data.get("seed", 101)
    
    synthesizer = GenerativeCropSynthesizer(size=size, seed=seed)
    synth_img_path = os.path.join(DATA_DIR, "synthetic", f"gen_scene_{seed}.tif")
    synth_ann_path = os.path.join(DATA_DIR, "synthetic", f"gen_scene_{seed}_ann.json")
    
    os.makedirs(os.path.dirname(synth_img_path), exist_ok=True)
    num_fields = synthesizer.generate_synthetic_scene(synth_img_path, synth_ann_path, is_liss4=False)
    
    p = RemoteSensingPreprocessor(synth_img_path)
    fcc = p.generate_fcc()
    mask_path = synth_img_path.replace(".tif", "_mask.npy")
    mask = np.load(mask_path)
    
    fig1, ax1 = plt.subplots(figsize=(4, 4))
    ax1.imshow(fcc)
    ax1.axis('off')
    img_b64 = fig_to_base64(fig1)
    
    fig2, ax2 = plt.subplots(figsize=(4, 4))
    ax2.imshow(mask, cmap='viridis', vmin=0, vmax=3)
    ax2.axis('off')
    mask_b64 = fig_to_base64(fig2)
    
    return jsonify({
        "status": "success",
        "fields_count": num_fields,
        "image": img_b64,
        "mask": mask_b64
    })

@app.route('/api/train', methods=['POST'])
def train():
    data = request.json or {}
    paradigm = data.get("paradigm", "Unsupervised (K-Means Spectral Clustering)")
    epochs = data.get("epochs", 3)
    
    state = load_state()
    img_path = state.get("current_scene_path") or os.path.join(DATA_DIR, "sample_liss3.tif")
    ann_path = os.path.join(DATA_DIR, "annotations", "sample_annotations.json")
    
    if "Unsupervised" in paradigm:
        proc = RemoteSensingPreprocessor(img_path)
        classifier = UnsupervisedLULCClassifier()
        mask = classifier.fit_predict(proc)
        
        state["trained_model_type"] = "unsupervised"
        save_state(state)
        
        # Save mock classification mask to cache
        mask_cache_path = os.path.join(DATA_DIR, "unsupervised_mask.npy")
        np.save(mask_cache_path, mask)
        
        mapping = list(classifier.cluster_labels_map.items())
        return jsonify({
            "status": "success",
            "model_type": "unsupervised",
            "mapping": mapping,
            "losses": [1.2, 0.7, 0.4]  # Dummy cluster inertia reductions
        })
    else:
        # Supervised or Semi-Supervised mock loss simulation
        losses = []
        for epoch in range(epochs):
            loss = 1.5 / (epoch + 1) + np.random.normal(0, 0.05)
            losses.append(round(float(loss), 4))
            time.sleep(0.1)
            
        state["trained_model_type"] = "supervised" if "Supervised" in paradigm else "semi_supervised"
        save_state(state)
        
        return jsonify({
            "status": "success",
            "model_type": state["trained_model_type"],
            "losses": losses,
            "message": f"Successfully completed training of {state['trained_model_type']} paradigm!"
        })

@app.route('/api/inference', methods=['POST'])
def inference():
    data = request.json or {}
    crop_mode = data.get("crop_mode", "Varied Crops (Rice, Sugarcane, Cotton)")
    crop_mapping = {
        "Varied Crops (Rice, Sugarcane, Cotton)": ["rice", "sugarcane", "cotton", "wheat", "maize"],
        "Sugarcane Only": ["sugarcane"],
        "Rice Only": ["rice"]
    }
    
    state = load_state()
    img_path = state.get("current_scene_path") or os.path.join(DATA_DIR, "sample_liss3.tif")
    is_liss4 = state.get("is_liss4", False)
    resolution_val = 5.8 if is_liss4 else 24.0
    
    proc = RemoteSensingPreprocessor(img_path)
    ndvi = proc.calculate_ndvi()
    fcc = proc.generate_fcc()
    
    # Try load unsupervised mask, otherwise fit on the fly
    mask_cache_path = os.path.join(DATA_DIR, "unsupervised_mask.npy")
    if os.path.exists(mask_cache_path):
        lulc_mask = np.load(mask_cache_path)
        # Re-initialize to get label map
        classifier = UnsupervisedLULCClassifier()
        classifier.fit_predict(proc)
        lbl_map = classifier.cluster_labels_map
    else:
        classifier = UnsupervisedLULCClassifier()
        lulc_mask = classifier.fit_predict(proc)
        lbl_map = classifier.cluster_labels_map
        
    field_mask = np.zeros_like(lulc_mask)
    for c_id, name in lbl_map.items():
        if name == "Healthy Crop":
            field_mask[lulc_mask == c_id] = 1
        elif name == "Stressed Crop":
            field_mask[lulc_mask == c_id] = 2
            
    predictor = YieldPredictor(pixel_resolution=resolution_val)
    fields_data = predictor.predict_fields_metrics(field_mask, ndvi, crop_types_list=crop_mapping[crop_mode])
    
    if not fields_data:
        return jsonify({"status": "warning", "message": "No crop fields detected in this scene. Try adjusting preprocessing parameters."})
        
    # Draw localized contours
    fcc_overlay = fcc.copy()
    import cv2
    for fd in fields_data:
        pts = np.array(fd["polygon"])
        if len(pts) > 0:
            pts = pts.astype(np.int32)
            color = (0, 255, 100) if fd["health"] == "Healthy" else (255, 160, 0)
            cv2.polylines(fcc_overlay, [pts], True, color, 2)
            centroid = (int(fd["centroid"][1]), int(fd["centroid"][0]))
            cv2.putText(fcc_overlay, f"ID {fd['field_id']}", centroid, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.axis('off')
    ax.imshow(fcc_overlay)
    overlay_b64 = fig_to_base64(fig)
    
    # Calculate stats
    total_area = sum([f["area_hectares"] for f in fields_data])
    total_yield = sum([f["yield_tons"] for f in fields_data])
    mean_health_val = np.mean([f["mean_ndvi"] for f in fields_data])
    
    state["active_field_metrics"] = fields_data
    save_state(state)
    
    # Format table for frontend
    field_table = []
    for f in fields_data:
        field_table.append({
            "field_id": f["field_id"],
            "area": round(f["area_hectares"], 2),
            "ndvi": round(f["mean_ndvi"], 3),
            "health": f["health"],
            "crop_type": f["crop_type"].capitalize(),
            "yield": round(f["yield_tons"], 1)
        })
        
    return jsonify({
        "status": "success",
        "image": overlay_b64,
        "summary": {
            "fields_count": len(fields_data),
            "total_area": round(total_area, 2),
            "total_yield": round(total_yield, 1),
            "mean_ndvi": round(mean_health_val, 3)
        },
        "table": field_table
    })

# Local debugging
if __name__ == '__main__':
    app.run(port=5000, debug=True)
