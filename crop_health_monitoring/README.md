# AgriSense: Crop Health Monitoring & Yield Prediction Platform

AgriSense is a multi-spectral remote sensing analysis and machine learning platform designed to identify crop fields, monitor crop health, and predict yield tonnage from satellite imagery (specifically LISS-3 and LISS-4 imagery from ISRO's Bhoonidhi portal).

---

## Project Structure

```text
crop_health_monitoring/
├── app.py                     # Premium Streamlit web dashboard interface
├── requirements.txt           # Project package dependencies
├── README.md                  # Project documentation & execution guide
├── src/                       # Source modules for processing & models
│   ├── ingestion.py           # Bhoonidhi STAC client & mock satellite scene generator
│   ├── preprocessing.py       # Band stacking, False Color Composite (FCC), NDVI, NDWI & enhancements
│   ├── annotation_parser.py   # VGG Image Annotator (VIA) JSON polygon parser & mask generator
│   ├── augmentation.py        # Albumentations geometric & radiometric augmentation pipelines
│   ├── synthetic_generator.py # Generative AI crop field texture and label synthesizer
│   ├── models_train.py        # Unsupervised (K-Means), Supervised (Mask R-CNN), & Semi-Supervised models
│   └── yield_prediction.py    # Connected components field localization, health grading, & yield regression
└── data/                      # Sample data and annotations (auto-populated by client/generator)
    ├── sample_liss3.tif       # Simulated 4-band LISS-3 scene (Green, Red, NIR, SWIR)
    ├── sample_liss4.tif       # Simulated 3-band LISS-4 scene (Green, Red, NIR)
    └── annotations/
        └── sample_annotations.json # Mock VGG Image Annotator JSON file
```

---

## 🛠️ Step-by-Step Pipeline Implementations

1. **Step 1: Data Acquisition (Bhoonidhi Portal)**
   - Query ISRO's STAC catalog search programmatically for LISS-3 or LISS-4 scenes.
   - Fallback simulation downloads simulated multi-spectral TIFFs replicating agricultural zones near Hyderabad, India.

2. **Step 2: Preprocessing & Enhancements**
   - Loads GeoTIFF bands via `rasterio`.
   - Computes False Color Composite (FCC) stacks (mapping Near-Infrared to Red, Red to Green, Green to Blue) to visually separate vegetation health.
   - Calculates vegetation (NDVI) and water (NDWI) indices.
   - Enhances contrast using Adaptive CLAHE and removes sensor anomalies using Bilateral filtering.

3. **Step 3: Demarcation & Annotation Parser**
   - Parses boundary polygons (X and Y coordinates list) exported from VGG Image Annotator (VIA).
   - Rasterizes polygons into multi-class pixel segmentation masks.

4. **Step 4: Data Augmentation & Generative AI Data Synthesizer**
   - Coordinated data augmentation using `albumentations` applying random shifts, rotations, scaling, flips, and radiometric perturbations to image-mask pairs.
   - **Generative AI Sim**: The `GenerativeCropSynthesizer` generates new synthetic crop fields using Voronoi partitioning, adds aligned crop row patterns, lays down roads and irrigation canals, and outputs corresponding ground-truth segmentation masks.

5. **Step 5: Model Training**
   - **Unsupervised**: Fits `KMeans` to spectral pixel signatures to automatically cluster vegetation, bare soil, and water canal zones without labels.
   - **Supervised**: Trains a PyTorch `MaskRCNN` model to detect and segment individual fields.
   - **Semi-Supervised**: Pseudo-labels unlabeled scenes using a trained model, filters predictions, and retrains with pseudo-annotations.

6. **Step 6: Localization & Yield Prediction**
   - Segments crop fields, computes areas in hectares, measures average NDVI inside each field boundary, assigns a health grade (Healthy, Moderate Stress, or Severe Stress), and outputs predicted yield.

---

## 🚀 Setup & Launch Instructions

### 1. Set Workspace Directory
We recommend setting the active workspace directory in your IDE to:
```text
C:\Users\iitn0\.gemini\antigravity-ide\scratch\crop_health_monitoring
```

### 2. Install Dependencies
Make sure Python is installed (Python 3.10 to 3.12 recommended), then run:
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit Dashboard
Launch the web interface locally using the following command:
```bash
streamlit run app.py
```
After executing, Streamlit will print the local URL (typically `http://localhost:8501`). Open it in your web browser to interact with the platform.
