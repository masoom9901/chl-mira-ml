import os
import json
import numpy as np

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

class VIAAnnotationParser:
    """
    Parses VGG Image Annotator (VIA) JSON outputs and handles rasterization to masks.
    """
    def __init__(self, json_path):
        self.json_path = json_path
        self.annotations = {}
        self.load_annotations()

    def load_annotations(self):
        """
        Loads and parses the VIA JSON structure.
        """
        if not os.path.exists(self.json_path):
            raise FileNotFoundError(f"Annotation file not found: {self.json_path}")
            
        with open(self.json_path, 'r') as f:
            data = json.load(f)
            
        # VIA version 2.x standard format
        if '_via_img_metadata' in data:
            metadata = data['_via_img_metadata']
        else:
            # Sometimes VIA JSON is just a flat dict of files
            metadata = data
            
        for key, val in metadata.items():
            filename = val.get('filename')
            regions = val.get('regions', [])
            
            parsed_regions = []
            for region in regions:
                shape = region.get('shape_attributes', {})
                attribs = region.get('region_attributes', {})
                
                # We only support polygon shapes for cropland demarcation
                if shape.get('name') == 'polygon':
                    xs = shape.get('all_points_x', [])
                    ys = shape.get('all_points_y', [])
                    points = np.stack([xs, ys], axis=-1).astype(np.int32)
                    
                    # Extract class label (defaulting to healthy_crop)
                    label = attribs.get('class') or attribs.get('label') or "healthy_crop"
                    
                    parsed_regions.append({
                        "points": points,
                        "label": label,
                        "attributes": attribs
                    })
            
            if parsed_regions:
                self.annotations[filename] = parsed_regions

    def get_parsed_images(self):
        """Returns list of image filenames with annotations."""
        return list(self.annotations.keys())

    def get_regions_for_image(self, filename):
        """Returns raw region definitions for a specific image."""
        return self.annotations.get(filename, [])

    def create_mask_for_image(self, filename, height=512, width=512):
        """
        Generates a multiclass mask for a given image.
        0: Background
        1: Healthy Crop
        2: Stressed Crop
        3: Bare Soil
        """
        mask = np.zeros((height, width), dtype=np.uint8)
        class_mapping = {
            "background": 0,
            "healthy_crop": 1,
            "stressed_crop": 2,
            "bare_soil": 3
        }
        
        regions = self.get_regions_for_image(filename)
        for region in regions:
            points = region["points"]
            label = region["label"].lower().strip()
            class_val = class_mapping.get(label, 1) # Default to 1 (healthy) if class is custom
            
            if OPENCV_AVAILABLE:
                # Fill the polygon inside the mask
                cv2.fillPoly(mask, [points], class_val)
            else:
                # Simple numpy-based polygon rasterizer as fallback if OpenCV is missing
                # (Normally opencv-python will be installed via requirements.txt)
                mask = self._fallback_fill_poly(mask, points, class_val)
                
        return mask

    def _fallback_fill_poly(self, mask, points, val):
        """Simple bounding box approximation for polygon filling in case OpenCV is absent."""
        h, w = mask.shape
        xs, ys = points[:, 0], points[:, 1]
        min_x, max_x = max(0, xs.min()), min(w - 1, xs.max())
        min_y, max_y = max(0, ys.min()), min(h - 1, ys.max())
        
        # Simple point-in-polygon check or simple fill
        # Here we just fill the bounding box for simplicity in fallback mode
        mask[min_y:max_y+1, min_x:max_x+1] = val
        return mask

def generate_mock_via_annotations(json_path, filename="sample_liss3.tif", size=512):
    """
    Generates a mock VIA JSON annotation file corresponding to the mock LISS image.
    This creates 16 grid-based polygons corresponding to the crop fields.
    """
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    grid_size = size // 4
    regions = []
    
    for i in range(4):
        for j in range(4):
            # Define field corners (clockwise)
            x_start, x_end = j * grid_size, (j + 1) * grid_size
            y_start, y_end = i * grid_size, (i + 1) * grid_size
            
            # Add slight jitter to simulate hand-drawn polygons
            jitter = 4
            x1 = x_start + np.random.randint(0, jitter)
            y1 = y_start + np.random.randint(0, jitter)
            x2 = x_end - np.random.randint(0, jitter)
            y2 = y_start + np.random.randint(0, jitter)
            x3 = x_end - np.random.randint(0, jitter)
            y3 = y_end - np.random.randint(0, jitter)
            x4 = x_start + np.random.randint(0, jitter)
            y4 = y_end - np.random.randint(0, jitter)
            
            # Map label
            if i < 2:
                label = "healthy_crop"
            elif i == 2:
                label = "stressed_crop"
            else:
                label = "bare_soil"
                
            regions.append({
                "shape_attributes": {
                    "name": "polygon",
                    "all_points_x": [int(x1), int(x2), int(x3), int(x4)],
                    "all_points_y": [int(y1), int(y2), int(y3), int(y4)]
                },
                "region_attributes": {
                    "class": label,
                    "crop_type": "rice" if (i+j)%2 == 0 else "wheat"
                }
            })
            
    # Structure matching VIA JSON
    img_size = size * size * 3 # mock size
    file_key = f"{filename}{img_size}"
    
    via_data = {
        "_via_settings": {},
        "_via_img_metadata": {
            file_key: {
                "filename": filename,
                "size": img_size,
                "regions": regions,
                "file_attributes": {}
            }
        },
        "_via_attributes": {
            "region": {
                "class": {
                    "type": "dropdown",
                    "description": "Health classification",
                    "options": {
                        "healthy_crop": "Healthy Crop",
                        "stressed_crop": "Stressed Crop",
                        "bare_soil": "Bare Soil"
                    },
                    "default_value": "healthy_crop"
                }
            },
            "file": {}
        }
    }
    
    with open(json_path, 'w') as f:
        json.dump(via_data, f, indent=2)
    print(f"Generated mock VIA annotations at {json_path}")

if __name__ == "__main__":
    json_path = "../data/annotations/sample_annotations.json"
    generate_mock_via_annotations(json_path)
    parser = VIAAnnotationParser(json_path)
    print("Parsed images:", parser.get_parsed_images())
    mask = parser.create_mask_for_image("sample_liss3.tif")
    print("Mask shape:", mask.shape, "Unique values in mask:", np.unique(mask))
