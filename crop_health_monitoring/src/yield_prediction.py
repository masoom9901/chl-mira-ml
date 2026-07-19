import numpy as np

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

class YieldPredictor:
    """
    Localizes fields from predicted segmentation masks, calculates field areas,
    evaluates average NDVI values, and predicts crop yield.
    """
    def __init__(self, pixel_resolution=24.0):
        """
        pixel_resolution: float, pixel size in meters.
        - LISS-3: ~24.0 meters
        - LISS-4: ~5.8 meters
        """
        self.pixel_resolution = pixel_resolution
        # Standard yield multipliers (in metric tons per hectare) for crop types at NDVI=1.0
        self.yield_coefficients = {
            "rice": 4.5,
            "wheat": 3.8,
            "sugarcane": 80.0,
            "cotton": 2.2,
            "maize": 5.5,
            "default": 4.0
        }

    def predict_fields_metrics(self, mask, ndvi, crop_types_list=None):
        """
        Processes a classified mask and NDVI array to analyze each detected field.
        
        Args:
            mask: 2D numpy array containing labeled field regions (1=healthy, 2=stressed, 3=bare soil).
            ndvi: 2D numpy array of NDVI values.
            crop_types_list: list of crop types corresponding to fields (optional).
            
        Returns:
            list of dicts containing localized field metrics.
        """
        if not OPENCV_AVAILABLE:
            # Fallback analysis if OpenCV is missing
            return self._fallback_predict(mask, ndvi, crop_types_list)
            
        h, w = mask.shape
        # Create a binary crop mask (all classes > 0 are fields)
        binary_fields = (mask > 0).astype(np.uint8)
        
        # Remove water body if any from the field mask (usually water has low/negative NDVI)
        # In our unsupervised map, water is cluster 0
        
        num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(binary_fields)
        
        results = []
        for i in range(1, num_labels):
            field_mask = (labels_im == i)
            pixel_count = stats[i, cv2.CC_STAT_AREA]
            
            # Area calculations
            # 1 hectare = 10,000 square meters
            area_sq_meters = pixel_count * (self.pixel_resolution ** 2)
            area_hectares = area_sq_meters / 10000.0
            
            # Extract NDVI values within this field
            field_ndvis = ndvi[field_mask]
            mean_ndvi = float(np.mean(field_ndvis))
            
            # Determine health grade based on mean NDVI
            if mean_ndvi > 0.5:
                health_grade = "Healthy"
            elif mean_ndvi >= 0.25:
                health_grade = "Moderate Stress"
            else:
                health_grade = "Severe Stress"
                
            # Assign crop type
            if crop_types_list and (i - 1) < len(crop_types_list):
                crop_type = crop_types_list[i - 1]
            else:
                # Default assignments based on field index
                crops = ["rice", "wheat", "sugarcane", "cotton", "maize"]
                crop_type = crops[(i - 1) % len(crops)]
                
            # Yield prediction: Area * mean_ndvi * yield_factor
            coeff = self.yield_coefficients.get(crop_type.lower(), self.yield_coefficients["default"])
            # Yield is suppressed if NDVI is low
            predicted_yield = area_hectares * max(0.0, mean_ndvi) * coeff
            
            # Bounding box
            x, y, w_box, h_box = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            
            # Find boundary contours to represent polygons
            contours, _ = cv2.findContours((field_mask * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            polygon_points = []
            if len(contours) > 0:
                # Get the largest contour for the field
                largest_contour = max(contours, key=cv2.contourArea)
                polygon_points = largest_contour.reshape(-1, 2).tolist()
            
            results.append({
                "field_id": i,
                "centroid": (float(centroids[i][1]), float(centroids[i][0])), # (row, col)
                "bbox": [int(x), int(y), int(x + w_box), int(y + h_box)],
                "polygon": polygon_points,
                "area_hectares": round(area_hectares, 3),
                "mean_ndvi": round(mean_ndvi, 3),
                "health": health_grade,
                "crop_type": crop_type.capitalize(),
                "yield_tons": round(predicted_yield, 2)
            })
            
        return results

    def _fallback_predict(self, mask, ndvi, crop_types_list):
        """
        Numpy fallback for field metrics when OpenCV is not available.
        Splits image into equal tiles as simulated fields.
        """
        h, w = mask.shape
        grid_size = h // 4
        results = []
        field_id = 1
        
        for r in range(4):
            for c in range(4):
                r_start, r_end = r * grid_size, (r + 1) * grid_size
                c_start, c_end = c * grid_size, (c + 1) * grid_size
                
                tile_mask = mask[r_start:r_end, c_start:c_end]
                tile_ndvi = ndvi[r_start:r_end, c_start:c_end]
                
                # Check if it contains field pixels
                if np.sum(tile_mask > 0) > (grid_size * grid_size * 0.1):
                    pixel_count = np.sum(tile_mask > 0)
                    area_hectares = (pixel_count * (self.pixel_resolution ** 2)) / 10000.0
                    mean_ndvi = float(np.mean(tile_ndvi[tile_mask > 0]))
                    
                    if mean_ndvi > 0.5:
                        health = "Healthy"
                    elif mean_ndvi >= 0.25:
                        health = "Moderate Stress"
                    else:
                        health = "Severe Stress"
                        
                    crop_type = "Rice"
                    coeff = self.yield_coefficients["rice"]
                    predicted_yield = area_hectares * max(0.0, mean_ndvi) * coeff
                    
                    results.append({
                        "field_id": field_id,
                        "centroid": (r_start + grid_size/2, c_start + grid_size/2),
                        "bbox": [c_start, r_start, c_end, r_end],
                        "polygon": [[c_start, r_start], [c_end, r_start], [c_end, r_end], [c_start, r_end]],
                        "area_hectares": round(area_hectares, 3),
                        "mean_ndvi": round(mean_ndvi, 3),
                        "health": health,
                        "crop_type": crop_type,
                        "yield_tons": round(predicted_yield, 2)
                    })
                    field_id += 1
                    
        return results

if __name__ == "__main__":
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:40, 10:40] = 1 # field 1
    mask[50:80, 50:80] = 2 # field 2
    
    ndvi = np.zeros((100, 100), dtype=np.float32)
    ndvi[10:40, 10:40] = 0.65
    ndvi[50:80, 50:80] = 0.35
    
    predictor = YieldPredictor(pixel_resolution=24.0)
    res = predictor.predict_fields_metrics(mask, ndvi)
    print("Detected fields count:", len(res))
    for f in res:
        print(f"Field {f['field_id']}: Area={f['area_hectares']} ha, Health={f['health']}, Yield={f['yield_tons']} tons")
