import os
import numpy as np
try:
    import rasterio
    from rasterio.transform import from_origin
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False
import json

class GenerativeCropSynthesizer:
    """
    A generative engine that creates synthetic multi-spectral satellite cropland imagery
    and matching pixel-perfect segmentation masks. This acts as a Generative AI simulator
    for data augmentation.
    """
    def __init__(self, size=512, seed=None):
        self.size = size
        if seed is not None:
            np.random.seed(seed)

    def _generate_voronoi_fields(self, num_points=12):
        """
        Generates a Voronoi diagram to split the image into organic-looking crop field polygons.
        Returns a label map where each pixel contains the ID of its field.
        """
        # Generate random centroids for the fields
        points = np.random.randint(0, self.size, size=(num_points, 2))
        
        # Grid of coordinates
        x = np.arange(self.size)
        y = np.arange(self.size)
        xx, yy = np.meshgrid(x, y)
        
        # Compute distance to each centroid
        # shape: (num_points, size, size)
        distances = np.zeros((num_points, self.size, self.size))
        for idx, (px, py) in enumerate(points):
            distances[idx] = (xx - px)**2 + (yy - py)**2
            
        # Find closest centroid index for each pixel
        field_map = np.argmin(distances, axis=0)
        return field_map, points

    def generate_synthetic_scene(self, output_img_path, output_json_path, is_liss4=False):
        """
        Generates a synthetic scene with:
        - Multi-spectral image (TIFF/npy)
        - Segmentation mask (0=bg/road, 1=healthy crop, 2=stressed crop, 3=bare soil)
        - VIA format JSON annotations containing the bounding boxes/polygons of fields
        """
        num_bands = 3 if is_liss4 else 4
        field_map, centroids = self._generate_voronoi_fields(num_points=15)
        
        # Assign properties to each field ID
        # Fields categories: 1 = healthy (40%), 2 = stressed (40%), 3 = bare soil (20%)
        num_fields = len(centroids)
        field_types = np.random.choice([1, 2, 3], size=num_fields, p=[0.4, 0.4, 0.2])
        
        # Define base crop row orientations (angles in radians)
        field_angles = np.random.uniform(0, np.pi, size=num_fields)
        # Crop row spacings (frequencies)
        field_freqs = np.random.uniform(0.1, 0.2, size=num_fields)
        
        # Initialize multi-spectral band data
        # shape: (bands, size, size)
        data = np.zeros((num_bands, self.size, self.size), dtype=np.uint16)
        mask = np.zeros((self.size, self.size), dtype=np.uint8)
        
        # Grid coordinates for crop rows calculation
        x = np.arange(self.size)
        y = np.arange(self.size)
        xx, yy = np.meshgrid(x, y)
        
        for field_id in range(num_fields):
            ftype = field_types[field_id]
            angle = field_angles[field_id]
            freq = field_freqs[field_id]
            
            # Mask of this field
            f_mask = (field_map == field_id)
            mask[f_mask] = ftype
            
            # Define band base reflectances
            if ftype == 1: # Healthy crop
                g_base, r_base, nir_base, swir_base = 0.14, 0.05, 0.75, 0.12
            elif ftype == 2: # Stressed crop
                g_base, r_base, nir_base, swir_base = 0.17, 0.14, 0.42, 0.24
            else: # Bare soil
                g_base, r_base, nir_base, swir_base = 0.12, 0.23, 0.25, 0.40
                
            # Create crop row lines using oriented sine waves
            # row_val ranges [-0.5, 0.5]
            row_val = np.sin((xx * np.cos(angle) + yy * np.sin(angle)) * freq) * 0.5
            
            # If bare soil, we don't have prominent crop rows (or very weak soil tilling lines)
            if ftype == 3:
                row_val *= 0.15
                
            # Add fractal Perlin-like low frequency soil noise
            soil_noise = np.sin(xx / 40.0) * np.cos(yy / 40.0) * 0.05
            
            # Combine bases, crop rows, and soil noise with local variations
            g_band = g_base + row_val * 0.02 + soil_noise * 0.01 + np.random.normal(0, 0.01, (self.size, self.size))
            r_band = r_base - row_val * 0.01 + soil_noise * 0.02 + np.random.normal(0, 0.01, (self.size, self.size))
            
            # For NIR, healthy crops have highly active reflectance
            nir_row_amp = 0.15 if ftype == 1 else (0.08 if ftype == 2 else 0.02)
            nir_band = nir_base + row_val * nir_row_amp + soil_noise * 0.03 + np.random.normal(0, 0.02, (self.size, self.size))
            
            swir_band = swir_base - row_val * 0.03 + soil_noise * 0.05 + np.random.normal(0, 0.02, (self.size, self.size))
            
            # Apply to this field
            data[0, f_mask] = np.clip(g_band[f_mask] * 10000, 0, 10000).astype(np.uint16)
            data[1, f_mask] = np.clip(r_band[f_mask] * 10000, 0, 10000).astype(np.uint16)
            data[2, f_mask] = np.clip(nir_band[f_mask] * 10000, 0, 10000).astype(np.uint16)
            if num_bands == 4:
                data[3, f_mask] = np.clip(swir_band[f_mask] * 10000, 0, 10000).astype(np.uint16)

        # Draw a dirty agricultural road network (unlabeled background)
        road_width = 8
        # Horizontal road
        road_y = int(self.size * 0.45)
        data[:, road_y - road_width : road_y + road_width, :] = int(0.20 * 10000) # bare soil signature
        mask[road_y - road_width : road_y + road_width, :] = 0 # background
        
        # Save multispectral image
        if RASTERIO_AVAILABLE:
            transform = from_origin(78.5, 17.5, 0.00025, 0.00025)
            with rasterio.open(
                output_img_path,
                'w',
                driver='GTiff',
                height=self.size,
                width=self.size,
                count=num_bands,
                dtype='uint16',
                crs='+proj=latlong',
                transform=transform
            ) as dst:
                for band_idx in range(num_bands):
                    dst.write(data[band_idx], band_idx + 1)
        else:
            np.save(output_img_path.replace(".tif", ".npy"), data)
            
        # Also save the segmentation mask (for training verification)
        np.save(output_img_path.replace(".tif", "_mask.npy"), mask)

        # Generate VIA JSON Annotations from the field polygons
        regions = []
        filename = os.path.basename(output_img_path)
        
        # Map class names
        class_names = {1: "healthy_crop", 2: "stressed_crop", 3: "bare_soil"}
        
        for field_id in range(num_fields):
            ftype = field_types[field_id]
            f_mask = (field_map == field_id)
            
            # Simple boundary tracing or contour extraction if OpenCV is available
            import cv2
            contours, _ = cv2.findContours((f_mask * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
            
            for cnt in contours:
                # Approximate contour to reduce points count
                epsilon = 0.02 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                
                # We need at least 3 points for a polygon
                if len(approx) >= 3:
                    xs = [int(p[0][0]) for p in approx]
                    ys = [int(p[0][1]) for p in approx]
                    
                    regions.append({
                        "shape_attributes": {
                            "name": "polygon",
                            "all_points_x": xs,
                            "all_points_y": ys
                        },
                        "region_attributes": {
                            "class": class_names[ftype],
                            "crop_type": "sugarcane" if field_id % 3 == 0 else ("cotton" if field_id % 3 == 1 else "maize")
                        }
                    })
                    
        img_size = self.size * self.size * 3
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
                        "options": {
                            "healthy_crop": "Healthy Crop",
                            "stressed_crop": "Stressed Crop",
                            "bare_soil": "Bare Soil"
                        },
                        "default_value": "healthy_crop"
                    }
                }
            }
        }
        
        with open(output_json_path, 'w') as f:
            json.dump(via_data, f, indent=2)
            
        print(f"Generated GenAI synthetic scene: {output_img_path}")
        print(f"Generated GenAI synthetic annotations: {output_json_path}")
        return num_fields

if __name__ == "__main__":
    synthesizer = GenerativeCropSynthesizer(seed=42)
    os.makedirs("../data/synthetic", exist_ok=True)
    synthesizer.generate_synthetic_scene(
        "../data/synthetic/gen_scene_1.tif", 
        "../data/synthetic/gen_scene_1_annotations.json"
    )
