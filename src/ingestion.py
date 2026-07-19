import os
import numpy as np
try:
    import rasterio
    from rasterio.transform import from_origin
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False
import json

class BhoonidhiClient:
    """
    A client to demonstrate connection and queries to ISRO's Bhoonidhi Portal API.
    Bhoonidhi uses SpatioTemporal Asset Catalog (STAC) specifications for search.
    """
    def __init__(self, username=None, password=None):
        self.username = username
        self.password = password
        self.base_url = "https://bhoonidhi.nrsc.gov.in/api"
        self.token = None

    def authenticate(self):
        """
        Simulate authentication to Bhoonidhi to obtain a bearer token.
        In production, this would make a POST request to '/auth/token' with credentials.
        """
        if not self.username or not self.password:
            return False, "Credentials missing. Please request access from bhoonidhi@nrsc.gov.in."
        # Mocking successful authentication
        self.token = "mock-token-xyz-12345"
        return True, "Authenticated successfully with Bhoonidhi API."

    def search_scenes(self, bbox, start_date, end_date, satellite="RESOURCESAT-2", sensor="LISS-3"):
        """
        Search for available LISS-3/LISS-4 products based on bounding box and date range.
        Query parameters map to the Bhoonidhi STAC API.
        """
        print(f"Searching Bhoonidhi for {satellite} {sensor} scenes...")
        print(f"Criteria: BBox={bbox}, DateRange={start_date} to {end_date}")
        
        # In a real API call, this would request the STAC /search endpoint
        mock_results = [
            {
                "product_id": f"R2_L3_{start_date.replace('-', '')}_098_054",
                "satellite": satellite,
                "sensor": sensor,
                "cloud_cover": 4.2,
                "acquisition_date": start_date,
                "bbox": bbox,
                "download_url": f"{self.base_url}/download/R2_L3_098_054"
            }
        ]
        return mock_results

def generate_mock_liss_image(output_path, is_liss4=False, size=512):
    """
    Generates a realistic multi-spectral mock LISS satellite image (GeoTIFF).
    LISS-3 has 4 bands: Green, Red, NIR, SWIR.
    LISS-4 has 3 bands: Green, Red, NIR.
    
    Reflectance properties:
    - Healthy Crop: High NIR, Low Red, Med Green, Low SWIR
    - Stressed Crop: Med NIR, Med Red, Med Green, Med SWIR
    - Bare Soil: Med NIR, Med Red, Med Green, High SWIR
    - Water Canal: Very Low NIR, Very Low Red, Low Green, Zero SWIR
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    num_bands = 3 if is_liss4 else 4
    
    # Initialize bands: shape (bands, height, width)
    # Reflectance ranges from 0.0 to 1.0, represented as uint16 (scaled by 10000)
    data = np.zeros((num_bands, size, size), dtype=np.uint16)
    
    # Create fields layout (polygons/grid)
    grid_size = size // 4
    for i in range(4):
        for j in range(4):
            # Define field coordinates
            r_start, r_end = i * grid_size, (i + 1) * grid_size
            c_start, c_end = j * grid_size, (j + 1) * grid_size
            
            # Crop health categorization
            # i=0,1: healthy; i=2: stressed; i=3: bare soil
            if i < 2:
                # Healthy Crop
                g = np.random.uniform(0.12, 0.18)
                r = np.random.uniform(0.04, 0.08)
                nir = np.random.uniform(0.60, 0.80)
                swir = np.random.uniform(0.10, 0.15)
            elif i == 2:
                # Stressed Crop
                g = np.random.uniform(0.14, 0.20)
                r = np.random.uniform(0.12, 0.18)
                nir = np.random.uniform(0.35, 0.45)
                swir = np.random.uniform(0.20, 0.28)
            else:
                # Bare Soil
                g = np.random.uniform(0.10, 0.15)
                r = np.random.uniform(0.20, 0.26)
                nir = np.random.uniform(0.22, 0.28)
                swir = np.random.uniform(0.35, 0.45)
            
            # Generate random texture per field
            noise = np.random.normal(0, 0.02, (grid_size, grid_size))
            
            data[0, r_start:r_end, c_start:c_end] = np.clip((g + noise) * 10000, 0, 10000).astype(np.uint16)
            data[1, r_start:r_end, c_start:c_end] = np.clip((r + noise) * 10000, 0, 10000).astype(np.uint16)
            data[2, r_start:r_end, c_start:c_end] = np.clip((nir + noise) * 10000, 0, 10000).astype(np.uint16)
            if num_bands == 4:
                data[3, r_start:r_end, c_start:c_end] = np.clip((swir + noise) * 10000, 0, 10000).astype(np.uint16)
                
    # Add a water canal winding through the middle (columns 200 to 240)
    for row in range(size):
        # Sine wave canal path
        col_center = int(220 + 30 * np.sin(row / 40.0))
        width = 15
        c_start = max(0, col_center - width)
        c_end = min(size, col_center + width)
        
        # Water signature (Very low Red/NIR, moderate Green, zero SWIR)
        g_val = int(np.random.uniform(0.06, 0.10) * 10000)
        r_val = int(np.random.uniform(0.03, 0.05) * 10000)
        nir_val = int(np.random.uniform(0.02, 0.04) * 10000)
        
        data[0, row, c_start:c_end] = g_val
        data[1, row, c_start:c_end] = r_val
        data[2, row, c_start:c_end] = nir_val
        if num_bands == 4:
            data[3, row, c_start:c_end] = int(0.01 * 10000)
            
    # Write GeoTIFF
    if RASTERIO_AVAILABLE:
        # Define geotransform (UTM Zone 44N, near Hyderabad, India)
        transform = from_origin(78.4, 17.4, 0.00025, 0.00025) # Approx 24m resolution for LISS-3
        
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=size,
            width=size,
            count=num_bands,
            dtype='uint16',
            crs='+proj=latlong',
            transform=transform,
        ) as dst:
            for band_idx in range(num_bands):
                dst.write(data[band_idx], band_idx + 1)
        print(f"Saved GeoTIFF mock scene to {output_path}")
    else:
        # Fallback to standard numpy array
        np_path = output_path.replace(".tif", ".npy")
        np.save(np_path, data)
        print(f"Rasterio not loaded yet. Saved numpy fallback to {np_path}")

if __name__ == "__main__":
    generate_mock_liss_image("../data/sample_liss3.tif", is_liss4=False)
    generate_mock_liss_image("../data/sample_liss4.tif", is_liss4=True)
