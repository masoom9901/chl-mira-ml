import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
try:
    import rasterio
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False


class RemoteSensingPreprocessor:
    """
    A processor to load, enhance, and compute indices for LISS-3 and LISS-4 satellite images.
    """
    def __init__(self, file_path):
        self.file_path = file_path
        self.bands = None
        self.is_liss4 = False
        self.load_image()

    def load_image(self):
        """
        Loads the remote sensing image. Uses rasterio if available, otherwise falls back to numpy.
        """
        if RASTERIO_AVAILABLE and self.file_path.endswith('.tif'):
            with rasterio.open(self.file_path) as src:
                # rasterio reads as (bands, height, width)
                self.bands = src.read()
                # If 3 bands, it's LISS-4, if 4 bands, it's LISS-3
                self.is_liss4 = (self.bands.shape[0] == 3)
                print(f"Loaded TIFF image: {self.bands.shape[0]} bands, shape: {self.bands.shape[1:]}")
        else:
            # Fallback to numpy
            np_path = self.file_path.replace(".tif", ".npy")
            if os.path.exists(np_path):
                self.bands = np.load(np_path)
                self.is_liss4 = (self.bands.shape[0] == 3)
                print(f"Loaded numpy fallback image: {self.bands.shape[0]} bands, shape: {self.bands.shape[1:]}")
            else:
                # Generate sample on the fly
                from src.ingestion import generate_mock_liss_image
                generate_mock_liss_image(self.file_path, is_liss4=('.liss4' in self.file_path))
                self.load_image()

    def get_band(self, name):
        """
        Map band names (Green, Red, NIR, SWIR) to indices.
        Our mock generator structure:
        0: Green
        1: Red
        2: NIR
        3: SWIR (LISS-3 only)
        """
        mapping = {"green": 0, "red": 1, "nir": 2, "swir": 3}
        band_idx = mapping.get(name.lower())
        
        if band_idx is None:
            raise ValueError(f"Unknown band name: {name}")
        if band_idx == 3 and self.is_liss4:
            raise ValueError("LISS-4 does not contain a SWIR band.")
            
        # Return scaled reflectance (0.0 to 1.0)
        return self.bands[band_idx].astype(np.float32) / 10000.0

    def generate_fcc(self):
        """
        False Color Composite (FCC): Stacks NIR, Red, and Green bands as RGB.
        In this combination, healthy vegetation appears in shades of bright red.
        """
        nir = self.get_band("nir")
        red = self.get_band("red")
        green = self.get_band("green")

        # Stack into 3-channel image (H, W, 3)
        fcc = np.stack([nir, red, green], axis=-1)
        
        # Scale to [0, 255] for display
        fcc_normalized = self.contrast_stretch(fcc)
        return (fcc_normalized * 255).astype(np.uint8)

    def calculate_ndvi(self):
        """
        Normalized Difference Vegetation Index: (NIR - Red) / (NIR + Red)
        """
        nir = self.get_band("nir")
        red = self.get_band("red")
        
        denominator = nir + red
        # Avoid division by zero
        denominator[denominator == 0] = 1e-6
        
        ndvi = (nir - red) / denominator
        return np.clip(ndvi, -1.0, 1.0)

    def calculate_ndwi(self):
        """
        Normalized Difference Water Index: (Green - NIR) / (Green + NIR)
        Useful for highlighting water features (canals, reservoirs).
        """
        green = self.get_band("green")
        nir = self.get_band("nir")
        
        denominator = green + nir
        denominator[denominator == 0] = 1e-6
        
        ndwi = (green - nir) / denominator
        return np.clip(ndwi, -1.0, 1.0)

    def contrast_stretch(self, img, low_percentile=2, high_percentile=98):
        """
        Applies linear percentile contrast stretching.
        """
        img_out = np.zeros_like(img)
        # Check if single or multi-channel
        if len(img.shape) == 2:
            channels = [img]
        else:
            channels = [img[..., i] for i in range(img.shape[-1])]
            
        stretched_channels = []
        for channel in channels:
            vmin, vmax = np.percentile(channel, [low_percentile, high_percentile])
            if vmax == vmin:
                stretched_channels.append(channel)
            else:
                stretched = (channel - vmin) / (vmax - vmin)
                stretched_channels.append(np.clip(stretched, 0.0, 1.0))
                
        if len(img.shape) == 2:
            return stretched_channels[0]
        else:
            return np.stack(stretched_channels, axis=-1)

    def apply_clahe(self, rgb_img):
        """
        Contrast Limited Adaptive Histogram Equalization (CLAHE) on the L channel of LAB color space.
        """
        if not OPENCV_AVAILABLE:
            # Fallback if OpenCV is not installed (simple contrast stretch)
            return rgb_img
            
        # Convert RGB to LAB
        lab = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L-channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        
        # Merge channels and convert back to RGB
        limg = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
        return enhanced

    def remove_artifacts(self, img, filter_type="bilateral"):
        """
        Removes sensor artifacts and noise using Bilateral or Gaussian filtering.
        Bilateral filtering is preferred as it preserves crop boundaries while smoothing noise.
        """
        if not OPENCV_AVAILABLE:
            return img
            
        if filter_type == "bilateral":
            # bilateralFilter requires uint8 or float32. We pass uint8 RGB
            return cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
        elif filter_type == "gaussian":
            return cv2.GaussianBlur(img, (5, 5), 0)
        else:
            return img

    def mask_clouds(self):
        """
        Creates a cloud mask. Clouds have high reflectance in both Red and Green bands.
        """
        green = self.get_band("green")
        red = self.get_band("red")
        
        # Simple threshold for cloud masking
        cloud_mask = (green > 0.4) & (red > 0.4)
        return cloud_mask

if __name__ == "__main__":
    # Test execution
    processor = RemoteSensingPreprocessor("../data/sample_liss3.tif")
    fcc = processor.generate_fcc()
    ndvi = processor.calculate_ndvi()
    print("FCC shape:", fcc.shape)
    print("NDVI mean:", ndvi.mean())
