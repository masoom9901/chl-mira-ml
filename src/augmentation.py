import numpy as np
try:
    import albumentations as A
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False

class RemoteSensingAugmenter:
    """
    Applies coordinated spatial and radiometric augmentations to multispectral images and segmentation masks.
    """
    def __init__(self, use_albumentations=True):
        self.use_albumentations = use_albumentations and ALBUMENTATIONS_AVAILABLE
        if self.use_albumentations:
            # Albumentations expects images in (H, W, C) format.
            # We will define the transforms.
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.GaussNoise(p=0.3)
            ])
            print("Albumentations pipeline initialized.")
        else:
            print("Albumentations not available. Initializing numpy fallback pipeline.")

    def augment(self, image, mask):
        """
        Args:
            image: numpy array of shape (bands, H, W)
            mask: numpy array of shape (H, W)
        Returns:
            augmented_image: shape (bands, H, W)
            augmented_mask: shape (H, W)
        """
        # Convert image to (H, W, bands) for processing
        img_hwc = np.transpose(image, (1, 2, 0))
        
        if self.use_albumentations:
            augmented = self.transform(image=img_hwc, mask=mask)
            aug_img = augmented['image']
            aug_mask = augmented['mask']
        else:
            # Numpy fallback logic
            aug_img, aug_mask = self._numpy_fallback_augment(img_hwc, mask)
            
        # Convert back to (bands, H, W)
        aug_img_chw = np.transpose(aug_img, (2, 0, 1))
        return aug_img_chw, aug_mask

    def _numpy_fallback_augment(self, image, mask):
        """
        Standard numpy operations to flip, rotate, and add noise for augmentation.
        image is shape (H, W, bands), mask is (H, W)
        """
        aug_img = image.copy()
        aug_mask = mask.copy()
        
        # 1. Random Horizontal Flip
        if np.random.rand() > 0.5:
            aug_img = np.fliplr(aug_img)
            aug_mask = np.fliplr(aug_mask)
            
        # 2. Random Vertical Flip
        if np.random.rand() > 0.5:
            aug_img = np.flipud(aug_img)
            aug_mask = np.flipud(aug_mask)
            
        # 3. Random 90-degree Rotation
        rot_k = np.random.choice([0, 1, 2, 3])
        if rot_k > 0:
            aug_img = np.rot90(aug_img, k=rot_k, axes=(0, 1))
            aug_mask = np.rot90(aug_mask, k=rot_k, axes=(0, 1))
            
        # 4. Radiometric noise (only to image)
        if np.random.rand() > 0.5:
            # Scale of values is 0-10000 (uint16)
            noise = np.random.normal(0, 150, image.shape).astype(np.int32)
            aug_img = np.clip(aug_img.astype(np.int32) + noise, 0, 10000).astype(np.uint16)
            
        return aug_img, aug_mask

if __name__ == "__main__":
    # Test execution
    img = np.random.randint(0, 10000, (4, 128, 128), dtype=np.uint16)
    mask = np.random.randint(0, 4, (128, 128), dtype=np.uint8)
    
    augmenter = RemoteSensingAugmenter(use_albumentations=False)
    aug_img, aug_msk = augmenter.augment(img, mask)
    print("Original image shape:", img.shape)
    print("Augmented image shape:", aug_img.shape)
    print("Augmented mask shape:", aug_msk.shape)
