#!/usr/bin/env python
# coding: utf-8

"""
Image Enhancement Module for Astronomical Images
Contains two powerful algorithms:
1. Moon Detail Enhancement - Combines CLAHE, unsharp masking, and contrast boost
2. Histogram Equalization - Redistributes intensities across full dynamic range
"""

from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QComboBox, QStackedWidget, QDoubleSpinBox
from PySide6.QtCore import Qt, Signal
import numpy as np
import imageio
import skimage.filters
from skimage.color import rgb2gray
from scipy.ndimage import uniform_filter

from modules.i_image_module import IImageModule
from image_data_store import ImageDataStore


##
class BaseParamsWidget(QWidget):
    """Base class for parameter widgets to ensure a consistent interface."""
    def get_params(self) -> dict:
        raise NotImplementedError


class MoonEnhancementParamsWidget(BaseParamsWidget):
    """Widget for Moon Detail Enhancement parameters."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Sharpening strength parameter
        layout.addWidget(QLabel("Sharpening Strength:"))
        self.sharpen_spinbox = QDoubleSpinBox()
        self.sharpen_spinbox.setMinimum(0.0)
        self.sharpen_spinbox.setMaximum(5.0)
        self.sharpen_spinbox.setValue(2.0)
        self.sharpen_spinbox.setSingleStep(0.1)
        layout.addWidget(self.sharpen_spinbox)

        # Contrast enhancement parameter
        layout.addWidget(QLabel("Contrast Boost:"))
        self.contrast_spinbox = QDoubleSpinBox()
        self.contrast_spinbox.setMinimum(1.0)
        self.contrast_spinbox.setMaximum(3.0)
        self.contrast_spinbox.setValue(1.5)
        self.contrast_spinbox.setSingleStep(0.1)
        layout.addWidget(self.contrast_spinbox)

        layout.addStretch()

    def get_params(self) -> dict:
        return {
            'sharpen_strength': self.sharpen_spinbox.value(),
            'contrast_boost': self.contrast_spinbox.value()
        }


class HistogramEqualizationWidget(BaseParamsWidget):
    """Widget for Histogram Equalization - shows it has no parameters."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        info_label = QLabel("Histogram Equalization redistributes pixel intensities\n"
                           "to use the full dynamic range.\n\n"
                           "Perfect for low-contrast medical and astronomical images.")
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #666; padding: 10px;")
        layout.addWidget(info_label)
        layout.addStretch()

    def get_params(self) -> dict:
        return {}


##
class SampleControlsWidget(QWidget):
    """Control widget for selecting and applying image enhancement operations."""
    process_requested = Signal(dict)

    def __init__(self, module_manager, parent=None):
        super().__init__(parent)
        self.module_manager = module_manager
        self.param_widgets = {}
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("<h3>Image Enhancement Module</h3>"))

        layout.addWidget(QLabel("Operation:"))
        self.operation_selector = QComboBox()
        layout.addWidget(self.operation_selector)

        # Stacked widget holds parameter controls for each operation
        self.params_stack = QStackedWidget()
        layout.addWidget(self.params_stack)

        # Define the two enhancement operations
        operations = {
            "Moon Detail Enhancement": MoonEnhancementParamsWidget,
            "Histogram Equalization": HistogramEqualizationWidget
        }

        for name, widget_class in operations.items():
            widget = widget_class()
            self.params_stack.addWidget(widget)
            self.param_widgets[name] = widget
            self.operation_selector.addItem(name)

        self.apply_button = QPushButton("Apply Enhancement")
        layout.addWidget(self.apply_button)

        self.apply_button.clicked.connect(self._on_apply_clicked)
        self.operation_selector.currentTextChanged.connect(self._on_operation_changed)

    def _on_apply_clicked(self):
        """Triggered when user clicks Apply button."""
        operation_name = self.operation_selector.currentText()
        active_widget = self.param_widgets[operation_name]
        params = active_widget.get_params()
        params['operation'] = operation_name
        self.process_requested.emit(params)

    def _on_operation_changed(self, operation_name: str):
        """Switch parameter widget when operation selection changes."""
        if operation_name in self.param_widgets:
            self.params_stack.setCurrentWidget(self.param_widgets[operation_name])


##
class MariamModule(IImageModule):
    """Main image processing module implementing Moon Enhancement and Histogram Equalization."""
    
    def __init__(self):
        super().__init__()
        self._controls_widget = None
        
    def _normalize_to_uint8(self, data: np.ndarray) -> np.ndarray:
        """
        Normalize any data to 0-255 uint8 range.
        
        Formula: normalized = (data - min) / (max - min) * 255
        This stretches the data to use the full 8-bit range.
        """
        data_min = np.min(data)
        data_max = np.max(data)
        if data_max > data_min:
            normalized = ((data - data_min) / (data_max - data_min) * 255.0)
            return normalized.astype(np.uint8)
        return data.astype(np.uint8)

    def get_name(self) -> str:
        return "Mariam Module"

    def get_supported_formats(self) -> list[str]:
        return ["png", "jpg", "jpeg", "bmp", "tiff"]

    def create_control_widget(self, parent=None, module_manager=None) -> QWidget:
        """Create and return the control widget for this module."""
        if self._controls_widget is None:
            self._controls_widget = SampleControlsWidget(module_manager, parent)
            self._controls_widget.process_requested.connect(self._handle_processing_request)
        return self._controls_widget

    def _handle_processing_request(self, params: dict):
        """Handle processing request from control widget."""
        if self._controls_widget and self._controls_widget.module_manager:
            self._controls_widget.module_manager.apply_processing_to_current_image(params)

    def load_image(self, file_path: str):
        """Load an image from file path."""
        try:
            image_data = imageio.imread(file_path)
            metadata = {'name': file_path.split('/')[-1]}
            return True, image_data, metadata, None
        except Exception as e:
            print(f"Error loading image {file_path}: {e}")
            return False, None, {}, None

    def process_image(self, image_data: np.ndarray, metadata: dict, params: dict) -> np.ndarray:
        """
        Main processing function - routes to appropriate algorithm based on operation.
        """
        operation = params.get('operation')
        
        if operation == "Moon Detail Enhancement":
            return self._moon_detail_enhancement(image_data, params)
        elif operation == "Histogram Equalization":
            return self._histogram_equalization(image_data)
        
        return image_data

    def _moon_detail_enhancement(self, image_data: np.ndarray, params: dict) -> np.ndarray:
        """
        Moon Detail Enhancement Algorithm
        ===================================
        Combines three techniques to reveal lunar surface details:
        1. CLAHE (Contrast Limited Adaptive Histogram Equalization)
        2. Unsharp Masking (Sharpening)
        3. Global Contrast Boost
        
        Line-by-line explanation follows in code comments.
        """
        # Extract user-defined parameters
        sharpen_strength = params.get('sharpen_strength', 2.0)  # How much to sharpen (0-5)
        contrast_boost = params.get('contrast_boost', 1.5)      # How much to boost contrast (1-3)

        # Convert RGB/RGBA images to grayscale for processing
        if image_data.ndim == 3 and image_data.shape[2] in [3, 4]:
            # rgb2gray uses weighted average: 0.2125*R + 0.7154*G + 0.0721*B
            grayscale_img = rgb2gray(image_data[:,:,:3])
        else:
            # Already grayscale, just copy
            grayscale_img = image_data.copy()

        # Convert to float for mathematical operations (avoids integer overflow)
        img_float = grayscale_img.astype(float)

        # ============================================================================
        # STEP 1: CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # ============================================================================
        # CLAHE enhances local contrast by dividing image into tiles and equalizing each
        
        # Normalize to [0, 255] range for histogram computation
        img_min = img_float.min()
        img_max = img_float.max()
        if img_max > img_min:
            # Scale to 0-255 range
            normalized = ((img_float - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        else:
            # No range, just cast to uint8
            normalized = img_float.astype(np.uint8)

        # Divide image into 8x8 grid of tiles
        height, width = normalized.shape
        tile_size = 8                          # Number of tiles per dimension
        tile_h = height // tile_size           # Height of each tile in pixels
        tile_w = width // tile_size            # Width of each tile in pixels
        
        # Output array to store CLAHE-processed image
        clahe_img = np.zeros_like(normalized, dtype=np.float32)

        # Process each tile independently
        for i in range(tile_size):
            for j in range(tile_size):
                # Calculate tile boundaries
                y1 = i * tile_h                                      # Top edge
                y2 = (i + 1) * tile_h if i < tile_size - 1 else height  # Bottom edge (handle remainder)
                x1 = j * tile_w                                      # Left edge
                x2 = (j + 1) * tile_w if j < tile_size - 1 else width   # Right edge (handle remainder)

                # Extract current tile
                tile = normalized[y1:y2, x1:x2]

                # Compute histogram: count frequency of each intensity (0-255)
                hist, bins = np.histogram(tile.flatten(), bins=256, range=(0, 256))

                # Clip histogram to limit contrast (prevents noise amplification)
                clip_threshold = 2.0 * (tile.size / 256.0)  # Clip at 2x average
                clipped_hist = np.minimum(hist, clip_threshold)

                # Redistribute clipped pixels uniformly across all bins
                excess = np.sum(hist - clipped_hist)        # Total clipped pixels
                redistribute = excess / 256.0               # Amount to add to each bin
                clipped_hist = clipped_hist + redistribute

                # Compute Cumulative Distribution Function (CDF)
                cdf = np.cumsum(clipped_hist)               # Running sum
                cdf = cdf / cdf[-1]                         # Normalize to [0, 1]

                # Map old intensities to new using CDF (histogram equalization)
                # np.interp: linear interpolation to map tile pixels via CDF
                equalized_tile = np.interp(tile.flatten(), bins[:-1], cdf * 255)
                
                # Reshape back to 2D and store in output
                clahe_img[y1:y2, x1:x2] = equalized_tile.reshape(tile.shape)

        # ============================================================================
        # FIXING TILE BOUNDARIES: Smooth transitions between tiles
        # ============================================================================
        # Apply a smoothing filter across tile boundaries to remove visible grid lines
        
        # Create a mask of pixels near tile boundaries (±2 pixels from boundaries)
        boundary_mask = np.zeros_like(clahe_img, dtype=bool)
        for i in range(1, tile_size):
            # Horizontal boundaries
            y_boundary = i * tile_h
            if y_boundary < height:
                # Mark 5 pixels around boundary (2 above, boundary itself, 2 below)
                y_start = max(0, y_boundary - 2)
                y_end = min(height, y_boundary + 3)
                boundary_mask[y_start:y_end, :] = True
            
            # Vertical boundaries
            x_boundary = i * tile_w
            if x_boundary < width:
                # Mark 5 pixels around boundary (2 left, boundary itself, 2 right)
                x_start = max(0, x_boundary - 2)
                x_end = min(width, x_boundary + 3)
                boundary_mask[:, x_start:x_end] = True

        # Apply local averaging filter (5x5 neighborhood) to boundary pixels only
        # uniform_filter: replaces each pixel with average of its neighborhood
        smoothed = uniform_filter(clahe_img, size=5, mode='reflect')
        
        # Blend smoothed and original: use smoothed values only at boundaries
        clahe_img = np.where(boundary_mask, smoothed, clahe_img)

        # ============================================================================
        # STEP 2: Unsharp Masking (Sharpening)
        # ============================================================================
        # Enhance fine details by subtracting a blurred version
        
        # Create blurred version using Gaussian filter (sigma=1.0)
        # Gaussian blur: weighted average with Gaussian kernel, smooths image
        blurred = skimage.filters.gaussian(clahe_img, sigma=1.0)
        
        # Calculate high-frequency details (edges, textures)
        # detail = original - blurred reveals fine structures
        detail = clahe_img - blurred
        
        # Add enhanced details back to original
        # Formula: sharpened = original + strength × detail
        sharpened = clahe_img + (sharpen_strength * detail)

        # ============================================================================
        # STEP 3: Global Contrast Boost
        # ============================================================================
        # Increase separation between bright and dark regions
        
        # Calculate mean intensity (center point)
        mean_val = np.mean(sharpened)
        
        # Push values away from mean: (value - mean) × boost + mean
        # Values above mean get brighter, below mean get darker
        contrasted = mean_val + contrast_boost * (sharpened - mean_val)

        # Clip to valid range [0, 255] to prevent overflow
        result = np.clip(contrasted, 0, 255)

        # Convert back to uint8 (8-bit integer format)
        return result.astype(np.uint8)

    def _histogram_equalization(self, image_data: np.ndarray) -> np.ndarray:
        """
        Histogram Equalization Algorithm
        ==================================
        Redistributes pixel intensities to use full dynamic range [0, 255].
        Most effective for low-contrast images.
        
        Theory:
        - Calculate histogram (frequency of each intensity)
        - Compute CDF (cumulative distribution function)
        - Map old intensities to new values that spread uniformly
        
        Line-by-line explanation follows in code comments.
        """
        
        # ============================================================================
        # STEP 1: Convert to Grayscale
        # ============================================================================
        if image_data.ndim == 3 and image_data.shape[2] in [3, 4]:
            # RGB/RGBA image - convert to grayscale using standard weights
            grayscale = rgb2gray(image_data[:, :, :3])
            grayscale = self._normalize_to_uint8(grayscale)
        elif image_data.ndim == 2:
            # Already grayscale, just normalize
            grayscale = self._normalize_to_uint8(image_data)
        else:
            # Unexpected format, squeeze extra dimensions and normalize
            grayscale = self._normalize_to_uint8(image_data.squeeze())
        
        # ============================================================================
        # STEP 2: Calculate Histogram
        # ============================================================================
        # Count how many pixels have each intensity value (0-255)
        histogram = np.zeros(256, dtype=np.int32)
        for intensity in range(256):
            # np.sum(grayscale == intensity) counts pixels with this exact intensity
            histogram[intensity] = np.sum(grayscale == intensity)
        
        # ============================================================================
        # STEP 3: Calculate Cumulative Distribution Function (CDF)
        # ============================================================================
        # CDF[i] = sum of all histogram values from 0 to i
        # Represents "how many pixels have intensity ≤ i"
        cdf = np.zeros(256, dtype=np.float64)
        cdf[0] = histogram[0]                    # First value is just histogram[0]
        for i in range(1, 256):
            cdf[i] = cdf[i-1] + histogram[i]     # Running sum
        
        # ============================================================================
        # STEP 4: Normalize CDF to Range [0, 255]
        # ============================================================================
        # This creates the mapping from old intensity → new intensity
        
        # Find first non-zero CDF value (ignore empty bins at start)
        cdf_min = np.min(cdf[cdf > 0])
        cdf_max = cdf[255]                        # Total number of pixels
        
        # Normalization formula: new = (cdf - cdf_min) / (cdf_max - cdf_min) × 255
        # This stretches the CDF to fill [0, 255] range
        cdf_normalized = np.zeros(256, dtype=np.uint8)
        for i in range(256):
            if cdf_max > cdf_min:
                # Apply normalization formula
                cdf_normalized[i] = np.uint8(((cdf[i] - cdf_min) / (cdf_max - cdf_min)) * 255.0)
            else:
                # No range to normalize, keep original
                cdf_normalized[i] = i
        
        # ============================================================================
        # STEP 5: Map Old Intensities to New Intensities
        # ============================================================================
        # Create output image with equalized histogram
        equalized = np.zeros_like(grayscale, dtype=np.uint8)
        height, width = grayscale.shape
        
        # For each pixel, look up its new intensity in the normalized CDF
        for i in range(height):
            for j in range(width):
                old_intensity = grayscale[i, j]              # Original pixel value
                new_intensity = cdf_normalized[old_intensity] # Mapped value
                equalized[i, j] = new_intensity
        
        return equalized