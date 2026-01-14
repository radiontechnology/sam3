import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List
from PIL import Image
import time

# --- SAM 3.0 IMPORTS ---
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
# -----------------------

class SAM3Segmentor:
    """
    A class wrapper for the SAM 3.0 model to perform image segmentation 
    on a SINGLE image file.
    """
    
    def __init__(
        self, 
        checkpoint_path: str = "", # SAM3 often auto-downloads, but keeping param just in case
        device: Optional[str] = None
    ):
        """
        Initializes the SAM3Segmentor.
        """
        self.checkpoint_path = checkpoint_path
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.model = None
        self.processor = None
        
        print(f"SAM3Segmentor initialized. Device: {self.device}")

    # --------------------------------------------------------------------------
    # --- Helper Functions (Kept from original) --------------------------------
    # --------------------------------------------------------------------------

    @staticmethod
    def _create_masked_images(image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Applies the mask to the image to create foreground (with alpha) and background images."""
        
        # Background values are set to 0
        mask = mask

        # 1. Create Foreground (with transparent background)
        image_bgra = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        mask_3d = np.expand_dims(mask, axis=-1)
        
        foreground = np.where(mask_3d, image_bgra, 0)
        # Set alpha channel: opaque (255) where mask is True, transparent (0) otherwise
        foreground[..., 3] = (mask.astype(np.uint8) * 255)

        # 2. Create Background (with foreground blacked out)
        background = image.copy()
        background[mask.astype(bool)] = [0, 0, 0] # Set masked area to black
        
        return foreground, background

    @staticmethod
    def _show_masks_on_image(image_rgb: np.ndarray, masks: np.ndarray, output_path: str):
        """Overlays all provided masks on the image with random colors and saves it."""
        plt.figure(figsize=(12, 12))
        plt.imshow(image_rgb)

        # Ensure masks are in (N, H, W) format
        if masks.ndim == 4:
            if masks.shape[0] == 1:
                masks = masks.squeeze(0)  # Handle (1, N, H, W)
            elif masks.shape[1] == 1:
                masks = masks.squeeze(1)  # Handle (N, 1, H, W)
        
        composite = np.zeros((masks.shape[1], masks.shape[2], 4), dtype=np.float32)
        
        for i in range(masks.shape[0]):
            mask = masks[i]
            # Random color with 50% opacity
            color = np.concatenate([np.random.random(3), np.array([0.5])]) 
            mask_3d = np.expand_dims(mask, axis=-1)
            composite += np.where(mask_3d, color, 0)

        plt.imshow(composite)
        plt.axis('off')
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()

    # --------------------------------------------------------------------------
    # --- Core Logic Methods ---------------------------------------------------
    # --------------------------------------------------------------------------

    def load_model(self):
        """Builds and loads the SAM 3 model and processor."""
        if self.model is not None:
            print("Model already loaded.")
            return

        print(f"Loading SAM 3.0 model...")
        
        try:
            # 1. Build the Model
            # Note: SAM3 usually handles checkpoints internally or via config,
            # but if your build_sam3_image_model accepts a path, pass self.checkpoint_path
            self.model = build_sam3_image_model(self.checkpoint_path)
            
            # Move to device
            self.model.to(self.device)
            # 2. Create the processor
            self.processor = Sam3Processor(self.model)
            print("Model loaded successfully.")
            
        except Exception as e:
            print(f"Failed to load model: {e}")
            raise e

    def _post_process_mask(self, masks: np.ndarray, scores: np.ndarray, H: int, W: int, multiple_masks: bool = True) -> np.ndarray:
        """
        Selects the best mask and cleans it up using morphology.
        Handles input shapes (1, N, H, W) and (N, 1, H, W).
        """
        if len(masks) == 0:
            return np.zeros((H, W), dtype=bool)

        # 1. robustly handle 4D inputs to get (N, H, W)
        if masks.ndim == 4:
            if masks.shape[0] == 1:
                masks = masks.squeeze(0)  # Handle (1, N, H, W)
            elif masks.shape[1] == 1:
                masks = masks.squeeze(1)  # Handle (N, 1, H, W)

        # Ensure scores are 1D (N,)
        if scores.ndim == 2: 
            if scores.shape[0] == 1:
                scores = scores.squeeze(0)
            elif scores.shape[1] == 1:
                scores = scores.squeeze(1)

        # 2. Logic for multiple masks (Union of scores > 0.5)
        if multiple_masks:
            valid_indices = scores > 0.5
            
            if np.any(valid_indices):
                # Union of all valid masks
                return np.any(masks[valid_indices], axis=0)
            else:
                return np.zeros((H, W), dtype=bool)

        # 3. Fallback: Select single best mask
        best_idx = np.argmax(scores)
        # If the best score is less than 0.5, return an empty mask
        if scores[best_idx] < 0.5:
            return np.zeros((H, W), dtype=bool)
        best_mask = masks[best_idx]
        
        return best_mask

    def segment_image(self, input_path: str, output_base_path: str, text_prompt: str = "generic object", multiple_maks:bool = False, save_debug: bool = False):
        """
        Runs the segmentation process on a single image.
        
        Args:
            input_path (str): Path to image.
            output_base_path (str): Base path for saving.
            text_prompt (str): SAM 3 is text-based. Provide a description of what to segment.
            save_debug (bool): Save intermediate visualizations.
        """
        if self.model is None:
            self.load_model()

        print(f"\n--- Processing: {os.path.basename(input_path)} ---")

        # 1. Load Image with OpenCV (for final processing/saving)
        image_bgr = cv2.imread(input_path)
        if image_bgr is None:
            print(f"Error: cannot read image {input_path}")
            return

        H, W, _ = image_bgr.shape
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # 2. Convert to PIL for SAM 3 Input
        pil_image = Image.fromarray(image_rgb)

        # --- Run SAM 3 Processor ---
        print("Setting image...")
        inference_state = self.processor.set_image(pil_image)
        
        print(f"Running prediction with prompt: '{text_prompt}'...")
        output = self.processor.set_text_prompt(state=inference_state, prompt=text_prompt)

        # Extract results
        # SAM 3 returns Tensors. We need to move them to CPU and convert to Numpy.
        masks_tensor = output["masks"]
        scores_tensor = output["scores"]
        
        # Convert to numpy
        masks = masks_tensor.cpu().numpy() # Shape usually (Batch, N, H, W)
        scores = scores_tensor.cpu().numpy()
        
        if masks.size == 0:
            print("No masks returned.")
            return
        
        print("Scores found: ", scores)

        # --- Post-process to get the best foreground mask ---
        best_mask = self._post_process_mask(masks, scores, H, W, multiple_maks)

        # Create FG/BG images
        fg, bg = self._create_masked_images(image_bgr, best_mask)
        
        if output_base_path:
            # Define paths
            all_masks_path = f"{output_base_path}_all_masks.png"
            fg_path = f"{output_base_path}_foreground.png"
            bg_path = f"{output_base_path}_background.jpg"
            mask_path = f"{output_base_path}_mask.png"
            
            # Save Foreground
            cv2.imwrite(fg_path, fg)
            
            print(f"Saved results to: {os.path.dirname(output_base_path) or './'}")
            print(f"- Foreground: {os.path.basename(fg_path)}")
            
            if save_debug:
                self._show_masks_on_image(image_rgb, masks, all_masks_path)
                cv2.imwrite(bg_path, bg)
                cv2.imwrite(mask_path, (best_mask.astype(np.uint8) * 255))

                print(f"- Debug: All masks overlay saved to {os.path.basename(all_masks_path)}")

        return fg, bg

# --------------------------------------------------------------------------
# --- Example Usage --------------------------------------------------------
# --------------------------------------------------------------------------

if __name__ == "__main__":
    
    # --- CONFIGURATION ---
    INPUT_FILE = "./input_pics/glasses.png"

    OUTPUT_BASE = "./output_pics/sample" 
    
    # SAM 3 Specific Prompt
    TEXT_PROMPT = "face with glasses" 
    
    # ---------------------

    # Mock image creation for testing
    if not os.path.exists(INPUT_FILE):
        print(f"Creating mock image at {INPUT_FILE}")
        mock_img = np.zeros((256, 256, 3), dtype=np.uint8)
        cv2.circle(mock_img, (128, 128), 60, (255, 255, 255), -1) 
        os.makedirs(os.path.dirname(INPUT_FILE) or '.', exist_ok=True)
        cv2.imwrite(INPUT_FILE, mock_img)

    # Ensure output dir exists
    if os.path.dirname(OUTPUT_BASE):
        os.makedirs(os.path.dirname(OUTPUT_BASE), exist_ok=True)
    
    try:
        # Initialize
        sam3_root = "./sam3/"
        checkpoint = f"{sam3_root}/assets/bpe_simple_vocab_16e6.txt.gz"
        segmentor = SAM3Segmentor(checkpoint_path=checkpoint)
        
        
        # Run
        segmentor.segment_image(
            input_path=INPUT_FILE,
            output_base_path=OUTPUT_BASE,
            text_prompt=TEXT_PROMPT, # Pass the text prompt here
            save_debug=True,
            multiple_maks=True
        )
        
    except Exception as e:
        print(f"\nAn error occurred: {e}")