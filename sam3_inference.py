import torch
import time
import cv2
import numpy as np
from PIL import Image
from torchvision.transforms import v2
from sam3.model_builder import build_sam3_image_model
from sam3.model.data_misc import FindStage, interpolate
from sam3.model import box_ops
from sam3.model.sam3_image import Prompt

# --- Configuration ---
IMAGE_BATCH_SIZE = 6      # Number of images processed in parallel
PROMPT_BATCH_SIZE = 2     # Number of text prompts PER image
TOTAL_QUERIES = IMAGE_BATCH_SIZE * PROMPT_BATCH_SIZE

INPUT_PATH = "./input_pics/glasses.png"
THR = 0.5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def time_stage(func):
    """Decorator to time functions with CUDA synchronization."""
    def wrapper(*args, **kwargs):
        # Sync before starting timer to clear pending ops
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        
        result = func(*args, **kwargs)
        
        # Sync before stopping timer to measure actual execution
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.perf_counter()
        print(f"Stage [{func.__name__}]: {end - start:.4f}s")
        return result, end - start
    return wrapper

@time_stage
def preprocess_images(path, batch_size):
    transform = v2.Compose([
        v2.ToDtype(torch.uint8, scale=True),
        v2.Resize(size=(1008, 1008)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    
    image_bgr = cv2.imread(path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(image_rgb)
    
    img_tensor = v2.functional.to_image(pil_img).to(DEVICE)
    print("OG Image Shape:", img_tensor.shape)
    processed_img = transform(img_tensor)
    
    # Create batch by stacking (simulating multiple images)
    batch = torch.stack([processed_img] * batch_size)
    return batch

def prepare_constant_inputs(model, total_queries):
    """
    Safely expands the dummy prompt to match the total query batch size.
    Crucial: Uses .contiguous() to prevent CUDA memory stride errors.
    """
    dummy_prompt = model._get_dummy_prompt()
    
    def expand_attr(v, is_mask_type=False):
        if v is None:
            return None
        v = v.detach().clone() # Detach from graph to avoid side effects
        
        if is_mask_type:
            # Mask tensors are [Batch, Seq_Len]
            # We need [Total_Queries, Seq_Len]
            sizes = list(v.shape)
            sizes[0] = total_queries
            return v.expand(*sizes).contiguous()
        else:
            # Embedding/Label tensors are [Seq_Len, Batch, Channels]
            # We need [Seq_Len, Total_Queries, Channels]
            sizes = list(v.shape)
            sizes[1] = total_queries
            return v.expand(*sizes).contiguous()

    # Re-construct the Prompt object with valid, expanded tensors
    batched_prompt = Prompt(
        box_embeddings=expand_attr(dummy_prompt.box_embeddings, is_mask_type=False),
        box_mask=expand_attr(dummy_prompt.box_mask, is_mask_type=True),
        point_embeddings=expand_attr(dummy_prompt.point_embeddings, is_mask_type=False),
        point_mask=expand_attr(dummy_prompt.point_mask, is_mask_type=True),
        box_labels=expand_attr(dummy_prompt.box_labels, is_mask_type=False),
        point_labels=expand_attr(dummy_prompt.point_labels, is_mask_type=False),
        mask_embeddings=expand_attr(dummy_prompt.mask_embeddings, is_mask_type=False),
        mask_mask=expand_attr(dummy_prompt.mask_mask, is_mask_type=True),
        mask_labels=expand_attr(dummy_prompt.mask_labels, is_mask_type=False)
    )

    # Correct mapping: 
    # If 2 images, 2 prompts each -> img_ids=[0,0,1,1]
    img_ids = torch.arange(IMAGE_BATCH_SIZE, device=DEVICE).repeat_interleave(PROMPT_BATCH_SIZE)
    text_ids = torch.arange(total_queries, device=DEVICE)

    return batched_prompt, img_ids, text_ids

@time_stage
def run_image_encoder(model, image_batch):
    with torch.no_grad():
        backbone_out = model.backbone.forward_image(image_batch)
        
        # SAM2 interactivity logic checks
        if model.inst_interactive_predictor is not None and "sam2_backbone_out" in backbone_out:
            s2_out = backbone_out["sam2_backbone_out"]
            s2_out["backbone_fpn"][0] = model.inst_interactive_predictor.model.sam_mask_decoder.conv_s0(s2_out["backbone_fpn"][0])
            s2_out["backbone_fpn"][1] = model.inst_interactive_predictor.model.sam_mask_decoder.conv_s1(s2_out["backbone_fpn"][1])
        return backbone_out

@time_stage
def run_text_encoder(model, prompts):
    with torch.no_grad():
        text_outputs = model.backbone.forward_text(prompts, device=DEVICE)
    return text_outputs

@time_stage
def run_decoder(model, backbone_out, geometric_prompt, img_ids, text_ids):
    find_input = FindStage(
        img_ids=img_ids,
        text_ids=text_ids,
        input_boxes=None, input_boxes_mask=None, input_boxes_label=None,
        input_points=None, input_points_mask=None,
    )

    with torch.no_grad():
        outputs = model.forward_grounding(
            backbone_out=backbone_out,
            find_input=find_input,
            geometric_prompt=geometric_prompt,
            find_target=None,
        )
    return outputs

@time_stage
def postprocess(outputs, threshold, img_size=(1008, 1008)):
    # Outputs are already on GPU
    out_bbox = outputs["pred_boxes"]
    out_logits = outputs["pred_logits"]
    out_masks = outputs["pred_masks"]
    
    # Calculate probabilities
    out_probs = (out_logits.sigmoid() * outputs["presence_logit_dec"].sigmoid().unsqueeze(1)).squeeze(-1)
    
    # Filter by threshold
    keep = out_probs > threshold
    
    # Process valid detections
    if keep.sum() == 0:
        return {"scores": [], "masks": [], "boxes": []}

    final_masks = interpolate(out_masks[keep].unsqueeze(1), img_size, mode="bilinear").sigmoid() > 0.5
    final_boxes = box_ops.box_cxcywh_to_xyxy(out_bbox[keep]) * torch.tensor([img_size[1], img_size[0], img_size[1], img_size[0]], device=DEVICE)
    
    return {
        "scores": out_probs[keep],
        "masks": final_masks,
        "boxes": final_boxes
    }

def main():
    sam3_root = "./sam3/"
    checkpoint_path = f"{sam3_root}/assets/bpe_simple_vocab_16e6.txt.gz"
    
    print(f"--- Initializing Model on {DEVICE} ---")
    model = build_sam3_image_model(checkpoint_path).to(DEVICE).eval()

    # 0. Setup Constant Inputs (Done once)
    print(f"Config: {IMAGE_BATCH_SIZE} Images x {PROMPT_BATCH_SIZE} Prompts = {TOTAL_QUERIES} Total Queries")
    geo_prompt, img_ids, text_ids = prepare_constant_inputs(model, TOTAL_QUERIES)

    # 1. Preprocess
    image_batch, t_prep = preprocess_images(INPUT_PATH, IMAGE_BATCH_SIZE)

    # 2. Image Encoder
    backbone_out, t_enc = run_image_encoder(model, image_batch)

    # 3. Text Encoder
    # Example: 2 images. Image 1 gets "face" & "glasses". Image 2 gets "face" & "glasses".
    prompts = ["face with glasses"]* PROMPT_BATCH_SIZE * IMAGE_BATCH_SIZE
    
    text_out, t_text = run_text_encoder(model, prompts)
    backbone_out.update(text_out)

    # 4. Decoder (Runs in parallel for all queries)
    raw_outputs, t_dec = run_decoder(model, backbone_out, geo_prompt, img_ids, text_ids)

    # 5. Post-process
    final_results, t_post = postprocess(raw_outputs, THR)
    
    print(f"--- Detection Complete ---")
    num_detections = len(final_results['scores']) if isinstance(final_results['scores'], list) else final_results['scores'].shape[0]
    print(f"Total Detections found: {num_detections}")

    total_time = (t_prep * IMAGE_BATCH_SIZE) + t_enc + t_text + t_dec + t_post
    print(f"Total Time: {total_time:.4f}s")

if __name__ == "__main__":
    main()