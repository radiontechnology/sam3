import torch
import torch.nn as nn
import coremltools as ct
from sam3.model_builder import build_sam3_image_model
import numpy as np
import torchvision
import torch.nn.functional as F

def no_op(self, *args, **kwargs):
    return self

torch.Tensor.pin_memory = no_op


def roi_align_patched(input, boxes, output_size, spatial_scale=1.0, sampling_ratio=-1, aligned=False):
    """
    Core ML compatible RoIAlign using explicit index_select.
    """
    # 1. Handle Boxes format
    if isinstance(boxes, (list, tuple)):
        rois_list = []
        for i, b in enumerate(boxes):
            batch_idx = torch.full((b.shape[0], 1), i, dtype=b.dtype, device=b.device)
            rois_list.append(torch.cat([batch_idx, b], dim=1))
        rois = torch.cat(rois_list, dim=0) if rois_list else torch.zeros((0, 5), dtype=input.dtype, device=input.device)
        
    elif isinstance(boxes, torch.Tensor):
        if boxes.dim() == 3:
            B, N, _ = boxes.shape
            batch_inds = torch.arange(B, dtype=boxes.dtype, device=boxes.device).view(B, 1).expand(B, N).reshape(-1, 1)
            flat_boxes = boxes.reshape(-1, 4)
            rois = torch.cat([batch_inds, flat_boxes], dim=1)
        else:
            rois = boxes
    else:
        raise ValueError(f"Unsupported boxes type: {type(boxes)}")

    # 2. Parse Arguments
    if isinstance(output_size, int):
        out_h = out_w = output_size
    else:
        out_h, out_w = output_size
    
    # 3. Early Exit
    if rois.shape[0] == 0:
        return torch.zeros((0, input.shape[1], out_h, out_w), dtype=input.dtype, device=input.device)

    n_rois = rois.shape[0]
    in_h, in_w = input.shape[2], input.shape[3]
    
    # Indices and Coords
    batch_inds = rois[:, 0].long()
    coords = rois[:, 1:] * spatial_scale

    if aligned:
        coords -= 0.5

    start_x, start_y = coords[:, 0], coords[:, 1]
    end_x, end_y = coords[:, 2], coords[:, 3]

    roi_w = end_x - start_x
    roi_h = end_y - start_y
    
    roi_w = torch.clamp(roi_w, min=1.0)
    roi_h = torch.clamp(roi_h, min=1.0)

    # 4. Grid Generation
    bin_size_w = roi_w / out_w
    bin_size_h = roi_h / out_h
    
    xx = torch.arange(0, out_w, device=input.device, dtype=input.dtype)
    yy = torch.arange(0, out_h, device=input.device, dtype=input.dtype)
    
    grid_y, grid_x = torch.meshgrid(yy, xx, indexing='ij') 
    
    grid_x = grid_x.unsqueeze(0).expand(n_rois, -1, -1)
    grid_y = grid_y.unsqueeze(0).expand(n_rois, -1, -1)
    
    grid_x = start_x.view(-1, 1, 1) + (grid_x + 0.5) * bin_size_w.view(-1, 1, 1)
    grid_y = start_y.view(-1, 1, 1) + (grid_y + 0.5) * bin_size_h.view(-1, 1, 1)
    
    grid_x = (2.0 * grid_x / (in_w - 1)) - 1.0
    grid_y = (2.0 * grid_y / (in_h - 1)) - 1.0
    
    grid = torch.stack([grid_x, grid_y], dim=-1)
    
    # --- 5. Sample (THE FIX) ---
    # OLD: input_gathered = input[batch_inds] 
    # NEW: Explicit index_select. This creates a clean "Gather" layer in Core ML.
    input_gathered = torch.index_select(input, 0, batch_inds)
    
    output = F.grid_sample(
        input_gathered, 
        grid, 
        mode='bilinear', 
        padding_mode='zeros', 
        align_corners=True
    )
    
    return output

# Apply the Patch
torchvision.ops.roi_align = roi_align_patched

# Make sure these imports work in your environment
try:
    from sam3.model.geometry_encoders import Prompt
    from sam3.model.data_misc import FindStage
except ImportError:
    # Fallback/Mock if running without full sam3 source in this specific check
    print("Warning: Could not import Prompt/FindStage from sam3. Assuming they are available at runtime.")
    pass

def dummy_checkpoint(function, *args, **kwargs):
    # use_reentrant is a specific kwarg for checkpoint that we must strip
    # because the underlying function likely doesn't expect it
    kwargs.pop('use_reentrant', None) 
    return function(*args, **kwargs)

torch.utils.checkpoint.checkpoint = dummy_checkpoint
torch.utils.checkpoint.checkpoint_sequential = dummy_checkpoint


# --- 1. CONFIGURATION ---
sam3_root = "/home2/radion/users/gborras/work/derma/segmenter/sam3/sam3/"
checkpoint = f"{sam3_root}/assets/bpe_simple_vocab_16e6.txt.gz"
H, W = 1008, 1008 

print("Loading model...")
full_model = build_sam3_image_model(checkpoint)
full_model.eval().to("cpu")

# ==========================================
# WRAPPER 1: IMAGE ENCODER
# ==========================================
class SAM3ImageEncoderWrapper(nn.Module):
    def __init__(self, sam_model):
        super().__init__()
        self.backbone = sam_model.backbone 

    def forward(self, x):
        out_dict = self.backbone.forward_image(x)
        fpn_feats = out_dict["backbone_fpn"]
        pos_enc = out_dict["vision_pos_enc"]
        

        return fpn_feats[0], fpn_feats[1], fpn_feats[2], \
               pos_enc[0], pos_enc[1], pos_enc[2]



# ==========================================
# WRAPPER 2: MASK DECODER
# ==========================================
class SAM3DecoderWrapper(nn.Module):
    def __init__(self, sam_model, target_size=(1008, 1008)):
        super().__init__()
        self.model = sam_model
        self.target_size = target_size
        
    def forward(self, 
                fpn0, fpn1, fpn2, 
                pos0, pos1, pos2, 
                lang_feats, lang_mask,
                point_coords, point_labels,
                box_coords, box_labels):
        
        # 1. Safety Casts
        point_labels = point_labels.long()
        box_labels = box_labels.long()
        
        # FIX: Cast float mask (0.0/1.0) to boolean (False/True)
        # PyTorch MultiheadAttention treats Float masks as additive bias, but Boolean masks as ignore flags.
        # We want 1.0 (padding) -> True (Ignore).
        lang_mask = lang_mask > 0.5
        
        # 2. Reconstruct dictionaries
        backbone_out = {
            "backbone_fpn": [fpn0, fpn1, fpn2],
            "vision_pos_enc": [pos0, pos1, pos2],
            "language_features": lang_feats,
            "language_mask": lang_mask,
        }

        # 3. Dummy FindStage input (CPU fallback for tracing)
        find_input = FindStage(
            img_ids=torch.zeros(1, dtype=torch.long),
            text_ids=torch.zeros(1, dtype=torch.long),
            input_boxes=None, input_boxes_mask=None, input_boxes_label=None,
            input_points=None, input_points_mask=None, object_ids=None
        )

        # 4. Prepare Geometric Inputs (Batch, N, ...) -> (N, Batch, ...)
        geo_points = point_coords.transpose(0, 1)
        geo_labels = point_labels.transpose(0, 1)
        geo_boxes = box_coords.transpose(0, 1)
        geo_box_labels = box_labels.transpose(0, 1)

        geometric_prompt = Prompt(
            point_embeddings=geo_points,
            point_labels=geo_labels,
            point_mask=None,
            box_embeddings=geo_boxes,
            box_labels=geo_box_labels,
            box_mask=None,
            mask_labels=torch.zeros(0, 1, dtype=torch.int64),
            mask_mask=torch.zeros(1, 0, dtype=torch.bool),
            mask_embeddings=None
        )

        # 5. Run Encoder
        prompt, prompt_mask, backbone_out = self.model._encode_prompt(
            backbone_out, find_input, geometric_prompt
        )
        
        backbone_out, encoder_out, _ = self.model._run_encoder(
            backbone_out, find_input, prompt, prompt_mask
        )
        
        out = {
            "encoder_hidden_states": encoder_out["encoder_hidden_states"],
            "prev_encoder_out": {
                "encoder_out": encoder_out,
                "backbone_out": backbone_out,
            },
        }

        # 6. Run Decoder
        out, hs = self.model._run_decoder(
            memory=out["encoder_hidden_states"],
            pos_embed=encoder_out["pos_embed"],
            src_mask=encoder_out["padding_mask"],
            out=out,
            prompt=prompt,
            prompt_mask=prompt_mask,
            encoder_out=encoder_out,
        )

        self.model._run_segmentation_heads(
            out=out,
            backbone_out=backbone_out,
            img_ids=find_input.img_ids,
            vis_feat_sizes=encoder_out["vis_feat_sizes"],
            encoder_hidden_states=out["encoder_hidden_states"],
            prompt=prompt,
            prompt_mask=prompt_mask,
            hs=hs,
        )
        
        # --- KEY FIX START ---
        low_res_masks = out["pred_masks"] # Typically 256x256
        scores = out["pred_logits"]       # Shape: (B, Queries) or (B, Queries, 1)
        print("scores", scores[:5])
        # Interpolate to target size (1008x1008)
        # Using bilinear for smooth mask edges
        upscaled_masks = F.interpolate(
            low_res_masks, 
            size=self.target_size, 
            mode='bilinear', 
            align_corners=False
        )
        # --- KEY FIX END ---
        
        return upscaled_masks, scores
    
# ==========================================
# WRAPPER 3: TEXT ENCODER
# ==========================================
class SAM3TextEncoderCoreML(nn.Module):
    def __init__(self, sam_model):
        super().__init__()
        # 1. Access the container (VETextEncoder)
        ve_encoder_container = sam_model.backbone.language_backbone
        
        # 2. Grab the Transformer (TextTransformer)
        self.encoder = ve_encoder_container.encoder
        
        # 3. Grab the Resizer (nn.Linear)
        self.resizer = ve_encoder_container.resizer

    def forward(self, input_ids):
        """
        input_ids: (Batch, Seq_Len) Int32 Tensor
        """
        # --- 1. Token & Positional Embedding ---
        x = self.encoder.token_embedding(input_ids)
        
        seq_len = input_ids.shape[1]
        x = x + self.encoder.positional_embedding[:seq_len]

        # --- 2. Attention Mask ---
        attn_mask = self.encoder.attn_mask
        if attn_mask is not None:
            attn_mask = attn_mask[:seq_len, :seq_len]

        # --- 3. Transformer Block ---
        x = self.encoder.transformer(x, attn_mask=attn_mask)

        # --- 4. Layer Norm ---
        # Current shape: [Batch, Seq_Len, 1024]
        x = self.encoder.ln_final(x)

        # --- 5. Transpose (Sequence First) ---
        # Change from [Batch, Seq_Len, Dim] -> [Seq_Len, Batch, Dim]
        # Example: [1, 32, 1024] -> [32, 1, 1024]
        x = x.transpose(0, 1)

        # --- 6. Resizer ---
        # Map dim: 1024 -> 256
        # Linear layer applies to the last dim, so shapes are preserved
        # Final shape: [32, 1, 256]
        x = self.resizer(x)
        
        # --- 7. Output Formatting ---
        # lang_mask usually stays [Batch, Seq_Len], but you can transpose it too if needed
        lang_mask = (input_ids == 0).float()
        
        return x, lang_mask

# ==========================================
# EXECUTION: EXPORT SCRIPT
# ==========================================
print("\n--- Converting Text Encoder (Corrected) ---")

try:
    # 1. Initialize Wrapper
    wrapper = SAM3TextEncoderCoreML(full_model).eval()
    
    # 2. Prepare Dummy Input
    # Use the context_length defined in the model (usually 77)
    ctx_len = full_model.backbone.language_backbone.encoder.context_length
    print(f"Using Context Length: {ctx_len}")
    
    # Create dummy tokens (Batch=1, Len=ctx_len)
    # 49406 is start, 49407 is end.
    dummy_ids = torch.zeros((1, ctx_len), dtype=torch.long)
    dummy_ids[0, 0] = 49406 
    dummy_ids[0, 1] = 49407
    
    # 3. Trace
    # We use strict=False to allow the dynamic slicing of positional embeddings
    traced_text = torch.jit.trace(wrapper, dummy_ids, strict=False)
    lang_feats, lang_mask = wrapper(dummy_ids)
    
    # 4. Convert
    text_mlmodel = ct.convert(
        traced_text,
        inputs=[
            ct.TensorType(name="input_ids", shape=(1, ctx_len), dtype=np.int64)
        ],
        outputs=[
            ct.TensorType(name="lang_feats"), # The full sequence (Batch, 77, Dim)
            ct.TensorType(name="lang_mask")   # The padding mask (Batch, 77)
        ],
        compute_precision=ct.precision.FLOAT16
    )
    
    text_mlmodel.save("SAM3_TextEncoder.mlpackage")
    print("Success! Saved SAM3_TextEncoder.mlpackage")

except Exception as e:
    print(f"\nERROR: {e}")
    # Debugging tip:
    import traceback
    traceback.print_exc()

# ==========================================
# EXECUTION: ENCODER
# ==========================================
print("\n--- Converting Image Encoder ---")
encoder_wrapper = SAM3ImageEncoderWrapper(full_model)
dummy_image = torch.randn(1, 3, H, W)

traced_encoder = torch.jit.trace(encoder_wrapper, dummy_image, strict=False)

encoder_mlmodel = ct.convert(
    traced_encoder,
    inputs=[ct.ImageType(name="input_image", shape=dummy_image.shape, scale=1/255.0, bias=[0,0,0])],
    outputs=[
        ct.TensorType(name="fpn0"),       # Matches fpn_feats[0]
        ct.TensorType(name="fpn1"),       # Matches fpn_feats[1]
        ct.TensorType(name="fpn2"),       # Matches fpn_feats[2]
        ct.TensorType(name="pos0"),       # Matches pos_enc[0]
        ct.TensorType(name="pos1"),       # Matches pos_enc[1]
        ct.TensorType(name="pos2")       # Matches pos_enc[2]
    ],
    skip_model_load=True,
)
encoder_mlmodel.save("SAM3_Encoder.mlpackage")
print("Saved SAM3_Encoder.mlpackage")

# ==========================================
# EXECUTION: DECODER
# ==========================================
print("\n--- Converting Mask Decoder (Fix) ---")

# 1. Get dummy outputs from encoder (Same as before)
with torch.no_grad():
    outs = encoder_wrapper(dummy_image)
    fpn0, fpn1, fpn2, pos0, pos1, pos2 = outs

# 2. Create Dummy Prompts (CRITICAL: Must be non-empty for Trace)
# Points
dummy_points = torch.tensor([[[500.0, 500.0]]], dtype=torch.float64) # (1, 1, 2)
dummy_labels = torch.tensor([[1]], dtype=torch.long) # (1, 1)

# Boxes (Adding 1 dummy box to satisfy the Tracer)
dummy_boxes = torch.tensor([[[0.0, 0.0, 10.0, 10.0]]], dtype=torch.float64) # (1, 1, 4)
dummy_box_labels = torch.tensor([[1]], dtype=torch.int64) # (1, 1)

decoder_wrapper = SAM3DecoderWrapper(full_model)
decoder_wrapper.eval()

# Dummy Data (Must have shape > 0 to establish valid ranks in the graph)
dummy_points = torch.zeros(1, 50, 2)
dummy_labels = torch.zeros(1, 50, dtype=torch.int32)
dummy_boxes  = torch.zeros(1, 50, 4)
dummy_box_lbls = torch.zeros(1, 50, dtype=torch.int32)


# Get encoder features (using random for tracing is fine if shapes match)
# Or use the outputs from your encoder trace earlier
""" f_dim = 256
fpn0 = torch.randn(1, f_dim, 288, 288)   # Adjust based on your H/W (1008/16 approx)
fpn1 = torch.randn(1, f_dim, 144, 144)
fpn2 = torch.randn(1, f_dim, 72, 72)
pos0 = torch.randn(1, f_dim, 288, 288)
pos1 = torch.randn(1, f_dim, 144, 144)
pos2 = torch.randn(1, f_dim, 72, 72)
print("gborras")
lang_feats = torch.randn(1, 1, f_dim)
lang_mask = torch.zeros(1, 1, dtype=torch.bool) """

inputs = (fpn0, fpn1, fpn2, pos0, pos1, pos2, lang_feats, lang_mask, 
          dummy_points, dummy_labels, dummy_boxes, dummy_box_lbls)

# Trace strictly
traced_decoder = torch.jit.trace(decoder_wrapper, inputs, strict=False, check_trace=False)

# --- 3. Conversion with Correct Types ---
print("\n--- Converting to Core ML ---")

# Dynamic Dimensions
# We set start=1 to avoid "zero-dimension" crashes. 
# Handle "no box" logic in your Swift/Python app (pass a [-1,-1,-1,-1] dummy box).
N_POINTS_FIXED = 5
N_BOXES_FIXED  = 5

# 1. Get dummy outputs from encoder (Same as before)
with torch.no_grad():
    outs = encoder_wrapper(dummy_image)
    fpn0, fpn1, fpn2, pos0, pos1, pos2 = outs

decoder_wrapper = SAM3DecoderWrapper(full_model)
decoder_wrapper.eval()

# 2. Create Dummy Prompts with EXACT Target Shape
# --- FIX: Use N_POINTS_FIXED here, not 50 ---
dummy_points = torch.zeros(1, N_POINTS_FIXED, 2)
dummy_labels = torch.zeros(1, N_POINTS_FIXED, dtype=torch.int32)
dummy_boxes  = torch.zeros(1, N_BOXES_FIXED, 4)
dummy_box_lbls = torch.zeros(1, N_BOXES_FIXED, dtype=torch.int32)

# Set at least one valid point/box to ensure graph validity
dummy_points[0,0,:] = 50.0
dummy_labels[0,0] = 1

# Get encoder features (shapes must match encoder output)
""" f_dim = 256
TEXT_LEN = full_model.backbone.language_backbone.encoder.context_length
fpn0 = torch.randn(1, f_dim, 288, 288)
fpn1 = torch.randn(1, f_dim, 144, 144)
fpn2 = torch.randn(1, f_dim, 72, 72)
pos0 = torch.randn(1, f_dim, 288, 288)
pos1 = torch.randn(1, f_dim, 144, 144)
pos2 = torch.randn(1, f_dim, 72, 72)
lang_feats = torch.randn(32, 1, f_dim)
lang_mask = torch.zeros(1, 32, dtype=torch.bool) """

inputs = (fpn0, fpn1, fpn2, pos0, pos1, pos2, lang_feats, lang_mask, 
          dummy_points, dummy_labels, dummy_boxes, dummy_box_lbls)

# 3. Trace
print(f"Tracing with {N_POINTS_FIXED} points and {N_BOXES_FIXED} boxes...")
traced_decoder = torch.jit.trace(decoder_wrapper, inputs, strict=False, check_trace=True)

# 4. Convert
print("\n--- Converting to Core ML ---")

decoder_mlmodel = ct.convert(
    traced_decoder,
    inputs=[
        ct.TensorType(name="fpn0", shape=fpn0.shape),
        ct.TensorType(name="fpn1", shape=fpn1.shape),
        ct.TensorType(name="fpn2", shape=fpn2.shape),
        ct.TensorType(name="pos0", shape=pos0.shape),
        ct.TensorType(name="pos1", shape=pos1.shape),
        ct.TensorType(name="pos2", shape=pos2.shape),
        ct.TensorType(name="lang_feats", shape=lang_feats.shape),
        ct.TensorType(name="lang_mask", shape=lang_mask.shape),
        
        # --- FIX: Match the Trace Shapes exactly ---
        ct.TensorType(name="point_coords", shape=(1, N_POINTS_FIXED, 2)),
        ct.TensorType(name="point_labels", shape=(1, N_POINTS_FIXED), dtype=np.int64),
        ct.TensorType(name="box_coords", shape=(1, N_BOXES_FIXED, 4)),
        ct.TensorType(name="box_labels", shape=(1, N_BOXES_FIXED), dtype=np.int64),
    ],
    outputs=[
        ct.TensorType(name="masks"),
        ct.TensorType(name="iou_scores")
    ],
    compute_precision=ct.precision.FLOAT16
)

decoder_mlmodel.save("SAM3_Decoder.mlpackage")
print("Saved SAM3_Decoder.mlpackage")