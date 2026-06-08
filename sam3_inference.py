import torch
import time
import cv2
import os
from PIL import Image
from torchvision.transforms import v2
from sam3.model_builder import build_sam3_image_model
from sam3.model.data_misc import FindStage, interpolate
from sam3.model import box_ops
from sam3.model.sam3_image import Prompt

class SAM3Inferencer:
    def __init__(self, model, device="cuda"):
        self.model = model
        self.device = device
        self.transform = v2.Compose([
            v2.ToDtype(torch.uint8, scale=True),
            v2.Resize(size=(1008, 1008)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def preprocess_images(self, image_bgr, batch_size):
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(image_rgb)
        img_tensor = v2.functional.to_image(pil_img).to(self.device)
        processed_img = self.transform(img_tensor)
        batch = torch.stack([processed_img] * batch_size)
        return batch

    def prepare_constant_inputs(self, total_queries, image_batch_size, prompt_batch_size):
        dummy_prompt = self.model._get_dummy_prompt()
        
        def expand_attr(v, is_mask_type=False):
            if v is None:
                return None
            v = v.detach().clone()
            if is_mask_type:
                sizes = list(v.shape)
                sizes[0] = total_queries
                return v.expand(*sizes).contiguous()
            else:
                sizes = list(v.shape)
                sizes[1] = total_queries
                return v.expand(*sizes).contiguous()

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

        img_ids = torch.arange(image_batch_size, device=self.device).repeat_interleave(prompt_batch_size)
        text_ids = torch.arange(total_queries, device=self.device)
        return batched_prompt, img_ids, text_ids

    def run_image_encoder(self, image_batch):
        with torch.no_grad():
            backbone_out = self.model.backbone.forward_image(image_batch)
            if getattr(self.model, "inst_interactive_predictor", None) is not None and "sam2_backbone_out" in backbone_out:
                s2_out = backbone_out["sam2_backbone_out"]
                s2_out["backbone_fpn"][0] = self.model.inst_interactive_predictor.model.sam_mask_decoder.conv_s0(s2_out["backbone_fpn"][0])
                s2_out["backbone_fpn"][1] = self.model.inst_interactive_predictor.model.sam_mask_decoder.conv_s1(s2_out["backbone_fpn"][1])
            return backbone_out

    def run_text_encoder(self, prompts):
        with torch.no_grad():
            text_outputs = self.model.backbone.forward_text(prompts, device=self.device)
        return text_outputs

    def run_decoder(self, backbone_out, geometric_prompt, img_ids, text_ids):
        find_input = FindStage(
            img_ids=img_ids,
            text_ids=text_ids,
            input_boxes=None, input_boxes_mask=None, input_boxes_label=None,
            input_points=None, input_points_mask=None,
        )
        with torch.no_grad():
            outputs = self.model.forward_grounding(
                backbone_out=backbone_out,
                find_input=find_input,
                geometric_prompt=geometric_prompt,
                find_target=None,
            )
        return outputs

    def postprocess(self, outputs, threshold, img_size=(1008, 1008)):
        out_bbox = outputs["pred_boxes"]
        out_logits = outputs["pred_logits"]
        out_masks = outputs["pred_masks"]
        
        out_probs = (out_logits.sigmoid() * outputs["presence_logit_dec"].sigmoid().unsqueeze(1)).squeeze(-1)
        keep = out_probs > threshold
        
        if keep.sum() == 0:
            return {"scores": [], "masks": [], "boxes": []}

        final_masks = interpolate(out_masks[keep].unsqueeze(1), img_size, mode="bilinear").sigmoid() > 0.5
        final_boxes = box_ops.box_cxcywh_to_xyxy(out_bbox[keep]) * torch.tensor([img_size[1], img_size[0], img_size[1], img_size[0]], device=self.device)
        
        return {
            "scores": out_probs[keep],
            "masks": final_masks,
            "boxes": final_boxes
        }

    @torch.inference_mode()
    def infer(self, image_bgr, prompts, threshold=0.5, orig_size=None):
        image_batch_size = 1
        prompt_batch_size = len(prompts)
        total_queries = image_batch_size * prompt_batch_size

        geo_prompt, img_ids, text_ids = self.prepare_constant_inputs(total_queries, image_batch_size, prompt_batch_size)
        image_batch = self.preprocess_images(image_bgr, image_batch_size)
        backbone_out = self.run_image_encoder(image_batch)
        text_out = self.run_text_encoder(prompts)
        
        # Use a fresh dictionary to avoid mutating cached model state internally
        combined_out = dict(backbone_out)
        combined_out.update(text_out)
        
        raw_outputs = self.run_decoder(combined_out, geo_prompt, img_ids, text_ids)
        
        final_results = self.postprocess(raw_outputs, threshold, img_size=(1008, 1008))
        
        # Explicitly move tensors to CPU to free VRAM for the caller
        for k, v in final_results.items():
            if isinstance(v, torch.Tensor):
                final_results[k] = v.cpu()
                
        if orig_size is not None and len(final_results["scores"]) > 0:
            masks = final_results["masks"].float()
            masks = torch.nn.functional.interpolate(
                masks, 
                size=orig_size, 
                mode="bilinear",
                align_corners=False
            ) > 0.5
            final_results["masks"] = masks
            
            scale_x = orig_size[1] / 1008.0
            scale_y = orig_size[0] / 1008.0
            boxes = final_results["boxes"].float()
            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y
            final_results["boxes"] = boxes

        return final_results

def main():
    INPUT_PATH = "/home2/radion/common/projects/fotofinder/atbm_bodyscan_stitching/dataset/1_development/2_curated/001033/201703/scanF_img2_FotoFinderUniverse_2803171746_FFI_000002.jpeg"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    sam3_root = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(sam3_root, "sam3/assets/sam3.pt")
    
    print(f"--- Initializing Model on {DEVICE} ---")
    model = build_sam3_image_model(checkpoint_path=checkpoint_path).to(DEVICE).eval()

    inferencer = SAM3Inferencer(model, device=DEVICE)
    
    image_bgr = cv2.imread(INPUT_PATH)
    prompts = ["face with glasses"]
    
    start = time.perf_counter()
    final_results = inferencer.infer(image_bgr, prompts, threshold=0.5, orig_size=image_bgr.shape[:2])
    end = time.perf_counter()
    
    print(f"--- Detection Complete ---")
    num_detections = len(final_results['scores']) if isinstance(final_results['scores'], list) else final_results['scores'].shape[0]
    print(f"Total Detections found: {num_detections}")
    print(f"Total Time: {end - start:.4f}s")

if __name__ == "__main__":
    main()