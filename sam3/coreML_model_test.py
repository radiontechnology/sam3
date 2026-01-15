import gzip
import html
import io
import os
import string
from functools import lru_cache
from typing import List, Optional, Union
import ftfy
import regex as re
import torch
import coremltools as ct
import PIL.Image
import numpy as np
import matplotlib.pyplot as plt
from iopath.common.file_io import g_pathmgr
# --- 1. CONFIGURATION ---
IMAGE_PATH = "./no_glasses.jpg"
ENCODER_PATH = "sam3_modules/SAM3_Encoder.mlpackage"
DECODER_PATH = "sam3_modules/SAM3_Decoder.mlpackage"
TEXT_ENCODER_PATH = "sam3_modules/SAM3_TextEncoder.mlpackage"
# Update this to point to your actual vocab file
BPE_PATH = "./bpe_simple_vocab_16e6.txt.gz"
H, W = 1008, 1008
CTX_LEN = 32
# --- 2. TOKENIZER CLASS (Embed directly) ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"
DEFAULT_CONTEXT_LENGTH = 77
@lru_cache()
def bytes_to_unicode():
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))
def get_pairs(word):
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs
def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()
def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text
def _clean_canonicalize(x):
    return canonicalize_text(basic_clean(x))
def _clean_lower(x):
    return whitespace_clean(basic_clean(x)).lower()
def _clean_whitespace(x):
    return whitespace_clean(basic_clean(x))
def get_clean_fn(type: str):
    if type == "canonicalize":
        return _clean_canonicalize
    elif type == "lower":
        return _clean_lower
    elif type == "whitespace":
        return _clean_whitespace
    else:
        assert False, f"Invalid clean function ({type})."
def canonicalize_text(text, *, keep_punctuation_exact_string=None):
    text = text.replace("_", " ")
    if keep_punctuation_exact_string:
        text = keep_punctuation_exact_string.join(
            part.translate(str.maketrans("", "", string.punctuation))
            for part in text.split(keep_punctuation_exact_string)
        )
    else:
        text = text.translate(str.maketrans("", "", string.punctuation))
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()
class SimpleTokenizer(object):
    def __init__(
        self,
        bpe_path: Union[str, os.PathLike],
        additional_special_tokens: Optional[List[str]] = None,
        context_length: Optional[int] = DEFAULT_CONTEXT_LENGTH,
        clean: str = "lower",
    ):
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        with g_pathmgr.open(bpe_path, "rb") as fh:
            bpe_bytes = io.BytesIO(fh.read())
            merges = gzip.open(bpe_bytes).read().decode("utf-8").split("\n")
        merges = merges[1 : 49152 - 256 - 2 + 1]
        merges = [tuple(merge.split()) for merge in merges]
        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v + "</w>" for v in vocab]
        for merge in merges:
            vocab.append("".join(merge))
        special_tokens = ["<start_of_text>", "<end_of_text>"]
        if additional_special_tokens:
            special_tokens += additional_special_tokens
        vocab.extend(special_tokens)
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {t: t for t in special_tokens}
        special = "|".join(special_tokens)
        self.pat = re.compile(
            special + r"""|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
            re.IGNORECASE,
        )
        self.vocab_size = len(self.encoder)
        self.all_special_ids = [self.encoder[t] for t in special_tokens]
        self.sot_token_id = self.all_special_ids[0]
        self.eot_token_id = self.all_special_ids[1]
        self.context_length = context_length
        self.clean_fn = get_clean_fn(clean)
    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        pairs = get_pairs(word)
        if not pairs:
            return token + "</w>"
        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = " ".join(word)
        self.cache[token] = word
        return word
    def encode(self, text):
        bpe_tokens = []
        text = self.clean_fn(text)
        for token in re.findall(self.pat, text):
            token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            bpe_tokens.extend(
                self.encoder[bpe_token] for bpe_token in self.bpe(token).split(" ")
            )
        return bpe_tokens
    def __call__(
        self, texts: Union[str, List[str]], context_length: Optional[int] = None
    ) -> torch.LongTensor:
        if isinstance(texts, str):
            texts = [texts]
        context_length = context_length or self.context_length
        assert context_length, "Please set a valid context length"
        all_tokens = [
            [self.sot_token_id] + self.encode(text) + [self.eot_token_id]
            for text in texts
        ]
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
        for i, tokens in enumerate(all_tokens):
            if len(tokens) > context_length:
                tokens = tokens[:context_length]
                tokens[-1] = self.eot_token_id
            result[i, : len(tokens)] = torch.tensor(tokens)
        return result

# --- 3. HELPER FUNCTIONS ---
def load_and_resize_image(path):
    img = PIL.Image.open(path).convert('RGB')
    img = img.resize((W, H), PIL.Image.Resampling.BILINEAR)
    return img
def get_text_inputs(text_prompt, tokenizer, model):
    """
    Tokenizes text using SimpleTokenizer and runs the Core ML Text Encoder.
    """
    print(f"Encoding text prompt: '{text_prompt}'")
    # 1. Tokenize (Results in shape [1, 77])
    # The tokenizer returns a Tensor, convert to numpy
    tokens = tokenizer(text_prompt)
    input_ids = tokens.detach().cpu().numpy().astype(np.int32)
    # 2. Run Core ML Text Encoder
    out = model.predict({"input_ids": input_ids})
    return out["lang_feats"], out["lang_mask"]
# --- 4. MAIN ---
def main():
    print(f"Loading Core ML models...")
    compute_config = ct.ComputeUnit.CPU_ONLY
    encoder = ct.models.MLModel(ENCODER_PATH, compute_units=compute_config)
    decoder = ct.models.MLModel(DECODER_PATH, compute_units=compute_config)
    # Init Tokenizer
    try:
        tokenizer = SimpleTokenizer(bpe_path=BPE_PATH, context_length=CTX_LEN)
        print("Tokenizer initialized successfully.")
    except Exception as e:
        print(f"Failed to load Tokenizer: {e}")
        tokenizer = None
    try:
        text_encoder = ct.models.MLModel(TEXT_ENCODER_PATH, compute_units=compute_config)
        has_text_model = True
    except:
        print("Could not load Text Encoder. Text prompts will be skipped.")
        has_text_model = False
    # 1. Prepare Image
    print(f"Processing image: {IMAGE_PATH}")
    pil_img = load_and_resize_image(IMAGE_PATH)
    # 2. Run Image Encoder
    print("Running Image Encoder...")
    encoder_out = encoder.predict({"input_image": pil_img})
    print("Encoder complete.")
    # 3. Setup Text Inputs
    # ---------------------------------------------------------
    TEXT_PROMPT = "eyebrow"
    # ---------------------------------------------------------
    if has_text_model and tokenizer and TEXT_PROMPT:
        lang_feats, lang_mask = get_text_inputs(TEXT_PROMPT, tokenizer, text_encoder)
        lang_feats = lang_feats.astype(np.float32)
        lang_mask = lang_mask.astype(np.float32)
    else:
        print("Using dummy text embeddings (No prompt or tokenizer missing).")
        # Ensure these shapes match your Decoder's expectations!
        # If your decoder expects (1, 77, 256), make sure this fallback does too.
        lang_feats = np.zeros((1, 1, 256), dtype=np.float32)
        lang_mask = np.zeros((1, 1), dtype=np.float32)
    # 4. Prepare Decoder Inputs
    decoder_inputs = {
        "fpn0": encoder_out["fpn0"],
        "fpn1": encoder_out["fpn1"],
        "fpn2": encoder_out["fpn2"],
        "pos0": encoder_out["pos0"],
        "pos1": encoder_out["pos1"],
        "pos2": encoder_out["pos2"],
        "lang_feats": lang_feats,
        "lang_mask": lang_mask
    }
    def to_fp16(x):
        if isinstance(x, np.ndarray) and x.dtype == np.float32:
            return x.astype(np.float16)
        return x
    for k in decoder_inputs:
        decoder_inputs[k] = to_fp16(decoder_inputs[k])
        decoder_inputs[k] = np.clip(decoder_inputs[k], -20, 20)
    # 5. Add Prompts (Points/Boxes)
    N_POINTS = 5
    N_BOXES = 5
    dummy_points = np.zeros((1, N_POINTS, 2), dtype=np.float32)
    dummy_labels = np.zeros((1, N_POINTS), dtype=np.int32)
    dummy_labels[:] = 1
    dummy_boxes = np.zeros((1, N_BOXES, 4), dtype=np.float32)
    dummy_box_lbls = np.zeros((1, N_BOXES), dtype=np.int32)
    decoder_inputs["point_coords"] = dummy_points
    decoder_inputs["point_labels"] = dummy_labels
    decoder_inputs["box_coords"] = dummy_boxes
    decoder_inputs["box_labels"] = dummy_box_lbls
    # 6. Run Decoder
    print("Running Mask Decoder...")
    try:
        decoder_out = decoder.predict(decoder_inputs)
        masks = decoder_out["masks"]
        iou = decoder_out["iou_scores"]
        iou = 1 / (1 + np.exp(-iou)) # Apply sigmoid to IOU scores
        print(f"Output Mask Shape: {masks.shape}")
        # Filter masks based on an IOU threshold (e.g., 0.5)
        IOU_THRESHOLD = 0.6
        high_iou_indices = np.where(iou[0] > IOU_THRESHOLD)[0]
        if len(high_iou_indices) == 0:
            high_iou_indices = np.array([np.argmax(iou[0])])
            print("No masks found for", TEXT_PROMPT)
            print(". Highest score founnd was", np.max(iou[0]))
            #return 0
        # Combine all masks that meet the IOU threshold using an OR operation
        combined_mask = np.zeros_like(masks[0, 0, :, :], dtype=np.bool_)
        for idx in high_iou_indices:
            combined_mask = np.logical_or(combined_mask, masks[0, idx, :, :] > 0)
        binary_mask = combined_mask.astype(np.uint8) # Convert to binary mask
        # List of best IOU scores
        best_iou_scores = iou[0, high_iou_indices]
        print(f"Best IOU scores for prompt '{TEXT_PROMPT}': {best_iou_scores}")
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title(f"Image\nPrompt: '{TEXT_PROMPT}'")
        plt.imshow(pil_img)
        plt.subplot(1, 2, 2)
        plt.title(f"Prediction (IOU: {', '.join([f'{s.item():.2f}' for s in best_iou_scores])})")
        plt.imshow(binary_mask, cmap='gray')
        save_path = "sam3_inference_text_result.png"
        plt.savefig(save_path)
        print(f"\nSuccess! Result saved to {save_path}")
    except RuntimeError as e:
        print(f"\nCore ML Prediction Failed: {e}")
        print("Debug Info - Input Shapes/Types:")
        for k, v in decoder_inputs.items():
            if hasattr(v, 'shape'):
                print(f"  {k}: {v.shape} ({v.dtype})")
            else:
                print(f"  {k}: {type(v)}")
if __name__ == "__main__":
    main()