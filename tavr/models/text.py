import re

import torch
from diffusers.pipelines.wan.pipeline_wan import prompt_clean
from transformers import AutoTokenizer, UMT5Config, UMT5EncoderModel

from tavr.contract import TEXT_LEN

__all__ = ["TextEncoder"]

UMT5_XXL = {
    "vocab_size": 256384,
    "d_model": 4096,
    "d_ff": 10240,
    "num_layers": 24,
    "num_heads": 64,
    "is_encoder_decoder": False,
    "use_cache": False,
}

_RENAMES = (
    (r"^token_embedding\.weight$", "encoder.embed_tokens.weight"),
    (r"^norm\.weight$", "encoder.final_layer_norm.weight"),
    (r"^blocks\.(\d+)\.norm1\.", r"encoder.block.\1.layer.0.layer_norm."),
    (r"^blocks\.(\d+)\.norm2\.", r"encoder.block.\1.layer.1.layer_norm."),
    (r"^blocks\.(\d+)\.attn\.([qkvo])\.", r"encoder.block.\1.layer.0.SelfAttention.\2."),
    (
        r"^blocks\.(\d+)\.pos_embedding\.embedding\.",
        r"encoder.block.\1.layer.0.SelfAttention.relative_attention_bias.",
    ),
    (r"^blocks\.(\d+)\.ffn\.gate\.0\.", r"encoder.block.\1.layer.1.DenseReluDense.wi_0."),
    (r"^blocks\.(\d+)\.ffn\.fc1\.", r"encoder.block.\1.layer.1.DenseReluDense.wi_1."),
    (r"^blocks\.(\d+)\.ffn\.fc2\.", r"encoder.block.\1.layer.1.DenseReluDense.wo."),
)


def to_transformers_key(key: str) -> str:
    for pattern, replacement in _RENAMES:
        renamed, hits = re.subn(pattern, replacement, key)
        if hits:
            return renamed
    raise KeyError(f"no rule for umT5 checkpoint entry {key!r}")


class TextEncoder:
    def __init__(self, checkpoint_path: str, tokenizer_path: str, device: str = "cpu"):
        with torch.device("meta"):
            model = UMT5EncoderModel(UMT5Config(**UMT5_XXL))
        weights = {
            to_transformers_key(name): tensor
            for name, tensor in torch.load(checkpoint_path, map_location="cpu", mmap=True).items()
        }
        weights["shared.weight"] = weights["encoder.embed_tokens.weight"]
        model.load_state_dict(weights, strict=True, assign=True)

        self.device = device
        self.model = model.eval().requires_grad_(False).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)

    @torch.no_grad()
    def __call__(self, texts: str | list[str], device: str | torch.device | None = None) -> list[torch.Tensor]:
        if isinstance(texts, str):
            texts = [texts]
        device = self.device if device is None else device
        batch = self.tokenizer(
            [prompt_clean(text) for text in texts],
            padding="max_length",
            truncation=True,
            max_length=TEXT_LEN,
            add_special_tokens=True,
            return_tensors="pt",
        )
        ids = batch.input_ids.to(device)
        mask = batch.attention_mask.to(device)
        hidden = self.model(ids, mask).last_hidden_state
        lengths = mask.gt(0).sum(dim=1).long()
        return [stream[:length] for stream, length in zip(hidden, lengths, strict=True)]
