"""Export the torchaudio MMS_FA and ForceAlign CTC networks for Rust/ORT."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
import torchaudio


def exportable_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, **_kwargs):
    scale = query.shape[-1] ** -0.5
    scores = torch.matmul(query, key.transpose(-2, -1)) * scale
    if is_causal:
        length = query.shape[-2]
        causal = torch.ones((length, length), dtype=torch.bool, device=query.device).triu(1)
        scores = scores.masked_fill(causal, float("-inf"))
    if attn_mask is not None:
        scores = scores + attn_mask
    weights = torch.softmax(scores, dim=-1)
    if dropout_p:
        weights = torch.nn.functional.dropout(weights, p=dropout_p)
    return torch.matmul(weights, value)


def exportable_wav2vec_attention(self, x, attention_mask=None, position_bias=None, key_padding_mask=None):
    query_heads = self.q_proj(x).split(self.head_dim, dim=-1)
    key_heads = self.k_proj(x).split(self.head_dim, dim=-1)
    value_heads = self.v_proj(x).split(self.head_dim, dim=-1)
    attended_heads = []
    for query, key, value in zip(query_heads, key_heads, value_heads):
        scores = torch.matmul(query, key.transpose(-2, -1)) * (self.head_dim ** -0.5)
        if attention_mask is not None:
            scores = scores + attention_mask[:, 0]
        attended_heads.append(torch.matmul(torch.softmax(scores, dim=-1), value))
    attended = torch.cat(attended_heads, dim=-1)
    return self.out_proj(attended), None


class Emissions(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, log_probabilities: bool) -> None:
        super().__init__()
        self.model = model.eval()
        self.log_probabilities = log_probabilities

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        result = self.model(audio)
        emissions = result[0] if isinstance(result, tuple) else result
        if self.log_probabilities:
            return emissions
        return torch.log_softmax(emissions, dim=-1)


def export(model: torch.nn.Module, destination: Path, *, log_probabilities: bool) -> None:
    wrapped = Emissions(model, log_probabilities).eval()
    # Fixed 30 second inputs keep export compatible with torchaudio's
    # wav2vec2 graph; Rust pads and windows every song into fixed-size chunks.
    example = torch.zeros(1, 16000 * 30, dtype=torch.float32)
    torch.onnx.export(
        wrapped,
        (example,),
        str(destination),
        input_names=["audio"],
        output_names=["log_probs"],
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("backend/models/alignment"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # The stock PyTorch SDPA ONNX decomposition in torch 2.8 fails to preserve
    # dynamic sequence dimensions for wav2vec2. The equivalent primitive ops
    # below export as standard ONNX MatMul/Softmax nodes.
    torch.nn.functional.scaled_dot_product_attention = exportable_attention
    torchaudio.models.wav2vec2.components.SelfAttention.forward = exportable_wav2vec_attention

    mms = torchaudio.pipelines.MMS_FA.get_model(with_star=False).cpu().eval()
    export(mms, args.output_dir / "mms_fa.onnx", log_probabilities=True)
    mms_vocab = torchaudio.pipelines.MMS_FA.get_dict(star=None)
    del mms
    gc.collect()

    force_bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    force = force_bundle.get_model().cpu().eval()
    export(force, args.output_dir / "forcealign_en.onnx", log_probabilities=False)
    force_labels = list(force_bundle.get_labels())
    del force
    gc.collect()

    (args.output_dir / "alignment_vocab.json").write_text(
        json.dumps(
            {
                "sample_rate": 16000,
                "mms_fa": {"blank": 0, "labels": list(mms_vocab)},
                "forcealign_en": {"blank": 0, "labels": force_labels},
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Exported Rust alignment models to {args.output_dir}")


if __name__ == "__main__":
    main()
