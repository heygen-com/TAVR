import flash_attn_interface
import torch

__all__ = ["flash_attention", "flash_attention_videoref"]

_HALF = (torch.float16, torch.bfloat16)


def _half(x: torch.Tensor) -> torch.Tensor:
    return x if x.dtype in _HALF else x.to(torch.bfloat16)


def _pack(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, ...]:
    q, k, v = _half(q.flatten(0, 1)), _half(k.flatten(0, 1)), _half(v.flatten(0, 1))
    return q.to(v.dtype), k.to(v.dtype), v


def _offsets(batch: int, seq_len: int, device: torch.device) -> torch.Tensor:
    return torch.arange(batch + 1, dtype=torch.int32, device=device) * seq_len


def _attend(q, k, v, cu_q, cu_k, max_q, max_k):
    return flash_attn_interface.flash_attn_varlen_func(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        seqused_q=None,
        seqused_k=None,
        max_seqlen_q=max_q,
        max_seqlen_k=max_k,
        softmax_scale=None,
        causal=False,
        deterministic=False,
    )[0]


def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    assert q.device.type == "cuda" and q.size(-1) <= 256
    batch, len_q, len_k, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype
    q, k, v = _pack(q, k, v)

    out = _attend(
        q,
        k,
        v,
        _offsets(batch, len_q, q.device),
        _offsets(batch, len_k, q.device),
        len_q,
        len_k,
    )
    return out.unflatten(0, (batch, len_q)).type(out_dtype)


def flash_attention_videoref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_videoref: torch.Tensor,
    k_videoref: torch.Tensor,
    v_videoref: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert q.device.type == "cuda" and q.size(-1) <= 256
    assert q_videoref.device.type == "cuda" and q_videoref.size(-1) <= 256
    batch, len_q, len_k, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype
    len_q_videoref, len_k_videoref = q_videoref.size(1), k_videoref.size(1)
    assert batch == 1 == q_videoref.size(0), "the videoref split assumes batch size 1"

    q, k, v = _pack(q, k, v)
    q_videoref, k_videoref, v_videoref = _pack(q_videoref, k_videoref, v_videoref)
    cu_k = _offsets(batch, len_k, q.device)
    cu_k_videoref = _offsets(batch, len_k_videoref, q.device)

    out = _attend(
        q,
        torch.cat([k, k_videoref], dim=0),
        torch.cat([v, v_videoref], dim=0),
        _offsets(batch, len_q, q.device),
        cu_k + cu_k_videoref,
        len_q,
        len_k + len_k_videoref,
    )
    out_videoref = _attend(
        q_videoref,
        k_videoref,
        v_videoref,
        _offsets(batch, len_q_videoref, q.device),
        cu_k_videoref,
        len_q_videoref,
        len_k_videoref,
    )
    return (
        out.unflatten(0, (batch, len_q)).type(out_dtype),
        out_videoref.unflatten(0, (batch, len_q_videoref)).type(out_dtype),
    )
