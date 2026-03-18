import math
import re
import warnings

import torch
import torch.nn as nn


_SENTENCE_END_RE = re.compile(r"(?:[.!?。！？]+[\"'”’）)\]]*|\n+)\s*$")
_STRUCTURAL_ONLY_RE = re.compile(
    r"^(?:\s|\[/[A-Z_]+\]|\[[A-Z_]+\]|<\|[^|]+\|>|<[^>]+>|#+\s*(?:assistant|user|system)\s*:|(?:assistant|user|system)\s*:)+$",
    re.IGNORECASE,
)
_SUPPORTED_SENTENCE_POOLING = {"avgpool", "maxpool", "avg", "max", "sentence_avg", "sentence_max"}
_TOKENIZER_WARNING_EMITTED = False


# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def bind_tokenizer_to_model(model, tokenizer):
    model._snapkv_tokenizer = tokenizer
    return model


def _decode_single_token(tokenizer, token_id):
    return tokenizer.decode(
        [token_id],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def _is_sentence_boundary(text):
    return bool(text) and _SENTENCE_END_RE.search(text) is not None


def _is_whitespace_only_text(text):
    return not text or text.strip() == ""


def _is_structural_only_text(text):
    stripped = text.strip()
    if not stripped:
        return True
    return _STRUCTURAL_ONLY_RE.fullmatch(stripped) is not None


def _split_input_ids_into_sentence_spans(tokenizer, token_ids):
    sentence_spans = []
    sentence_start = 0
    sentence_text = ""
    pending_boundary = None
    sentence_has_non_whitespace = False

    for idx, token_id in enumerate(token_ids):
        token_text = _decode_single_token(tokenizer, token_id)

        if pending_boundary is not None:
            if _is_whitespace_only_text(token_text):
                continue
            if sentence_start < pending_boundary and sentence_has_non_whitespace:
                sentence_spans.append((sentence_start, pending_boundary))
            sentence_start = idx
            sentence_text = ""
            sentence_has_non_whitespace = False
            pending_boundary = None

        sentence_text += token_text
        if not _is_whitespace_only_text(token_text):
            sentence_has_non_whitespace = True
        if _is_sentence_boundary(sentence_text):
            pending_boundary = idx + 1

    sentence_end = pending_boundary if pending_boundary is not None else len(token_ids)
    if sentence_start < sentence_end and sentence_has_non_whitespace:
        sentence_spans.append((sentence_start, sentence_end))

    return sentence_spans


def _extract_semantic_sentence_metadata(tokenizer, token_ids, sentence_spans):
    if not sentence_spans:
        return sentence_spans, None, None

    semantic_spans = list(sentence_spans)
    trailing_structural_start = len(token_ids)
    while len(semantic_spans) > 1:
        start, end = semantic_spans[-1]
        span_text = tokenizer.decode(
            token_ids[start:end],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if not _is_structural_only_text(span_text):
            break
        trailing_structural_start = start
        semantic_spans.pop()

    if semantic_spans:
        last_content_span = semantic_spans[-1]
    else:
        last_content_span = sentence_spans[-1]
        semantic_spans = [last_content_span]
        trailing_structural_start = last_content_span[1]

    trailing_structural_span = None
    if trailing_structural_start < len(token_ids):
        trailing_structural_span = (trailing_structural_start, len(token_ids))

    return semantic_spans, last_content_span, trailing_structural_span


def cache_snapkv_prompt_metadata(model, input_ids, attention_mask=None):
    tokenizer = getattr(model, "_snapkv_tokenizer", None)
    metadata = None

    if tokenizer is not None:
        input_ids_cpu = input_ids.detach().cpu()
        attention_mask_cpu = attention_mask.detach().cpu() if attention_mask is not None else None
        sentence_spans = []
        prompt_lengths = []
        last_content_spans = []
        trailing_structural_spans = []

        for batch_idx in range(input_ids_cpu.shape[0]):
            if attention_mask_cpu is not None:
                prompt_length = int(attention_mask_cpu[batch_idx].sum().item())
            else:
                prompt_length = int(input_ids_cpu.shape[1])
            prompt_lengths.append(prompt_length)
            token_ids = input_ids_cpu[batch_idx, :prompt_length].tolist()
            all_sentence_spans = _split_input_ids_into_sentence_spans(tokenizer, token_ids)
            semantic_spans, last_content_span, trailing_structural_span = _extract_semantic_sentence_metadata(
                tokenizer, token_ids, all_sentence_spans
            )
            sentence_spans.append(semantic_spans)
            last_content_spans.append(last_content_span)
            trailing_structural_spans.append(trailing_structural_span)

        metadata = {
            "sentence_spans": sentence_spans,
            "prompt_lengths": prompt_lengths,
            "last_content_spans": last_content_spans,
            "trailing_structural_spans": trailing_structural_spans,
        }
    else:
        global _TOKENIZER_WARNING_EMITTED
        if not _TOKENIZER_WARNING_EMITTED:
            warnings.warn(
                "SnapKV sentence-aware compression requires a tokenizer bound to the model. "
                "Call `bind_tokenizer_to_model(model, tokenizer)` after loading the model. "
                "Falling back to fixed-window token-level SnapKV until then."
            )
            _TOKENIZER_WARNING_EMITTED = True

    model._snapkv_prompt_metadata = metadata
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        for layer in model.model.layers:
            if hasattr(layer, "self_attn"):
                layer.self_attn.snapkv_prompt_metadata = metadata

    return metadata


class SnapKVCluster:
    def __init__(
        self,
        window_size=64,
        max_capacity_prompt=256 + 64,
        kernel_size=5,
        pooling="avgpool",
        obs_window_mode="adaptive",
    ):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.window_size > 0 and self.max_capacity_prompt > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.obs_window_mode = obs_window_mode

    def reset(
        self,
        window_size=64,
        max_capacity_prompt=256 + 64,
        kernel_size=5,
        pooling="avgpool",
        obs_window_mode="adaptive",
    ):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.window_size > 0 and self.max_capacity_prompt > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.obs_window_mode = obs_window_mode

    def _pooling_mode(self):
        if self.pooling in {"avgpool", "avg", "sentence_avg"}:
            return "avg"
        if self.pooling in {"maxpool", "max", "sentence_max"}:
            return "max"
        raise ValueError("Pooling method not supported")

    def _get_window_info(self, q_len, sentence_metadata):
        base_window_size = min(int(self.window_size), int(q_len))
        query_start = q_len - base_window_size
        query_end = q_len
        keep_start = query_start

        if self.obs_window_mode == "fixed":
            return max(base_window_size, 1), query_start, query_end, keep_start

        if not sentence_metadata:
            return max(base_window_size, 1), query_start, query_end, keep_start

        last_content_spans = sentence_metadata.get("last_content_spans") or []
        if not last_content_spans or last_content_spans[0] is None:
            return max(base_window_size, 1), query_start, query_end, keep_start

        last_content_start, last_content_end = last_content_spans[0]
        last_sentence_len = max(int(last_content_end - last_content_start), 1)
        window_size = max(min(base_window_size, last_sentence_len, int(q_len)), 1)
        query_end = last_content_end
        query_start = max(last_content_start, query_end - window_size)
        keep_start = query_start
        return window_size, query_start, query_end, keep_start

    def _build_causal_mask(self, query_start, query_end, total_len, dtype, device):
        query_positions = torch.arange(query_start, query_end, device=device).unsqueeze(-1)
        key_positions = torch.arange(total_len, device=device).unsqueeze(0)
        invalid = key_positions > query_positions
        mask = torch.zeros((query_end - query_start, total_len), dtype=dtype, device=device)
        mask.masked_fill_(invalid, torch.finfo(dtype).min)
        return mask

    def _compute_prefix_token_scores(self, key_states, query_states, head_dim, query_start, query_end, prefix_len):
        if prefix_len <= 0 or query_end <= query_start:
            return None

        query_slice = query_states[..., query_start:query_end, :]
        attn_weights = torch.matmul(query_slice, key_states.transpose(2, 3)) / math.sqrt(head_dim)
        causal_mask = self._build_causal_mask(
            query_start,
            query_end,
            key_states.shape[-2],
            attn_weights.dtype,
            attn_weights.device,
        )
        attn_weights += causal_mask[None, None, :, :]
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        return attn_weights[..., :prefix_len].sum(dim=-2)

    def _gather_by_indices(self, states, indices):
        head_dim = states.shape[-1]
        indices = indices.unsqueeze(0).unsqueeze(0).unsqueeze(-1).expand(states.shape[0], states.shape[1], -1, head_dim)
        return states.gather(dim=2, index=indices)

    def _token_level_compress(self, key_states, query_states, value_states, head_dim, sentence_metadata=None):
        q_len = key_states.shape[-2]
        window_size, query_start, query_end, keep_start = self._get_window_info(q_len, sentence_metadata)
        prefix_len = keep_start
        keep_current_len = q_len - keep_start
        keep_prefix = max(self.max_capacity_prompt - keep_current_len, 0)
        if prefix_len <= 0 or keep_prefix <= 0:
            return key_states[:, :, keep_start:, :], value_states[:, :, keep_start:, :]

        token_scores = self._compute_prefix_token_scores(
            key_states,
            query_states,
            head_dim,
            query_start,
            query_end,
            prefix_len,
        )
        if token_scores is None:
            return key_states[:, :, keep_start:, :], value_states[:, :, keep_start:, :]

        keep_prefix = min(keep_prefix, prefix_len)
        indices = token_scores.topk(keep_prefix, dim=-1).indices
        indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
        k_past = key_states[:, :, :prefix_len, :].gather(dim=2, index=indices)
        v_past = value_states[:, :, :prefix_len, :].gather(dim=2, index=indices)
        k_cur = key_states[:, :, keep_start:, :]
        v_cur = value_states[:, :, keep_start:, :]
        return torch.cat([k_past, k_cur], dim=2), torch.cat([v_past, v_cur], dim=2)

    def _sentence_level_compress(self, key_states, query_states, value_states, head_dim, sentence_metadata):
        bsz, _, q_len, _ = query_states.shape
        if bsz != 1:
            return self._token_level_compress(key_states, query_states, value_states, head_dim, sentence_metadata)

        _, query_start, query_end, keep_start = self._get_window_info(q_len, sentence_metadata)
        prefix_len = keep_start
        keep_current_len = q_len - keep_start
        keep_prefix_budget = max(self.max_capacity_prompt - keep_current_len, 0)
        if prefix_len <= 0 or keep_prefix_budget <= 0:
            return key_states[:, :, keep_start:, :], value_states[:, :, keep_start:, :]

        sentence_spans_batch = sentence_metadata.get("sentence_spans") or []
        if not sentence_spans_batch or not sentence_spans_batch[0]:
            return self._token_level_compress(key_states, query_states, value_states, head_dim, sentence_metadata)

        candidate_spans = []
        for start, end in sentence_spans_batch[0]:
            if end <= prefix_len:
                candidate_spans.append((int(start), int(end)))
                continue
            if start < prefix_len:
                candidate_spans.append((int(start), int(prefix_len)))
                break
            break

        if not candidate_spans:
            return key_states[:, :, keep_start:, :], value_states[:, :, keep_start:, :]

        token_scores = self._compute_prefix_token_scores(
            key_states,
            query_states,
            head_dim,
            query_start,
            query_end,
            prefix_len,
        )
        if token_scores is None:
            return key_states[:, :, keep_start:, :], value_states[:, :, keep_start:, :]

        token_scores = token_scores.mean(dim=1)[0]
        pooling_mode = self._pooling_mode()
        ranked_sentences = []
        for start, end in candidate_spans:
            sentence_scores = token_scores[start:end]
            if sentence_scores.numel() == 0:
                continue
            if pooling_mode == "max":
                score = sentence_scores.max()
            else:
                score = sentence_scores.mean()
            ranked_sentences.append((float(score.item()), start, end))

        ranked_sentences.sort(key=lambda item: item[0], reverse=True)
        selected_spans = []
        used_tokens = 0
        for _, start, end in ranked_sentences:
            sentence_len = end - start
            if used_tokens + sentence_len > keep_prefix_budget:
                continue
            selected_spans.append((start, end))
            used_tokens += sentence_len
            if used_tokens >= keep_prefix_budget:
                break

        selected_spans.sort(key=lambda item: item[0])

        index_chunks = []
        for start, end in selected_spans:
            index_chunks.append(torch.arange(start, end, device=key_states.device, dtype=torch.long))
        index_chunks.append(torch.arange(keep_start, q_len, device=key_states.device, dtype=torch.long))
        indices = torch.cat(index_chunks, dim=0)

        return self._gather_by_indices(key_states, indices), self._gather_by_indices(value_states, indices)

    def update_kv(
        self,
        key_states,
        query_states,
        value_states,
        attention_mask,
        num_key_value_groups,
        sentence_metadata=None,
    ):
        assert key_states.shape[-2] == query_states.shape[-2]
        _, _, q_len, head_dim = query_states.shape
        if q_len < self.max_capacity_prompt:
            return key_states, value_states

        if sentence_metadata and self.pooling in _SUPPORTED_SENTENCE_POOLING:
            return self._sentence_level_compress(
                key_states,
                query_states,
                value_states,
                head_dim,
                sentence_metadata,
            )
        return self._token_level_compress(key_states, query_states, value_states, head_dim, sentence_metadata)


def init_snapkv(self):
    if not hasattr(self, "kv_cluster"):
        if not hasattr(self.config, "window_size"):
            self.config.window_size = 32
        if not hasattr(self.config, "max_capacity_prompt"):
            self.config.max_capacity_prompt = 2048
        if not hasattr(self.config, "kernel_size"):
            self.config.kernel_size = 5
        if not hasattr(self.config, "pooling"):
            self.config.pooling = "avgpool"
        if not hasattr(self.config, "obs_window_mode"):
            self.config.obs_window_mode = "adaptive"
    self.kv_cluster = SnapKVCluster(
        window_size=self.config.window_size,
        max_capacity_prompt=self.config.max_capacity_prompt,
        kernel_size=self.config.kernel_size,
        pooling=self.config.pooling,
        obs_window_mode=self.config.obs_window_mode,
    )
