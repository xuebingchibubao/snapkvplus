
import torch
import time
import torch.nn.functional as F
import torch.nn as nn
import math

# perform qk calculation and get indices
# this version will not update in inference mode

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

class SnapKVCluster():
    def __init__(self, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool'):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling

    def reset(self, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool'):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling

    def get_last_sentence_length(self, input_ids, tokenizer):
        sentence_end_tokens = set()
        
        for punct in ['.', '!', '?', '。', '！', '？', '；', ';', '\n']:
            token_id = tokenizer.convert_tokens_to_ids(punct)
            if token_id != tokenizer.unk_token_id:
                sentence_end_tokens.add(token_id)
        
        if tokenizer.eos_token_id is not None:
            sentence_end_tokens.add(tokenizer.eos_token_id)
        
        for i in range(len(input_ids) - 1, -1, -1):
            if input_ids[i] in sentence_end_tokens:
                return len(input_ids) - i - 1
        
        return len(input_ids)

    def update_kv(self, key_states, query_states, value_states, attention_mask, num_key_value_groups, self_attn=None):
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape
        if q_len < self.max_capacity_prompt:
            return key_states, value_states
        
        window_sizes = torch.full((bsz,), self.window_size, dtype=torch.long, device=key_states.device)
        
        input_ids = None
        tokenizer = None
        if self_attn is not None:
            if hasattr(self_attn, 'snapkv_input_ids'):
                input_ids = self_attn.snapkv_input_ids
            if hasattr(self_attn, 'snapkv_tokenizer'):
                tokenizer = self_attn.snapkv_tokenizer
        
        if input_ids is not None and tokenizer is not None:
            for batch_idx in range(bsz):
                last_sentence_length = self.get_last_sentence_length(input_ids[batch_idx], tokenizer)
                window_sizes[batch_idx] = min(self.window_size, last_sentence_length)
        
        key_states_list = []
        value_states_list = []
        
        for batch_idx in range(bsz):
            window_size = window_sizes[batch_idx].item()
            
            key_states_batch = key_states[batch_idx:batch_idx+1]
            query_states_batch = query_states[batch_idx:batch_idx+1]
            value_states_batch = value_states[batch_idx:batch_idx+1]
            
            attn_weights = torch.matmul(query_states_batch[..., -window_size:, :], key_states_batch.transpose(2, 3)) / math.sqrt(head_dim)
            mask = torch.full((window_size, window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
            mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
            mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
            mask = mask.to(attn_weights.device)
            attention_mask_batch = mask[None, None, :, :]

            attn_weights[:, :, -window_size:, -window_size:] += attention_mask_batch

            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights_sum = attn_weights[:, :, -window_size:, : -window_size].sum(dim = -2)
            if self.pooling == 'avgpool':
                attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            elif self.pooling == 'maxpool':
                attn_cache = F.max_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            else:
                raise ValueError('Pooling method not supported')
            indices = attn_cache.topk(self.max_capacity_prompt - window_size, dim=-1).indices
            indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
            k_past_compress = key_states_batch[:, :, :-window_size, :].gather(dim = 2, index = indices)
            v_past_compress = value_states_batch[:, :, :-window_size, :].gather(dim = 2, index = indices)
            k_cur = key_states_batch[:, :, -window_size:, :]
            v_cur = value_states_batch[:, :, -window_size:, :]
            key_states_batch = torch.cat([k_past_compress, k_cur], dim = 2)
            value_states_batch = torch.cat([v_past_compress, v_cur], dim = 2)
            
            key_states_list.append(key_states_batch)
            value_states_list.append(value_states_batch)
        
        key_states = torch.cat(key_states_list, dim=0)
        value_states = torch.cat(value_states_list, dim=0)
        
        return key_states, value_states

def init_snapkv(self):
    if not hasattr(self, "kv_cluster"):
        if not hasattr(self.config, 'window_size'):
            self.config.window_size = 32
        if not hasattr(self.config, 'max_capacity_prompt'):
            self.config.max_capacity_prompt = 2048
        if not hasattr(self.config, 'kernel_size'):
            self.config.kernel_size = 5
        if not hasattr(self.config, 'pooling'):
            self.config.pooling = 'avgpool'
    self.kv_cluster = SnapKVCluster( 
        window_size = self.config.window_size, 
        max_capacity_prompt = self.config.max_capacity_prompt, 
        kernel_size = self.config.kernel_size,
        pooling = self.config.pooling
        )
    if not hasattr(self, 'snapkv_tokenizer'):
        self.snapkv_tokenizer = None
    if not hasattr(self, 'snapkv_input_ids'):
        self.snapkv_input_ids = None