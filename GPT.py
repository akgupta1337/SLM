from SLM.TransformerBlock import TransformerBlock
from SLM.LayerNormalisation import LayerNorm
import torch
import torch.nn as nn

class GPTModel(nn.Module):
    def getcfg(self, size):
        BASE_CONFIG = {
            "vocab_size": 50257,     # Vocabulary size
            "context_length": 1024,  # Context length
            "drop_rate": 0.0,        # Dropout rate
            "qkv_bias": True         # Query-key-value bias
        }
        model_configs = {
            "124M": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
            "355M": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
            "774M": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
            "1558M": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
        }
        BASE_CONFIG.update(model_configs[size])
        
        return BASE_CONFIG

    def __init__(self, size):
        self.size = size
        super().__init__()

        self.cfg = self.getcfg(size)
        
        self.tok_emb = nn.Embedding(self.cfg["vocab_size"], self.cfg["emb_dim"])
        self.pos_emb = nn.Embedding(self.cfg["context_length"], self.cfg["emb_dim"])
        self.drop_emb = nn.Dropout(self.cfg["drop_rate"])
        
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(self.cfg) for _ in range(self.cfg["n_layers"])])
        
        self.final_norm = LayerNorm(self.cfg["emb_dim"])
        self.out_head = nn.Linear(
            self.cfg["emb_dim"], self.cfg["vocab_size"], bias=False
        )

    
    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
    