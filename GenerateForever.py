import torch
from Converter import token_ids_to_text
def generateForever(model, idx, context_size, tokenizer = None, temperature=0.0, top_k=None, eos_id=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # For-loop is the same as before: Get logits, and only focus on last time step
    while True:
        idx_cond = idx[:, -context_size:]
        idx_cond = idx_cond.to(device)
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        # New: Filter logits with top_k sampling
        if top_k is not None:
            # Keep only top_k values
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(device), logits)

        # New: Apply temperature scaling
        if temperature > 0.0:
            logits = logits / temperature

            # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)

            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

        # Otherwise same as before: get idx of the vocab entry with the highest logits value
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

        if idx_next == eos_id:  # Stop generating early if end-of-sequence token is encountered and eos_id is specified
            break
        print(token_ids_to_text(idx_next, tokenizer), end="")

        # Same as before: append sampled index to the running sequence
        idx = torch.cat((idx.to(device), idx_next), dim=1)  # (batch_size, num_tokens+1)
        

    return idx