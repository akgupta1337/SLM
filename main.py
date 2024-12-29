import torch
import tiktoken
from GPT import GPTModel
from LoadModel import load_weights_into_gpt
from GenerateText import generate
from GPT_CONFIG_124M import GPT_CONFIG_124M
tokenizer = tiktoken.get_encoding("gpt2")

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) # add batch dimension
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0) # remove batch dimension
    return tokenizer.decode(flat.tolist())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device.")
gpt = GPTModel()
gpt.eval();
# total_params = sum(p.numel() for p in gpt.parameters())
# print(f"Total number of parameters: {total_params:,}")
load_weights_into_gpt(gpt)
gpt.to(device);

token_ids = generate(
    model=gpt,
    idx=text_to_token_ids("the world is", tokenizer).to(device),
    max_new_tokens=25,
    context_size=GPT_CONFIG_124M["context_length"],
    top_k=50,
    temperature=1.5
)

print("Output text:\n", token_ids_to_text(token_ids, tokenizer))
