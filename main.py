import torch
import tiktoken
from GPT import GPTModel
from LoadModel import load_weights_into_gpt
from GenerateText import generate
from GenerateForever import generateForever
from GPT_CONFIG_124M import GPT_CONFIG_124M
from Converter import text_to_token_ids, token_ids_to_text
tokenizer = tiktoken.get_encoding("gpt2")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device.")
gpt = GPTModel()
gpt.eval()

load_weights_into_gpt(gpt)
gpt.to(device)

token_ids = generateForever(
    model=gpt,
    idx=text_to_token_ids("api=", tokenizer).to(device),
    context_size=GPT_CONFIG_124M["context_length"],
    tokenizer=tokenizer,
    top_k=40,
    eos_id=50256,
    temperature=1.2
)
# while True:
#     text = input("Enter: ")
#     token_ids = generate(
#         model=gpt,
#         idx=text_to_token_ids(text, tokenizer).to(device),
#         max_new_tokens=350,
#         context_size=GPT_CONFIG_124M["context_length"],
#         top_k=40,
#         eos_id=50256,
#         temperature=1.2
#     )

#     output = token_ids_to_text(token_ids, tokenizer)
#     # print("Output text:\n", output )

#     with open("hello.txt", "w") as f:
#         f.write(output)

output = token_ids_to_text(token_ids, tokenizer)
# print("Output text:\n", output )

with open("hello.txt", "w") as f:
    f.write(output)
