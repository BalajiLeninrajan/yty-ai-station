from transformers import AutoTokenizer,GPTJForCausalLM
import torch

model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", revision="float16", torch_dtype=torch.float16)
torch.save(model, "./ai_models/gptj.pt")
