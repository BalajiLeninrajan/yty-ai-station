from django.shortcuts import render
from django import forms
from transformers import GPTJForCausalLM, AutoTokenizer

import torch

device = "cpu"
model = GPTJForCausalLM.from_pretrained(
    "EleutherAI/gpt-j-6B", torch_dtype=torch.float16).to(device)
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")


class PromptForm():
    prompt = forms.CharField(label='Prompt')


def index(request):
    output = None
    if request.method == 'POST':
        form = PromptForm(request.POST)
        if form.is_valid():
            prompt = form.cleaned_data['prompt']
            input_ids = tokenizer(
                prompt, return_tensors="pt").input_ids.to(device)
            gen_tokens = model.generate(
                input_ids,
                do_sample=True,
                temperature=0.9,
                max_length=100,
            )
            output = tokenizer.batch_decode(gen_tokens)[0]
    else:
        form = PromptForm()

    render(request, 'index.html', {'form': form, 'output': output})
