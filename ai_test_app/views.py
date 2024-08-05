from django.shortcuts import render
from django import forms
from transformers import AutoModelForCausalLM, AutoTokenizer

import torch

model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")


class PromptForm(forms.Form):
    prompt = forms.CharField(label='Prompt')


def index(request):
    output = None
    if request.method == 'POST':
        form = PromptForm(request.POST)
        if form.is_valid():
            prompt = form.cleaned_data['prompt']
            input_ids = tokenizer(
                prompt, return_tensors="pt").input_ids
            gen_tokens = model.generate(
                input_ids,
                do_sample=True,
                temperature=0.9,
                max_length=100,
            )
            output = tokenizer.batch_decode(gen_tokens)[0]
    else:
        form = PromptForm()

    return render(request, 'index.html', {'form': form, 'output': output})
