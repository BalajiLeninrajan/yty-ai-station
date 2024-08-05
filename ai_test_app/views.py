from django.shortcuts import render
from django import forms
from transformers import AutoTokenizer, pipeline

import torch

model = torch.load("../ai_models/gptj.pt")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
gen = pipeline("text-generation",model=model,tokenizer=tokenizer,device=0)


class PromptForm(forms.Form):
    prompt = forms.CharField(label='Prompt')


def index(request):
    output = None
    if request.method == 'POST':
        form = PromptForm(request.POST)
        if form.is_valid():
            prompt = form.cleaned_data['prompt']
            output = gen(prompt)
    else:
        form = PromptForm()

    return render(request, 'index.html', {'form': form, 'output': output})
