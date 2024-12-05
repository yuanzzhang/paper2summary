import os
import torch
from transformers import AutoTokenizer, LlamaForCausalLM, pipeline

from config import GlobalConfig
from paper_dataset import load_scientific_papers


global_config = GlobalConfig()
os.environ["CUDA_VISIBLE_DEVICES"] = global_config.device
SYSTEM_MSG = {
    "role": "system",
    "content": "You are a scientific paper summarization assistant that follows a structured approach to generate comprehensive abstracts. You consider the paper's discourse structure including problem definition, methodology, experiments/results, and conclusions to create a cohesive summary."
}

def setup_testing(model_path):
    model = LlamaForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
        use_cache=True,
        low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    summarizer = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    dataset = load_scientific_papers("test[:10]")

    return summarizer, dataset


def test_single_sample(summarizer, dataset, index):
    sample = dataset[index]
    article = sample["article"]
    abstract = sample["abstract"]
    
    input_text = (
        f"Summarize this scientific paper into an abstract that covers key elements in sequence:\n"
        f"1. The problem or research objective\n"
        f"2. The proposed methodology or approach\n"
        f"3. Key experimental results and findings\n"
        f"4. Main conclusions or implications\n\n"
        f"Paper:\n{article}\n\n"
        f"Generate a single paragraph summary that flows naturally between these elements."
    )
    messages = [SYSTEM_MSG, {"role": "user", "content": input_text}]
    
    outputs = summarizer(
        messages,
        max_new_tokens=210,
        min_new_tokens=100,
        num_beams=4,
        temperature=1,
        top_k=50,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3
    )
    summary = outputs[0]["generated_text"][-1]["content"]
    return abstract, summary