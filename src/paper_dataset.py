from datasets import load_dataset, Features, Value


def load_scientific_papers(split):
    return load_dataset(
        "armanc/scientific_papers",
        name="arxiv",
        trust_remote_code=True,
        split=split,
        features=Features({
            'article': Value('string'),
            'abstract': Value('string')
        })
    )


def get_paper_dataset(config, tokenizer, split):
    dataset = load_scientific_papers(split)

    prompt = (
        f"Summarize this scientific paper:\n{{article}}\n---\nSummary:\n"
    )

    def apply_prompt_template(sample):
        return {
            "prompt": prompt.format(article=sample["article"]),
            "summary": sample["abstract"],
        }

    dataset = dataset.map(
        apply_prompt_template,
        remove_columns=list(dataset.features),
        num_proc=8,
        desc="Applying prompt template"
    )

    def tokenize_add_label(sample):
        prompt = tokenizer.encode(
            tokenizer.bos_token + sample["prompt"],
            add_special_tokens=False,
            padding=config.padding,
            truncation=config.truncation,
            max_length=config.context_length
        )
        summary = tokenizer.encode(
            sample["summary"] + tokenizer.eos_token,
            add_special_tokens=False,
            padding=config.padding,
            truncation=config.truncation,
            max_length=config.summary_length
        )

        sample = {
            "input_ids": prompt + summary,
            "attention_mask": [1] * (len(prompt) + len(summary)),
            "labels": [-100] * len(prompt) + summary,
        }

        return sample

    dataset = dataset.map(
        tokenize_add_label, 
        remove_columns=list(dataset.features),
        num_proc=8,
        desc="Tokenizing dataset"
    )

    return dataset