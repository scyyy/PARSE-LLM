import json
import os
import random
from rich import print
from tqdm import tqdm
from functools import partial
from loguru import logger
from .utils import read_jsonl, write_jsonl


def load_prompt(name, prompt_file="./prompts/prompts.json"):
    with open(prompt_file, "r", encoding="utf-8") as f:
        prompts = json.load(f)
    return prompts.get(name, "")


def naive_infer(llm, query):
    prompt = load_prompt("naive_infer")
    response = llm.invoke(prompt + "\n" + query)
    return response.content.strip()


def contrastive_infer(llm, query):
    prompt = load_prompt("contrastive_infer")
    response = llm.invoke(prompt + "\n`" + query + "`")
    return response.content.strip()


def contrasting_infer_v2(llm, query):
    prompt = load_prompt("contrasting_infer_v2")
    response = llm.invoke(prompt + "\n" + query)
    return response.content.strip()


def demonstration_infer(llm, query, demonstrations):
    instruction = load_prompt("demonstration_infer_instruction")
    shots = "\n".join(
        [f"{instruction}\nInput: `{d['query']}`\nOutput: `{d['template']}`" for d in demonstrations]
    )
    example = f"{instruction}\nInput: `{query}`\nOutput:"
    prompt = shots + "\n" + example
    response = llm.invoke(prompt)
    return response.content.strip()


def fixed_demonstration_infer(llm, query, demonstrations, max_shots=5):
    instruction = load_prompt("demonstration_infer_instruction")
    if len(demonstrations) > max_shots:
        demonstrations = random.sample(demonstrations, k=max_shots)
    shots = "\n".join(
        [f"{instruction}\nInput: `{d['query']}`\nOutput: `{d['template']}`" for d in demonstrations]
    )
    example = f"{instruction}\nInput: `{query}`\nOutput:"
    prompt = shots + "\n" + example
    response = llm.invoke(prompt)
    return response.content.strip()


def infer_dataset(llm, data_path, output_path, method="naive", max_shots=5, demo_path=None):
    data = read_jsonl(data_path)
    if demo_path:
        demonstrations = read_jsonl(demo_path)
    else:
        demonstrations = []

    method_fn_map = {
        "naive": naive_infer,
        "contrastive": contrastive_infer,
        "contrastive_v2": contrasting_infer_v2,
        "demonstration": partial(demonstration_infer, demonstrations=demonstrations),
        "fixed_demonstration": partial(fixed_demonstration_infer, demonstrations=demonstrations, max_shots=max_shots)
    }

    method_fn = method_fn_map[method]
    logger.info(f"Using method: {method}, demo_path: {demo_path}, max_shots: {max_shots}")

    new_data = []
    for sample in tqdm(data, desc="Inferring"):
        template = method_fn(llm, sample["query"])
        sample["pred"] = template
        new_data.append(sample)

    write_jsonl(output_path, new_data)
    logger.success(f"Saved results to {output_path}")
