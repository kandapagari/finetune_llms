from functools import partial
from typing import Union

import torch
from datasets import (Dataset, DatasetDict, IterableDataset,
                      IterableDatasetDict, load_dataset)
from transformers import AutoTokenizer


def get_dataset(
    dataset_name: str,
) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]:
    return load_dataset(dataset_name)


def create_prompt_formats(sample: str) -> str:
    """
    Format various fields of the sample ('instruction','output')
    Then concatenate them using two newline characters
    :param sample: Sample dictionary
    """
    INTRO_BLURB = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
    INSTRUCTION_KEY = "### Instruct: Summarize the below conversation."
    RESPONSE_KEY = "### Output:"
    END_KEY = "### End"

    blurb = f"\n{INTRO_BLURB}"
    instruction = f"{INSTRUCTION_KEY}"
    input_context = f"{sample['dialogue']}" if sample["dialogue"] else None
    response = f"{RESPONSE_KEY}\n{sample['summary']}"
    end = f"{END_KEY}"

    parts = [
        part for part in [blurb, instruction, input_context, response, end]
        if part
    ]

    formatted_prompt = "\n\n".join(parts)
    sample["text"] = formatted_prompt

    return sample


def preprocess_dataset(
    tokenizer: AutoTokenizer,
    max_length: int,
    seed: int,
    dataset: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset],
) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]:
    """Format & tokenize it so it is ready for training
    :param tokenizer (AutoTokenizer): Model Tokenizer
    :param max_length (int): Maximum number of tokens to emit from tokenizer
    """

    # Add prompt to each sample
    print("Preprocessing dataset...")
    dataset = dataset.map(create_prompt_formats)  # , batched=True)

    # Apply preprocessing to each batch of the dataset & and remove 'instruction', 'context', 'response', 'category' fields
    _preprocessing_function = partial(preprocess_batch,
                                      max_length=max_length,
                                      tokenizer=tokenizer)
    dataset = dataset.map(
        _preprocessing_function,
        batched=True,
        remove_columns=["id", "topic", "dialogue", "summary"],
    )

    # Filter out samples that have input_ids exceeding max_length
    dataset = dataset.filter(
        lambda sample: len(sample["input_ids"]) < max_length)

    # Shuffle dataset
    dataset = dataset.shuffle(seed=seed)

    return dataset


def preprocess_batch(batch: dict, tokenizer: AutoTokenizer,
                     max_length: int) -> torch.Tensor:
    """
    Tokenizing a batch
    """
    return tokenizer(
        batch["text"],
        max_length=max_length,
        truncation=True,
    )
