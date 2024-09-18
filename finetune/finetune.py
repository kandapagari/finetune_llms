import os
import time

import dotenv
import typer
from huggingface_hub import login
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import set_seed

from finetune.dataset import get_dataset, preprocess_dataset
from finetune.model import get_model
from finetune.utils import (generate_dialogue_summary, get_max_length,
                            initialize_perf_training,
                            print_number_of_trainable_model_parameters)

_ = dotenv.load_dotenv(dotenv.find_dotenv())
login(os.getenv("HF_TOKEN"))


def finetune(
    dataset_name: str = "neil-code/dialogsum-test",
    model_name: str = "microsoft/phi-2",
    seed: int = 42,
    output_dir: str | None = None,
) -> None:
    set_seed(seed)
    dataset = get_dataset(dataset_name)
    original_model, tokenizer = get_model(model_name)

    generate_dialogue_summary(dataset, original_model, tokenizer)

    max_length = get_max_length(original_model)
    print(max_length)

    train_dataset, eval_dataset = (
        preprocess_dataset(tokenizer, max_length, seed, dataset["train"]),
        preprocess_dataset(tokenizer, max_length, seed, dataset["validation"]),
    )

    original_model = prepare_model_for_kbit_training(original_model)

    config = LoraConfig(
        r=32,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "dense"],
        bias="none",
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )

    original_model.gradient_checkpointing_enable()
    print(print_number_of_trainable_model_parameters(original_model))

    perf_model = get_peft_model(original_model, config)
    print(print_number_of_trainable_model_parameters(perf_model))
    print(perf_model)
    output_dir = (f"./peft-dialogue-summary-training-{int(time.time())}"
                  if output_dir is None else output_dir)
    perf_training = initialize_perf_training(output_dir, tokenizer,
                                             train_dataset, eval_dataset,
                                             perf_model)
    perf_training.train()
    print(f"Model saved to {output_dir}")


if __name__ == "__main__":
    typer.run(finetune)
