import os

import torch
import transformers
from dotenv import load_dotenv
from pynvml import nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlInit
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments


def print_gpu_utilization() -> None:
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


def generate(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_length: int = 100,
    sample: bool = True,
) -> list[str]:
    tokens = tokenizer(prompt, return_tensors="pt")
    res = model.generate(
        **tokens.to(model.device),
        max_new_tokens=max_length,
        do_sample=sample,
        num_return_sequences=1,
        temperature=0.1,
        num_beams=1,
        top_p=0.95,
    ).to("cpu")
    return tokenizer.batch_decode(res, skip_special_tokens=True)


def generate_dialogue_summary(dataset, model, tokenizer) -> None:
    index = 10
    prompt = dataset["test"][index]["dialogue"]
    summary = dataset["test"][index]["summary"]

    formatted_prompt = (
        f"Instruct: Summarize the following conversation:\n{prompt}\nOutput:\n"
    )
    res = generate(model, tokenizer, formatted_prompt)
    output = res[0].split("Output:\n")[1]
    dash_line = "-" * 100
    print(dash_line)
    print(f"INPUT PROMPT:\n{formatted_prompt}")
    print(dash_line)
    print(f"BASELINE HUMAN SUMMARY:\n{summary}\n")
    print(dash_line)
    print(f"MODEL GENERATION:\n{output}")


def get_max_length(model: AutoModelForCausalLM) -> int:
    conf = model.config
    max_length = None
    for length_setting in ["n_positions", "max_position_embeddings", "seq_length"]:
        max_length = getattr(conf, length_setting, None)
        if max_length:
            print(f"Found max length: {max_length}")
            break
    if not max_length:
        max_length = 1024
        print(f"Using default max length: {max_length}")
    return max_length


def print_number_of_trainable_model_parameters(model: AutoModelForCausalLM) -> str:
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    return (
        f"trainable model parameters: {trainable_model_params}\n"
        f"all model parameters: {all_model_params}\n"
        f"percentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%"
    )


def initialize_perf_training(
    output_dir: str, tokenizer: AutoTokenizer, train_dataset, eval_dataset, perf_model
) -> transformers.Trainer:
    peft_training_args = TrainingArguments(
        output_dir=output_dir,
        warmup_steps=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        max_steps=1000,
        learning_rate=2e-4,
        optim="paged_adamw_8bit",
        logging_steps=25,
        logging_dir="./logs",
        save_strategy="steps",
        save_steps=25,
        evaluation_strategy="steps",
        eval_steps=25,
        do_eval=True,
        gradient_checkpointing=True,
        report_to="none",
        overwrite_output_dir="True",
        group_by_length=True,
    )

    perf_model.config.use_cache = False

    return transformers.Trainer(
        model=perf_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=peft_training_args,
        data_collator=transformers.DataCollatorForLanguageModeling(
            tokenizer, mlm=False
        ),
    )


def get_compute_dtype() -> torch.DeviceObjType:
    load_dotenv()
    return getattr(torch, os.getenv("COMPUTE_DTYPE", "float16"))
