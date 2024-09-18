import evaluate
import numpy as np
import pandas as pd
import typer
from peft import PeftModel
from tqdm import tqdm
from transformers import set_seed

from finetune.dataset import get_dataset
from finetune.model import get_model
from finetune.utils import generate, generate_dialogue_summary, get_compute_dtype


def eval(dataset_name: str = "neil-code/dialogsum-test", seed: int = 42) -> None:
    set_seed(seed)
    base_model, tokenizer = get_model("microsoft/phi-2")

    dataset = get_dataset(dataset_name)

    dialogues = dataset["test"][:10]["dialogue"]
    human_baseline_summaries = dataset["test"][:10]["summary"]

    ft_model = PeftModel.from_pretrained(
        base_model,
        "peft-dialogue-summary-training/checkpoint-1000",
        torch_dtype=get_compute_dtype(),
        is_trainable=False,
    )

    generate_dialogue_summary(dataset, base_model, tokenizer)
    generate_dialogue_summary(dataset, ft_model, tokenizer)

    original_model_summaries = []
    peft_model_summaries = []

    for dialogue in tqdm(dialogues):
        prompt = (
            f"Instruct: Summarize the following conversation.\n{dialogue}\nOutput:\n"
        )

        original_model_res = generate(
            base_model,
            tokenizer,
            prompt,
        )
        original_model_text_output = original_model_res[0].split("Output:\n")[1]

        peft_model_res = generate(
            ft_model,
            tokenizer,
            prompt,
        )
        peft_model_output = peft_model_res[0].split("Output:\n")[1]
        peft_model_text_output, success, result = peft_model_output.partition("###")

        original_model_summaries.append(original_model_text_output)
        peft_model_summaries.append(peft_model_text_output)

    zipped_summaries = list(
        zip(human_baseline_summaries, original_model_summaries, peft_model_summaries)
    )

    df = pd.DataFrame(
        zipped_summaries,
        columns=[
            "human_baseline_summaries",
            "original_model_summaries",
            "peft_model_summaries",
        ],
    )

    print(df)

    rouge = evaluate.load("rouge")

    original_model_results = rouge.compute(
        predictions=original_model_summaries,
        references=human_baseline_summaries[: len(original_model_summaries)],
        use_aggregator=True,
        use_stemmer=True,
    )

    peft_model_results = rouge.compute(
        predictions=peft_model_summaries,
        references=human_baseline_summaries[: len(peft_model_summaries)],
        use_aggregator=True,
        use_stemmer=True,
    )

    print("ORIGINAL MODEL:")
    print(original_model_results)
    print("PEFT MODEL:")
    print(peft_model_results)
    print("Absolute percentage improvement of PEFT MODEL over ORIGINAL MODEL")

    improvement = np.array(list(peft_model_results.values())) - np.array(
        list(original_model_results.values())
    )
    for key, value in zip(peft_model_results.keys(), improvement):
        print(f"{key}: {value*100:.2f}%")


if __name__ == "__main__":
    typer.run(eval)
