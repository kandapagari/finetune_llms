from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)

from finetune.utils import get_compute_dtype


def get_model(model_name: str) -> tuple[AutoModelForCausalLM, AutoTokenizer]:

    compute_dtype = get_compute_dtype()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=False,
    )
    device_map = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        quantization_config=bnb_config,
        trust_remote_code=True,
        token=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="left",
        add_eos_token=True,
        add_bos_token=True,
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer
