# ai/llm.py
from transformers import AutoModelForCausalLM, AutoTokenizer # type: ignore

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

# Load once, at import time
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype="auto",
    device_map="auto",
)

def chat(messages, max_new_tokens=512, max_input_tokens=6000):
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        [text],
        return_tensors="pt",
        truncation=True,
        max_length=max_input_tokens,
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.2,
    )

    # strip prompt tokens
    out = outputs[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(out, skip_special_tokens=True)
