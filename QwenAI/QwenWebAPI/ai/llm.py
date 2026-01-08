# ai/llm.py
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig # type: ignore
import torch # type: ignore

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)

SYSTEM = {
  "role": "system",
  "content": (
    "Output Markdown.\n"
    "All equations must be in LaTeX using \\[ ... \\] for display and \\( ... \\) inline.\n"
    "Never use [ ... ] or ( x ) for math.\n"
    "When defining variables, ALWAYS format as a bullet list, one per line:\n"
    "- \\(symbol\\): description\n"
    "Structure your responses for readability:\n"
    "- Break content into sections with short paragraphs.\n"
    "- Prefer bullet lists for definitions and explanations.\n"
    "- Indent sub-points as nested bullet lists.\n"
    "- Place equations on separate lines.\n"
    "- Avoid dense text blocks longer than 3–4 lines.\n"
  )
}

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb,
    device_map="auto"
)

def chat(messages, max_new_tokens=512, max_input_tokens=6000) -> dict[str, any]:
    messages = [SYSTEM] + messages
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
        do_sample=False,   # keep greedy for speed
        use_cache=True,
    )

    out = outputs[0][inputs.input_ids.shape[1]:]
    decoded = tokenizer.decode(out, skip_special_tokens=True).strip()
    decoded = normalize_math(decoded)

    fmt = "markdown"
    return {"format": fmt, "content": decoded}


import re
from typing import Any, Dict, List

MD_HINTS = [
    r"^#{1,6}\s", 
    r"```",       
    r"^\s*[-*+]\s+",  
    r"^\s*\d+\.\s+",  
    r"\$\$[\s\S]*\$\$",
    r"\\\(", r"\\\)",
    r"\\\[", r"\\\]",  
]

def normalize_math(text: str) -> str:
    def repl_block(m: re.Match) -> str:
        inner = m.group(1).strip()
        return f"\\[\n{inner}\n\\]"
    text = re.sub(r"(?ms)^\s*\[\s*(.+?)\s*\]\s*$", repl_block, text)
    def repl_inline(m: re.Match) -> str:
        inner = m.group(1).strip()
        return f"\\[{inner}\\]"
    text = re.sub(
        r"\[\s*([^\[\]\n]*?(?:=|\\[A-Za-z]+|[_^]|\\frac|\\Delta|\\int|\\sum|\d)[^\[\]\n]*?)\s*\]",
        repl_inline,
        text,
    )
    text = re.sub(
        r"\(\s*([A-Za-z]+(?:_[A-Za-z0-9]+)?|\\[A-Za-z]+|[^()\n]*?(?:\\[A-Za-z]+|[_^]|=|\d)[^()\n]*?)\s*\)",
        r"\\(\1\\)",
        text,
    )
    return text


def looks_like_markdown(text: str) -> bool:
    t = text.strip()
    for pat in MD_HINTS:
        if re.search(pat, t, flags=re.MULTILINE):
            return True
    return False



