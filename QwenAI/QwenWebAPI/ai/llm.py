# ai/llm.py
from ast import pattern
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig # type: ignore
import torch # type: ignore
from ai.scrape import scrape_url # type: ignore
import re
from typing import Any, Dict, List

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)

SYSTEM = {
  "role": "system",
  "content": (
    "Output Markdown.\n"
    "Be complete: do not stop early. If solving a math problem, show the full work and the final result.\n"
    "If you introduce \\(A\\), compute what is asked (eigenvalues, determinant, etc.) instead of stopping.\n"
    "\n"
    "Math formatting:\n"
    "- Inline: \\( ... \\)\n"
    "- Display: \\[ ... \\]\n"
    "- Matrices: \\begin{bmatrix} ... \\end{bmatrix}\n"
    "\n"
    "Restrictions:\n"
    "- No other LaTeX environments.\n"
    "- No unicode math symbols; use TeX commands.\n"
    "\n"
    "Name:\n"
    " - You go by 'Martin'.\n"
    " - Never start your responses noting who you are.\n"
  )
}

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb,
    device_map={"": 0},
)

URL_RE = re.compile(r"https?://\S+")

def extract_first_url(s: str) -> str | None:
    m = URL_RE.search(s)
    return m.group(0) if m else None

def is_unusable(text: str | None) -> bool:
    if not text:
        return True
    t = text.lower()
    if len(text) < 800:  # tune
        return True
    blocked_markers = [
        "enable javascript",
        "something went wrong",
        "log in",
        "sign up",
        "cookies",
        "consent",
        "you have been blocked",
    ]
    return any(m in t for m in blocked_markers)

def chat(messages, max_new_tokens=512, max_input_tokens=6000) -> dict[str, any]:
    messages = [SYSTEM] + messages
    url = None
    page_text = None
    last = messages[-1].get("content", "")
    url = extract_first_url(last)

    if url:
        page_text = scrape_url(url)[:10000]
        if is_unusable(page_text):
            page_text = None

        scrape = (
            "The following is reference context from a webpage.\n"
            "If CONTENT is empty or unavailable, you MUST say you could not fetch the page.\n"
            "If the user asks for 'word for word', 'exact text', or 'quote', "
            "ONLY output verbatim text from CONTENT.\n"
            "If CONTENT does not contain the requested text, say so and ask the user to paste it.\n\n"
            f"URL: {url}\n\n"
            f"CONTENT:\n{page_text or '[NO USABLE CONTENT RETRIEVED]'}"
        )

        messages.insert(
                -1,
                {
                    "role": "system",
                    "content": scrape,
                },
        )
    
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
        temperature=None,
        top_p=None,
        repetition_penalty=1.05, 
        no_repeat_ngram_size=0, 
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )

    out = outputs[0][inputs.input_ids.shape[1]:]
    decoded = tokenizer.decode(out, skip_special_tokens=True).strip()
    decoded = normalize_math(decoded)
    print(decoded[decoded.find("\\left")-10:decoded.find("\\left")+10])
    print(decoded[decoded.find("\\\\left")-10:decoded.find("\\\\left")+10])

    fmt = "markdown"
    return {"format": fmt, "content": decoded}


CODE_FENCE = re.compile(r"(```[\s\S]*?```)", re.MULTILINE)
DISPLAY_MATH = re.compile(r"(\\\[[\s\S]*?\\\])")
INLINE_MATH = re.compile(r"(\\\([\s\S]*?\\\))")

def split_keep(pattern, s):
    parts = pattern.split(s)
    return [p for p in parts if p]

def normalize_matrix_envs(s: str) -> str:
    s = re.sub(r"\\begin\{[A-Za-z]*matri[a-z]*\}", r"\\begin{bmatrix}", s)
    s = re.sub(r"\\end\{[A-Za-z]*matri[a-z]*\}", r"\\end{bmatrix}", s)
    s = re.sub(r"\\begin\{bmatri[a-z]*\}", r"\\begin{bmatrix}", s)
    s = re.sub(r"\\end\{bmatri[a-z]*\}", r"\\end{bmatrix}", s)
    return s

def fix_unicode_math(s: str) -> str:
    UNICODE_TEX_MAP = {
        "Δ": r"\Delta",
        "μ": r"\mu",
        "π": r"\pi",
        "→": r"\rightarrow",
        "←": r"\leftarrow",
        "≤": r"\le",
        "≥": r"\ge",
    }
    s = re.sub(r"\\\s*λ", r"\\lambda", s)
    s = re.sub(r"\\\s*θ", r"\\theta", s)
    s = re.sub(r"\\\s*σ", r"\\sigma", s)
    s = re.sub(r"\\\s*δ", r"\\delta", s)
    s = re.sub(r"\\\s+\\lambda", r"\\lambda", s)
    s = s.replace("λ", r"\lambda")
    for ch, tex in UNICODE_TEX_MAP.items():
        s = s.replace(ch, tex)
    return s

def normalize_math(text: str) -> str:
    chunks = split_keep(CODE_FENCE, text)
    out = []

    for chunk in chunks:
        # 1) Never touch code blocks
        if chunk.startswith("```"):
            out.append(chunk)
            continue

        # 2) Split display math
        parts = split_keep(DISPLAY_MATH, chunk)
        for part in parts:
            if part.startswith(r"\[") and part.endswith(r"\]"):
                # math block — minimal safe fixes only
                part = fix_unicode_math(part)
                part = normalize_matrix_envs(part)
                out.append(part)
            else:
                # 3) Split inline math inside text
                sub = split_keep(INLINE_MATH, part)
                for s in sub:
                    if s.startswith(r"\(") and s.endswith(r"\)"):
                        s = fix_unicode_math(s)
                        s = normalize_matrix_envs(s)
                        out.append(s)
                    else:
                        # plain text cleanup only
                        s = re.sub(r"\\{2,}(?=[A-Za-z])", r"\\", s)
                        out.append(s)

    return "".join(out)
