# Qwen — Local AI Web Interface (Django + Qwen2.5)

A local AI web application built with **Django** and **Qwen2.5-7B-Instruct**, running fully **on-device** with **no API keys** and no cloud dependencies.

This project is my foundation for **tool-using agents**, structured outputs, and a custom AI interface with full control over inference, context, and execution.

---

## Features

- Local LLM inference using **Qwen2.5-7B-Instruct**
- No API keys required (Fully privately hosted)
- GPU-accelerated inference (tested on my RTX 3060 laptop GPU)
- Django-based API backend
- Custom HTML/CSS frontend
- Hugging Face model caching

---

## Tech Stack

### Backend
- Python 3.13
- Django
- PyTorch
- Hugging Face Transformers
- BitsAndBytes (4-bit quantization)

### Model
- `Qwen/Qwen2.5-7B-Instruct`

### Frontend
- HTML
- CSS (custom, no framework)
- JavaScript (minimal, expandable)

---

## Project Structure

Qwen/
├── manage.py
├── QwenWebAPI/            # Django project config
│   ├── settings.py
│   ├── urls.py
│   └── ...
├── polls/                 # Web Front
│   ├──templates/
│   │  └── index.html             
├── ai/                    # AI / LLM app
│   ├── llm.py             # Model + tokenizer
│   ├── views.py           # API endpoint
│   ├── urls.py
│   └── ...
└── README.md              # Read Me (THIS OBVIOUSLY DUH)


## Dependencies

`pip install torch transformers accelerate bitsandbytes django`

## Author

`Built by: StefanJ20`