# ai/views.py
import json, re
from django.http import JsonResponse # type: ignore
from django.views.decorators.csrf import csrf_exempt # type: ignore
from .llm import chat

MATH_HEAVY = re.compile(
    r"\b(rref|reduced row echelon|row echelon|gauss|gaussian|elim(inat)?ion|determinant|eigen)\b"
    r"|(\b\d+\s*[x×]\s*\d+\b)"
    r"|\\begin\{bmatrix\}",
    re.I
)
MATH_LIGHT = re.compile(
    r"(\\\[|\\\(|\\frac|\\sqrt|\\sum|\\int|\\sin|\\cos|\\tan|\\theta|\\lambda)"
    r"|[=^_]|(\b\d+\s*[\+\-\*/]\s*\d+\b)",
    re.I
)
LONG_ESSAY_STYLE_PROMPT = re.compile(
    r"\b(write|compose|draft|create).{0,20}\b(essay|article|story|narrative|report|summary|description|explanation|analysis)\b",
    re.I
)

def pick_limit(messages):
    s = next((m.get("content","") for m in reversed(messages) if m.get("role")=="user"), "")
    if MATH_HEAVY.search(s):
        return 2000
    if MATH_LIGHT.search(s):
        return 1200
    if LONG_ESSAY_STYLE_PROMPT.search(s):
        return 1800
    return 800

@csrf_exempt
def chat_api(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST only"}, status=405)

    payload = json.loads(request.body.decode("utf-8"))
    history = payload.get("history", [])
    max_new_tokens = int(payload.get("max_new_tokens", pick_limit(history)))

    response = chat(history, max_new_tokens=max_new_tokens)
    return JsonResponse(response)
