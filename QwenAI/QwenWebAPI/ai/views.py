# ai/views.py
import json
from django.http import JsonResponse # type: ignore
from django.views.decorators.csrf import csrf_exempt # type: ignore
from .llm import chat

@csrf_exempt
def chat_api(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST only"}, status=405)

    payload = json.loads(request.body.decode("utf-8"))
    messages = payload.get("messages", [])
    max_new_tokens = int(payload.get("max_new_tokens", 512))

    response = chat(messages, max_new_tokens=max_new_tokens)
    return JsonResponse(response)
