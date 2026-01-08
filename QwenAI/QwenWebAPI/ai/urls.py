# ai/urls.py
from django.urls import path # type: ignore
from .views import chat_api

urlpatterns = [
    path("chat/", chat_api),
]
