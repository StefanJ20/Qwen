from django.urls import path # type: ignore
from django.contrib import admin # type: ignore
from . import views 

urlpatterns = [
    path('', views.index, name='index'),
]