from django.shortcuts import render # type: ignore

def index(request):
    return render(request, 'index.html')

def tests(request):
    return render(request, 'tests.html')

# Create your views here.
