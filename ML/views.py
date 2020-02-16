from django.shortcuts import render, redirect
from django.http import HttpResponse

from .forms import InputForm
from .models import show

def index_template(request):
    if request.method == 'POST':
        form = InputForm(request.POST)
        url = request.POST['url']
    else:
        form = InputForm()
        url = ''
    
    if form.is_valid():
        message = 'データ検出に成功しました'
        nb_res, rf_res, be_res = show(url)
    else:
        message = 'データ検出に失敗しました'
        nb_res, rf_res, be_res = '', '', ''
    
    return render(
            request, 
            'index.html', 
            {'form': form,
             'naive_bayes_result': nb_res,
             'randomforest_result': rf_res,
             'bert_result': be_res
            }
            )


