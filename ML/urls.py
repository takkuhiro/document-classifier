#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
URLの紐付け
"""
from django.conf.urls import url 
from . import views


urlpatterns = [
    url(r'^$', views.index_template, name='index_template'),
] 
