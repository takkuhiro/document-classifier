#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
入力フォーム
"""
from django import forms


class InputForm(forms.Form):
    """
    入力フォーム
    """
    url = forms.URLField(
            label='URL',
            max_length=100,
            required=True,
            )    
