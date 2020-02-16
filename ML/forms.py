from django import forms

class InputForm(forms.Form):
    url = forms.URLField(
            label='URL',
            max_length=100,
            required=True,
            )

    
