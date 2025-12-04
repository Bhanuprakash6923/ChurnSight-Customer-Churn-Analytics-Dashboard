from django import forms
from .models import Dataset

class BatchPredictionForm(forms.Form):
    file = forms.FileField(label='Upload CSV file for batch prediction')

class DatasetUploadForm(forms.ModelForm):
    class Meta:
        model = Dataset
        fields = ['name', 'file']

class SinglePredictionForm(forms.Form):
    customer_id = forms.CharField(max_length=100)
    tenure = forms.FloatField()
    monthly_charges = forms.FloatField()
    total_charges = forms.FloatField()
    contract_type = forms.ChoiceField(choices=[
        ('Month-to-month', 'Month-to-month'),
        ('One year', 'One year'),
        ('Two year', 'Two year'),

    ])
