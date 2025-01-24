from django.forms import ModelForm 
from .models import Dlmodel

class DlmodelForm(ModelForm):
    
    class Meta:
        model = Dlmodel
        fields = ['id','dlimage','owner']
       