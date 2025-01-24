from django.test import TestCase
from django.contrib.auth.models import User
from Myapp.forms import MlmodelForm
from Myapp.models import Mlmodel

class MlmodelFormTestCase(TestCase):
    def setUp(self):
        User.objects.filter(username='testuser').delete()
        self.user = User.objects.create_user(username='testuser', password='12345')
        self.mlmodel_data = {
            'Time': 1,
            'V1': 0.1,
            'V2': 0.2,
            'V3': 0.3,
            'V4': 0.4,
            'V5': 0.5,
            'V6': 0.6,
            'V7': 0.7,
            'V8': 0.8,
            'V9': 0.9,
            'V10': 1.0,
            'V11': 1.1,
            'V12': 1.2,
            'V13': 1.3,
            'V14': 1.4,
            'V15': 1.5,
            'V16': 1.6,
            'V17': 1.7,
            'V18': 1.8,
            'V19': 1.9,
            'V20': 2.0,
            'V21': 2.1,
            'V22': 2.2,
            'V23': 2.3,
            'V24': 2.4,
            'V25': 2.5,
            'V26': 2.6,
            'V27': 2.7,
            'V28': 2.8,
            'Amount': 100,
            'owner': self.user.id
        }

    def test_valid_mlmodel_form(self):
        form = MlmodelForm(data=self.mlmodel_data)
        self.assertTrue(form.is_valid())

    def test_invalid_mlmodel_form(self):
        invalid_data = self.mlmodel_data.copy()
        invalid_data['Time'] = ''  # Making Time field invalid
        form = MlmodelForm(data=invalid_data)
        self.assertFalse(form.is_valid())
        self.assertIn('Time', form.errors)