from django.test import TestCase
from Myapp.apps import MyappConfig

class MyappConfigTestCase(TestCase):
    def test_app_config(self):
        self.assertEqual(MyappConfig.name, 'Myapp')
        self.assertEqual(MyappConfig.default_auto_field, 'django.db.models.BigAutoField')