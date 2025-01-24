from django.test import TestCase
from django.contrib.auth.models import User
from Myapp.models import Mlmodel

class MlmodelTestCase(TestCase):
    def setUp(self):
        
        self.user = User.objects.create_user(username='testuser', password='12345')
        self.mlmodel = Mlmodel.objects.create(
            Time=1,
            V1=0.1,
            V2=0.2,
            V3=0.3,
            V4=0.4,
            V5=0.5,
            V6=0.6,
            V7=0.7,
            V8=0.8,
            V9=0.9,
            V10=1.0,
            V11=1.1,
            V12=1.2,
            V13=1.3,
            V14=1.4,
            V15=1.5,
            V16=1.6,
            V17=1.7,
            V18=1.8,
            V19=1.9,
            V20=2.0,
            V21=2.1,
            V22=2.2,
            V23=2.3,
            V24=2.4,
            V25=2.5,
            V26=2.6,
            V27=2.7,
            V28=2.8,
            Amount=100,
            owner=self.user
        )

    def test_mlmodel_creation(self):
        self.assertEqual(self.mlmodel.Time, 1)
        self.assertEqual(self.mlmodel.V1, 0.1)
        self.assertEqual(self.mlmodel.V2, 0.2)
        self.assertEqual(self.mlmodel.V3, 0.3)
        self.assertEqual(self.mlmodel.V4, 0.4)
        self.assertEqual(self.mlmodel.V5, 0.5)
        self.assertEqual(self.mlmodel.V6, 0.6)
        self.assertEqual(self.mlmodel.V7, 0.7)
        self.assertEqual(self.mlmodel.V8, 0.8)
        self.assertEqual(self.mlmodel.V9, 0.9)
        self.assertEqual(self.mlmodel.V10, 1.0)
        self.assertEqual(self.mlmodel.V11, 1.1)
        self.assertEqual(self.mlmodel.V12, 1.2)
        self.assertEqual(self.mlmodel.V13, 1.3)
        self.assertEqual(self.mlmodel.V14, 1.4)
        self.assertEqual(self.mlmodel.V15, 1.5)
        self.assertEqual(self.mlmodel.V16, 1.6)
        self.assertEqual(self.mlmodel.V17, 1.7)
        self.assertEqual(self.mlmodel.V18, 1.8)
        self.assertEqual(self.mlmodel.V19, 1.9)
        self.assertEqual(self.mlmodel.V20, 2.0)
        self.assertEqual(self.mlmodel.V21, 2.1)
        self.assertEqual(self.mlmodel.V22, 2.2)
        self.assertEqual(self.mlmodel.V23, 2.3)
        self.assertEqual(self.mlmodel.V24, 2.4)
        self.assertEqual(self.mlmodel.V25, 2.5)
        self.assertEqual(self.mlmodel.V26, 2.6)
        self.assertEqual(self.mlmodel.V27, 2.7)
        self.assertEqual(self.mlmodel.V28, 2.8)
        self.assertEqual(self.mlmodel.Amount, 100)
        self.assertEqual(self.mlmodel.owner, self.user)
        self.assertIsNotNone(self.mlmodel.created)
        

    def test_mlmodel_str(self):
        self.assertEqual(str(self.mlmodel), f'{self.user}{self.mlmodel.created}')