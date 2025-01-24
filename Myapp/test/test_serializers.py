from django.test import TestCase
from django.contrib.auth.models import User
from rest_framework.test import APIRequestFactory
from Myapp.serializers import MlmodelSerializer, UserSerializer
from Myapp.models import Mlmodel

class MlmodelSerializerTestCase(TestCase):
    def setUp(self):
        User.objects.filter(username='testuser').delete()
        self.user = User.objects.create_user(username='testuser',password="12345")
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
        self.factory = APIRequestFactory()
        self.request = self.factory.get('/')

    def test_mlmodel_serializer(self):
        serializer = MlmodelSerializer(self.mlmodel, context={'request': self.request})
        data = serializer.data
        self.assertEqual(data['Time'], 1)
        self.assertEqual(data['V1'], 0.1)
        self.assertEqual(data['V2'], 0.2)
        self.assertEqual(data['V3'], 0.3)
        self.assertEqual(data['V4'], 0.4)
        self.assertEqual(data['V5'], 0.5)
        self.assertEqual(data['V6'], 0.6)
        self.assertEqual(data['V7'], 0.7)
        self.assertEqual(data['V8'], 0.8)
        self.assertEqual(data['V9'], 0.9)
        self.assertEqual(data['V10'], 1.0)
        self.assertEqual(data['V11'], 1.1)
        self.assertEqual(data['V12'], 1.2)
        self.assertEqual(data['V13'], 1.3)
        self.assertEqual(data['V14'], 1.4)
        self.assertEqual(data['V15'], 1.5)
        self.assertEqual(data['V16'], 1.6)
        self.assertEqual(data['V17'], 1.7)
        self.assertEqual(data['V18'], 1.8)
        self.assertEqual(data['V19'], 1.9)
        self.assertEqual(data['V20'], 2.0)
        self.assertEqual(data['V21'], 2.1)
        self.assertEqual(data['V22'], 2.2)
        self.assertEqual(data['V23'], 2.3)
        self.assertEqual(data['V24'], 2.4)
        self.assertEqual(data['V25'], 2.5)
        self.assertEqual(data['V26'], 2.6)
        self.assertEqual(data['V27'], 2.7)
        self.assertEqual(data['V28'], 2.8)
        self.assertEqual(data['Amount'], 100)
        self.assertEqual(data['owner'], self.user.username)

class UserSerializerTestCase(TestCase):
    def setUp(self):
        User.objects.filter(username='testuser').delete()

        self.user = User.objects.create_user(username='testuser', password='12345')
        self.factory = APIRequestFactory()
        self.request = self.factory.get('/')

    def test_user_serializer(self):
        serializer = UserSerializer(self.user, context={'request': self.request})
        data = serializer.data
        self.assertEqual(data['username'], 'testuser')
        self.assertEqual(data['mlmodels'], [])