from django.urls import reverse
from rest_framework.test import APITestCase
from rest_framework import status
from Myproject.Myapp.views1 import api_root, get_mlmodel, predict, train, thanks

class ViewsTestCase(APITestCase):

    def test_api_root(self):
        url = reverse('api_root')
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('users', response.data)
        self.assertIn('mlmodels', response.data)

    def test_get_mlmodel_get(self):
        url = reverse('get_mlmodel')
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertTemplateUsed(response, 'mlmodel.html')

    def test_get_mlmodel_post(self):
        url = reverse('get_mlmodel')
        data = {'field1': 'value1', 'field2': 'value2'}  # Replace with actual form data
        response = self.client.post(url, data)
        self.assertEqual(response.status_code, status.HTTP_302_FOUND)
        self.assertRedirects(response, '/thanks/')

    def test_predict_get(self):
        url = reverse('predict')
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertTemplateUsed(response, 'predict.html')

    def test_predict_post(self):
        url = reverse('predict')
        data = {'field1': 'value1', 'field2': 'value2'}  # Replace with actual form data
        response = self.client.post(url, data)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertTemplateUsed(response, 'predict.html')

    def test_train_get(self):
        url = reverse('train')
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertTemplateUsed(response, 'train.html')

    def test_train_post(self):
        url = reverse('train')
        data = {'field1': 'value1', 'field2': 'value2'}  # Replace with actual form data
        response = self.client.post(url, data)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertTemplateUsed(response, 'train.html')

    def test_thanks_get(self):
        url = reverse('thanks')
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertTemplateUsed(response, 'mlmodel.html')