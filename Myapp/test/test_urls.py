from django.urls import reverse, resolve
from rest_framework.test import APITestCase
from Myproject.Myapp.views1 import api_root, get_mlmodel, predict, train, thanks

class UrlsTestCase(APITestCase):
   def test_api_root_url(self):
      url = reverse('api_root')
      self.assertEqual(resolve(url).func, api_root)

   def test_get_mlmodel_url(self):
      url = reverse('get_mlmodel')
      self.assertEqual(resolve(url).func, get_mlmodel)

   def test_predict_url(self):
      url = reverse('predict')
      self.assertEqual(resolve(url).func, predict)

   def test_predict_with_id_url(self):
      url = reverse('predict', kwargs={'id': 1})
      self.assertEqual(resolve(url).func, predict)

   def test_train_url(self):
      url = reverse('train')
      self.assertEqual(resolve(url).func, train)

   def test_thanks_url(self):
      url = reverse('thanks')
      self.assertEqual(resolve(url).func, thanks)

   def test_swagger_ui_url(self):
      url = reverse('schema-swagger-ui')
      self.assertEqual(resolve(url).func.__name__, 'SwaggerUIView')

   def test_redoc_url(self):
      url = reverse('schema-redoc')
      self.assertEqual(resolve(url).func.__name__, 'ReDocView')