from django.contrib.auth.models import User
from rest_framework.test import APIRequestFactory, APITestCase
from .permissions import IsOwnerOrReadOnly

class IsOwnerOrReadOnlyTestCase(APITestCase):
    def setUp(self):
        self.factory = APIRequestFactory()
        self.user = User.objects.create_user(username='user1', password='pass')
        self.other_user = User.objects.create_user(username='user2', password='pass')
        self.permission = IsOwnerOrReadOnly()

    def test_has_object_permission_read(self):
        request = self.factory.get('/')
        request.user = self.user
        obj = type('obj', (object,), {'owner': self.other_user})  # Mock object with owner attribute

        self.assertTrue(self.permission.has_object_permission(request, None, obj))

    def test_has_object_permission_write_owner(self):
        request = self.factory.post('/')
        request.user = self.user
        obj = type('obj', (object,), {'owner': self.user})  # Mock object with owner attribute

        self.assertTrue(self.permission.has_object_permission(request, None, obj))

    def test_has_object_permission_write_not_owner(self):
        request = self.factory.post('/')
        request.user = self.user
        obj = type('obj', (object,), {'owner': self.other_user})  # Mock object with owner attribute

        self.assertFalse(self.permission.has_object_permission(request, None, obj))