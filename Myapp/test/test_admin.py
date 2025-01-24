from django.test import TestCase
from django.contrib.admin.sites import AdminSite
from django.contrib import admin

from Myapp.models import Mlmodel
from django.contrib.auth.models import User

class MlmodelAdmin(admin.ModelAdmin):
    list_display = ('id', 'Time', 'Amount', 'owner')


class MockRequest:
    pass

class MlmodelAdminTestCase(TestCase):
    def setUp(self):
        self.site = AdminSite()
        User.objects.filter(username='testuser').delete()
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
        self.admin = MlmodelAdmin(model=Mlmodel, admin_site=self.site)

    def test_mlmodel_str(self):
        self.assertEqual(str(self.mlmodel), f'Mlmodel {self.mlmodel.id}')

    def test_mlmodel_admin(self):
        request = MockRequest()
        queryset = Mlmodel.objects.all()
        response = self.admin.changelist_view(request)
        self.assertEqual(response.status_code, 200)