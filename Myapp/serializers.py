from rest_framework import serializers
from django.contrib.auth.models import User
from .models import Dlmodel
from versatileimagefield.serializers import VersatileImageFieldSerializer

class DlmodelSerializer(serializers.ModelSerializer):
    id = serializers.IntegerField(read_only = True)
    owner = serializers.ReadOnlyField(source='owner.username')
    dlimage = VersatileImageFieldSerializer(
        sizes='dlmodel_dlimage'
    )
    class Meta:
        model = Dlmodel
        fields = ['owner', 'id','dlimage']


class UserSerializer(serializers.ModelSerializer):
    dlmodels = serializers.PrimaryKeyRelatedField(many=True, queryset=Dlmodel.objects.all())

    class Meta:
        model = User    
        fields = ['id', 'username', 'dlmodels'] 