from django.db import models
from versatileimagefield.fields import VersatileImageField, PPOIField
# Create your models here.
class Dlmodel(models.Model):
    created = models.DateTimeField(auto_now_add=True)
    dlimage = VersatileImageField(
        'Dlimage',
        upload_to = 'dlimage/',
        ppoi_field = 'dlimage_ppoi'
    )
    dlimage_ppoi =  PPOIField()
    owner = models.ForeignKey('auth.User', related_name='dlmodels', on_delete=models.CASCADE)
    

    class Meta:
        ordering = ['created']
        verbose_name ="Dlmodel"
        verbose_name_plural = "dlmodels"
   
    def save(self, *args, **kwargs):
        dlimage = kwargs.get('image',0)
        
        super(Dlmodel,self).save(*args, **kwargs)
    