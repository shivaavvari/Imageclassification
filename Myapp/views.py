
"""
This module contains views for the Django application, including API endpoints and viewsets for handling Dlmodel and User objects.
It also includes functions for training a machine learning model, making predictions, and rendering forms.
Classes:
    DlmodelViewSet: A viewset for viewing and editing Dlmodel instances.
    UserViewSet: A viewset for viewing and editing User instances.
Functions:
    api_root(request, format=None): API root endpoint providing links to user and dlmodel lists.
    get_dlmodel(request, format=None): Handles GET and POST requests for the Mlmodel form.
    predict(request, format=None): Handles POST requests to make predictions using the latest Mlmodel instance.
    Train(request): Handles POST requests to train a machine learning model using a CSV file.
    thanks(request, format=None): Renders a thank you page after data is saved to the model.
"""
from rest_framework.response import Response
from .permissions import IsOwnerOrReadOnly
from django.shortcuts import render
from rest_framework import viewsets
from django.contrib.auth.models import User
from rest_framework import permissions
from .serializers import DlmodelSerializer ,UserSerializer
from .models import Dlmodel 
from rest_framework.decorators import api_view ,permission_classes
from rest_framework.reverse import reverse
from rest_framework import renderers
from rest_framework.decorators import action
from rest_framework.renderers import StaticHTMLRenderer
from django.http import HttpResponseRedirect, HttpResponse
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from .forms import DlmodelForm
from rest_framework.permissions import IsAuthenticated
from PIL import Image
import os
from django.conf import settings
import tensorflow.keras as keras
from keras import datasets, layers, models
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input, decode_predictions
from keras.layers import Input
import matplotlib.pyplot as plt
import tensorflow as tf
# Create  your views here.

@api_view(['GET', 'POST','PUT'])
@permission_classes(( ))
def api_root(request, format=None):
    
    try:
         return Response({
        'users':    reverse('user-list', request=request, format=format),
        'dlmodels': reverse('dlmodel-list', request=request, format=format),
          })      
    except Exception as e:
        return Response({"error": str(e)}, status=500)
    




class DlmodelViewSet(viewsets.ModelViewSet):
   
    queryset = Dlmodel.objects.all()
    serializer_class = DlmodelSerializer
    permission_classes = [permissions.IsAuthenticated,
                          IsOwnerOrReadOnly]  
    
    @action(detail=True, methods=['post','get'],renderer_classes=[renderers.StaticHTMLRenderer])  
    def perform_predict(self, request ,pk,**kwargs):
        try:
           
            dlmodel = Dlmodel.objects.get(id=pk)
            serializer = DlmodelSerializer(dlmodel, context={'request': self.request})
 
            if request.method=="POST":
                context = {

               'users': reverse('user-list', request=request, format=None),
                'dlmodels': reverse('dlmodel-list', request=request, format=None),

                }              

                return render(self.request, 'predict.html', context=context)
            else:
                context = {
 
                'users': reverse('user-list', request=request, format=None),
                'dlmodels': reverse('dlmodel-list', request=request, format=None),

                }              



                return render(self.request, 'predict.html', context=context)
        except Exception as e:
                        print(f"Error during prediction:(e)")
        except FileNotFoundError:
                print("Model File not found .")
        
            
        
        except Exception as e:
            return Response({"error": str(e)}, status=500)
    
    
    
    def perform_create(self, serializer ):
        try:
            
            serializer.save(owner=self.request.user)
            context =  {
                        "data": serializer.data,
                        'users': reverse('user-list', request=self.request, format=None),
                        'dlmodels': reverse('dlmodel-list', request=self.request, format=None),
       
              }              
            return render(self.request, 'predict.html', context=context)
        except Exception as e:
            return Response({"error": str(e)}, status=500)

    @action(detail=False, methods=['post','get'],renderer_classes=[renderers.StaticHTMLRenderer])  
    def navigate_form(self, request ,**kwargs):
        try:

                return HttpResponseRedirect('/thanks')
        except Exception as e:
            return Response({"error": str(e)}, status=500)

class UserViewSet(viewsets.ModelViewSet):
    queryset = User.objects.all()
    serializer_class = UserSerializer




@api_view(['GET', 'POST'])
@permission_classes(())
def get_mlmodel(request, format=None):
    if request.method == 'POST':
        form = DlmodelForm(request.POST,request.FILES)
        try:  
            if form.is_valid():
                form.save()

        except Exception as e:
           return Response({"error": str(e)}, status=500)

        return HttpResponseRedirect('/thanks/')
         
      
    else:
        form = DlmodelForm()
        queryset = Dlmodel.objects.all()
        context = {
            "qs": queryset,
            'users': reverse('user-list', request=request, format=format),
            'dlmodels': reverse('dlmodel-list', request=request, format=format),
        }              
        return render(request, 'mlmodel.html', context=context)


@api_view(['GET', 'POST'])
@permission_classes(())
def predict(request,format=None):
  try:
    
      if request.method=="POST":  
        dlmodel = Dlmodel.objects.get(id=self.pk)
        serializer = DlmodelSerializer(dlmodel, context={'request': request})
        print(serializer.data)
        serializer_dict = serializer.data
        image_url = serializer_dict['dlimage']['cifar_square_crop']
        image_url = image_url.split('/')
        file = image_url[-1]
    
        file_path = os.path.join(settings.BASE_DIR, "media", *image_url[-3:])
        image = Image.open(file_path)
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
        image = np.array(image)
        image = image // 255
        image = np.expand_dims(image, axis=0)
        
        model = model = keras.models.load_model("tensorflow_model.keras")
        predict_res =model.predict(image)
        result =  class_names[tf.argmax(predict_res[0])]
        context = {
                
                "qs":dlmodel ,
                 "result" :result,
               
                'users': reverse('user-list', request=request, format=format),
                'dlmodels': reverse('dlmodel-list', request=request, format=format),
            }              
        return render(request, 'predicttrue.html', context=context)
      else:
        dlmodel = Dlmodel.objects.last()
        serializer = DlmodelSerializer(dlmodel, context={'request': request})
        print(serializer.data)
        serializer_dict = serializer.data
        image_url = serializer_dict['dlimage']['cifar_square_crop']
        image_url = image_url.split('/')
        file = image_url[-1]
    
        file_path = os.path.join(settings.BASE_DIR, "media", *image_url[-3:])
        image = Image.open(file_path)
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
        image = np.array(image)
        image = image // 255
        image = np.expand_dims(image, axis=0)
        
        model = keras.models.load_model("tensorflow_model.keras")
        predict_res =model.predict(image)
        result =  class_names[tf.argmax(predict_res[0])]
        context = {
                
                "qs":dlmodel ,
                 "result" :result,
               
                'users': reverse('user-list', request=request, format=format),
                'dlmodels': reverse('dlmodel-list', request=request, format=format),
            }              
        return render(request, 'predicttrue.html', context=context)
       
            
  except Exception as e:
        return Response({"error": str(e)}, status=500)





@api_view(['GET', 'POST'])
@permission_classes(())
def train(request,format=None):
 try:
    if request.method =="POST":   



        (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

        # Normalize pixel values to be between 0 and 1
        train_images, test_images = train_images / 255.0, test_images / 255.0


        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                    'dog', 'frog', 'horse', 'ship', 'truck']


        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))


        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(10))


        suummary = model.summary()
        model.compile(optimizer='adam',
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])

        history = model.fit(train_images, train_labels, epochs=10,
                        validation_data=(test_images, test_labels))

        keras.saving.save_model("simplecnn.keras")
        keras.saving.load_model("simplecnn.keras")
        test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

         
        return(render(request,'train.html',{"data": Summary,
                                                'users': reverse('user-list', request=request, format=format),
                                                'dlmodels': reverse('dlmodel-list', request=request, format=format),
                                               "accuracy" :test_acc    
                                               }))
    else:
        
        return(render(request,'train.html',{"data":"Data ready for training ",
                                                 'users': reverse('user-list', request=request, format=None),
                                                 'dlmodels': reverse('dlmodel-list', request=request, format=None),
      }))
 except Exception as e:
    return Response({"error": str(e)}, status=500)



@api_view(['GET', 'POST'])
@permission_classes(())
def thanks(request, format=None):
    
        try:
            queryset = Dlmodel.objects.all()
            return render(request, "mlmodel.html", {
                "qs": queryset,
                'users': reverse('user-list', request=request, format=format),
                'dlmodels': reverse('dlmodel-list', request=request, format=format),
            })
        except Exception as e:
            return Response({"error": str(e)}, status=500)
