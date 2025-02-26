from django.contrib import admin
from django.urls import path
from prediction.views import predict_obesity

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', predict_obesity, name='predict'),
]
