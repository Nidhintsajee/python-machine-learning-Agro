from django.conf.urls import url
from . import views



urlpatterns = [
    url(r'^AboutUs/', views.About, name='about_us_url'),
    url(r'^ContactUs/', views.Contact, name='contact_url'),
    url (r'^predict/',views.Predict,name='agropredict_url'),
    url (r'^Clustering/',views.Cluster,name='clustered_url'),
    url (r'^show/',views.show,name='show_url'),
    
    # url(r'^charts/pie/(?P<id>\d+)/$', views.piechart , name='image_url'),

    # url(r'^images/', views.images, name='show_url'),
    # url(r'^shows/', views.showimage1, name='shows_url'),
]