from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.http import HttpResponse, HttpResponseRedirect, HttpResponseNotFound,JsonResponse
from django.shortcuts import render,render_to_response
from sklearn.cluster import MeanShift, estimate_bandwidth
import numpy as np

from django.core.urlresolvers import reverse
from django.core.mail import send_mail
import random
import datetime
import time
from fusioncharts import FusionCharts
import django
import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.dates import DateFormatter
from matplotlib import pylab
from pylab import *
import PIL, PIL.Image, StringIO
from django.db.models import Avg,Sum
from AgriPredict.settings import BASE_DIR
import pandas as pd
from sklearn import linear_model
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from graphos.sources.simple import SimpleDataSource
from graphos.renderers.highcharts import LineChart,BarChart,ScatterChart,ColumnChart
from django.db.models import Avg
from AgroMain.models import *

# def create_js(name='ERNAKULAM',html_id='highchart_div',title = 'ERNAKULAM'):
#     district = Agridata.objects.filter(district_name__icontains=name)
#     years = [int(value[0]) for value in district.values_list('crop_year').distinct()]
#     averages = [
#         (Agridata.objects.filter(district_name=name, crop_year=year).aggregate(
#             Avg('production'))).get('production__avg') for year in years
#     ]
#     data = [
#         ['Years', 'Production (Avg)'],
#     ]
#     for year, avg in zip(years, averages):
#         data.append([year, avg])
#     sd = SimpleDataSource(data)
#     hc = LineChart(
#         sd, html_id, height=450, width=450,
#         options={'title': title, 'xAxis': {'title': {'text':'years'}}, 'style': 'float:right;'}
#     )
#     return hc


def create_js_chart(name='ERNAKULAM',html_id='highchart_div',title = 'ERNAKULAM'):
	district = Agridata.objects.filter(district_name__icontains=name)
	years = [int(value[0]) for value in district.values_list('crop_year').distinct()]
	averages = [
		(Agridata.objects.filter(district_name=name, crop_year=year).aggregate(
			Avg('production'))).get('production__avg') for year in years
	]
	data = [
		['Years', 'Production (Avg)'],
	]
	for year, avg in zip(years, averages):
		data.append([year, avg])
	sd = SimpleDataSource(data)
	hc = LineChart(
		sd, html_id, height=450, width=450,
		options={'title': title, 'xAxis': {'title': {'text':'years'}}, 'style': 'float:right;'}
	)
	return hc
def create_chart(name='ERNAKULAM',html_id='highchart_div',title = 'ERNAKULAM'):
	agri = Agridata.objects.all()
	raindata = [i.rainfall for i in agri]
	# print"raindata",raindata
	# raintuple = tuple(raindata)
	X = np.array(zip(raindata,range(len(raindata))), dtype=np.int)
	bandwidth = estimate_bandwidth(X, quantile=0.1)
	ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
	ms.fit(X)
	labels = ms.labels_
	cluster_centers = ms.cluster_centers_
	# print cluster_centers

	labels_unique = np.unique(labels)
	# n_clustfers_ = 7
	n_clusters_ = len(labels_unique)
	# print("Number of estimated clusters:", n_clusters_)
	for k in range(n_clusters_):
		my_members = labels == k
		# print "cluster {0}: {1}".format(k, X[my_members,0])
	colors = 10*['r.','g.','b.','c.','k.','y.','m.']
	r,g,b,c,k,y,m=[],[],[],[],[],[],[]
	r=[['Years', 'Production (Avg)'],]
	g=[['Years', 'Production (Avg)'],]
	b=[['Years', 'Production (Avg)'],]
	c=[['Years', 'Production (Avg)'],]
	k=[['Years', 'Production (Avg)'],]
	y=[['Years', 'Production (Avg)'],]
	m=[['Years', 'Production (Avg)'],]
	# ,b,c,k,y,m=,[['Years', 'Production (Avg)'],],
	# [['Years', 'Production (Avg)'],],[['Years', 'Production (Avg)'],],[['Years', 'Production (Avg)'],],
	# [['Years', 'Production (Avg)'],],[['Years', 'Production (Avg)'],]
	# f=[['Years', 'Production (Avg)'],]
	f=[]
	for i in range(len(X)):
		if colors[labels[i]] == 'r.':
			r.append([X[i][0],X[i][1]])
		elif colors[labels[i]] == 'g.':
			g.append([X[i][0],X[i][1]])
		elif colors[labels[i]] == 'b.':
			b.append([X[i][0],X[i][1]])
		elif colors[labels[i]] == 'c.':
			c.append([X[i][0],X[i][1]])
		elif colors[labels[i]] == 'k.':
			k.append([X[i][0],X[i][1]])
		elif colors[labels[i]] == 'y.':
			y.append([X[i][0],X[i][1]])
		else:
			m.append([X[i][0],X[i][1]])
	f.append(r)
	f.append(g)
	print"fffffffffffff",f
	f.append(b)
	f.append(c)
	f.append(k)
	f.append(y)
	f.append(m)


	# district = Agridata.objects.filter(district_name__icontains=name)
	# years = [int(value[0]) for value in district.values_list('crop_year').distinct()]
	# averages = [
	#     (Agridata.objects.filter(district_name=name, crop_year=year).aggregate(
	#         Avg('production'))).get('production__avg') for year in years
	# ]
	data = [
		['Years', 'Production (Avg)'],
	]
	# for year, avg in zip(years, averages):
	#     data.append([year, avg])
	f=SimpleDataSource(f)
	# r = SimpleDataSource(r)
	# g = SimpleDataSource(g)
	# b = SimpleDataSource(b)
	# c = SimpleDataSource(c)
	# k = SimpleDataSource(k)
	# y = SimpleDataSource(y)
	# m = SimpleDataSource(m)
	# print "sfsf",f.get_header()
	# {'series':[{'color':'blue','data':r},{'data':g},{'data':b},{'data':c},{'data':k},{'data':y},{'data':m}],}
	hc = ScatterChart(f,html_id, height=450, width=450,
		options={'title': title, 'xAxis': {'title': {'text':'years'}}, 'style': 'float:right;'}
	)
	# print "hchch",hc.as_html()
	return hc
# def create_chart():
#     # district = Agridata.objects.filter(district_name__icontains=name)
#     # years = [int(value[0]) for value in district.values_list('crop_year').distinct()]
#     # averages = [
#     #     (Agridata.objects.filter(district_name=name, crop_year=year).aggregate(
#     #         Avg('production'))).get('production__avg') for year in years
#     # ]
#     data = [
#         ['Years', 'Production (Avg)'],
#     ]
#     data.append([20,100])
#     data.append([40,110])
#     sd = SimpleDataSource(data)
#     hc = ColumnChart(sd,'highbar_div',height=450,width=450,
#         options={'title':"barchart",'xAxis': {'title': {'text':'years'}}, 'style': 'float:right;'})
#     return hc
# def create_chart(name='ERNAKULAM'):
#     district = Agridata.objects.filter(district_name__icontains=name)
#     years = [int(value[0]) for value in district.values_list('crop_year').distinct()]
#     averages = [
#         (Agridata.objects.filter(district_name=name, crop_year=year).aggregate(
#             Avg('production'))).get('production__avg') for year in years
#     ]
#     data = [
#         ['Years', 'Production (Avg)'],
#     ]
#     for year, avg in zip(years, averages):
#         data.append([year, avg])
#     sd = SimpleDataSource(data)
#     hc = BarChart(sd,'highbar_div',height=450,width=450,
#         options={'title':"barchart",'xAxis': {'title': {'text':'years'}}, 'style': 'float:right;'})
#     return hc
def home(request):
	if request.method == 'POST':
		if "file1" in request.FILES:
			file1 = request.FILES['file1']
			analysis = AnalysisFiles(data_file=file1)
			analysis.save()
			ernakulam = create_js_chart()
			alappuzha = create_js_chart('ALAPPUZHA','highchartala_div','ALAPPUZHA')
			idukki = create_js_chart('IDUKKI','highchartidukki_div','IDUKKI')
			kannur = create_js_chart('KANNUR','highchartkannur_div','KANNUR')
			kasaragod = create_js_chart('KASARAGOD','highchartkasaragod_div','KASARAGOD')
			kollam = create_js_chart('KOLLAM','highchartkollam_div','KOLLAM')
			kottayam = create_js_chart('KOTTAYAM','highchartkottayam_div','KOTTAYAM')
			kozhikode = create_js_chart('KOZHIKODE','highchartkozhikode_div','KOZHIKODE')
			malappuram = create_js_chart('MALAPPURAM','highchartmala_div','MALAPPURAM')
			palakad = create_js_chart('PALAKKAD','highchartpala_div','PALAKKAD')
			pathanamthitta = create_js_chart('PATHANAMTHITTA','highchartpathanam_div','PATHANAMTHITTA')
			tvm = create_js_chart('THIRUVANANTHAPURAM','highcharttvn_div','THIRUVANANTHAPURAM')
			tsr = create_js_chart('THRISSUR','highcharttsr_div','THRISSUR')
			wayanad = create_js_chart('WAYANAD','highchartwayanad_div','WAYANAD')
			# data = pd.read_csv(file1)
			# Agridata.objects.all().delete()
			# state =[i for i in data['state']]
			# district_name = [i for i in data['district']]
			# crop_year = [i for i in data['year']]
			# season = [i for i in data['season']]
			# crop = [i for i in data['crop']]
			# area = [i for i in data['area']]
			# production = [i for i in data['production']]
			# rainfall = [i for i in data['rainfall']]
			# for i in range(len(state)):
			#     agri = Agridata(state_name=state[i],district_name=district_name[i],crop_year=crop_year[i],
			#         season=season[i],crop=crop[i],area=area[i],production=production[i],rainfall=rainfall[i])
			#     agri.save()
			msg = "ok"

			return render(request, 'home.html', {'msg':msg,'wayanad':wayanad,'tsr':tsr,
				'tvm':tvm,'pathanamthitta':pathanamthitta,'palakad':palakad,
				'malappuram':malappuram,'kozhikode':kozhikode,'kottayam':kottayam,
				'kollam':kollam,'kasaragod':kasaragod,
				'kannur':kannur,'idukki':idukki, 'alappuzha':alappuzha,'ernakulam':ernakulam})
		else:

			file1 = None
			return render(request, 'home.html', {})
		
	else:
		# ernakulam = create_js_chart()
		return render(request, 'home.html', {})


def About(request):
	ernakulam = create_chart()

	return render(request, 'about.html', {})


def Cluster(request):
	data = AnalysisFiles.objects.get(active=True)
	filename = "{}/media/{}".format(BASE_DIR,data.data_file)
	df = pd.read_csv(filename)
	print(type(df))
	print(df['rainfall'])
	rf = df['rainfall']
	gf = df['district_name'].unique()
	dist = gf.tolist()
	x=[]

	for i in rf:
		x.append(i)

	y = tuple(x)

	# x = df['rainfall']
	X = np.array(zip(x,range(len(x))), dtype=np.int)
	# print(X)
	
	bandwidth = estimate_bandwidth(X, quantile=0.1)
	print ("bandwidth",bandwidth)
	ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
	print ms
	ms.fit(X)
	labels = ms.labels_
	print("labels",labels)
	cluster_centers = ms.cluster_centers_
	n_clusters_ = len(dist)
	for k in range(n_clusters_):
		my_members = labels == k
		print "cluster {0}: {1}".format(k, X[my_members,0])
	colors = 10*['r.','g.','b.','c.','k.','y.','m.']
	r,g,b,c,k,y,m=[],[],[],[],[],[],[]
	for i in range(len(X)):
		if colors[labels[i]] == 'r.':
			r.append([X[i][0],X[i][1]])
		elif colors[labels[i]] == 'g.':
			g.append([X[i][0],X[i][1]])
		elif colors[labels[i]] == 'b.':
			b.append([X[i][0],X[i][1]])
		elif colors[labels[i]] == 'c.':
			c.append([X[i][0],X[i][1]])
		elif colors[labels[i]] == 'k.':
			k.append([X[i][0],X[i][1]])
		elif colors[labels[i]] == 'y.':
			y.append([X[i][0],X[i][1]])
		else:
			m.append([X[i][0],X[i][1]])

		plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize = 10)

	

	print("ggggggggggggggggg",g)


	vals = X.tolist()

	return render(request,'cluster.html',{'vals':vals,'r':r,'g':g,'b':b,'c':c,'k':k,'y':y,'m':m})

def Predict(request):
	try :
		if request.method == 'POST':
		# datas  = create_chart()
			year = request.POST.get('sel1')
			dist = request.POST.get('sel2')
			area = request.POST.get('area')
			rainfall = request.POST.get('rainfall')

			print ("inserted",year,rainfall,dist,area)
			result = show(rainfall)
			result = float(result)
			return render(request,'predict.html',{'result':result,'year':year,'rainfall':rainfall})

		else:
			return render(request,'predict.html',{})
	except Exception as e:
		print("error",e)
		return render(request,'home.html',{})
def Contact(request):
	return render(request, 'contact.html', {})

def show(rainfall):
	
	data = AnalysisFiles.objects.get(active=True)
	filename = "{}/media/{}".format(BASE_DIR,data.data_file)
	df = pd.read_csv(filename)
	DISTRICTS = df['district_name'].unique()
	rain = int(rainfall)
	print(rain)
	# fig = plt.figure(figsize=(20, 20))
	# ax = fig.add_subplot(111)

	X = 'rainfall'
	Y = 'production'

	df2 = df[[X, Y]]
	df2.columns = np.array([X, Y])
	x_mean = df2[X].mean()
	y_mean = df2[Y].mean()

	df2 = df2.copy()
	df2.fillna({X: x_mean, Y: y_mean}, inplace=True)
	print("DF2",df2)
	# msk = np.random.rand(len(df2)) < 0.8

	# df_train = df2[msk]
	# df_test = df2[~msk]

	reg = linear_model.LinearRegression()
	
	print "Rain ", rain
	df_test = [[rain]]
	print "Test ", df_test	
	model = reg.fit(df2[X].values.reshape(-1, 1), df2[Y].values.reshape(-1, 1))
	# print(model.summary())
	print("reg",reg)
	print("coef",reg.coef_)
	# to_test = df_test[X].values.reshape(-1, 1)
	predicted_values = reg.predict(df_test)
	print("pere",predicted_values)
	print(df_test)
	print("\n")
	print("Predicted Values")
	for rainfall, predicted_production in zip(df_test, predicted_values):
		print("RainFall : {} mm | Production : {:.2f}".format(rainfall[0], float(predicted_production[0])))
	return(predicted_values)