from __future__ import unicode_literals

from django.db import models

# Create your models here.



class Agridata(models.Model):
	state_name = models.CharField(max_length=50)
	district_name = models.CharField(max_length=50)
	crop_year = models.IntegerField(default=2002)
	season = models.CharField(max_length=50)
	crop = models.CharField(max_length=50)
	area = models.DecimalField(default=0.0,decimal_places=2,max_digits=10)
	production = models.DecimalField(default=0.0,decimal_places=2,max_digits=10)
	rainfall = models.IntegerField(default=0)

	def __str__(self):
		return self.district_name

class Images(models.Model):
	chart = models.ImageField(blank=True,null=True)


class AnalysisFiles(models.Model):
	data_file = models.FileField(upload_to='data/', null=True, blank=True)
	active = models.BooleanField(default=True)


	@staticmethod
	def _set_inactive():
		
		data_set = AnalysisFiles.objects.all()
		print(data_set)
		if data_set:
			
			for data in data_set:
				data.active = False
				data.save()


	def save(self, *args, **kwargs):
		
		if self.pk is None:
			self._set_inactive()
		super(AnalysisFiles, self).save(*args, **kwargs)