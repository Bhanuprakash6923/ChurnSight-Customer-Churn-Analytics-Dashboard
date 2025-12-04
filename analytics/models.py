from django.db import models
from django.contrib.auth.models import User

class Dataset(models.Model):
    name = models.CharField(max_length=200)
    file = models.FileField(upload_to='datasets/')
    uploaded_by = models.ForeignKey(User, on_delete=models.CASCADE)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    num_rows = models.IntegerField(null=True, blank=True)
    num_columns = models.IntegerField(null=True, blank=True)

    def __str__(self):
        return self.name

class ModelVersion(models.Model):
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE)
    name = models.CharField(max_length=200, default='Churn Model')
    created_at = models.DateTimeField(auto_now_add=True)
    model_file = models.FileField(upload_to='models/')
    accuracy = models.FloatField(null=True, blank=True)
    precision = models.FloatField(null=True, blank=True)
    recall = models.FloatField(null=True, blank=True)
    f1_score = models.FloatField(null=True, blank=True)
    is_active = models.BooleanField(default=True)

    def __str__(self):
        return f"{self.name} ({self.created_at.date()})"

class Prediction(models.Model):
    model = models.ForeignKey(ModelVersion, on_delete=models.CASCADE)
    customer_id = models.CharField(max_length=100)
    probability = models.FloatField()
    label = models.BooleanField()
    predicted_at = models.DateTimeField(auto_now_add=True)

    def risk_level(self):
        if self.probability >= 0.8:
            return 'High'
        elif self.probability >= 0.5:
            return 'Medium'
        return 'Low'

    def __str__(self):
        return f"{self.customer_id} - {self.probability:.2f}"
