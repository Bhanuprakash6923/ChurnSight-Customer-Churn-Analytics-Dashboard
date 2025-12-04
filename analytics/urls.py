from django.urls import path
from django.views.generic import RedirectView
from . import views

app_name = 'analytics'

urlpatterns = [
    # When user hits "/analytics/", send them to dashboard
    path(
        '',
        RedirectView.as_view(pattern_name='analytics:dashboard', permanent=False),
        name='analytics_root'
    ),

    # Dashboard
    path('dashboard/', views.dashboard, name='dashboard'),

    # Dataset upload
    path('datasets/upload/', views.upload_dataset, name='upload_dataset'),

    # Model training (note: URL starts with models/train/)
    path('models/train/<int:dataset_id>/', views.train_model_view, name='train_model'),

    # Model list + details
    path('models/', views.model_list, name='model_list'),
    path('models/<int:model_id>/', views.model_detail, name='model_detail'),

    # Single prediction
    path('models/<int:model_id>/predict/', views.single_predict_view, name='single_predict'),
    path('models/<int:model_id>/batch-predict/', views.batch_predict_view, name='batch_predict'),
path('models/<int:model_id>/export-predictions/', views.export_predictions_csv, name='export_predictions'),
path('api/models/', views.api_models, name='api_models'),
path('api/models/<int:model_id>/predict/', views.api_model_predict, name='api_model_predict'),


]
