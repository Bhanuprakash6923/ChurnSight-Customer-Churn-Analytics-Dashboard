from django.contrib import admin
from django.urls import path, include
from django.contrib.auth import views as auth_views
from django.views.generic import RedirectView
from analytics import views   # IMPORTANT LINE ADDED


urlpatterns = [
    path('admin/', admin.site.urls),

    path(
        'accounts/login/',
        auth_views.LoginView.as_view(template_name='accounts/login.html'),
        name='login'
    ),

    path(
        'accounts/logout/',
        auth_views.LogoutView.as_view(),
        name='logout'
    ),

    # root → dashboard
    path(
        '',
        RedirectView.as_view(pattern_name='analytics:dashboard', permanent=False),
        name='home'
    ),

    # include analytics URLs
    path(
        'analytics/',
        include(('analytics.urls', 'analytics'), namespace='analytics')
    ),
    path(
    'models/<int:model_id>/export-predictions/',
    views.export_predictions_csv,
    name='export_predictions'
),

]
