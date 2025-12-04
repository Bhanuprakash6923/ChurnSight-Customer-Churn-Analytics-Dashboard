from django.contrib import admin
from django.utils.html import format_html

from .models import Dataset, ModelVersion, Prediction


@admin.register(Dataset)
class DatasetAdmin(admin.ModelAdmin):
    list_display = (
        'name',
        'uploaded_by',
        'uploaded_at',
        'num_rows',
        'num_columns',
        'open_in_analytics',
    )
    search_fields = ('name', 'uploaded_by__username')
    list_filter = ('uploaded_at',)
    date_hierarchy = 'uploaded_at'
    readonly_fields = ('num_rows', 'num_columns', 'uploaded_at')
    list_per_page = 25
    ordering = ('-uploaded_at',)

    def open_in_analytics(self, obj):
        # Button to open dataset details in your frontend app
        return format_html(
            '<a href="/analytics/datasets/{}/" class="btn btn-sm btn-outline-primary">View</a>',
            obj.id
        )
    open_in_analytics.short_description = "Analytics UI"


@admin.register(ModelVersion)
class ModelVersionAdmin(admin.ModelAdmin):
    list_display = (
        'name',
        'dataset',
        'created_at',
        'accuracy',
        'precision',
        'recall',
        'f1_score',
        'is_active',
    )
    search_fields = ('name', 'dataset__name')
    list_filter = ('created_at', 'is_active')
    date_hierarchy = 'created_at'
    list_editable = ('is_active',)
    list_per_page = 25
    ordering = ('-created_at',)

    fieldsets = (
        ('Basic Info', {
            'fields': ('name', 'dataset', 'is_active')
        }),
        ('Performance Metrics', {
            'fields': ('accuracy', 'precision', 'recall', 'f1_score'),
        }),
        ('Model File', {
            'fields': ('model_file',),
        }),
    )
class CustomAdminSite(admin.AdminSite):
    class Media:
        css = {
            'all': ('admin/custom_admin.css',)
        }

admin.site.__class__ = CustomAdminSite




@admin.register(Prediction)
class PredictionAdmin(admin.ModelAdmin):
    list_display = (
        'customer_id',
        'model',
        'probability',
        'label',
        'risk_badge',
        'predicted_at',
    )
    search_fields = ('customer_id', 'model__name')
    list_filter = ('label', 'model', 'predicted_at')
    date_hierarchy = 'predicted_at'
    list_per_page = 50
    ordering = ('-predicted_at',)
    
    

    def risk_badge(self, obj):
        # Your model has def risk_level(self): ... so we CALL it
        level = obj.risk_level()  # <-- call the method

        color = {
            'High': 'danger',
            'Medium': 'warning',
            'Low': 'success',
        }.get(level, 'secondary')

        return format_html('<span class="badge bg-{}">{}</span>', color, level)


    risk_badge.short_description = "Risk"
    
