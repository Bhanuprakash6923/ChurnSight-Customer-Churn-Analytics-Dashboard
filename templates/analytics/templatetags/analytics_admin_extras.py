from django import template
from analytics.models import Dataset, ModelVersion, Prediction

register = template.Library()


@register.simple_tag
def dataset_count():
    """Total datasets."""
    return Dataset.objects.count()


@register.simple_tag
def model_count():
    """Total models."""
    return ModelVersion.objects.count()


@register.simple_tag
def prediction_count():
    """Total predictions."""
    return Prediction.objects.count()


@register.simple_tag
def recent_predictions(limit=5):
    """
    Latest N predictions for the admin dashboard.
    Usage in template:
        {% recent_predictions 5 as last_preds %}
    """
    return Prediction.objects.select_related('model').order_by('-predicted_at')[:limit]
