from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
import pandas as pd
import io
import json
import csv  # 👈 IMPORTANT: for CSV export

from .models import Dataset, ModelVersion, Prediction
from .forms import DatasetUploadForm, SinglePredictionForm, BatchPredictionForm
from .ml.model_utils import train_churn_model, load_model


# =========================
# Dashboard
# =========================
@login_required
def dashboard(request):
    datasets_count = Dataset.objects.count()
    models_count = ModelVersion.objects.count()
    predictions_count = Prediction.objects.count()

    latest_models = ModelVersion.objects.order_by('-created_at')[:5]
    datasets = Dataset.objects.all().order_by('-uploaded_at')[:5]

    # For charts: churn vs no churn from Prediction table
    churn_count = Prediction.objects.filter(label=True).count()
    no_churn_count = Prediction.objects.filter(label=False).count()

    context = {
        'datasets_count': datasets_count,
        'models_count': models_count,
        'predictions_count': predictions_count,
        'latest_models': latest_models,
        'datasets': datasets,
        'churn_count': churn_count,
        'no_churn_count': no_churn_count,
    }
    return render(request, 'analytics/dashboard.html', context)


# =========================
# Dataset upload + preview
# =========================
@login_required
def upload_dataset(request):
    if request.method == 'POST':
        form = DatasetUploadForm(request.POST, request.FILES)
        if form.is_valid():
            dataset = form.save(commit=False)
            dataset.uploaded_by = request.user
            dataset.save()

            # Basic stats
            df = pd.read_csv(dataset.file.path)
            dataset.num_rows = df.shape[0]
            dataset.num_columns = df.shape[1]
            dataset.save()

            messages.success(request, 'Dataset uploaded successfully!')
            return redirect('analytics:dashboard')
    else:
        form = DatasetUploadForm()
    return render(request, 'analytics/upload_dataset.html', {'form': form})


@login_required
def dataset_detail(request, dataset_id):
    dataset = get_object_or_404(Dataset, id=dataset_id)

    df = pd.read_csv(dataset.file.path)
    preview = df.head(20)
    columns = list(preview.columns)
    rows = preview.to_dict(orient='records')

    context = {
        'dataset': dataset,
        'columns': columns,
        'rows': rows,
        'total_rows': df.shape[0],
        'total_cols': df.shape[1],
    }
    return render(request, 'analytics/dataset_detail.html', context)


# =========================
# Model training & listing
# =========================
@login_required
def train_model_view(request, dataset_id):
    dataset = get_object_or_404(Dataset, id=dataset_id)
    if request.method == 'POST':
        try:
            metrics = train_churn_model(dataset.file.path)
            mv = ModelVersion.objects.create(
      dataset=dataset,
    name='Churn Model',
    accuracy=metrics['accuracy'],
    precision=metrics['precision'],
    recall=metrics['recall'],
    f1_score=metrics['f1_score'],
    tn=metrics.get('tn', 0),
    fp=metrics.get('fp', 0),
    fn=metrics.get('fn', 0),
    tp=metrics.get('tp', 0),
)

            # Assign file path manually (FileField normally expects a File, but path works for demo)
            mv.model_file.name = metrics['model_path']
            mv.save()
            messages.success(request, 'Model trained successfully!')
            return redirect('analytics:model_detail', model_id=mv.id)
        except Exception as e:
            messages.error(request, f'Error training model: {e}')
    return render(request, 'analytics/train_model.html', {'dataset': dataset})


@login_required
def model_list(request):
    models = ModelVersion.objects.all().order_by('-created_at')
    return render(request, 'analytics/model_list.html', {'models': models})


@login_required
def model_detail(request, model_id):
    model = get_object_or_404(ModelVersion, id=model_id)
    predictions = Prediction.objects.filter(model=model).order_by('-predicted_at')[:50]
    return render(request, 'analytics/model_detail.html', {
        'model_obj': model,
        'predictions': predictions,
    })


# =========================
# Single prediction
# =========================
@login_required
def single_predict_view(request, model_id):
    model = get_object_or_404(ModelVersion, id=model_id)
    if request.method == 'POST':
        form = SinglePredictionForm(request.POST)
        if form.is_valid():
            data = form.cleaned_data
            clf = load_model(model.model_file.path)

            # include customer_id so columns match training
            df = pd.DataFrame([{
                'customer_id': data['customer_id'],
                'tenure': data['tenure'],
                'monthly_charges': data['monthly_charges'],
                'total_charges': data['total_charges'],
                'contract_type': data['contract_type'],
            }])

            prob = clf.predict_proba(df)[0][1]
            label = prob >= 0.5

            Prediction.objects.create(
                model=model,
                customer_id=data['customer_id'],
                probability=prob,
                label=label,
            )

            messages.success(request, f'Predicted churn probability: {prob:.2f}')
            return redirect('analytics:model_detail', model_id=model.id)
    else:
        form = SinglePredictionForm()

    return render(request, 'analytics/single_predict.html', {
        'form': form,
        'model_obj': model,
    })


# =========================
# Batch prediction (CSV)
# =========================
@login_required
def batch_predict_view(request, model_id):
    model = get_object_or_404(ModelVersion, id=model_id)
    clf = load_model(model.model_file.path)

    if request.method == 'POST':
        form = BatchPredictionForm(request.POST, request.FILES)
        if form.is_valid():
            f = form.cleaned_data['file']
            try:
                # Read uploaded CSV into DataFrame
                df = pd.read_csv(f)

                required_cols = ['customer_id', 'tenure', 'monthly_charges',
                                 'total_charges', 'contract_type']
                missing = [c for c in required_cols if c not in df.columns]
                if missing:
                    messages.error(request, f'Missing columns in CSV: {missing}')
                    return redirect('analytics:batch_predict', model_id=model.id)

                # Prepare input for model
                X = df[required_cols].copy()

                # Predict probabilities
                probs = clf.predict_proba(X)[:, 1]
                labels = probs >= 0.5

                # Attach predictions to dataframe
                df['churn_probability'] = probs
                df['churn_label'] = labels.astype(int)

                # Save each prediction to DB
                for idx, row in df.iterrows():
                    Prediction.objects.create(
                        model=model,
                        customer_id=row['customer_id'],
                        probability=row['churn_probability'],
                        label=bool(row['churn_label']),
                    )

                # Return CSV as download
                output = io.StringIO()
                df.to_csv(output, index=False)
                response = HttpResponse(
                    output.getvalue(),
                    content_type='text/csv'
                )
                response['Content-Disposition'] = (
                    f'attachment; filename="batch_predictions_model_{model.id}.csv"'
                )
                messages.success(
                    request,
                    f'Batch prediction completed for {len(df)} records.'
                )
                return response

            except Exception as e:
                messages.error(request, f'Error processing file: {e}')
                return redirect('analytics:batch_predict', model_id=model.id)
    else:
        form = BatchPredictionForm()

    return render(request, 'analytics/batch_predict.html', {
        'form': form,
        'model_obj': model,
    })


# =========================
# Export predictions as CSV
# =========================
@login_required
def export_predictions_csv(request, model_id):
    model = get_object_or_404(ModelVersion, id=model_id)
    predictions = Prediction.objects.filter(model=model).order_by('-predicted_at')

    # Create the HttpResponse object with CSV headers
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = (
        f'attachment; filename="predictions_model_{model.id}.csv"'
    )

    writer = csv.writer(response)
    # Header row
    writer.writerow([
        'customer_id',
        'probability',
        'label',
        'risk_level',
        'predicted_at',
    ])

    # Data rows
    for p in predictions:
        writer.writerow([
            p.customer_id,
            f"{p.probability:.4f}",
            int(p.label),                     # 1 = churn, 0 = no churn
            getattr(p, 'risk_level', ''),     # if you have risk_level property
            p.predicted_at.strftime('%Y-%m-%d %H:%M:%S'),
        ])

    return response


# =========================
# Simple JSON APIs (optional)
# =========================
@login_required
def api_models(request):
    """
    Return list of models and their metrics as JSON.
    """
    models = ModelVersion.objects.all().order_by('-created_at')
    data = []
    for m in models:
        data.append({
            'id': m.id,
            'name': m.name,
            'dataset': m.dataset.name,
            'created_at': m.created_at.isoformat(),
            'accuracy': m.accuracy,
            'precision': m.precision,
            'recall': m.recall,
            'f1_score': m.f1_score,
        })
    return JsonResponse(data, safe=False)


@csrf_exempt
def api_model_predict(request, model_id):
    """
    JSON API for prediction.

    POST JSON:
    {
      "customer_id": "C001",
      "tenure": 5,
      "monthly_charges": 700,
      "total_charges": 3500,
      "contract_type": "Month-to-month"
    }
    """
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST allowed'}, status=405)

    try:
        body = request.body.decode('utf-8')
        data = json.loads(body)
    except Exception:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)

    required = ['customer_id', 'tenure', 'monthly_charges', 'total_charges', 'contract_type']
    missing = [k for k in required if k not in data]
    if missing:
        return JsonResponse({'error': f'Missing fields: {missing}'}, status=400)

    model = get_object_or_404(ModelVersion, id=model_id)
    clf = load_model(model.model_file.path)

    df = pd.DataFrame([{
        'customer_id': data['customer_id'],
        'tenure': data['tenure'],
        'monthly_charges': data['monthly_charges'],
        'total_charges': data['total_charges'],
        'contract_type': data['contract_type'],
    }])

    prob = float(clf.predict_proba(df)[0][1])
    label = prob >= 0.5

    # Save prediction
    Prediction.objects.create(
        model=model,
        customer_id=data['customer_id'],
        probability=prob,
        label=label,
    )

    return JsonResponse({
        'model_id': model.id,
        'customer_id': data['customer_id'],
        'churn_probability': prob,
        'churn_label': int(label),
    })
