import os
import numpy as np


def load_model_dynamic(model_path):
    ext = os.path.splitext(model_path)[1].lower()

    if ext in ('.h5', '.keras'):
        import tensorflow as tf
        model = tf.keras.models.load_model(model_path)
        return model, 'tensorflow'

    elif ext in ('.pt', '.pth'):
        return _load_pytorch(model_path)

    elif ext in ('.pkl', '.joblib'):
        try:
            import joblib
            model = joblib.load(model_path)
        except Exception:
            import pickle
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
        return model, 'sklearn'

    else:
        raise ValueError(f'Desteklenmeyen model formatı: {ext}')


def _load_pytorch(model_path):
    import torch

    # 1) TorchScript dene
    try:
        model = torch.jit.load(model_path, map_location='cpu')
        model.eval()
        return model, 'pytorch_script'
    except Exception:
        pass

    # 2) Tam model (torch.save(model, path))
    try:
        obj = torch.load(model_path, map_location='cpu', weights_only=False)
        if hasattr(obj, 'parameters'):
            obj.eval()
            return obj, 'pytorch_full'
    except Exception:
        pass

    # 3) state_dict — mimariyi dosya adından bul
    try:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    except Exception:
        checkpoint = torch.load(model_path, map_location='cpu')

    state_dict = checkpoint
    if isinstance(checkpoint, dict):
        for key in ('state_dict', 'model', 'net'):
            if key in checkpoint:
                state_dict = checkpoint[key]
                break

    model = _build_model_from_name(model_path, state_dict)
    model.eval()
    return model, 'pytorch_full'


def _build_model_from_name(model_path, state_dict):
    import torch.nn as nn
    import torchvision.models as models

    name = os.path.basename(model_path).lower()

    arch_map = [
        ('resnet152',       lambda: models.resnet152(weights=None)),
        ('resnet101',       lambda: models.resnet101(weights=None)),
        ('resnet50',        lambda: models.resnet50(weights=None)),
        ('resnet34',        lambda: models.resnet34(weights=None)),
        ('resnet18',        lambda: models.resnet18(weights=None)),
        ('vgg19_bn',        lambda: models.vgg19_bn(weights=None)),
        ('vgg19',           lambda: models.vgg19(weights=None)),
        ('vgg16_bn',        lambda: models.vgg16_bn(weights=None)),
        ('vgg16',           lambda: models.vgg16(weights=None)),
        ('vgg13_bn',        lambda: models.vgg13_bn(weights=None)),
        ('vgg13',           lambda: models.vgg13(weights=None)),
        ('vgg11_bn',        lambda: models.vgg11_bn(weights=None)),
        ('vgg11',           lambda: models.vgg11(weights=None)),
        ('alexnet',         lambda: models.alexnet(weights=None)),
        ('mobilenet_v3',    lambda: models.mobilenet_v3_small(weights=None)),
        ('mobilenet',       lambda: models.mobilenet_v2(weights=None)),
        ('efficientnet_b7', lambda: models.efficientnet_b7(weights=None)),
        ('efficientnet_b6', lambda: models.efficientnet_b6(weights=None)),
        ('efficientnet_b5', lambda: models.efficientnet_b5(weights=None)),
        ('efficientnet_b4', lambda: models.efficientnet_b4(weights=None)),
        ('efficientnet_b3', lambda: models.efficientnet_b3(weights=None)),
        ('efficientnet_b2', lambda: models.efficientnet_b2(weights=None)),
        ('efficientnet_b1', lambda: models.efficientnet_b1(weights=None)),
        ('efficientnet_b0', lambda: models.efficientnet_b0(weights=None)),
        ('densenet201',     lambda: models.densenet201(weights=None)),
        ('densenet169',     lambda: models.densenet169(weights=None)),
        ('densenet121',     lambda: models.densenet121(weights=None)),
        ('inception',       lambda: models.inception_v3(weights=None)),
        ('squeezenet',      lambda: models.squeezenet1_1(weights=None)),
        ('shufflenet',      lambda: models.shufflenet_v2_x1_0(weights=None)),
        ('mnasnet',         lambda: models.mnasnet1_0(weights=None)),
    ]

    base_model = None
    for keyword, builder in arch_map:
        if keyword in name:
            base_model = builder()
            break

    if base_model is None:
        raise ValueError(
            f'Mimari tanınamadı: "{os.path.basename(model_path)}"\n'
            f'Dosya adında mimari adı geçmeli. Örnek: resnet50_model.pt'
        )

    # Sınıf sayısını tespit et ve son katmanı ayarla
    num_classes = _detect_num_classes(state_dict)
    if num_classes and num_classes != 1000:
        base_model = _adjust_classifier(base_model, num_classes)

    # Yükle — strict=False ile küçük uyumsuzlukları geç
    base_model.load_state_dict(state_dict, strict=False)
    return base_model


def _detect_num_classes(state_dict):
    for key in reversed(list(state_dict.keys())):
        if 'weight' in key and len(state_dict[key].shape) == 2:
            return state_dict[key].shape[0]
    return None


def _adjust_classifier(model, num_classes):
    import torch.nn as nn

    # ResNet, EfficientNet vb. — fc katmanı
    if hasattr(model, 'fc') and hasattr(model.fc, 'in_features'):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    # VGG, AlexNet, MobileNet — classifier
    if hasattr(model, 'classifier'):
        clf = model.classifier
        if isinstance(clf, nn.Sequential):
            for i in reversed(range(len(clf))):
                if hasattr(clf[i], 'in_features'):
                    clf[i] = nn.Linear(clf[i].in_features, num_classes)
                    break
        elif hasattr(clf, 'in_features'):
            model.classifier = nn.Linear(clf.in_features, num_classes)
        return model

    # DenseNet — classifier doğrudan
    if hasattr(model, 'classifier') and hasattr(model.classifier, 'in_features'):
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)

    return model


def predict_image(model, framework, image_path, input_size=(224, 224)):
    img_array = _preprocess_image(image_path, input_size)

    if framework == 'tensorflow':
        raw = model.predict(img_array, verbose=0)[0]
        # Binary sigmoid: tek değer döner (0-1 arası)
        if raw.shape == () or (hasattr(raw, '__len__') and len(raw) == 1):
            score = float(np.squeeze(raw))
            return _format_binary(score)
        # Softmax: çok sınıflı
        return _format_predictions(raw)

    elif framework in ('pytorch_script', 'pytorch_full'):
        import torch
        import torch.nn.functional as F
        tensor = torch.tensor(img_array, dtype=torch.float32).permute(0, 3, 1, 2)
        with torch.no_grad():
            outputs = model(tensor)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            # Binary: tek nöron
            if outputs.shape[1] == 1:
                score = float(torch.sigmoid(outputs).squeeze())
                return _format_binary(score)
            preds = F.softmax(outputs, dim=1).numpy()[0]
        return _format_predictions(preds)

    elif framework == 'sklearn':
        flat = img_array.reshape(1, -1)
        if hasattr(model, 'predict_proba'):
            preds = model.predict_proba(flat)[0]
        else:
            pred = model.predict(flat)[0]
            preds = np.eye(2)[int(pred)][0]
        return _format_predictions(preds)

    else:
        raise ValueError(f'Bilinmeyen framework: {framework}')


def _preprocess_image(image_path, input_size):
    from PIL import Image
    img = Image.open(image_path).convert('RGB').resize(input_size)
    return np.array(img, dtype=np.float32)[np.newaxis, ...] / 255.0


def _format_binary(score):
    """
    Binary sigmoid çıkışını yorumlar.
    Eğitimde class_indices genellikle {'fake':0, 'real':1} veya tersi olur.
    score → 1'e yakınsa 2. sınıf, 0'a yakınsa 1. sınıf.
    ImageDataGenerator alfabetik sıralar: fake=0, real=1 → score=P(real)
    """
    p_real = float(score)
    p_fake = 1.0 - p_real
    results = [
        {'label': 'Gerçek', 'probability': round(p_real * 100, 2)},
        {'label': 'Yapay',  'probability': round(p_fake * 100, 2)},
    ]
    return sorted(results, key=lambda x: x['probability'], reverse=True)


def _format_predictions(preds):
    labels = ['Gerçek', 'Yapay'] if len(preds) == 2 else [f'Sınıf {i}' for i in range(len(preds))]
    sorted_idx = np.argsort(preds)[::-1]
    return [
        {
            'label': labels[i] if i < len(labels) else f'Sınıf {i}',
            'probability': float(round(float(preds[i]) * 100, 2))
        }
        for i in sorted_idx
    ]