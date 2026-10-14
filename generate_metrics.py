"""
generate_metrics.py
====================
Eğitim bittikten sonra bu scripti çalıştırarak her model için
metrics/<model_adı>.json dosyasını oluşturun.

Kullanım:
    python generate_metrics.py

Scripti kendi eğitim kodunuza göre düzenleyin.
"""

import json
import os
import numpy as np


def save_metrics(model_name: str, data: dict):
    """metrics/ klasörüne JSON kaydeder."""
    os.makedirs('metrics', exist_ok=True)
    path = os.path.join('metrics', f'{model_name}.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f'Kaydedildi: {path}')


# ──────────────────────────────────────────────
# ÖRNEK: TensorFlow / Keras modeli için
# ──────────────────────────────────────────────
def generate_for_tensorflow(model_name, model, X_train, y_train, X_val, y_val, X_test, y_test):
    from sklearn.metrics import (
        f1_score, accuracy_score, precision_score, recall_score,
        confusion_matrix, roc_curve, auc
    )

    def evaluate_split(X, y_true):
        y_pred_prob = model.predict(X)
        if y_pred_prob.shape[1] == 2:
            y_pred = np.argmax(y_pred_prob, axis=1)
            probs_pos = y_pred_prob[:, 1]
        else:
            y_pred = (y_pred_prob[:, 0] > 0.5).astype(int)
            probs_pos = y_pred_prob[:, 0]

        fpr, tpr, _ = roc_curve(y_true, probs_pos, pos_label=1)
        return {
            'accuracy':  float(accuracy_score(y_true, y_pred)),
            'f1_macro':  float(f1_score(y_true, y_pred, average='macro')),
            'f1_weighted': float(f1_score(y_true, y_pred, average='weighted')),
            'precision': float(precision_score(y_true, y_pred, average='weighted')),
            'recall':    float(recall_score(y_true, y_pred, average='weighted')),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
            'roc': {'fpr': fpr.tolist(), 'tpr': tpr.tolist(), 'auc': float(auc(fpr, tpr))}
        }

    # Keras history nesnesi varsa — model.history.history
    # Aşağıya kendi history değişkeninizi girin
    history = {}  # örn: history_obj.history

    metrics_data = {
        'model_name': model_name,
        'framework': 'tensorflow',
        'classes': ['Gerçek', 'Yapay'],
        'train': evaluate_split(X_train, y_train),
        'validation': evaluate_split(X_val, y_val),
        'test': evaluate_split(X_test, y_test),
        'history': {
            'train_loss': history.get('loss', []),
            'val_loss':   history.get('val_loss', []),
            'train_acc':  history.get('accuracy', []),
            'val_acc':    history.get('val_accuracy', [])
        }
    }
    save_metrics(model_name, metrics_data)


# ──────────────────────────────────────────────
# ÖRNEK MANUEL GİRİŞ (eğer değerleri zaten biliyorsanız)
# ──────────────────────────────────────────────
def generate_manual_example():
    """
    Değerleri zaten hesapladıysanız bu fonksiyonu kullanın.
    model_name = models/ klasöründeki dosya adıyla aynı olmalı (uzantısız)
    """
    example = {
        'model_name': 'efficientnet_v1',
        'framework': 'tensorflow',
        'classes': ['Gerçek', 'Yapay'],
        'train': {
            'accuracy': 0.97,
            'f1_macro': 0.96,
            'f1_weighted': 0.97,
            'precision': 0.97,
            'recall': 0.96,
            'confusion_matrix': [[450, 12], [8, 430]],
            'roc': {'fpr': [0, 0.02, 0.05, 1], 'tpr': [0, 0.95, 0.98, 1], 'auc': 0.98}
        },
        'validation': {
            'accuracy': 0.94,
            'f1_macro': 0.93,
            'f1_weighted': 0.94,
            'precision': 0.94,
            'recall': 0.93,
            'confusion_matrix': [[95, 7], [5, 93]],
            'roc': {'fpr': [0, 0.04, 0.08, 1], 'tpr': [0, 0.92, 0.96, 1], 'auc': 0.96}
        },
        'test': {
            'accuracy': 0.93,
            'f1_macro': 0.92,
            'f1_weighted': 0.93,
            'precision': 0.93,
            'recall': 0.92,
            'confusion_matrix': [[88, 9], [6, 97]],
            'roc': {'fpr': [0, 0.05, 0.1, 1], 'tpr': [0, 0.9, 0.95, 1], 'auc': 0.95}
        },
        'history': {
            'train_loss': [0.9, 0.6, 0.4, 0.3, 0.2, 0.15, 0.12, 0.10],
            'val_loss':   [0.95, 0.65, 0.45, 0.38, 0.30, 0.28, 0.27, 0.26],
            'train_acc':  [0.60, 0.75, 0.85, 0.90, 0.93, 0.95, 0.96, 0.97],
            'val_acc':    [0.58, 0.72, 0.82, 0.87, 0.90, 0.91, 0.93, 0.94]
        }
    }
    save_metrics('efficientnet_v1', example)


if __name__ == '__main__':
    # Örnek veriyi oluştur (test için)
    generate_manual_example()
    print('Örnek metrics/efficientnet_v1.json oluşturuldu.')
    print('Kendi modelleriniz için fonksiyonları düzenleyin.')