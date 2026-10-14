import os
import glob
import json
import base64
import numpy as np
from io import BytesIO
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
from utils.model_loader import load_model_dynamic, predict_image
from utils.gradcam import generate_gradcam
from utils.augmentation import apply_augmentation, get_catalog

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'bmp'}

os.makedirs('static/uploads', exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_available_models():
    models = []
    extensions = ['*.h5','*.keras', '*.pt', '*.pth', '*.pkl', '*.joblib']
    for ext in extensions:
        for path in glob.glob(os.path.join('models', ext)):
            name = os.path.splitext(os.path.basename(path))[0]
            fmt  = os.path.splitext(path)[1].replace('.', '').upper()
            models.append({'id': name, 'name': name, 'format': fmt, 'path': path})
    return models

def pil_to_base64(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()

@app.route('/')
def index():
    models  = get_available_models()
    catalog = get_catalog()
    return render_template('index.html', models=models, catalog=catalog)

@app.route('/api/models')
def api_models():
    return jsonify(get_available_models())

@app.route('/api/augment', methods=['POST'])
def augment():
    """Önizleme: görsele dönüşüm uygular, base64 döner."""
    if 'image' not in request.files:
        return jsonify({'error': 'Görsel bulunamadı'}), 400
    file   = request.files['image']
    aug_id = request.form.get('aug_id', '')
    if not aug_id:
        return jsonify({'error': 'Dönüşüm seçilmedi'}), 400
    try:
        img = Image.open(file).convert('RGB')
        augmented = apply_augmentation(img, aug_id)
        return jsonify({'image': pil_to_base64(augmented)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'Görsel bulunamadı'}), 400

    file          = request.files['image']
    model_id      = request.form.get('model_id', '')
    generate_cam  = request.form.get('gradcam', 'false') == 'true'
    aug_id        = request.form.get('aug_id', '')   # boşsa orijinal

    if not allowed_file(file.filename):
        return jsonify({'error': 'Desteklenmeyen dosya formatı'}), 400

    # Görseli kaydet
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    img_orig = Image.open(file).convert('RGB')
    img_orig.save(filepath)

    # Augmentation uygula (varsa)
    img_to_predict = img_orig
    aug_b64 = None
    if aug_id:
        try:
            img_to_predict = apply_augmentation(img_orig, aug_id)
            aug_b64 = pil_to_base64(img_to_predict)
            aug_path = os.path.join(app.config['UPLOAD_FOLDER'], 'aug_' + filename)
            img_to_predict.save(aug_path)
        except Exception as e:
            return jsonify({'error': f'Dönüşüm hatası: {e}'}), 500
    else:
        aug_path = filepath

    # Model bul
    models     = get_available_models()
    model_info = next((m for m in models if m['id'] == model_id), None)
    if not model_info:
        return jsonify({'error': f'Model bulunamadı: {model_id}'}), 404

    try:
        model, framework = load_model_dynamic(model_info['path'])
        predictions = predict_image(model, framework, aug_path)

        result = {
            'predictions': predictions,
            'model_name':  model_info['name'],
            'framework':   framework,
            'aug_image':   aug_b64
        }

        if generate_cam:
            cam_path = generate_gradcam(model, framework, aug_path, app.config['UPLOAD_FOLDER'])
            if cam_path:
                result['gradcam_path'] = '/' + cam_path + '?t=' + str(os.path.getmtime(cam_path))

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)