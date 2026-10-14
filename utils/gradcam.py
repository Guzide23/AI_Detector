import os
import numpy as np


def generate_gradcam(model, framework, image_path, output_dir, input_size=(224, 224)):
    try:
        if framework == 'tensorflow':
            return _gradcam_tensorflow(model, image_path, output_dir, input_size)
        elif framework in ('pytorch_script', 'pytorch_full'):
            return _gradcam_pytorch(model, image_path, output_dir, input_size)
        else:
            return None
    except Exception as e:
        print(f'GradCAM hatası: {e}')
        return None


# ── TensorFlow ────────────────────────────────────────────
def _gradcam_tensorflow(model, image_path, output_dir, input_size):
    import tensorflow as tf
    from PIL import Image

    img = Image.open(image_path).convert('RGB').resize(input_size)
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_tensor = tf.expand_dims(img_array, axis=0)

    last_conv = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv = layer.name
            break

    if not last_conv:
        return None

    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        if predictions.shape[-1] == 1:
            class_channel = predictions[:, 0]
        else:
            pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)

    return _overlay_heatmap(heatmap.numpy(), image_path, output_dir)


# ── PyTorch ───────────────────────────────────────────────
def _gradcam_pytorch(model, image_path, output_dir, input_size):
    import torch
    import torch.nn as nn
    from PIL import Image

    # VGG ve benzeri modellerde inplace ReLU'ları kapat
    _disable_inplace_relu(model)

    img = Image.open(image_path).convert('RGB').resize(input_size)
    arr = np.array(img, dtype=np.float32) / 255.0
    tensor = torch.tensor(arr).permute(2, 0, 1).unsqueeze(0)
    tensor.requires_grad_(True)

    last_conv = _find_last_conv(model)
    if last_conv is None:
        print('GradCAM: Conv2d katmanı bulunamadı')
        return None

    activations = {}
    gradients   = {}

    def fwd_hook(module, input, output):
        activations['v'] = output.clone()

    def bwd_hook(module, grad_in, grad_out):
        gradients['v'] = grad_out[0].clone()

    fh = last_conv.register_forward_hook(fwd_hook)
    bh = last_conv.register_full_backward_hook(bwd_hook)

    try:
        model.zero_grad()
        outputs = model(tensor)
        if isinstance(outputs, tuple):
            outputs = outputs[0]

        if outputs.shape[1] == 1:
            score = torch.sigmoid(outputs[0, 0])
        else:
            idx = outputs.argmax(dim=1).item()
            score = outputs[0, idx]

        score.backward()
    finally:
        fh.remove()
        bh.remove()

    if 'v' not in gradients or 'v' not in activations:
        print('GradCAM: gradient/activation yakalanamadı')
        return None

    grads = gradients['v'][0]    # (C, H, W)
    acts  = activations['v'][0]  # (C, H, W)

    weights = grads.mean(dim=(1, 2))
    cam = (weights[:, None, None] * acts).sum(0)
    cam = torch.relu(cam).detach().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    return _overlay_heatmap(cam, image_path, output_dir)


def _disable_inplace_relu(model):
    """Tüm inplace ReLU'ları devre dışı bırakır (VGG için gerekli)."""
    import torch.nn as nn
    for module in model.modules():
        if isinstance(module, nn.ReLU):
            module.inplace = False


def _find_last_conv(model):
    import torch.nn as nn
    last = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last = module
    return last


# ── Ortak: ısı haritasını görsele bindirme ────────────────
def _overlay_heatmap(heatmap, original_image_path, output_dir):
    import cv2
    from PIL import Image

    orig = np.array(Image.open(original_image_path).convert('RGB'))
    h, w = orig.shape[:2]

    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8   = np.uint8(255 * heatmap_resized)
    colored         = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    colored_rgb     = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    superimposed    = (orig * 0.6 + colored_rgb * 0.4).astype(np.uint8)

    base     = os.path.splitext(os.path.basename(original_image_path))[0]
    out_path = os.path.join(output_dir, f'{base}_gradcam.jpg')
    Image.fromarray(superimposed).save(out_path)
    return out_path