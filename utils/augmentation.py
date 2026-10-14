"""
augmentation.py
================
Görüntü dönüşüm / augmentation işlemleri.
Her fonksiyon bir PIL Image alır, dönüştürülmüş PIL Image döner.
"""

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import io


# ── Gürültü ────────────────────────────────────────────────

def gaussian_noise(img: Image.Image, std: float = 25.0) -> Image.Image:
    """Gaussian gürültüsü ekler."""
    arr = np.array(img, dtype=np.float32)
    noise = np.random.normal(0, std, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def salt_pepper_noise(img: Image.Image, amount: float = 0.05) -> Image.Image:
    """Salt & Pepper gürültüsü ekler."""
    arr = np.array(img).copy()
    total = arr.shape[0] * arr.shape[1]
    n = int(total * amount)

    # Salt (beyaz)
    ys = np.random.randint(0, arr.shape[0], n)
    xs = np.random.randint(0, arr.shape[1], n)
    arr[ys, xs] = 255

    # Pepper (siyah)
    ys = np.random.randint(0, arr.shape[0], n)
    xs = np.random.randint(0, arr.shape[1], n)
    arr[ys, xs] = 0

    return Image.fromarray(arr)


def speckle_noise(img: Image.Image, intensity: float = 0.15) -> Image.Image:
    """Speckle (çarpımsal) gürültüsü ekler."""
    arr = np.array(img, dtype=np.float32)
    noise = np.random.randn(*arr.shape) * intensity
    arr = np.clip(arr + arr * noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


# ── Bulanıklaştırma ────────────────────────────────────────

def gaussian_blur(img: Image.Image, radius: float = 3.0) -> Image.Image:
    """Gaussian blur uygular."""
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def motion_blur(img: Image.Image, size: int = 15, angle: int = 0) -> Image.Image:
    """Yatay hareket bulanıklığı simüle eder."""
    import cv2
    arr = np.array(img)
    kernel = np.zeros((size, size))
    kernel[size // 2, :] = np.ones(size) / size
    # Açıya göre döndür
    M = cv2.getRotationMatrix2D((size / 2 - 0.5, size / 2 - 0.5), angle, 1)
    kernel = cv2.warpAffine(kernel, M, (size, size))
    blurred = cv2.filter2D(arr, -1, kernel)
    return Image.fromarray(blurred)


def median_blur(img: Image.Image, size: int = 5) -> Image.Image:
    """Medyan filtresi uygular."""
    import cv2
    arr = np.array(img)
    blurred = cv2.medianBlur(arr, size if size % 2 == 1 else size + 1)
    return Image.fromarray(blurred)


# ── Parlaklık / Kontrast ───────────────────────────────────

def adjust_brightness(img: Image.Image, factor: float = 1.5) -> Image.Image:
    """Parlaklık ayarı. 1.0 = orijinal, >1 = daha parlak."""
    return ImageEnhance.Brightness(img).enhance(factor)


def adjust_contrast(img: Image.Image, factor: float = 2.0) -> Image.Image:
    """Kontrast ayarı. 1.0 = orijinal."""
    return ImageEnhance.Contrast(img).enhance(factor)


def adjust_saturation(img: Image.Image, factor: float = 0.0) -> Image.Image:
    """Doygunluk. 0.0 = gri tonlama, 1.0 = orijinal."""
    return ImageEnhance.Color(img).enhance(factor)


def gamma_correction(img: Image.Image, gamma: float = 0.5) -> Image.Image:
    """Gamma düzeltme. <1 = karartır, >1 = aydınlatır."""
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.power(arr, gamma)
    arr = (arr * 255).astype(np.uint8)
    return Image.fromarray(arr)


# ── JPEG Sıkıştırma ────────────────────────────────────────

def jpeg_compression(img: Image.Image, quality: int = 10) -> Image.Image:
    """JPEG sıkıştırma artefaktı ekler. quality 1-95."""
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=quality)
    buf.seek(0)
    return Image.open(buf).copy()


# ── Geometrik ─────────────────────────────────────────────

def random_crop_resize(img: Image.Image, crop_pct: float = 0.85) -> Image.Image:
    """Rastgele kırp ve orijinal boyuta geri büyüt."""
    w, h = img.size
    new_w = int(w * crop_pct)
    new_h = int(h * crop_pct)
    left  = np.random.randint(0, w - new_w)
    top   = np.random.randint(0, h - new_h)
    cropped = img.crop((left, top, left + new_w, top + new_h))
    return cropped.resize((w, h), Image.LANCZOS)


def horizontal_flip(img: Image.Image) -> Image.Image:
    return img.transpose(Image.FLIP_LEFT_RIGHT)


def rotate(img: Image.Image, degrees: float = 15.0) -> Image.Image:
    return img.rotate(degrees, expand=False, fillcolor=(128, 128, 128))


# ── Dönüşüm kataloğu (arayüze gönderilir) ─────────────────

AUGMENTATIONS = {
    # id: (görünen ad, kategori, fonksiyon, varsayılan parametre açıklaması)
    'gaussian_noise':     ('Gaussian Gürültü',        'Gürültü',        gaussian_noise,       'std=25'),
    'salt_pepper':        ('Salt & Pepper Gürültü',   'Gürültü',        salt_pepper_noise,    'amount=0.05'),
    'speckle':            ('Speckle Gürültü',         'Gürültü',        speckle_noise,        'intensity=0.15'),
    'gaussian_blur':      ('Gaussian Bulanık',        'Bulanıklaştırma', gaussian_blur,        'radius=3'),
    'motion_blur':        ('Hareket Bulanıklığı',     'Bulanıklaştırma', motion_blur,          'size=15'),
    'median_blur':        ('Medyan Filtre',           'Bulanıklaştırma', median_blur,          'size=5'),
    'brightness_up':      ('Parlaklık Artır',         'Parlaklık/Kontrast', lambda img: adjust_brightness(img, 1.6), ''),
    'brightness_down':    ('Parlaklık Azalt',         'Parlaklık/Kontrast', lambda img: adjust_brightness(img, 0.5), ''),
    'contrast_up':        ('Kontrast Artır',          'Parlaklık/Kontrast', lambda img: adjust_contrast(img, 2.0), ''),
    'grayscale':          ('Gri Tonlama',             'Parlaklık/Kontrast', lambda img: adjust_saturation(img, 0.0), ''),
    'gamma_dark':         ('Gamma Karartma',          'Parlaklık/Kontrast', lambda img: gamma_correction(img, 0.4), ''),
    'gamma_light':        ('Gamma Aydınlatma',        'Parlaklık/Kontrast', lambda img: gamma_correction(img, 2.0), ''),
    'jpeg_low':           ('JPEG Sıkıştırma (Düşük)', 'Sıkıştırma',     lambda img: jpeg_compression(img, 10), ''),
    'jpeg_mid':           ('JPEG Sıkıştırma (Orta)',  'Sıkıştırma',     lambda img: jpeg_compression(img, 35), ''),
    'crop_resize':        ('Kırp & Yeniden Boyutla',  'Geometrik',      random_crop_resize,   ''),
    'flip':               ('Yatay Çevir',             'Geometrik',      horizontal_flip,      ''),
    'rotate':             ('Döndür (15°)',            'Geometrik',      lambda img: rotate(img, 15), ''),
}


def apply_augmentation(img: Image.Image, aug_id: str) -> Image.Image:
    """Verilen aug_id'ye göre dönüşümü uygular."""
    if aug_id not in AUGMENTATIONS:
        raise ValueError(f'Bilinmeyen dönüşüm: {aug_id}')
    _, _, fn, _ = AUGMENTATIONS[aug_id]
    return fn(img.convert('RGB'))


def get_catalog():
    """Frontend'e gönderilecek katalog listesi."""
    catalog = {}
    for aug_id, (name, category, _, hint) in AUGMENTATIONS.items():
        if category not in catalog:
            catalog[category] = []
        catalog[category].append({'id': aug_id, 'name': name, 'hint': hint})
    return catalog