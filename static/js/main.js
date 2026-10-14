/* ─── AI Dedektör — main.js ──────────────────── */

const uploadZone    = document.getElementById('uploadZone');
const fileInput     = document.getElementById('fileInput');
const browseBtn     = document.getElementById('browseBtn');
const uploadContent = document.getElementById('uploadContent');
const uploadPreview = document.getElementById('uploadPreview');
const previewImg    = document.getElementById('previewImg');
const removeBtn     = document.getElementById('removeBtn');
const predictBtn    = document.getElementById('predictBtn');
const gradcamToggle = document.getElementById('gradcamToggle');
const gradcamPanel  = document.getElementById('gradcamPanel');
const gradcamImg    = document.getElementById('gradcamImg');
const compareRow    = document.getElementById('compareRow');
const compareOrig   = document.getElementById('compareOrig');
const compareAug    = document.getElementById('compareAug');
const compareAugLbl = document.getElementById('compareAugLabel');
const clearAugBtn   = document.getElementById('clearAugBtn');

let selectedFile    = null;
let origDataURL     = null;

/* ── Dosya seçimi ── */
browseBtn.addEventListener('click', e => { e.stopPropagation(); fileInput.click(); });
uploadZone.addEventListener('click', () => { if (!selectedFile) fileInput.click(); });
fileInput.addEventListener('change', () => { if (fileInput.files[0]) handleFile(fileInput.files[0]); });

uploadZone.addEventListener('dragover',  e => { e.preventDefault(); uploadZone.classList.add('drag-over'); });
uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('drag-over'));
uploadZone.addEventListener('drop', e => {
  e.preventDefault(); uploadZone.classList.remove('drag-over');
  if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
});

removeBtn.addEventListener('click', e => { e.stopPropagation(); clearFile(); });

function handleFile(file) {
  if (!file.type.startsWith('image/')) { alert('Lütfen bir görsel dosyası seçin.'); return; }
  selectedFile = file;
  const reader = new FileReader();
  reader.onload = ev => {
    origDataURL = ev.target.result;
    previewImg.src = origDataURL;
    uploadContent.style.display = 'none';
    uploadPreview.style.display = 'block';
    predictBtn.disabled = false;
    clearResults();
    compareRow.style.display = 'none';
    gradcamPanel.style.display = 'none';
  };
  reader.readAsDataURL(file);
}

function clearFile() {
  selectedFile = null; origDataURL = null;
  fileInput.value = '';
  uploadContent.style.display = 'flex';
  uploadPreview.style.display = 'none';
  predictBtn.disabled = true;
  clearResults();
  compareRow.style.display = 'none';
  gradcamPanel.style.display = 'none';
}

/* ── Dönüşüm temizle ── */
clearAugBtn.addEventListener('click', () => {
  document.querySelectorAll('input[name="augmentation"]').forEach(r => r.checked = false);
  compareRow.style.display = 'none';
});

/* ── Dönüşüm chip seçilince önizleme ── */
document.querySelectorAll('input[name="augmentation"]').forEach(radio => {
  radio.addEventListener('change', async () => {
    if (!selectedFile) return;
    const fd = new FormData();
    fd.append('image', selectedFile);
    fd.append('aug_id', radio.value);
    try {
      const res  = await fetch('/api/augment', { method: 'POST', body: fd });
      const data = await res.json();
      if (data.error) { console.error(data.error); return; }

      compareOrig.src   = origDataURL;
      compareAug.src    = data.image;
      compareAugLbl.textContent = radio.closest('label').querySelector('span').textContent;
      compareRow.style.display  = 'flex';
    } catch (e) { console.error(e); }
  });
});

/* ── Tahmin ── */
predictBtn.addEventListener('click', async () => {
  if (!selectedFile) return;
  const modelId = document.querySelector('input[name="model"]:checked')?.value;
  if (!modelId) { alert('Lütfen bir model seçin.'); return; }

  const augId = document.querySelector('input[name="augmentation"]:checked')?.value || '';

  document.getElementById('resultsPlaceholder').style.display = 'none';
  document.getElementById('resultsContent').style.display     = 'none';
  document.getElementById('loadingState').style.display       = 'block';
  gradcamPanel.style.display = 'none';

  const fd = new FormData();
  fd.append('image',    selectedFile);
  fd.append('model_id', modelId);
  fd.append('gradcam',  gradcamToggle.checked ? 'true' : 'false');
  fd.append('aug_id',   augId);

  try {
    const res  = await fetch('/api/predict', { method: 'POST', body: fd });
    const data = await res.json();
    document.getElementById('loadingState').style.display = 'none';

    if (data.error) { alert('Hata: ' + data.error); clearResults(); return; }

    renderResults(data);

    if (data.gradcam_path) {
      gradcamImg.src = data.gradcam_path;
      gradcamPanel.style.display = 'block';
    }
  } catch (err) {
    document.getElementById('loadingState').style.display = 'none';
    alert('Sunucu hatası: ' + err.message);
    clearResults();
  }
});

/* ── Sonuçları render et ── */
function renderResults(data) {
  document.getElementById('resultsPlaceholder').style.display = 'none';
  document.getElementById('loadingState').style.display       = 'none';
  const content = document.getElementById('resultsContent');
  content.style.display = 'block';

  document.getElementById('resultMeta').textContent =
    `${data.model_name}  ·  ${data.framework.toUpperCase()}`;

  const barsEl = document.getElementById('predictionBars');
  barsEl.innerHTML = '';
  data.predictions.forEach((pred, idx) => {
    const item = document.createElement('div');
    item.className = `pred-item${idx === 0 ? ' top' : ''}`;
    item.innerHTML = `
      <div class="pred-header">
        <span class="pred-label">${pred.label}</span>
        <span class="pred-pct">%${pred.probability.toFixed(1)}</span>
      </div>
      <div class="pred-bar-bg">
        <div class="pred-bar-fill" style="width:0%" data-w="${pred.probability}%"></div>
      </div>`;
    barsEl.appendChild(item);
  });
  requestAnimationFrame(() => {
    document.querySelectorAll('.pred-bar-fill').forEach(b => b.style.width = b.dataset.w);
  });

  const top      = data.predictions[0];
  const isReal   = top.label.toLowerCase().includes('gerçek');
  const verdict  = document.getElementById('resultVerdict');
  verdict.className = `result-verdict ${isReal ? 'verdict-real' : 'verdict-fake'}`;
  verdict.textContent = isReal
    ? `✓ Büyük olasılıkla gerçek (%${top.probability.toFixed(1)})`
    : `✗ Büyük olasılıkla yapay (%${top.probability.toFixed(1)})`;
}

function clearResults() {
  document.getElementById('resultsPlaceholder').style.display = 'block';
  document.getElementById('resultsContent').style.display     = 'none';
  document.getElementById('loadingState').style.display       = 'none';
}

/* ─── Accordion ── */
const augBtn  = document.getElementById('augAccordionBtn');
const augBody = document.getElementById('augAccordionBody');
augBtn.addEventListener('click', () => {
  const isOpen = augBody.style.display !== 'none';
  augBody.style.display = isOpen ? 'none' : 'block';
  augBtn.classList.toggle('open', !isOpen);
});