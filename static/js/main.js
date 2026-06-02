/* ══════════════════════════════════════════════════
   MediPredict AI — Main JavaScript
   ══════════════════════════════════════════════════ */

let lastPredictionData = null;

// ── Symptom Chips ────────────────────────────────────────────────────────────
document.querySelectorAll('.symptom-chip').forEach(chip => {
    chip.addEventListener('click', () => {
        chip.classList.toggle('active');
        const cb = chip.querySelector('input[type="checkbox"]');
        cb.checked = !cb.checked;
        updateSymptomCount();
    });
});

function updateSymptomCount() {
    const count = document.querySelectorAll('.symptom-chip.active').length;
    document.getElementById('symptomCount').textContent = count;
}

// ── Model Selector ───────────────────────────────────────────────────────────
document.querySelectorAll('.model-option').forEach(opt => {
    opt.addEventListener('click', () => {
        document.querySelectorAll('.model-option').forEach(o => o.classList.remove('selected'));
        opt.classList.add('selected');
        const radio = opt.querySelector('input[type="radio"]');
        radio.checked = true;
    });
});

// ── Validation ───────────────────────────────────────────────────────────────
function validateForm() {
    const name = document.getElementById('patientName').value.trim();
    const age = document.getElementById('patientAge').value.trim();
    const dob = document.getElementById('patientDob').value.trim();
    const blood = document.getElementById('bloodGroup').value;
    const phone = document.getElementById('patientPhone').value.trim();
    const symptoms = document.querySelectorAll('.symptom-chip.active');

    if (!name) { showToast('Please enter patient name', 'error'); return false; }
    if (!age || age < 1 || age > 120) { showToast('Please enter a valid age (1–120)', 'error'); return false; }
    if (!dob) { showToast('Please enter date of birth', 'error'); return false; }
    if (!blood) { showToast('Please select blood group', 'error'); return false; }
    if (!phone || phone.replace(/\D/g,'').length < 10) { showToast('Please enter a valid phone number', 'error'); return false; }
    if (symptoms.length === 0) { showToast('Please select at least one symptom', 'error'); return false; }
    return true;
}

// ── Predict Disease ──────────────────────────────────────────────────────────
async function predictDisease() {
    if (!validateForm()) return;

    const selectedSymptoms = Array.from(document.querySelectorAll('.symptom-chip.active'))
        .map(c => c.getAttribute('data-symptom'));
    const model = document.querySelector('input[name="model"]:checked').value;

    const payload = {
        name: document.getElementById('patientName').value.trim(),
        age: document.getElementById('patientAge').value.trim(),
        dob: document.getElementById('patientDob').value.trim(),
        blood_group: document.getElementById('bloodGroup').value,
        phone: document.getElementById('patientPhone').value.trim(),
        symptoms: selectedSymptoms,
        model: model
    };

    showLoading('Analyzing symptoms...');

    try {
        await sleep(600); // slight delay for UX
        setLoadingText('Running ML models...');
        await sleep(600);
        setLoadingText('Generating prediction...');

        const res = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const data = await res.json();

        hideLoading();

        if (data.success) {
            lastPredictionData = data;
            displayResults(data);
        } else {
            showToast('Prediction failed. Please try again.', 'error');
        }
    } catch (err) {
        hideLoading();
        showToast('Server error. Please ensure the Flask app is running.', 'error');
        console.error(err);
    }
}

// ── Display Results ──────────────────────────────────────────────────────────
function displayResults(data) {
    const resultsSection = document.getElementById('resultsSection');
    resultsSection.style.display = 'block';

    // Disease & emoji
    document.getElementById('resultEmoji').textContent = data.info.emoji || '🏥';
    document.getElementById('resultDisease').textContent = data.disease;
    document.getElementById('recommendationText').textContent = data.info.recommendation;

    // Severity badge
    const sev = data.info.severity || '';
    const sevBadge = document.getElementById('severityBadge');
    document.getElementById('severityText').textContent = `Severity: ${sev}`;
    sevBadge.className = 'severity-badge';
    if (sev.toLowerCase().includes('mild') && !sev.toLowerCase().includes('moderate')) {
        sevBadge.classList.add('severity-mild');
    } else if (sev.toLowerCase().includes('high')) {
        sevBadge.classList.add('severity-high');
    } else if (sev.toLowerCase().includes('chronic')) {
        sevBadge.classList.add('severity-chronic');
    } else {
        sevBadge.classList.add('severity-moderate');
    }

    // Models comparison
    const modelNames = {
        random_forest: { label: 'Random Forest', icon: '🌳' },
        svm: { label: 'SVM', icon: '⚡' },
        naive_bayes: { label: 'Naïve Bayes', icon: '🧠' }
    };
    const grid = document.getElementById('modelsGrid');
    grid.innerHTML = '';
    for (const [key, mdata] of Object.entries(data.all_results)) {
        const isActive = key === data.model_used;
        const conf = mdata.confidence || 0;
        const card = document.createElement('div');
        card.className = `model-result-card ${isActive ? 'active-model' : ''}`;
        card.innerHTML = `
            <div class="model-result-icon">${modelNames[key]?.icon || '🤖'}</div>
            <div class="model-result-name">${modelNames[key]?.label || key}</div>
            <div class="model-result-disease">${mdata.disease}</div>
            <div class="confidence-bar-wrap">
                <div class="confidence-bar" style="width: 0%" data-width="${conf}%"></div>
            </div>
            <div class="confidence-text">Confidence: ${conf}%</div>
            ${isActive ? '<div style="margin-top:0.5rem;font-size:0.72rem;color:var(--primary);font-weight:600;">✓ Selected Model</div>' : ''}
        `;
        grid.appendChild(card);
    }

    // Animate confidence bars
    setTimeout(() => {
        document.querySelectorAll('.confidence-bar').forEach(bar => {
            bar.style.width = bar.getAttribute('data-width');
        });
    }, 200);

    // Reset action status
    const status = document.getElementById('actionStatus');
    status.style.display = 'none';

    // Scroll to results
    setTimeout(() => {
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 100);
}

// ── Generate PDF ─────────────────────────────────────────────────────────────
async function generatePDF() {
    if (!lastPredictionData) {
        showToast('No prediction data. Please predict first.', 'error');
        return;
    }

    showLoading('Generating PDF report...');

    try {
        const res = await fetch('/generate_pdf', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(lastPredictionData)
        });

        hideLoading();

        if (res.ok) {
            const blob = await res.blob();
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `MediPredict_${lastPredictionData.patient.name.replace(/\s+/g,'_')}_Report.pdf`;
            a.click();
            URL.revokeObjectURL(url);
            showActionStatus('✅ PDF report downloaded successfully!', 'success');
        } else {
            showActionStatus('❌ PDF generation failed. Please try again.', 'error');
        }
    } catch (err) {
        hideLoading();
        showActionStatus('❌ Error generating PDF: ' + err.message, 'error');
    }
}

// ── Send WhatsApp ─────────────────────────────────────────────────────────────
async function sendWhatsApp() {
    if (!lastPredictionData) {
        showToast('No prediction data. Please predict first.', 'error');
        return;
    }

    showLoading('Scheduling WhatsApp message...');

    try {
        const payload = {
            phone: lastPredictionData.patient.phone,
            patient_name: lastPredictionData.patient.name,
            disease: lastPredictionData.disease,
            recommendation: lastPredictionData.info.recommendation,
            severity: lastPredictionData.info.severity,
            model_used: lastPredictionData.model_used
        };

        const res = await fetch('/send_whatsapp', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        const data = await res.json();
        hideLoading();

        if (data.success) {
            showActionStatus('✅ ' + data.message + '\n\n⚠️ Make sure WhatsApp Web is open in your browser!', 'success');
        } else {
            showActionStatus('⚠️ ' + data.message, 'error');
        }
    } catch (err) {
        hideLoading();
        showActionStatus('❌ Error: ' + err.message, 'error');
    }
}

// ── Reset Form ────────────────────────────────────────────────────────────────
function resetForm() {
    document.getElementById('patientName').value = '';
    document.getElementById('patientAge').value = '';
    document.getElementById('patientDob').value = '';
    document.getElementById('bloodGroup').value = '';
    document.getElementById('patientPhone').value = '';

    document.querySelectorAll('.symptom-chip').forEach(chip => {
        chip.classList.remove('active');
        chip.querySelector('input').checked = false;
    });

    document.getElementById('symptomCount').textContent = '0';
    document.getElementById('resultsSection').style.display = 'none';
    lastPredictionData = null;

    // Reset to first model
    document.querySelectorAll('.model-option').forEach(o => o.classList.remove('selected'));
    document.querySelectorAll('.model-option')[0].classList.add('selected');
    document.querySelectorAll('input[name="model"]')[0].checked = true;

    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// ── Loading Helpers ───────────────────────────────────────────────────────────
function showLoading(text = 'Processing...') {
    document.getElementById('loadingText').textContent = text;
    document.getElementById('loadingOverlay').style.display = 'flex';
}
function setLoadingText(text) {
    document.getElementById('loadingText').textContent = text;
}
function hideLoading() {
    document.getElementById('loadingOverlay').style.display = 'none';
}

// ── Toast Notification ────────────────────────────────────────────────────────
function showToast(msg, type = 'info') {
    const existing = document.querySelector('.toast');
    if (existing) existing.remove();

    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.innerHTML = `<i class="fas fa-${type === 'error' ? 'exclamation-circle' : 'info-circle'}"></i> ${msg}`;
    toast.style.cssText = `
        position: fixed; bottom: 2rem; right: 2rem; z-index: 9999;
        background: ${type === 'error' ? 'rgba(239,68,68,0.9)' : 'rgba(56,189,248,0.9)'};
        color: white; padding: 0.85rem 1.4rem; border-radius: 12px;
        font-size: 0.88rem; font-weight: 600; font-family: Inter, sans-serif;
        backdrop-filter: blur(16px); box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        display: flex; align-items: center; gap: 0.5rem;
        animation: slideInRight 0.3s ease-out;
        max-width: 340px;
    `;
    document.body.appendChild(toast);
    setTimeout(() => toast.remove(), 4000);
}

function showActionStatus(msg, type) {
    const status = document.getElementById('actionStatus');
    status.textContent = msg;
    status.className = `action-status ${type}`;
    status.style.display = 'block';
    status.style.whiteSpace = 'pre-line';
    setTimeout(() => { status.style.display = 'none'; }, 8000);
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

// ── Keypress Enter support ────────────────────────────────────────────────────
document.addEventListener('keydown', e => {
    if (e.key === 'Enter' && e.ctrlKey) predictDisease();
});
