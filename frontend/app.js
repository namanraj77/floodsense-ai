/* =============================================
   FloodSense AI - Main JavaScript Application
   ============================================= */

const API_BASE = 'http://localhost:5000/api';

// Global state
let radarChart = null;
let performanceChart = null;
let distributionChart = null;
let featureChart = null;
let scatterChart = null;
let metricsData = null;

/* ==================== INIT ==================== */
document.addEventListener('DOMContentLoaded', () => {
  createParticles();
  checkServerStatus();
  setInterval(checkServerStatus, 15000);
  loadAnalytics();

  // Navbar scroll effect
  window.addEventListener('scroll', () => {
    const navbar = document.getElementById('navbar');
    if (window.scrollY > 20) navbar.classList.add('scrolled');
    else navbar.classList.remove('scrolled');
  });
});

/* ==================== PARTICLES ==================== */
function createParticles() {
  const container = document.getElementById('particles');
  for (let i = 0; i < 25; i++) {
    const p = document.createElement('div');
    p.className = 'particle';
    const size = Math.random() * 6 + 2;
    p.style.cssText = `
      width: ${size}px; height: ${size}px;
      left: ${Math.random() * 100}%;
      animation-duration: ${Math.random() * 20 + 15}s;
      animation-delay: ${Math.random() * 20}s;
      opacity: ${Math.random() * 0.5};
    `;
    container.appendChild(p);
  }
}

/* ==================== NAVIGATION ==================== */
function showSection(id) {
  // Hide all sections
  document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
  document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));

  // Show target
  const section = document.getElementById(id);
  if (section) section.classList.add('active');

  // Activate nav link
  const link = document.querySelector(`.nav-link[href="#${id}"]`);
  if (link) link.classList.add('active');

  // Load charts on analytics tab
  if (id === 'analytics') renderAnalyticsCharts();
  
  // Scroll to top
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

/* ==================== STATUS CHECK ==================== */
async function checkServerStatus() {
  const dot = document.getElementById('statusDot');
  const text = document.getElementById('statusText');
  try {
    const res = await fetch(`${API_BASE}/status`, { signal: AbortSignal.timeout(5000) });
    if (res.ok) {
      const data = await res.json();
      dot.className = 'status-dot online';
      text.textContent = `${data.models_count} Models Online`;
      if (data.training_complete) loadDashboardMetrics();
    } else {
      setOffline(dot, text);
    }
  } catch (e) {
    setOffline(dot, text);
  }
}

function setOffline(dot, text) {
  dot.className = 'status-dot offline';
  text.textContent = 'Server Offline';
}

/* ==================== LOAD METRICS ==================== */
async function loadDashboardMetrics() {
  try {
    const res = await fetch(`${API_BASE}/metrics`);
    if (!res.ok) return;
    const data = await res.json();
    metricsData = data;

    const m = data.metrics;

    // Update dashboard stat card
    if (m.ensemble) {
      document.getElementById('ensembleAcc').textContent = m.ensemble.r2.toFixed(4);
    }

    // Update model cards
    const modelMap = {
      lstm: 'lstm', random_forest: 'rf', xgboost: 'xgb', svm: 'svm'
    };
    for (const [key, id] of Object.entries(modelMap)) {
      if (m[key]) {
        const el = document.getElementById(`${id}-stats`);
        if (el) {
          el.querySelector('span:nth-child(1) strong').textContent = m[key].r2.toFixed(4);
          el.querySelector('span:nth-child(2) strong').textContent = m[key].rmse.toFixed(4);
        }
        // Detail metrics
        updateDetailMetrics(id, m[key]);
      }
    }

    // Fill metrics table
    fillMetricsTable(m);

    // Update dataset info
    const di = data.dataset_info;
    if (di) {
      document.getElementById('totalSamples').textContent = di.n_samples.toLocaleString();
    }

  } catch (e) {
    console.warn('Metrics load failed:', e);
  }
}

function updateDetailMetrics(id, metrics) {
  const el = document.getElementById(`${id}-detail-metrics`);
  if (!el) return;
  const items = el.querySelectorAll('.dm-item strong');
  if (items[0]) items[0].textContent = metrics.r2.toFixed(4);
  if (items[1]) items[1].textContent = metrics.rmse.toFixed(4);
  if (items[2]) items[2].textContent = metrics.mae.toFixed(4);
}

function fillMetricsTable(metrics) {
  const tbody = document.getElementById('metricsBody');
  if (!tbody) return;

  const rows = [
    { key: 'lstm',          label: 'LSTM',          icon: '🔴', badge: 'lstm' },
    { key: 'random_forest', label: 'Random Forest',  icon: '🌲', badge: 'rf' },
    { key: 'xgboost',       label: 'XGBoost',        icon: '⚡', badge: 'xgb' },
    { key: 'svm',           label: 'SVM',            icon: '🔵', badge: 'svm' },
    { key: 'ensemble',      label: 'Ensemble Avg',   icon: '⭐', badge: 'ensemble' },
  ];

  tbody.innerHTML = rows.map(r => {
    const m = metrics[r.key];
    if (!m) return '';
    return `
      <tr>
        <td><span class="model-badge badge-${r.badge}">${r.icon} ${r.label}</span></td>
        <td><strong>${m.r2.toFixed(4)}</strong></td>
        <td>${m.rmse.toFixed(5)}</td>
        <td>${m.mae.toFixed(5)}</td>
        <td>${m.accuracy ? m.accuracy.toFixed(1) + '%' : '—'}</td>
        <td><span class="status-pill pill-trained">✅ Trained</span></td>
      </tr>
    `;
  }).join('');
}

/* ==================== ANALYTICS CHARTS ==================== */
async function loadAnalytics() {
  try {
    const [metricsRes, distRes, sampleRes] = await Promise.all([
      fetch(`${API_BASE}/metrics`).catch(() => null),
      fetch(`${API_BASE}/historical-data`).catch(() => null),
      fetch(`${API_BASE}/sample-predictions`).catch(() => null),
    ]);

    if (metricsRes?.ok) {
      const data = await metricsRes.json();
      metricsData = data;
      loadDashboardMetrics();
    }
    if (distRes?.ok) {
      const distData = await distRes.json();
      window._distData = distData;
    }
    if (sampleRes?.ok) {
      const sampleData = await sampleRes.json();
      window._sampleData = sampleData;
    }
  } catch (e) {
    console.warn('Analytics load error:', e);
  }
}

function renderAnalyticsCharts() {
  const data = metricsData;
  if (!data) {
    showToast('Start the backend server and train models first', 'info');
    return;
  }

  renderPerformanceChart(data.metrics);
  renderFeatureChart(data.feature_importance);
  if (window._distData) renderDistributionChart(window._distData);
  if (window._sampleData) renderScatterChart(window._sampleData);
}

function renderPerformanceChart(metrics) {
  const ctx = document.getElementById('performanceChart');
  if (!ctx || !metrics) return;
  if (performanceChart) performanceChart.destroy();

  const labels = ['LSTM', 'Random Forest', 'XGBoost', 'SVM', 'Ensemble'];
  const keys = ['lstm', 'random_forest', 'xgboost', 'svm', 'ensemble'];
  const colors = ['#f87171', '#34d399', '#fbbf24', '#818cf8', '#38bdf8'];

  performanceChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [
        {
          label: 'R² Score',
          data: keys.map(k => metrics[k]?.r2?.toFixed(4) || 0),
          backgroundColor: colors.map(c => c + '33'),
          borderColor: colors,
          borderWidth: 2,
          borderRadius: 6,
        },
        {
          label: 'RMSE (×10)',
          data: keys.map(k => ((metrics[k]?.rmse || 0) * 10).toFixed(4)),
          backgroundColor: colors.map(c => c + '18'),
          borderColor: colors.map(c => c + '88'),
          borderWidth: 2,
          borderRadius: 6,
          borderDash: [5, 5],
        },
      ]
    },
    options: {
      responsive: true,
      plugins: {
        legend: { labels: { color: '#94a3b8', font: { size: 12 } } },
        tooltip: { backgroundColor: '#0f172a', borderColor: '#1e293b', borderWidth: 1, titleColor: '#f8fafc', bodyColor: '#94a3b8' },
      },
      scales: {
        x: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.04)' } },
        y: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.04)' }, beginAtZero: true },
      }
    }
  });
}

function renderDistributionChart(distData) {
  const ctx = document.getElementById('distributionChart');
  if (!ctx || !distData?.distribution) return;
  if (distributionChart) distributionChart.destroy();

  const dist = distData.distribution;
  distributionChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: dist.labels || [],
      datasets: [{
        label: 'Number of Samples',
        data: dist.counts || [],
        backgroundColor: [
          '#22c55e55', '#86efac55', '#fbbf2455', '#f9731655', '#ef444455', '#dc262655', '#7f1d1d55'
        ],
        borderColor: ['#22c55e', '#86efac', '#fbbf24', '#f97316', '#ef4444', '#dc2626', '#7f1d1d'],
        borderWidth: 2,
        borderRadius: 6,
      }]
    },
    options: {
      responsive: true,
      plugins: {
        legend: { display: false },
        tooltip: { backgroundColor: '#0f172a', borderColor: '#1e293b', borderWidth: 1, titleColor: '#f8fafc', bodyColor: '#94a3b8' },
      },
      scales: {
        x: { ticks: { color: '#94a3b8', font: { size: 11 } }, grid: { color: 'rgba(255,255,255,0.04)' } },
        y: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.04)' } },
      }
    }
  });
}

function renderFeatureChart(featureImportance) {
  const ctx = document.getElementById('featureChart');
  if (!ctx || !featureImportance) return;
  if (featureChart) featureChart.destroy();

  const entries = Object.entries(featureImportance).slice(0, 10);
  const labels = entries.map(([k]) => k.replace(/([A-Z])/g, ' $1').trim());
  const values = entries.map(([, v]) => (v * 100).toFixed(2));

  const gradient = ctx.getContext('2d');
  const grad = gradient.createLinearGradient(0, 0, 400, 0);
  grad.addColorStop(0, '#38bdf8');
  grad.addColorStop(1, '#a78bfa');

  featureChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'Importance %',
        data: values,
        backgroundColor: grad,
        borderRadius: 6,
      }]
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      plugins: {
        legend: { display: false },
        tooltip: { backgroundColor: '#0f172a', borderColor: '#1e293b', borderWidth: 1, titleColor: '#f8fafc', bodyColor: '#94a3b8' },
      },
      scales: {
        x: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.04)' } },
        y: { ticks: { color: '#94a3b8', font: { size: 11 } }, grid: { color: 'rgba(255,255,255,0.04)' } },
      }
    }
  });
}

function renderScatterChart(sampleData) {
  const ctx = document.getElementById('scatterChart');
  if (!ctx || !sampleData?.sample_predictions) return;
  if (scatterChart) scatterChart.destroy();

  const sp = sampleData.sample_predictions;
  const actual = sp.actual || [];
  const predictions = {
    'LSTM': sp.lstm,
    'Random Forest': sp.rf,
    'XGBoost': sp.xgb,
    'SVM': sp.svm,
    'Ensemble': sp.ensemble
  };
  const colors = ['#f87171', '#34d399', '#fbbf24', '#818cf8', '#38bdf8'];

  const datasets = Object.entries(predictions).map(([label, preds], i) => ({
    label,
    data: actual.map((a, j) => ({ x: a, y: preds?.[j] || 0 })),
    backgroundColor: colors[i] + '66',
    borderColor: colors[i],
    borderWidth: 1,
    pointRadius: 3,
    pointHoverRadius: 5,
    showLine: false,
  }));

  // Perfect prediction line
  const minV = Math.min(...actual);
  const maxV = Math.max(...actual);
  datasets.push({
    label: 'Perfect Prediction',
    data: [{ x: minV, y: minV }, { x: maxV, y: maxV }],
    borderColor: '#ffffff33',
    borderWidth: 1,
    borderDash: [5, 5],
    showLine: true,
    pointRadius: 0,
    fill: false,
  });

  scatterChart = new Chart(ctx, {
    type: 'scatter',
    data: { datasets },
    options: {
      responsive: true,
      plugins: {
        legend: { labels: { color: '#94a3b8', font: { size: 12 } } },
        tooltip: { backgroundColor: '#0f172a', borderColor: '#1e293b', borderWidth: 1, titleColor: '#f8fafc', bodyColor: '#94a3b8' },
      },
      scales: {
        x: { title: { display: true, text: 'Actual Probability', color: '#94a3b8' }, ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.04)' } },
        y: { title: { display: true, text: 'Predicted Probability', color: '#94a3b8' }, ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.04)' } },
      }
    }
  });
}

/* ==================== SLIDER UPDATES ==================== */
function updateVal(input) {
  const el = document.getElementById(`val-${input.id}`);
  if (el) el.textContent = input.value;
  updateRangeColor(input);
}

function updateRangeColor(input) {
  const pct = (input.value / input.max) * 100;
  input.style.background = `linear-gradient(to right, #38bdf8 ${pct}%, #1e293b ${pct}%)`;
}

// Initialize all sliders
document.querySelectorAll('input[type="range"]').forEach(input => {
  updateRangeColor(input);
  input.addEventListener('input', () => updateRangeColor(input));
});

/* ==================== PRESETS ==================== */
const PRESETS = {
  low: {
    MonsoonIntensity: 3, TopographyDrainage: 8, RiverManagement: 8,
    Deforestation: 2, Urbanization: 2, ClimateChange: 2, DamsQuality: 9,
    Siltation: 2, AgriculturalPractices: 2, Encroachments: 2,
    IneffectiveDisasterPreparedness: 2, DrainageSystems: 9,
    CoastalVulnerability: 2, Landslides: 2, Watersheds: 2,
    DeterioratingInfrastructure: 2, PopulationScore: 2, WetlandLoss: 2,
    InadequatePlanning: 2, PoliticalFactors: 2
  },
  medium: {
    MonsoonIntensity: 6, TopographyDrainage: 6, RiverManagement: 5,
    Deforestation: 5, Urbanization: 6, ClimateChange: 6, DamsQuality: 5,
    Siltation: 5, AgriculturalPractices: 5, Encroachments: 5,
    IneffectiveDisasterPreparedness: 5, DrainageSystems: 5,
    CoastalVulnerability: 5, Landslides: 5, Watersheds: 5,
    DeterioratingInfrastructure: 5, PopulationScore: 5, WetlandLoss: 5,
    InadequatePlanning: 5, PoliticalFactors: 5
  },
  high: {
    MonsoonIntensity: 12, TopographyDrainage: 2, RiverManagement: 2,
    Deforestation: 10, Urbanization: 12, ClimateChange: 11, DamsQuality: 2,
    Siltation: 10, AgriculturalPractices: 9, Encroachments: 10,
    IneffectiveDisasterPreparedness: 11, DrainageSystems: 2,
    CoastalVulnerability: 10, Landslides: 10, Watersheds: 2,
    DeterioratingInfrastructure: 10, PopulationScore: 11, WetlandLoss: 10,
    InadequatePlanning: 10, PoliticalFactors: 10
  }
};

function loadPreset(type) {
  const preset = PRESETS[type];
  if (!preset) return;
  Object.entries(preset).forEach(([key, val]) => {
    const input = document.getElementById(key);
    if (input) {
      input.value = val;
      updateVal(input);
    }
  });
  showToast(`Loaded "${type}" risk preset`, 'info');
}

function resetForm() {
  document.querySelectorAll('.form-categories input[type="range"]').forEach(input => {
    input.value = 5;
    updateVal(input);
  });
  showToast('Form reset to default values', 'info');
}

/* ==================== PREDICTION ==================== */
async function runPrediction() {
  const btn = document.getElementById('predictBtn');
  const btnText = btn.querySelector('.btn-text');
  const btnLoader = btn.querySelector('.btn-loader');

  // Collect input values
  const inputData = {};
  const features = [
    'MonsoonIntensity', 'TopographyDrainage', 'RiverManagement',
    'Deforestation', 'Urbanization', 'ClimateChange', 'DamsQuality',
    'Siltation', 'AgriculturalPractices', 'Encroachments',
    'IneffectiveDisasterPreparedness', 'DrainageSystems',
    'CoastalVulnerability', 'Landslides', 'Watersheds',
    'DeterioratingInfrastructure', 'PopulationScore', 'WetlandLoss',
    'InadequatePlanning', 'PoliticalFactors'
  ];

  features.forEach(f => {
    const el = document.getElementById(f);
    if (el) inputData[f] = parseFloat(el.value);
  });

  // Loading state
  btn.disabled = true;
  btnText.style.display = 'none';
  btnLoader.style.display = 'flex';

  try {
    const res = await fetch(`${API_BASE}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(inputData),
      signal: AbortSignal.timeout(30000)
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.error || 'Prediction failed');
    }

    const result = await res.json();
    displayResults(result, inputData);
    showToast('Prediction complete! ✅', 'success');

  } catch (e) {
    if (e.name === 'TimeoutError') {
      showToast('Request timed out. Is the server running?', 'error');
    } else {
      showToast(`Error: ${e.message}`, 'error');
    }
    console.error('Prediction error:', e);
  } finally {
    btn.disabled = false;
    btnText.style.display = 'flex';
    btnLoader.style.display = 'none';
  }
}

/* ==================== DISPLAY RESULTS ==================== */
function displayResults(result, inputData) {
  document.getElementById('resultsPlaceholder').style.display = 'none';
  document.getElementById('resultsContent').style.display = 'flex';

  const preds = result.predictions;
  const risk = result.risk_assessment;
  const ensemble = preds.ensemble_average;
  const pct = preds.flood_probability_percent;

  // Update gauge
  updateGauge(pct);

  // Risk badge
  document.getElementById('riskIcon').textContent = risk.icon;
  document.getElementById('riskLevel').textContent = risk.level + ' Risk';
  const badge = document.getElementById('riskBadge');
  badge.style.borderColor = risk.color + '44';
  badge.style.background = risk.color + '15';

  // Confidence
  const conf = risk.confidence;
  document.getElementById('confFill').style.width = conf + '%';
  document.getElementById('confValue').textContent = conf.toFixed(1) + '%';

  // Model bars
  const indiv = preds.individual;
  setTimeout(() => {
    setBar('lstm', (indiv['LSTM'] || 0) * 100);
    setBar('rf',   (indiv['Random Forest'] || 0) * 100);
    setBar('xgb',  (indiv['XGBoost'] || 0) * 100);
    setBar('svm',  (indiv['SVM'] || 0) * 100);
    setBar('ensemble', pct);
  }, 100);

  // Recommendations
  const recsEl = document.getElementById('recsList');
  recsEl.innerHTML = result.recommendations.map(r =>
    `<div class="rec-item">${r}</div>`
  ).join('');

  // Radar chart
  renderRadarChart(inputData);
}

function updateGauge(percent) {
  const gaugeEl = document.getElementById('gaugeFill');
  const percentEl = document.getElementById('gaugePercent');

  const pct = Math.min(100, Math.max(0, percent));
  const totalLength = 251.3; // Half-circle circumference
  const offset = totalLength - (totalLength * pct / 100);

  const color = pct < 30 ? '#22c55e' :
                pct < 45 ? '#86efac' :
                pct < 55 ? '#fbbf24' :
                pct < 70 ? '#f97316' : '#ef4444';

  gaugeEl.style.strokeDashoffset = offset;
  gaugeEl.style.stroke = color;

  // Animate counter
  animateCounter(percentEl, 0, pct, 1000, v => v.toFixed(1) + '%');
}

function animateCounter(el, from, to, duration, formatter) {
  const start = performance.now();
  const update = (time) => {
    const elapsed = time - start;
    const progress = Math.min(elapsed / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3);
    el.textContent = formatter(from + (to - from) * eased);
    if (progress < 1) requestAnimationFrame(update);
  };
  requestAnimationFrame(update);
}

function setBar(id, percent) {
  const bar = document.getElementById(`bar-${id}`);
  const val = document.getElementById(`val-${id}`);
  if (bar) bar.style.width = Math.min(100, percent) + '%';
  if (val) val.textContent = percent.toFixed(1) + '%';
}

/* ==================== RADAR CHART ==================== */
function renderRadarChart(inputData) {
  const ctx = document.getElementById('radarChart');
  if (!ctx) return;
  if (radarChart) radarChart.destroy();

  const labels = ['Monsoon', 'Climate', 'Deforestation', 'Urbanization', 'Infrastructure', 'Drainage', 'Disaster Prep', 'Population'];
  const fieldMap = [
    'MonsoonIntensity', 'ClimateChange', 'Deforestation',
    'Urbanization', 'DeterioratingInfrastructure', 'DrainageSystems',
    'IneffectiveDisasterPreparedness', 'PopulationScore'
  ];
  const values = fieldMap.map(f => ((inputData[f] || 0) / 15) * 100);

  radarChart = new Chart(ctx, {
    type: 'radar',
    data: {
      labels,
      datasets: [{
        label: 'Risk Factors',
        data: values,
        backgroundColor: 'rgba(56,189,248,0.15)',
        borderColor: '#38bdf8',
        borderWidth: 2,
        pointBackgroundColor: '#38bdf8',
        pointBorderColor: '#0f172a',
        pointBorderWidth: 2,
        pointRadius: 5,
      }]
    },
    options: {
      responsive: true,
      scales: {
        r: {
          beginAtZero: true,
          max: 100,
          ticks: { display: false },
          grid: { color: 'rgba(255,255,255,0.08)' },
          pointLabels: { color: '#94a3b8', font: { size: 11 } },
          angleLines: { color: 'rgba(255,255,255,0.06)' },
        }
      },
      plugins: {
        legend: { display: false },
        tooltip: { backgroundColor: '#0f172a', borderColor: '#1e293b', borderWidth: 1, titleColor: '#f8fafc', bodyColor: '#94a3b8' },
      }
    }
  });
}

/* ==================== TOAST NOTIFICATIONS ==================== */
function showToast(message, type = 'info') {
  let container = document.querySelector('.toast-container');
  if (!container) {
    container = document.createElement('div');
    container.className = 'toast-container';
    document.body.appendChild(container);
  }

  const icons = { success: '✅', error: '❌', info: '💡' };
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.innerHTML = `<span>${icons[type] || '💬'}</span><span>${message}</span>`;
  container.appendChild(toast);

  setTimeout(() => {
    toast.style.animation = 'none';
    toast.style.opacity = '0';
    toast.style.transform = 'translateX(100px)';
    toast.style.transition = 'all 0.3s ease';
    setTimeout(() => toast.remove(), 300);
  }, 4000);
}

/* ==================== UTILITY ==================== */
function formatNum(n, decimals = 4) {
  return parseFloat(n).toFixed(decimals);
}

// Initialize range sliders on page load
window.addEventListener('load', () => {
  document.querySelectorAll('input[type="range"]').forEach(input => {
    updateRangeColor(input);
  });
});
