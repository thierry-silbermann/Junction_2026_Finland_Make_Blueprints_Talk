'use strict';
const $ = id => document.getElementById(id);
function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined && text !== null) node.textContent = String(text);
  return node;
}
async function api(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) {
    let detail = response.statusText;
    try { detail = (await response.json()).detail || detail; } catch (e) { /* keep status text */ }
    throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail));
  }
  return response.json();
}
const post = (url, body) => api(url, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
let toastTimer;
function toast(message, info) {
  const node = $('toast');
  node.textContent = message;
  node.className = 'toast show' + (info ? ' info' : '');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { node.className = 'toast'; }, 4200);
}
const pct = v => v === null || v === undefined ? '—' : (100 * v).toFixed(v >= 0.995 || v === 0 ? 0 : 1) + '%';
const fmt = v => v === null || v === undefined ? '—' : typeof v === 'number' ? (Number.isInteger(v) ? String(v) : v.toFixed(2)) : typeof v === 'string' ? v : JSON.stringify(v);
const num = v => Number(v || 0).toLocaleString('en-US');
const qs = target => target.upload ? 'upload=' + encodeURIComponent(target.upload) : 'path=' + encodeURIComponent(target.path);
const targetBody = target => target.upload ? {upload: target.upload} : {path: target.path};

const KINDS = {
  door: ['#1f7a4d', 'door', 'doors'], window: ['#2f6fb5', 'window', 'windows'],
  toilet: ['#b0651a', 'toilet', 'toilets'], washbasin: ['#c47f2c', 'washbasin', 'washbasins'],
  kitchen_sink: ['#a05a12', 'kitchen sink', 'kitchen sinks'], bathtub: ['#c98a3a', 'bathtub', 'bathtubs'],
  shower: ['#b8742a', 'shower', 'showers'], room: ['#7a4fa0', 'room', 'rooms'],
  apartment: ['#5c6670', 'apartment', 'apartments'], floor: ['#5c6670', 'floor', 'floors'],
  stair: ['#9b3b2e', 'staircase', 'staircases'], column: ['#9b3b2e', 'column', 'columns'],
};
const CATEGORY_KIND = {doors: 'door', windows: 'window', toilets: 'toilet', sinks: 'washbasin', washbasins: 'washbasin',
  showers_bathtubs: 'shower', bathtubs: 'bathtub', showers: 'shower', rooms: 'room', bedrooms: 'room', kitchens: 'room',
  bathrooms: 'room', saunas: 'room', wc_rooms: 'room', balconies: 'room', floors: 'floor', stairs: 'stair', columns: 'column'};
const ROOM_TYPES = {bedroom: ['bedroom', 'bedrooms'], kitchen: ['kitchen', 'kitchens'], bathroom: ['bathroom', 'bathrooms'],
  wc: ['WC', 'WCs'], sauna: ['sauna', 'saunas'], balcony: ['balcony', 'balconies'], terrace: ['terrace', 'terraces'],
  living: ['living room', 'living rooms'], hallway: ['hallway', 'hallways'], storage: ['storage', 'storage rooms'], other: ['other room', 'other rooms']};
const MODEL_LABELS = {nemotron_nvidia: 'Nemotron Nano Omni · NVIDIA API', nemotron_gpu: 'Nemotron Nano Omni · own H100',
  nemotron: 'Nemotron Nano Omni · local vLLM', qwen: 'Qwen3-VL 8B · local'};
const kindColor = kind => (KINDS[kind] || ['#2c412b'])[0];
const plural = (n, pair) => n + ' ' + (n === 1 ? pair[0] : pair[1]);

/* ---------- Drawing viewer with evidence overlay ---------- */
function Viewer(root) {
  root.innerHTML = '<div class="v-toolbar"><span class="v-pages" hidden>Page <select></select></span><span class="grow"></span>' +
    '<button data-z="out" aria-label="Zoom out">−</button><span class="v-zoom mono">Fit</span><button data-z="in" aria-label="Zoom in">+</button>' +
    '<button data-z="fit">Fit</button></div><div class="v-stage"><div class="v-canvas"><img alt=""><div class="v-overlay"></div></div>' +
    '<div class="v-empty"></div></div><div class="scan"></div><div class="busy-label"><span class="busy-text"></span><span class="mono busy-time"></span></div>';
  const stage = root.querySelector('.v-stage'), canvas = root.querySelector('.v-canvas'), img = canvas.querySelector('img');
  const overlay = root.querySelector('.v-overlay'), empty = root.querySelector('.v-empty');
  const pageWrap = root.querySelector('.v-pages'), pageSelect = pageWrap.querySelector('select');
  const state = {pages: {}, page: null, boxes: [], zoom: 1, filter: null, hl: null, timer: null, onBox: null};

  function layout() {
    if (!img.naturalWidth) return;
    const w = stage.clientWidth - 32, h = stage.clientHeight - 32;
    const fit = Math.min(w / img.naturalWidth, h / img.naturalHeight);
    const width = img.naturalWidth * fit * state.zoom, height = img.naturalHeight * fit * state.zoom;
    canvas.style.width = width + 'px';
    canvas.style.marginTop = Math.max(16, (stage.clientHeight - height) / 2) + 'px';
    canvas.style.marginBottom = '16px';
    root.querySelector('.v-zoom').textContent = state.zoom === 1 ? 'Fit' : Math.round(state.zoom * 100) + '%';
  }
  function drawBoxes() {
    overlay.replaceChildren();
    const visible = state.boxes.filter(b => String(b.page || 1) === String(state.page) && b.bbox);
    visible.forEach((b, i) => {
      const [x1, y1, x2, y2] = b.bbox;
      const box = el('div', 'box' + (b.room ? ' room' : '') + (b.uncertain ? ' uncertain' : ''));
      box.style.cssText = `left:${x1 * 100}%;top:${y1 * 100}%;width:${(x2 - x1) * 100}%;height:${(y2 - y1) * 100}%;--c:${b.color};animation-delay:${Math.min(i * 22, 900)}ms`;
      box.title = b.title || b.tag || '';
      if (b.tag && (b.room || b.showTag)) box.append(el('span', 'tag', b.tag));
      if (b.num) box.append(el('span', 'num', b.num));
      box.onclick = event => { event.stopPropagation(); highlight(b.key); if (state.onBox) state.onBox(b); };
      box.dataset.key = b.key;
      overlay.append(box);
    });
    applyFilter();
  }
  function applyFilter() {
    for (const node of overlay.children) {
      const b = state.boxes.find(x => x.key === node.dataset.key);
      node.classList.toggle('dim', !!state.filter && !state.filter(b));
      node.classList.toggle('hl', state.hl === node.dataset.key);
      if (state.hl === node.dataset.key && b && b.tag && !node.querySelector('.tag')) node.append(el('span', 'tag', b.tag));
    }
  }
  function highlight(key) {
    state.hl = key;
    const b = state.boxes.find(x => x.key === key);
    if (b && b.page && String(b.page) !== String(state.page) && state.pages[b.page]) { setPage(String(b.page)); return; }
    applyFilter();
  }
  function setPage(page) {
    state.page = page;
    pageSelect.value = page;
    img.src = state.pages[page];
    drawBoxes();
  }
  img.onload = layout;
  img.onerror = () => { empty.textContent = 'The page image could not be loaded.'; empty.hidden = false; };
  pageSelect.onchange = () => setPage(pageSelect.value);
  root.querySelector('.v-toolbar').onclick = event => {
    const z = event.target.dataset && event.target.dataset.z;
    if (!z) return;
    state.zoom = z === 'fit' ? 1 : Math.min(6, Math.max(0.5, state.zoom * (z === 'in' ? 1.35 : 1 / 1.35)));
    layout();
  };
  new ResizeObserver(layout).observe(stage);
  const api = {
    show(pages, message) {
      state.pages = pages || {}; state.boxes = []; state.filter = null; state.hl = null; state.zoom = 1;
      const keys = Object.keys(state.pages).sort((a, b) => a - b);
      pageSelect.replaceChildren(...keys.map(k => { const o = el('option', null, k); o.value = k; return o; }));
      pageWrap.hidden = keys.length < 2;
      canvas.hidden = !keys.length;
      empty.hidden = !!keys.length;
      if (!keys.length) {
        empty.replaceChildren();
        if (message && typeof message === 'object') empty.append(message); else empty.textContent = message || '';
        img.removeAttribute('src');
        overlay.replaceChildren();
        return;
      }
      setPage(keys[0]);
    },
    get page() { return state.page; },
    get pageCount() { return Object.keys(state.pages).length; },
    setBoxes(boxes) { state.boxes = boxes || []; state.hl = null; drawBoxes(); },
    filter(fn) { state.filter = fn; applyFilter(); },
    highlight,
    onBox(fn) { state.onBox = fn; },
    busy(text) {
      root.classList.add('busy');
      root.querySelector('.busy-text').textContent = text;
      const started = performance.now(), clock = root.querySelector('.busy-time');
      clearInterval(state.timer);
      clock.textContent = '0.0 s';
      state.timer = setInterval(() => { clock.textContent = ((performance.now() - started) / 1000).toFixed(1) + ' s'; }, 100);
    },
    idle() { root.classList.remove('busy'); clearInterval(state.timer); },
  };
  return api;
}

function emptyMessage(title, text) {
  const wrap = el('div');
  wrap.append(el('span', 'display', title), el('span', null, text));
  return wrap;
}

function evidenceBoxes(p) {
  return (p.evidence || []).filter(e => e.bbox).map((e, i) => {
    const kind = CATEGORY_KIND[e.category] || null;
    const color = e.counted === false ? '#8a8f94' : kind ? kindColor(kind) : '#2c412b';
    return {key: 'ev' + i, page: e.page, bbox: e.bbox, color, num: i + 1, uncertain: e.uncertain,
      tag: e.label, title: (i + 1) + '. ' + e.label + (e.note ? ' — ' + e.note : ''), room: kind === 'room'};
  });
}

/* ---------- Answer card (shared by both tabs) ---------- */
function renderAnswer(container, p, viewer, options = {}) {
  container.replaceChildren();
  const native = !['Nemotron', 'Vision model'].includes(p.route_label);
  const head = el('div', 'card-title');
  head.append(el('span', null, options.title || 'Answer'));
  const badges = el('span', 'badges');
  badges.append(el('span', 'badge route' + (native ? ' native' : ''), p.route_label + (native ? ' · 0 tokens' : '')));
  if (p.live !== undefined && !native) badges.append(el('span', 'badge ' + (p.live ? 'live' : 'cached'), p.live ? 'live' : 'cached'));
  head.append(badges);
  const body = el('div', 'answer-body');
  if (options.showQuestion !== false) body.append(el('p', 'answer-q', p.question));
  const value = el('div', 'answer-value');
  const unit = p.unit && p.unit !== 'count' ? p.unit : '';
  if (p.answer !== null && p.answer !== undefined) {
    value.textContent = fmt(p.answer);
    if (unit) value.append(el('small', null, unit));
  } else if (p.proposed_answer !== null && p.proposed_answer !== undefined) {
    value.textContent = fmt(p.proposed_answer);
    value.append(el('small', null, (unit ? unit + ' · ' : '') + 'proposed'));
  } else {
    value.className = 'answer-value none';
    value.textContent = p.status === 'unsupported' ? 'Not supported yet' : p.status === 'error' ? 'No answer' : 'No final answer';
  }
  body.append(value);
  const status = el('div', 'badges');
  status.append(el('span', 'badge ' + p.status, p.status.replace('_', ' ')));
  if (p.has_ground_truth) {
    const proposedMatch = p.status === 'needs_review' && fmt(p.proposed_answer) === fmt(p.expected_answer);
    status.append(p.correct ? el('span', 'badge right', '✓ matches ' + fmt(p.expected_answer))
      : proposedMatch ? el('span', 'badge needs_review', 'proposal matches ' + fmt(p.expected_answer) + ' · held for review')
      : el('span', 'badge wrong', '✗ expected ' + fmt(p.expected_answer)));
  }
  body.append(status);
  const u = p.usage || {}, stats = el('div', 'stats-line');
  const seconds = p.elapsed_seconds !== undefined ? p.elapsed_seconds : u.latency_seconds;
  if (seconds !== undefined) stats.append(el('span', null, seconds.toFixed(1) + ' s'));
  stats.append(el('span', null, num(u.input_tokens) + ' in / ' + num(u.output_tokens) + ' out tokens'));
  if (u.requests) stats.append(el('span', null, u.requests + ' model request' + (u.requests > 1 ? 's' : '')));
  if (u.cached_requests) stats.append(el('span', null, u.cached_requests + ' from cache'));
  body.append(stats);
  if (p.rationale) body.append(el('p', 'rationale', p.rationale.length > 600 ? p.rationale.slice(0, 600) + '…' : p.rationale));
  if (p.error) body.append(el('p', 'rationale', p.error));
  const evidence = p.evidence || [];
  if (evidence.length) {
    const list = el('div', 'ev-list');
    evidence.forEach((e, i) => {
      const kind = CATEGORY_KIND[e.category];
      const item = el('button', 'ev');
      item.type = 'button';
      const n = el('span', 'n', i + 1);
      n.style.setProperty('--c', e.counted === false ? '#8a8f94' : kind ? kindColor(kind) : '#2c412b');
      item.append(n, el('span', null, e.label));
      item.append(el('small', null, e.entity_guid ? 'GUID ' + e.entity_guid : e.uncertain ? 'uncertain' : e.counted === false ? 'excluded' : e.provenance.replace('_', ' ')));
      item.onclick = () => { if (viewer && e.bbox) viewer.highlight('ev' + i); };
      list.append(item);
    });
    body.append(list);
  }
  const warnings = (p.warnings || []).filter(w => !/has not been human verified/.test(w));
  if (warnings.length) {
    const details = el('details', 'warnings');
    details.append(el('summary', null, warnings.length + ' note' + (warnings.length > 1 ? 's' : '') + ' for the reviewer'));
    const ul = el('ul');
    warnings.forEach(w => ul.append(el('li', null, w)));
    details.append(ul);
    body.append(details);
  }
  container.append(head, body);
  container.hidden = false;
}

/* ---------- Model picker ---------- */
let MODELS = [];
const currentModel = () => $('modelSelect').value;
function updateDot() {
  const m = MODELS.find(x => x.alias === currentModel()), dot = $('modelDot');
  dot.className = 'dot' + (m && m.reachable === true ? ' ok' : m && m.reachable === false ? ' bad' : '');
  dot.title = !m ? '' : m.reachable === true ? 'Endpoint reachable' : m.reachable === false ? 'Not reachable: ' + (m.detail || '') + '. Cached answers still work.' : 'Status not checked';
}
async function loadModels() {
  try {
    const data = await api('/api/models');
    MODELS = data.models;
    const select = $('modelSelect');
    select.replaceChildren(...MODELS.map(m => { const o = el('option', null, MODEL_LABELS[m.alias] || m.alias); o.value = m.alias; return o; }));
    const saved = localStorage.getItem('planproof-model');
    select.value = MODELS.some(m => m.alias === saved) ? saved : data.default;
    select.onchange = () => { localStorage.setItem('planproof-model', select.value); updateDot(); };
    updateDot();
  } catch (e) { toast('Could not load models: ' + e.message); }
}

/* ---------- Live blueprint tab ---------- */
const live = {target: null, detect: null, answer: null, layer: 'detect', focus: null};
const liveViewer = Viewer($('liveViewer'));
liveViewer.show({}, emptyMessage('Drop a blueprint to begin', 'Any floor plan, scan or PDF sheet. Then detect everything on it, or ask it a question.'));

function setLayer(layer) {
  live.layer = layer;
  for (const b of $('liveLayer').querySelectorAll('button')) b.classList.toggle('active', b.dataset.layer === layer);
  if (layer === 'answer' && live.answer) {
    liveViewer.setBoxes(evidenceBoxes(live.answer));
  } else if (live.detect) {
    liveViewer.setBoxes(detectBoxes(live.detect));
    liveViewer.filter(live.focus ? b => b.kind === live.focus : null);
  } else liveViewer.setBoxes([]);
}
$('liveLayer').onclick = e => { if (e.target.dataset.layer) setLayer(e.target.dataset.layer); };
function syncLayerSwitch() { $('liveLayer').hidden = !(live.detect && live.answer && (live.answer.evidence || []).length); }

function detectBoxes(result) {
  return result.items.map((item, i) => {
    let tag = (KINDS[item.kind] || [null, item.kind])[1];
    if (item.kind === 'room') {
      tag = [item.text || (item.room_type ? ROOM_TYPES[item.room_type][0] : 'room'), item.number,
        item.area !== null && item.area !== undefined ? item.area + ' m²' : ''].filter(Boolean).join(' · ');
    } else if (item.text) tag += ' · ' + item.text;
    return {key: 'd' + i, page: item.page, bbox: item.bbox, color: kindColor(item.kind), kind: item.kind,
      room: item.kind === 'room' || item.kind === 'apartment' || item.kind === 'floor', uncertain: item.uncertain, tag,
      title: tag + (item.uncertain ? ' (uncertain)' : '')};
  });
}

async function selectTarget(target, button) {
  live.target = target; live.detect = null; live.answer = null; live.focus = null;
  for (const b of $('samples').querySelectorAll('.sample')) b.classList.toggle('active', b === button);
  $('liveTitle').textContent = target.name;
  $('liveSubtitle').textContent = target.dataset ? 'Dataset · ' + target.dataset : 'Your upload';
  $('kindChips').replaceChildren(); $('roomChips').replaceChildren(); $('detectMeta').textContent = '';
  $('answerCard').hidden = true; $('askInput').value = ''; syncLayerSwitch();
  for (const id of ['detectBtn', 'askBtn', 'askInput']) $(id).disabled = true;
  document.querySelector('.live-grid').scrollIntoView({behavior: 'smooth', block: 'start'});
  liveViewer.busy('Rendering the drawing');
  try {
    const page = await api('/api/render?' + qs(target));
    liveViewer.show(page.pages, page.kind === 'ifc' ? emptyMessage('BIM model', 'Ask a question: answers come straight from the IFC entities, with their GlobalIds.') : '');
    $('detectBtn').disabled = page.kind === 'ifc';
    $('askBtn').disabled = false; $('askInput').disabled = false;
    $('askInput').focus();
  } catch (e) {
    liveViewer.show({}, 'This drawing could not be rendered: ' + e.message);
    toast(e.message);
  } finally { liveViewer.idle(); }
  suggestQuestions(target);
}

async function suggestQuestions(target) {
  let questions = SUGGESTIONS;
  if (target.path) {
    try {
      const data = await api('/api/image?path=' + encodeURIComponent(target.path));
      const own = [...new Set(data.questions.map(q => q.question))];
      if (own.length) questions = [...own.filter(q => q.length < 90).slice(0, 4), ...SUGGESTIONS].slice(0, 5);
    } catch (e) { /* defaults are fine */ }
  }
  const box = $('suggestions');
  box.replaceChildren(...questions.map(q => {
    const chip = el('button', 'chip', q.length > 64 ? q.slice(0, 62) + '…' : q);
    chip.type = 'button'; chip.title = q;
    chip.onclick = () => { $('askInput').value = q; ask(); };
    return chip;
  }));
}

async function detect() {
  if (!live.target) return;
  const button = $('detectBtn');
  button.disabled = true; button.classList.add('busy');
  liveViewer.busy('Nemotron is taking inventory of the drawing');
  try {
    const body = {...targetBody(live.target), model: currentModel()};
    if (liveViewer.pageCount > 1) body.page = Number(liveViewer.page);
    const result = await post('/api/detect', body);
    live.detect = result; live.focus = null;
    renderDetect(result);
    setLayer('detect'); syncLayerSwitch();
  } catch (e) { toast('Detection failed: ' + e.message); }
  finally { liveViewer.idle(); button.disabled = false; button.classList.remove('busy'); }
}

function renderDetect(result) {
  const chips = $('kindChips'), rooms = $('roomChips');
  chips.replaceChildren(); rooms.replaceChildren();
  Object.entries(result.counts).forEach(([kind, c], i) => {
    const pair = KINDS[kind] ? [KINDS[kind][1], KINDS[kind][2]] : [kind, kind];
    const chip = el('button', 'chip');
    chip.type = 'button'; chip.style.setProperty('--c', kindColor(kind)); chip.style.animationDelay = (i * 60) + 'ms';
    chip.append(el('i'), el('b', null, c.total), el('span', null, c.total === 1 ? pair[0] : pair[1]));
    if (c.uncertain) chip.append(el('span', 'muted small', '(' + c.uncertain + ' unsure)'));
    chip.onclick = () => {
      live.focus = live.focus === kind ? null : kind;
      for (const other of chips.children) other.classList.toggle('on', other === chip && !!live.focus);
      if (live.layer !== 'detect') setLayer('detect'); else liveViewer.filter(live.focus ? b => b.kind === live.focus : null);
    };
    chips.append(chip);
  });
  Object.entries(result.room_types).forEach(([type, n], i) => {
    const chip = el('span', 'chip');
    chip.style.animationDelay = (300 + i * 50) + 'ms';
    chip.append(el('b', null, n), el('span', null, n === 1 ? (ROOM_TYPES[type] || [type])[0] : (ROOM_TYPES[type] || [type, type])[1]));
    rooms.append(chip);
  });
  const u = result.usage, failed = Object.entries(result.groups).filter(([, g]) => g.failed).map(([name]) => name);
  const parts = [result.items.length + ' objects', result.elapsed_seconds.toFixed(1) + ' s'];
  parts.push(result.live ? num(u.input_tokens + u.output_tokens) + ' tokens · ' + u.requests + ' requests' : 'replayed from cache');
  if (failed.length) parts.push('failed: ' + failed.join(', '));
  $('detectMeta').textContent = parts.join(' · ');
  if (!result.items.length) toast('No objects were listed. Try another model or a sharper scan.', true);
}

async function ask() {
  const question = $('askInput').value.trim();
  if (!live.target || !question) return;
  $('askBtn').disabled = true;
  liveViewer.busy(/^(kuinka|montako|mikä|paljonko)/i.test(question) ? 'Luetaan piirustusta' : 'Reading the drawing');
  try {
    const body = {...targetBody(live.target), question, model: currentModel()};
    if (liveViewer.pageCount > 1) body.page = Number(liveViewer.page);
    const result = await post('/api/ask', body);
    live.answer = result;
    renderAnswer($('answerCard'), result, liveViewer);
    $('answerCard').scrollIntoView({behavior: 'smooth', block: 'nearest'});
    setLayer((result.evidence || []).some(e => e.bbox) ? 'answer' : 'detect');
    syncLayerSwitch();
  } catch (e) { toast('Question failed: ' + e.message); }
  finally { liveViewer.idle(); $('askBtn').disabled = false; }
}

$('detectBtn').onclick = detect;
$('askForm').onsubmit = e => { e.preventDefault(); ask(); };
$('askInput').onkeydown = e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); ask(); } };
liveViewer.onBox(b => {
  if (!b.key.startsWith('ev')) return;
  const items = $('answerCard').querySelectorAll('.ev');
  items.forEach((node, i) => node.classList.toggle('hl', 'ev' + i === b.key));
});

async function upload(file) {
  if (!file) return;
  const form = new FormData();
  form.append('file', file);
  liveViewer.busy('Uploading ' + file.name);
  try {
    const result = await api('/api/upload', {method: 'POST', body: form});
    liveViewer.idle();
    await selectTarget({upload: result.upload, name: file.name});
  } catch (e) { liveViewer.idle(); toast('Upload failed: ' + e.message); }
}
const drop = $('dropzone');
drop.ondragover = e => { e.preventDefault(); drop.classList.add('over'); };
drop.ondragleave = () => drop.classList.remove('over');
drop.ondrop = e => { e.preventDefault(); drop.classList.remove('over'); upload(e.dataTransfer.files[0]); };
drop.onkeydown = e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); $('fileInput').click(); } };
$('fileInput').onchange = e => { upload(e.target.files[0]); e.target.value = ''; };
document.addEventListener('dragover', e => e.preventDefault());
document.addEventListener('drop', e => { if (!drop.contains(e.target)) { e.preventDefault(); if (e.dataTransfer.files[0]) upload(e.dataTransfer.files[0]); } });

let SUGGESTIONS = [];
async function loadSamples(run) {
  try {
    const data = await api('/api/samples' + (run ? '?run=' + encodeURIComponent(run) : ''));
    SUGGESTIONS = data.questions;
    $('samples').replaceChildren(...data.samples.map(s => {
      const button = el('button', 'sample');
      button.type = 'button'; button.title = s.path;
      const img = el('img'); img.loading = 'lazy'; img.alt = s.name; img.src = s.thumb;
      button.append(img, el('span', null, s.dataset));
      button.onclick = () => selectTarget({path: s.path, name: s.name, dataset: s.dataset}, button);
      return button;
    }));
  } catch (e) { toast('Could not load samples: ' + e.message); }
}

/* ---------- Benchmark tab ---------- */
const bench = {run: null, loaded: false, dataset: null, image: null, rows: [], live: {}, selected: -1, offset: 0};
const benchViewer = Viewer($('benchViewer'));

async function loadRuns() {
  const data = await api('/api/runs');
  const select = $('runSelect');
  select.replaceChildren(...data.runs.map(name => { const o = el('option', null, name); o.value = name; return o; }));
  if (!data.runs.length) { $('kpis').textContent = 'No saved runs found. Run drawingqa evaluate first.'; return null; }
  select.value = data.default;
  select.onchange = () => loadRun(select.value);
  return data.default;
}

async function loadRun(name) {
  bench.run = name;
  $('galleryCard').hidden = true; $('benchWork').hidden = true;
  try {
    const [run, sets] = await Promise.all([api('/api/runs/' + encodeURIComponent(name)), api('/api/datasets?run=' + encodeURIComponent(name))]);
    renderKpis(run);
    renderBars(sets.datasets);
  } catch (e) { toast('Could not load run: ' + e.message); }
}

function renderKpis(run) {
  const s = run.summary, status = s.status_counts || {};
  const cards = [
    ['hero', 'Final accuracy', pct(s.exact_accuracy_all_labeled), num(s.correct_answers) + ' of ' + num(s.labeled_questions) + ' labeled questions'],
    ['', 'Accuracy when it answers', pct(s.exact_accuracy_answered_only), 'the rest is flagged for review, never guessed'],
    ['', 'Answer coverage', pct(s.answer_coverage), num(status.needs_review || 0) + ' need review · ' + num(status.unsupported || 0) + ' unsupported'],
    ['', 'Answered at zero tokens', pct(run.zero_token_share), 'straight from DXF and IFC files'],
  ];
  $('kpis').replaceChildren(...cards.map(([cls, label, value, note], i) => {
    const card = el('div', 'kpi ' + cls);
    card.style.animationDelay = (i * 70) + 'ms';
    card.append(el('div', 'label', label), el('div', 'value', value), el('div', 'note', note));
    return card;
  }));
}

function renderBars(datasets) {
  const evaluated = datasets.filter(d => d.run_metrics).sort((a, b) => (b.run_metrics.exact_accuracy_all_labeled || 0) - (a.run_metrics.exact_accuracy_all_labeled || 0));
  const others = datasets.filter(d => !d.run_metrics && d.questions);
  const container = $('datasetBars');
  container.replaceChildren();
  for (const d of [...evaluated, ...others]) {
    const m = d.run_metrics, row = el('div', 'bar-row');
    row.tabIndex = 0;
    const name = el('div', 'name', d.name);
    name.append(el('small', null, m ? num(m.questions) + ' questions' : num(d.questions) + ' questions · not in this run'));
    const track = el('div', 'track');
    const answered = el('div', 'fill answered'), final = el('div', 'fill');
    track.append(answered, final);
    const value = el('div', 'pct' + (m ? '' : ' na'), m ? pct(m.exact_accuracy_all_labeled) : 'run live');
    const routes = el('div', 'routes');
    for (const [route, n] of Object.entries(d.routes || {})) {
      const native = !['Nemotron', 'Vision model'].includes(route);
      routes.append(el('span', 'badge route' + (native ? ' native' : ''), route + ' ' + n));
    }
    if (m && m.status_counts && m.status_counts.unsupported) routes.append(el('span', 'badge unsupported', m.status_counts.unsupported + ' unsupported'));
    row.append(name, track, value, routes);
    if (m) requestAnimationFrame(() => setTimeout(() => {
      answered.style.width = (100 * (m.exact_accuracy_answered_only || 0)) + '%';
      final.style.width = (100 * (m.exact_accuracy_all_labeled || 0)) + '%';
    }, 60));
    row.onclick = row.onkeydown = e => {
      if (e.type === 'keydown' && e.key !== 'Enter') return;
      for (const r of container.children) r.classList.toggle('active', r === row);
      selectDataset(d.name, true);
    };
    container.append(row);
  }
  const legend = el('div', 'legend-line');
  legend.innerHTML = '<span><i style="background:var(--forest-800)"></i>Final accuracy, all labeled questions</span><span><i style="background:var(--moss-soft)"></i>Accuracy when an answer is given</span>';
  container.append(legend);
}

async function selectDataset(name, reset) {
  if (reset) { bench.dataset = name; bench.offset = 0; $('gallery').replaceChildren(); }
  $('galleryCard').hidden = false;
  $('galleryTitle').textContent = name;
  try {
    const data = await api(`/api/datasets/${encodeURIComponent(name)}/images?run=${encodeURIComponent(bench.run)}&offset=${bench.offset}&limit=48`);
    const gallery = $('gallery');
    gallery.querySelector('.more') && gallery.querySelector('.more').remove();
    for (const image of data.images) {
      const tile = el('button', 'sample');
      tile.type = 'button'; tile.title = image.path;
      if (image.path.toLowerCase().endsWith('.ifc')) tile.append(el('div', 'ifc', 'IFC model'));
      else { const img = el('img'); img.loading = 'lazy'; img.src = image.thumb; img.alt = image.name; tile.append(img); }
      const caption = el('span');
      if (image.saved) { caption.append(el('b', null, image.correct + '/' + image.saved + ' ✓')); caption.append(' · ' + image.name); }
      else caption.textContent = image.questions + ' q · ' + image.name;
      tile.append(caption);
      tile.onclick = () => {
        for (const t of gallery.querySelectorAll('.sample')) t.classList.toggle('active', t === tile);
        selectImage(image.path);
      };
      gallery.append(tile);
    }
    bench.offset += data.images.length;
    $('galleryCount').textContent = num(data.total) + ' drawings with questions';
    if (bench.offset < data.total) {
      const more = el('button', 'btn small more', 'Show more');
      more.onclick = () => selectDataset(name, false);
      gallery.append(more);
    }
    if (reset) $('galleryCard').scrollIntoView({behavior: 'smooth', block: 'start'});
  } catch (e) { toast(e.message); }
}

async function selectImage(path) {
  bench.image = path; bench.live = {}; bench.selected = -1;
  $('benchWork').hidden = false;
  $('benchTitle').textContent = path.split('/').pop();
  $('benchSubtitle').textContent = 'Dataset · ' + path.split('/')[0];
  $('benchDetail').replaceChildren();
  benchViewer.busy('Rendering the drawing');
  try {
    const [data, page] = await Promise.all([
      api('/api/image?path=' + encodeURIComponent(path) + '&run=' + encodeURIComponent(bench.run)),
      api('/api/render?path=' + encodeURIComponent(path))]);
    bench.rows = data.questions;
    benchViewer.show(page.pages, page.kind === 'ifc' ? emptyMessage('BIM model', 'Answers come from IFC entities; their GlobalIds are listed with each answer.') : '');
    renderTable();
    const first = bench.rows.findIndex(r => r.saved);
    if (first >= 0) selectRow(first);
    $('benchWork').scrollIntoView({behavior: 'smooth', block: 'start'});
  } catch (e) { toast(e.message); }
  finally { benchViewer.idle(); }
}

function markCell(p) {
  if (p.correct === true) return el('span', 'mark-ok', '✓');
  if (p.status === 'needs_review') {
    const flag = el('span', 'mark-review', '⚑');
    flag.title = 'Held for human review, not counted as a final answer';
    return flag;
  }
  return el('span', p.correct === false ? 'mark-bad' : 'mark-na', p.correct === false ? '✗' : '–');
}
function answerText(p) {
  if (!p) return '—';
  if (p.answer !== null && p.answer !== undefined) return fmt(p.answer);
  if (p.proposed_answer !== null && p.proposed_answer !== undefined) return fmt(p.proposed_answer) + '?';
  return p.status === 'unsupported' ? 'n/a' : '—';
}

function renderTable() {
  const table = el('table'), head = el('tr');
  for (const h of ['Question', 'Expected', 'Saved run', 'Live']) head.append(el('th', null, h));
  const thead = el('thead');
  thead.append(head);
  table.append(thead);
  const body = el('tbody');
  bench.rows.forEach((row, i) => {
    const tr = el('tr', bench.selected === i ? 'active' : '');
    tr.append(el('td', null, row.question));
    tr.append(el('td', 'ans', fmt(row.answer)));
    const saved = el('td', 'ans');
    if (row.saved) {
      saved.append(answerText(row.saved) + ' ', markCell(row.saved));
      saved.title = row.saved.status + ' · ' + row.saved.route_label;
    } else saved.append(el('span', 'mark-na small', 'not run'));
    tr.append(saved);
    const liveCell = el('td', 'live-cell'), result = bench.live[i];
    if (result === 'pending') liveCell.append(el('span', 'muted small', 'running…'));
    else if (result) { liveCell.className += ' ans'; liveCell.append(answerText(result) + ' ', markCell(result)); }
    else {
      const run = el('button', 'btn small', 'Run');
      run.onclick = e => { e.stopPropagation(); runLive(i); };
      liveCell.append(run);
    }
    tr.append(liveCell);
    tr.onclick = () => selectRow(i);
    body.append(tr);
  });
  table.append(body);
  $('questionTable').replaceChildren(table);
  if (!bench.rows.length) $('questionTable').append(el('p', 'pad muted', 'No questions for this drawing.'));
}

function selectRow(i) {
  bench.selected = i;
  const row = bench.rows[i], result = bench.live[i] && bench.live[i] !== 'pending' ? bench.live[i] : row.saved;
  for (const [index, tr] of [...$('questionTable').querySelectorAll('tbody tr')].entries()) tr.classList.toggle('active', index === i);
  if (!result) {
    benchViewer.setBoxes([]);
    $('benchDetail').replaceChildren(el('p', 'pad muted', 'Not in the saved run. Press Run to ask the model live.'));
    return;
  }
  benchViewer.setBoxes(evidenceBoxes(result));
  renderAnswer($('benchDetail'), result, benchViewer, {title: bench.live[i] && bench.live[i] !== 'pending' ? 'Live answer' : 'Saved answer'});
}

async function runLive(i) {
  bench.live[i] = 'pending';
  renderTable();
  try {
    const row = bench.rows[i];
    const body = {path: bench.image, question: row.question, model: currentModel()};
    if (row.page) body.page = row.page;
    if (bench.selected === i || bench.selected < 0) benchViewer.busy('Asking ' + (MODEL_LABELS[currentModel()] || currentModel()));
    bench.live[i] = await post('/api/ask', body);
  } catch (e) { delete bench.live[i]; toast('Live run failed: ' + e.message); }
  finally { benchViewer.idle(); }
  renderTable();
  if (bench.selected === i || bench.selected < 0) selectRow(i);
}

$('runAllBtn').onclick = async () => {
  const button = $('runAllBtn'), pending = bench.rows.map((_, i) => i).filter(i => !bench.live[i]);
  button.disabled = true;
  let done = 0;
  const worker = async () => {
    while (pending.length) {
      const i = pending.shift();
      await runLive(i);
      button.textContent = `Running ${++done}/${done + pending.length}`;
    }
  };
  await Promise.all([worker(), worker()]);
  const scored = Object.values(bench.live).filter(r => r && r.has_ground_truth);
  button.textContent = 'Run all live'; button.disabled = false;
  if (scored.length) toast(`${scored.filter(r => r.correct).length} of ${scored.length} correct live`, true);
};

/* ---------- Tabs ---------- */
async function showTab(name) {
  for (const tab of document.querySelectorAll('.tab')) {
    const on = tab.dataset.tab === name;
    tab.classList.toggle('active', on); tab.setAttribute('aria-selected', on);
  }
  $('tab-live').hidden = name !== 'live';
  $('tab-bench').hidden = name !== 'bench';
  if (location.hash !== '#' + name) history.replaceState(null, '', '#' + name);
  if (name === 'bench' && !bench.loaded) {
    bench.loaded = true;
    try { const run = await loadRuns(); if (run) await loadRun(run); } catch (e) { toast(e.message); }
  }
}
for (const tab of document.querySelectorAll('.tab')) tab.onclick = () => showTab(tab.dataset.tab);
window.addEventListener('hashchange', () => showTab(location.hash === '#bench' ? 'bench' : 'live'));

(async () => {
  loadModels();
  let defaultRun = null;
  try { defaultRun = (await api('/api/runs')).default; } catch (e) { /* samples work without runs */ }
  await loadSamples(defaultRun);
  showTab(location.hash === '#bench' ? 'bench' : 'live');
})();
