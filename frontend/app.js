/**
 * app.js — HR Policy Assistant Frontend
 * ────────────────────────────────────────
 * Responsibilities:
 *  - Manage five UI states: empty, loading, answer, not-found, error
 *  - POST to /ask, handle responses and network errors
 *  - Render source cards safely (no innerHTML with raw server data)
 *  - Suggestion chip auto-fill
 *  - Copy-to-clipboard button
 *  - Input validation with accessible error messages
 */

'use strict';

/* ── Constants ──────────────────────────────────────────────── */
const API_URL        = '/ask';
const MAX_Q_LENGTH   = 500;
const MIN_Q_LENGTH   = 3;
const FETCH_TIMEOUT  = 45000;   // 45 s — Gemini can take 10-20 s on free tier

/* ── DOM refs ───────────────────────────────────────────────── */
const form          = document.getElementById('ask-form');
const input         = document.getElementById('question-input');
const askBtn        = document.getElementById('ask-btn');
const inputError    = document.getElementById('input-error');
const copyBtn       = document.getElementById('copy-btn');
const retryBtn      = document.getElementById('retry-btn');

const stateEmpty    = document.getElementById('state-empty');
const stateLoading  = document.getElementById('state-loading');
const stateAnswer   = document.getElementById('state-answer');
const stateNotFound = document.getElementById('state-not-found');
const stateError    = document.getElementById('state-error');

const answerText    = document.getElementById('answer-text');
const sourcesSection= document.getElementById('sources-section');
const sourcesGrid   = document.getElementById('sources-grid');
const errorMessage  = document.getElementById('error-message');

/* ── State ──────────────────────────────────────────────────── */
let lastQuestion = '';

/* ── UI State Machine ───────────────────────────────────────── */
const STATES = ['empty', 'loading', 'answer', 'not-found', 'error'];

function showState(name) {
  stateEmpty   .hidden = (name !== 'empty');
  stateLoading .hidden = (name !== 'loading');
  stateAnswer  .hidden = (name !== 'answer');
  stateNotFound.hidden = (name !== 'not-found');
  stateError   .hidden = (name !== 'error');
}

/* ── Input validation ───────────────────────────────────────── */
function validateInput(value) {
  const trimmed = value.trim();
  if (trimmed.length < MIN_Q_LENGTH) return 'Please enter at least 3 characters.';
  if (trimmed.length > MAX_Q_LENGTH) return `Question must be ${MAX_Q_LENGTH} characters or fewer.`;
  return null;
}

function setInputError(msg) {
  inputError.textContent = msg || '';
}

input.addEventListener('input', () => {
  const trimmed = input.value.trim();
  askBtn.disabled = trimmed.length < MIN_Q_LENGTH;
  if (inputError.textContent) setInputError(null); // clear old error on typing
});

/* ── Safe DOM text helpers ──────────────────────────────────── */

/** Create a text node — never injects raw HTML. */
function safeText(str) {
  return document.createTextNode(String(str));
}

/** Set element text content safely. */
function setText(el, str) {
  el.textContent = String(str);
}

/* ── Source card renderer ───────────────────────────────────── */
function renderSourceCard(source) {
  const card = document.createElement('div');
  card.className = 'source-card';

  // Page badge
  const badge = document.createElement('div');
  badge.className = 'source-page-badge';
  badge.appendChild(safeText(`📄 Page ${source.page}`));
  card.appendChild(badge);

  // Excerpt
  const excerpt = document.createElement('p');
  excerpt.className = 'source-excerpt';
  setText(excerpt, source.excerpt);
  card.appendChild(excerpt);

  return card;
}

function renderSources(sources) {
  sourcesGrid.innerHTML = '';   // safe: removing children only
  if (!sources || sources.length === 0) {
    sourcesSection.hidden = true;
    return;
  }
  sourcesSection.hidden = false;
  sources.forEach(src => {
    sourcesGrid.appendChild(renderSourceCard(src));
  });
}

/* ── Loading timer ──────────────────────────────────────────── */
let _loadingTimer = null;

function startLoadingTimer() {
  const el = document.getElementById('loading-elapsed');
  if (!el) return;
  let secs = 0;
  el.textContent = '';
  _loadingTimer = setInterval(() => {
    secs++;
    el.textContent = `${secs}s — Gemini is thinking…`;
    if (secs >= 40) el.textContent += ' (almost there)';
  }, 1000);
}

function stopLoadingTimer() {
  if (_loadingTimer) { clearInterval(_loadingTimer); _loadingTimer = null; }
  const el = document.getElementById('loading-elapsed');
  if (el) el.textContent = '';
}

/* ── API call with timeout ──────────────────────────────────── */
async function askQuestion(question) {
  const controller = new AbortController();
  const timeoutId  = setTimeout(() => controller.abort(), FETCH_TIMEOUT);

  let response;
  try {
    response = await fetch(API_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question }),
      signal: controller.signal,
    });
  } catch (err) {
    if (err.name === 'AbortError') {
      throw new Error(
        `Request timed out after ${FETCH_TIMEOUT / 1000}s. The AI model may be under high load — please try again in a moment.`
      );
    }
    throw new Error(`Network error: ${err.message}. Check that the server is running.`);
  } finally {
    clearTimeout(timeoutId);
  }

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    const detail = errorData.detail || `Server error (${response.status}).`;
    if (response.status === 503) {
      throw new Error(
        'The knowledge base has not been built yet. Please ask your administrator to run the ingestion pipeline.'
      );
    }
    if (response.status === 504) {
      throw new Error(
        'The AI model is taking too long to respond (server timeout). This usually means the free-tier API is under load — please wait a moment and try again.'
      );
    }
    if (response.status === 422) {
      throw new Error('Invalid question. Please ensure it is between 3 and 500 characters.');
    }
    throw new Error(detail);
  }

  return response.json();
}

/* ── Main submit handler ────────────────────────────────────── */
async function handleSubmit(e) {
  e.preventDefault();

  const question = input.value.trim();

  // Client-side validation
  const validationError = validateInput(question);
  if (validationError) {
    setInputError(validationError);
    input.focus();
    return;
  }

  setInputError(null);
  lastQuestion = question;
  showState('loading');
  startLoadingTimer();
  askBtn.disabled = true;

  try {
    const data = await askQuestion(question);

    if (data.is_not_found) {
      showState('not-found');
    } else {
      setText(answerText, data.answer);
      renderSources(data.sources);
      showState('answer');
      copyBtn.classList.remove('copied');
      copyBtn.title = 'Copy answer';
    }
  } catch (err) {
    console.error('[HR Assistant] Error:', err);
    setText(errorMessage, err.message || 'An unexpected error occurred. Please try again.');
    showState('error');
  } finally {
    stopLoadingTimer();
    askBtn.disabled = input.value.trim().length < MIN_Q_LENGTH;
  }
}

form.addEventListener('submit', handleSubmit);

/* ── Retry button ───────────────────────────────────────────── */
retryBtn.addEventListener('click', () => {
  if (lastQuestion) {
    input.value = lastQuestion;
    askBtn.disabled = false;
    form.requestSubmit();
  } else {
    showState('empty');
  }
});

/* ── Copy-to-clipboard ─────────────────────────────────────── */
copyBtn.addEventListener('click', async () => {
  const text = answerText.textContent;
  if (!text) return;

  try {
    await navigator.clipboard.writeText(text);
    copyBtn.classList.add('copied');
    copyBtn.title = 'Copied!';
    setTimeout(() => {
      copyBtn.classList.remove('copied');
      copyBtn.title = 'Copy answer';
    }, 2000);
  } catch {
    // Clipboard not available — silently ignore
  }
});

/* ── Suggestion chips ───────────────────────────────────────── */
document.querySelectorAll('.chip').forEach(chip => {
  chip.addEventListener('click', () => {
    const question = chip.dataset.question;
    if (!question) return;
    input.value = question;
    askBtn.disabled = false;
    setInputError(null);
    form.requestSubmit();
  });
});

/* ── Keyboard shortcut: Enter in input ──────────────────────── */
input.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    if (!askBtn.disabled) form.requestSubmit();
  }
});

/* ── Init ───────────────────────────────────────────────────── */
// Wrap in DOMContentLoaded so DOM is guaranteed to exist
// even if script is ever moved to <head> without defer
function init() {
  showState('empty');
  input.focus();
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
