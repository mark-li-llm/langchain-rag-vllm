// ====== tiny utils ======
const $ = (sel) => document.querySelector(sel);
const byId = (id) => document.getElementById(id);
const sleep = (ms) => new Promise(r => setTimeout(r, ms));

const els = {
  baseUrl: byId('baseUrl'),
  token: byId('token'),
  toggleToken: byId('toggleToken'),
  query: byId('query'),
  topK: byId('topK'),
  maxTokens: byId('maxTokens'),
  stream: byId('stream'),
  btnSend: byId('btn-send'),
  btnStop: byId('btn-stop'),
  btnCopy: byId('btn-copy'),
  btnHealth: byId('btn-health'),
  btnClear: byId('btn-clear'),
  answer: byId('answer'),
  citations: byId('citations'),
  usage: byId('usage'),
  status: byId('status'),
  err: byId('error'),
  samples: byId('samples'),
};

let controller = null; // AbortController for streaming

// ====== local persistence ======
const LS = {
  baseUrl: 'rag.baseUrl',
  token: 'rag.token',
  topK: 'rag.topK',
  maxTokens: 'rag.maxTokens',
  stream: 'rag.stream',
  query: 'rag.query',
};

function loadPrefs(){
  els.baseUrl.value = localStorage.getItem(LS.baseUrl) || '';
  els.token.value   = localStorage.getItem(LS.token) || '';
  els.topK.value    = localStorage.getItem(LS.topK) || '5';
  els.maxTokens.value = localStorage.getItem(LS.maxTokens) || '300';
  els.stream.checked  = (localStorage.getItem(LS.stream) ?? 'true') === 'true';
  els.query.value   = localStorage.getItem(LS.query) || '';
}
function savePrefs(){
  localStorage.setItem(LS.baseUrl, els.baseUrl.value.trim());
  localStorage.setItem(LS.token, els.token.value);
  localStorage.setItem(LS.topK, els.topK.value);
  localStorage.setItem(LS.maxTokens, els.maxTokens.value);
  localStorage.setItem(LS.stream, String(els.stream.checked));
  localStorage.setItem(LS.query, els.query.value);
}

// ====== UI helpers ======
function setBusy(b){
  els.btnSend.disabled = b;
  els.btnStop.disabled = !b;
  els.btnCopy.disabled = b;
  els.baseUrl.disabled = b;
  els.token.disabled = b;
  els.topK.disabled = b;
  els.maxTokens.disabled = b;
  els.stream.disabled = b;
}
function clearOutput(){
  els.answer.textContent = '';
  els.citations.innerHTML = '';
  els.usage.textContent = '';
  els.err.classList.add('hidden');
  els.err.textContent = '';
}
function showError(msg){
  els.err.textContent = msg;
  els.err.classList.remove('hidden');
}
function renderUsage(u){
  if(!u){ els.usage.textContent = ''; return; }
  const parts = [];
  if(u.prompt_tokens != null) parts.push(`prompt_tokens=${u.prompt_tokens}`);
  if(u.output_tokens != null) parts.push(`output_tokens=${u.output_tokens}`);
  if(u.latency_ms != null) parts.push(`latency_ms=${u.latency_ms}`);
  els.usage.textContent = parts.join('  ·  ');
}
function renderCitations(cites){
  els.citations.innerHTML = '';
  if(!Array.isArray(cites) || !cites.length){
    return;
  }
  for(const c of cites){
    const li = document.createElement('li');
    const label = `[${c.marker ?? c.index ?? ''}] ${c.doc_id ?? ''}:${c.chunk_id ?? ''}`;
    const src = c.source ? ` • ${c.source}` : '';
    if(c.url){
      const a = document.createElement('a');
      a.href = c.url; a.target = '_blank';
      a.textContent = label;
      li.appendChild(a);
      const s = document.createElement('span');
      s.textContent = src;
      li.appendChild(s);
    }else{
      li.textContent = label + src;
    }
    els.citations.appendChild(li);
  }
}

// ====== core: call /v1/query ======
async function send(){
  clearOutput(); savePrefs();
  const base = els.baseUrl.value.trim().replace(/\/+$/,''); // '' or 'http://host:port'
  const urlQuery = (base || '') + '/v1/query';
  const urlTrace = (tid) => (base || '') + `/v1/traces/${encodeURIComponent(tid)}`;

  const token = els.token.value.trim();
  if(!token){ showError('Missing Bearer token (AUTH_BEARER_TOKEN).'); return; }

  const payload = {
    query: els.query.value.trim(),
    top_k: Number(els.topK.value || 5),
    stream: !!els.stream.checked,
    max_output_tokens: Number(els.maxTokens.value || 300),
  };
  if(!payload.query){ showError('Please enter a question.'); return; }

  setBusy(true);
  els.status.textContent = payload.stream ? 'Streaming…' : 'Requesting…';

  controller = new AbortController();
  const startedAt = performance.now();

  try{
    const res = await fetch(urlQuery, {
      method:'POST',
      headers:{
        'Authorization': `Bearer ${token}`,
        'Content-Type':'application/json'
      },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });

    if(!res.ok){
      // Try parse error body
      let body = '';
      try{ body = await res.text(); }catch{}
      showError(`HTTP ${res.status} ${res.statusText} — ${body?.slice(0,500) || ''}`);
      els.status.textContent = 'Failed';
      return;
    }

    // Non-streaming path: parse JSON
    const ctype = res.headers.get('content-type') || '';
    if(!payload.stream || ctype.includes('application/json')){
      const data = await res.json();
      els.answer.textContent = data.answer ?? '';
      renderCitations(data.citations);
      renderUsage(data.usage);
      els.status.textContent = 'Done';
      return;
    }

    // Streaming path: chunked text or SSE/NDJSON
    const reader = res.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let traceId = res.headers.get('x-trace-id') || null;
    let buffer = '';
    while(true){
      const {value, done} = await reader.read();
      if(done) break;
      const chunk = decoder.decode(value, {stream:true});
      buffer += chunk;

      // Heuristics:
      // 1) SSE: lines starting with "data:"
      // 2) NDJSON: newline-delimited JSON lines
      // 3) Raw text: append as-is
      const ctype = res.headers.get('content-type') || '';
      if(ctype.includes('text/event-stream')){
        // Split by lines and append "data:" parts
        const lines = chunk.split(/\r?\n/).filter(Boolean);
        for(const line of lines){
          if(line.startsWith('data:')){
            const dataStr = line.slice(5).trim();
            try{
              const obj = JSON.parse(dataStr);
              if(obj.delta){ els.answer.textContent += obj.delta; }
              if(obj.trace_id){ traceId = obj.trace_id; }
              if(obj.usage){ renderUsage(obj.usage); }
              if(obj.citations){ renderCitations(obj.citations); }
            }catch{
              // Not JSON, treat as plain text token
              els.answer.textContent += dataStr;
            }
          }
        }
      }else if(ctype.includes('ndjson') || ctype.includes('jsonlines')){
        const parts = buffer.split(/\r?\n/);
        buffer = parts.pop() ?? '';
        for(const line of parts){
          if(!line.trim()) continue;
          try{
            const obj = JSON.parse(line);
            if(obj.delta){ els.answer.textContent += obj.delta; }
            if(obj.trace_id){ traceId = obj.trace_id; }
            if(obj.usage){ renderUsage(obj.usage); }
            if(obj.citations){ renderCitations(obj.citations); }
          }catch{
            els.answer.textContent += line;
          }
        }
      }else{
        // Raw chunked text
        els.answer.textContent += chunk;
      }
    }

    // Try to fetch citations/usage via trace endpoint if we have a trace id
    if(!els.citations.childElementCount){
      const tidFromFooter = extractTraceIdFromFooter(els.answer.textContent);
      const tid = traceId || tidFromFooter;
      if(tid){
        try{
          const tRes = await fetch(urlTrace(tid), {
            headers:{ 'Authorization': `Bearer ${token}` }
          });
          if(tRes.ok){
            const t = await tRes.json();
            if(t.citations) renderCitations(t.citations);
            if(t.usage) renderUsage(t.usage);
          }
        }catch{}
      }
    }

    els.status.textContent = 'Done';
  }catch(err){
    if(err?.name === 'AbortError'){
      showError('Stopped by user.');
      els.status.textContent = 'Stopped';
    }else{
      showError(String(err));
      els.status.textContent = 'Failed';
    }
  }finally{
    const ms = Math.round(performance.now() - startedAt);
    if(!els.usage.textContent.includes('latency_ms=')){
      renderUsage({ latency_ms: ms });
    }
    setBusy(false);
    controller = null;
  }
}

function extractTraceIdFromFooter(text){
  // Optional: if your server appends something like "\n---\ntrace_id: <id>"
  const m = text.match(/trace[_\- ]?id\s*[:=]\s*([a-zA-Z0-9\-\_]+)/i);
  return m?.[1] ?? null;
}

// ====== health check ======
async function health(){
  savePrefs();
  const base = els.baseUrl.value.trim().replace(/\/+$/,'');
  const url = (base || '') + '/v1/health';
  els.status.textContent = 'Checking health…';
  try{
    const res = await fetch(url, {
      headers:{ 'Authorization': `Bearer ${els.token.value.trim()}` }
    });
    const body = await res.text();
    if(!res.ok){ showError(`HTTP ${res.status} ${res.statusText} — ${body}`); els.status.textContent='Health failed'; return; }
    els.answer.textContent = prettyJson(body);
    els.status.textContent = 'Health OK';
    els.err.classList.add('hidden');
  }catch(e){
    showError(String(e));
    els.status.textContent = 'Health failed';
  }
}

function prettyJson(s){
  try{ return JSON.stringify(JSON.parse(s), null, 2); }catch{ return s; }
}

// ====== wire events ======
function init(){
  loadPrefs();
  els.btnSend.addEventListener('click', send);
  els.query.addEventListener('keydown', (e)=>{
    if((e.metaKey || e.ctrlKey) && e.key === 'Enter'){ send(); }
  });
  els.btnStop.addEventListener('click', ()=> controller?.abort());
  els.btnCopy.addEventListener('click', async ()=>{
    try{ await navigator.clipboard.writeText(els.answer.textContent); }
    catch{}
  });
  els.btnHealth.addEventListener('click', health);
  els.btnClear.addEventListener('click', ()=>{
    els.query.value=''; clearOutput(); els.status.textContent='Idle';
  });
  els.toggleToken.addEventListener('click', ()=>{
    els.token.type = (els.token.type === 'password') ? 'text' : 'password';
  });
  els.samples.addEventListener('click', (e)=>{
    if(e.target.tagName === 'LI'){ els.query.value = e.target.textContent.trim(); }
  });
}
document.addEventListener('DOMContentLoaded', init);
