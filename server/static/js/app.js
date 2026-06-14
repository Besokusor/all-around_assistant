/**
 * 全局状态
 */
const App = {
  currentSessionId: null,
  sessions: [],
  pendingImage: null,
  isProcessing: false,
};

const API = {
  async request(method, path, body) {
    const opts = { method, headers: {} };
    if (body && !(body instanceof FormData)) {
      opts.headers['Content-Type'] = 'application/json';
      opts.body = JSON.stringify(body);
    } else if (body instanceof FormData) {
      opts.body = body;
    }
    const resp = await fetch(path, opts);
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ detail: resp.statusText }));
      throw new Error(err.detail || resp.statusText);
    }
    return resp.json();
  },
  get(path) { return this.request('GET', path); },
  post(path, body) { return this.request('POST', path, body); },
  delete(path) { return this.request('DELETE', path); },
};

document.addEventListener('DOMContentLoaded', async () => {
  Sessions.init();
  Documents.init();
  Chat.init();

  try {
    const data = await API.get('/api/sessions');
    App.sessions = data.sessions || [];
    if (App.sessions.length > 0) {
      await Sessions.switch(App.sessions[0].session_id);
    } else {
      await Sessions.create();
    }
  } catch (e) {
    console.error('Init failed:', e);
    await Sessions.create();
  }
});
