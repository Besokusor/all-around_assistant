/**
 * 知识库管理
 */
const Documents = {
  init() {
    const btn = document.getElementById('btn-toggle-knowledge');
    const close = document.getElementById('btn-close-knowledge');
    const panel = document.getElementById('knowledge-panel');

    btn.addEventListener('click', () => panel.classList.toggle('hidden'));
    close.addEventListener('click', () => panel.classList.add('hidden'));

    // Upload zone
    const zone = document.getElementById('doc-upload-zone');
    const fi = document.getElementById('file-doc-input');
    zone.addEventListener('click', () => fi.click());
    zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('drag-over'); });
    zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
    zone.addEventListener('drop', e => {
      e.preventDefault(); zone.classList.remove('drag-over');
      if (e.dataTransfer.files[0]) this.upload(e.dataTransfer.files[0]);
    });
    fi.addEventListener('change', () => { if (fi.files[0]) this.upload(fi.files[0]); });

    // QA
    document.getElementById('btn-qa-send').addEventListener('click', () => this.ask());
    document.getElementById('qa-input').addEventListener('keydown', e => {
      if (e.key === 'Enter') this.ask();
    });

    this.load();
  },

  async upload(file) {
    const zone = document.getElementById('doc-upload-zone');
    const orig = zone.innerHTML;
    zone.innerHTML = '⏳ 上传中...';
    try {
      const form = new FormData();
      form.append('file', file);
      const resp = await fetch('/api/documents/upload', { method: 'POST', body: form });
      const data = await resp.json();
      if (data.success) {
        zone.innerHTML = orig;
        await this.load();
      } else {
        zone.innerHTML = `<span style="color:var(--error)">❌ ${data.error || '失败'}</span>`;
        setTimeout(() => { zone.innerHTML = orig; }, 3000);
      }
    } catch (e) {
      zone.innerHTML = `<span style="color:var(--error)">❌ ${e.message}</span>`;
      setTimeout(() => { zone.innerHTML = orig; }, 3000);
    }
  },

  async load() {
    try {
      const data = await API.get('/api/documents');
      const docs = data.documents || [];
      const list = document.getElementById('doc-list');
      if (!docs.length) {
        list.innerHTML = '<div style="padding:16px;color:var(--text-secondary);font-size:13px;text-align:center">暂无文档</div>';
      } else {
        list.innerHTML = docs.map(d => `
          <div class="doc-item">
            <div class="doc-info">
              <div class="doc-name">📄 ${this.escapeHtml(d.source)}</div>
              <div class="doc-meta">${d.file_type} · ${d.chunk_count} chunks</div>
            </div>
            <span class="doc-delete" onclick="Documents.remove('${d.source.replace(/'/g, "\\'")}')">✕</span>
          </div>
        `).join('');
      }
    } catch (e) {
      console.error('Load docs failed:', e);
    }
  },

  async remove(name) {
    if (!confirm(`确定删除「${name}」？`)) return;
    try {
      await API.delete(`/api/documents/${encodeURIComponent(name)}`);
      await this.load();
    } catch (e) {
      console.error('Delete doc failed:', e);
    }
  },

  async ask() {
    const input = document.getElementById('qa-input');
    const query = input.value.trim();
    if (!query) return;
    const result = document.getElementById('qa-result');
    result.innerHTML = '⏳ 检索中...';
    try {
      const data = await API.post('/api/knowledge/qa', { query });
      if (data.answerable) {
        result.innerHTML = `<div style="white-space:pre-wrap;line-height:1.7">${this.escapeHtml(data.answer)}</div>`;
        if (data.sources?.length) {
          const srcs = [...new Set(data.sources.map(s => s.source))];
          result.innerHTML += `<div style="font-size:11px;color:var(--text-secondary);margin-top:8px">📚 ${srcs.join(', ')}</div>`;
        }
      } else {
        result.innerHTML = `<div style="color:var(--text-secondary)">${this.escapeHtml(data.answer)}</div>`;
      }
    } catch (e) {
      result.innerHTML = `<div style="color:var(--error)">❌ ${e.message}</div>`;
    }
  },

  escapeHtml(text) {
    const d = document.createElement('div');
    d.textContent = text;
    return d.innerHTML;
  },
};
