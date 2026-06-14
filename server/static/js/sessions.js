/**
 * 会话管理
 */
const Sessions = {
  init() {
    document.getElementById('btn-new-session').addEventListener('click', () => this.create());
  },

  async create() {
    try {
      const data = await API.post('/api/sessions', { user_id: 'web_user' });
      await this.load();
      await this.switch(data.session_id);
    } catch (e) {
      console.error('Create session failed:', e);
    }
  },

  async load() {
    try {
      const data = await API.get('/api/sessions');
      App.sessions = data.sessions || [];
      this.render();
    } catch (e) {
      console.error('Load sessions failed:', e);
    }
  },

  async switch(sessionId) {
    App.currentSessionId = sessionId;
    this.render();
    Chat.clearMessages();
    // 用第一条消息作标题，新会话显示默认名称
    const s = App.sessions.find(s => s.session_id === sessionId);
    const title = (s && s.preview) ? s.preview : '个人全能助手';
    document.getElementById('session-title').textContent = title;
    document.getElementById('chat-input').focus();
  },

  async remove(sessionId) {
    if (!confirm('确定删除此会话？')) return;
    try {
      await API.delete(`/api/sessions/${sessionId}`);
      await this.load();
      if (App.currentSessionId === sessionId) {
        if (App.sessions.length > 0) {
          await this.switch(App.sessions[0].session_id);
        } else {
          await this.create();
        }
      }
    } catch (e) {
      console.error('Delete session failed:', e);
    }
  },

  render() {
    const list = document.getElementById('session-list');
    list.innerHTML = App.sessions.map(s => `
      <li class="${s.session_id === App.currentSessionId ? 'active' : ''}"
          onclick="Sessions.switch('${s.session_id}')">
        <div style="font-weight:500">${s.preview || '新会话'}</div>
        <div style="font-size:11px;opacity:.7">${s.message_count} 轮</div>
        <span class="delete-session" onclick="event.stopPropagation();Sessions.remove('${s.session_id}')">✕</span>
      </li>
    `).join('');
  },
};
