/**
 * 聊天模块 — SSE 流式 + 图片上传 + 多轮对话
 */
const Chat = {
  init() {
    const input = document.getElementById('chat-input');
    const btn = document.getElementById('btn-send');
    const imgBtn = document.getElementById('btn-upload-image');
    const imgInput = document.getElementById('file-image-input');

    btn.addEventListener('click', () => this.send());
    input.addEventListener('keydown', e => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        this.send();
      }
    });
    input.addEventListener('input', () => {
      input.style.height = 'auto';
      input.style.height = Math.min(input.scrollHeight, 140) + 'px';
    });

    imgBtn.addEventListener('click', () => imgInput.click());
    imgInput.addEventListener('change', () => {
      if (imgInput.files[0]) this.setImage(imgInput.files[0]);
    });
    document.getElementById('btn-remove-image').addEventListener('click', () => this.clearImage());

    // Suggestion chips are bound dynamically in showWelcome()

    // Paste image
    document.addEventListener('paste', e => {
      const f = e.clipboardData?.files?.[0];
      if (f && f.type.startsWith('image/')) { e.preventDefault(); this.setImage(f); }
    });

    // Drag-drop
    const mc = document.getElementById('messages-container');
    mc.addEventListener('dragover', e => e.preventDefault());
    mc.addEventListener('drop', e => {
      e.preventDefault();
      const f = e.dataTransfer.files[0];
      if (f && f.type.startsWith('image/')) this.setImage(f);
    });

    // Sidebar toggle
    document.getElementById('btn-toggle-sidebar').addEventListener('click', () => {
      document.getElementById('sidebar').classList.toggle('open');
    });
  },

  setImage(file) {
    App.pendingImage = { file };
    const reader = new FileReader();
    reader.onload = () => {
      document.getElementById('image-preview').src = reader.result;
      document.getElementById('image-preview-container').style.display = 'inline-block';
    };
    reader.readAsDataURL(file);
  },

  clearImage() {
    App.pendingImage = null;
    document.getElementById('image-preview-container').style.display = 'none';
    document.getElementById('file-image-input').value = '';
  },

  _cleanup() {
    App.isProcessing = false;
    App.pendingImage = null;
    this.clearImage();
    document.getElementById('btn-send').disabled = false;
    document.getElementById('chat-input').focus();
    // 刷新侧边栏（更新标题预览）
    Sessions.load();
  },

  async send() {
    if (App.isProcessing) {
      console.log('send blocked: isProcessing');
      return;
    }

    const input = document.getElementById('chat-input');
    const message = input.value.trim();

    // 在清理前保存图片引用和预览
    const imageFile = App.pendingImage ? App.pendingImage.file : null;
    const imagePreviewSrc = document.getElementById('image-preview').src || '';

    console.log('send:', { message, hasImage: !!imageFile, sessionId: App.currentSessionId });

    if (!message && !imageFile) return;

    if (!App.currentSessionId) {
      try { await Sessions.create(); } catch (e) { console.error(e); return; }
    }

    // 先清理输入框和图片预览
    input.value = '';
    input.style.height = 'auto';
    this.clearImage();  // 清除 pendingImage + 预览
    App.isProcessing = true;
    document.getElementById('btn-send').disabled = true;
    this.clearWelcome();

    // User bubble（如果有图片预览，显示缩略图）
    const userBubble = this.addMessage('user', '');
    const userContent = userBubble.querySelector('.message-content');
    if (message) {
      userContent.textContent = message;
    }
    if (imagePreviewSrc && imagePreviewSrc.startsWith('data:')) {
      const img = document.createElement('img');
      img.src = imagePreviewSrc;
      img.style.cssText = 'max-width:200px;max-height:200px;border-radius:8px;display:block;';
      if (message) img.style.marginTop = '8px';
      userContent.appendChild(img);
    }
    if (!message && !imagePreviewSrc) {
      userContent.textContent = '[图片]';
    }

    // Assistant bubble
    const bubbleDiv = this.addMessage('assistant', '', false);
    const contentDiv = bubbleDiv.querySelector('.message-content');

    // Spinner below bubble
    const spinnerRow = document.createElement('div');
    spinnerRow.className = 'spinner-row';
    spinnerRow.innerHTML = '<div class="spinner"></div><span>思考中...</span>';
    bubbleDiv.appendChild(spinnerRow);

    // Build FormData（用保存的 imageFile，不依赖 App.pendingImage）
    const form = new FormData();
    form.append('message', message || '请分析这张图片');
    form.append('session_id', App.currentSessionId);
    if (imageFile) {
      form.append('image', imageFile);
      console.log('image appended:', imageFile.name, imageFile.size);
    }

    let fullContent = '';
    let doneReceived = false;

    const finish = (errorMsg) => {
      try { spinnerRow.remove(); } catch (e) { /* already removed */ }
      if (errorMsg) {
        contentDiv.textContent = `❌ ${errorMsg}`;
      } else if (!doneReceived && fullContent) {
        contentDiv.innerHTML = this.renderMarkdown(fullContent);
      }
      this.addTime(bubbleDiv);
      this._cleanup();
    };

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        body: form,
        headers: { 'Accept': 'text/event-stream' },
      });

      if (!response.ok) {
        const err = await response.json().catch(() => ({ detail: `HTTP ${response.status}` }));
        return finish(err.detail || '请求失败');
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      // 读 SSE 流，收到 done/error 后主动退出（不等连接关闭）
      const readLoop = async () => {
        while (true) {
          let result;
          try {
            result = await reader.read();
          } catch (e) {
            if (!doneReceived) finish('连接中断');
            return;
          }
          const { done, value } = result;
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const parts = buffer.split('\n\n');
          buffer = parts.pop() || '';

          for (const part of parts) {
            for (const line of part.split('\n')) {
              if (!line.startsWith('data: ')) continue;
              let json;
              try { json = JSON.parse(line.slice(6)); }
              catch (e) { continue; }

              switch (json.type) {
                case 'node_event':
                  spinnerRow.querySelector('span').textContent =
                    json.node || '处理中...';
                  break;
                case 'token':
                  fullContent += json.content;
                  contentDiv.textContent = fullContent;
                  this.scrollToBottom();
                  break;
                case 'done':
                  doneReceived = true;
                  fullContent = json.content || fullContent;
                  // 替换为 Markdown 渲染后的 HTML
                  contentDiv.innerHTML = this.renderMarkdown(fullContent);
                  reader.cancel();
                  return finish(null);
                case 'error':
                  return finish(json.content);
              }
            }
          }
        }
        finish(null);
      };

      // 加超时保护（120秒）
      const timeout = setTimeout(() => {
        if (!doneReceived) finish('请求超时');
      }, 120000);

      await readLoop();
      clearTimeout(timeout);
    } catch (e) {
      finish(e.message || '网络错误');
    }
  },

  addMessage(role, content) {
    const div = document.createElement('div');
    div.className = `message ${role}`;
    div.innerHTML = `<div class="message-content">${this.escapeHtml(content)}</div>`;
    document.getElementById('messages-container').appendChild(div);
    this.scrollToBottom();
    return div;
  },

  addTime(bubble) {
    const t = document.createElement('div');
    t.className = 'time';
    t.textContent = new Date().toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' });
    bubble.appendChild(t);
  },

  clearMessages() {
    document.getElementById('messages-container').innerHTML = '';
    this.showWelcome();
  },

  clearWelcome() {
    const w = document.querySelector('.welcome-message');
    if (w) w.remove();
  },

  showWelcome() {
    if (document.querySelector('.welcome-message')) return;
    const mc = document.getElementById('messages-container');
    mc.innerHTML = `
      <div class="welcome-message">
        <h1>🤖 个人全能助手</h1>
        <p>智能问答 · 联网搜索 · 出行规划 · 计算 · 菜谱推荐 · CLI · 图像分析 · 知识问答</p>
        <div class="suggestion-chips">
          <span class="chip">今天天气怎么样？</span>
          <span class="chip">计算 (100+50)*3.14</span>
          <span class="chip">鸡蛋和西红柿能做什么菜？</span>
          <span class="chip">搜索AI最新新闻</span>
        </div>
      </div>`;
    // Re-bind chips
    mc.querySelectorAll('.chip').forEach(chip => {
      chip.addEventListener('click', () => {
        document.getElementById('chat-input').value = chip.textContent;
        this.send();
      });
    });
  },

  scrollToBottom() {
    const c = document.getElementById('messages-container');
    requestAnimationFrame(() => { c.scrollTop = c.scrollHeight; });
  },

  escapeHtml(text) {
    const d = document.createElement('div');
    d.textContent = text;
    return d.innerHTML;
  },

  /** 简易 Markdown → HTML */
  renderMarkdown(text) {
    let html = this.escapeHtml(text);
    // 代码块 ```
    html = html.replace(/```(\w*)\n([\s\S]*?)```/g,
      '<pre><code>$2</code></pre>');
    // 行内代码 `...`
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
    // 粗体 **text**
    html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    // 斜体 *text*
    html = html.replace(/\*([^*]+)\*/g, '<em>$1</em>');
    // ### 标题
    html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
    html = html.replace(/^## (.+)$/gm, '<h2>$1</h2>');
    // - 无序列表
    html = html.replace(/^- (.+)$/gm, '<li>$1</li>');
    html = html.replace(/(<li>.*<\/li>)/s, '<ul>$1</ul>');
    // 1. 有序列表
    html = html.replace(/^\d+\. (.+)$/gm, '<li>$1</li>');
    // URL 链接
    html = html.replace(/(https?:\/\/[^\s<>"]+)/g,
      '<a href="$1" target="_blank">$1</a>');
    // 换行
    html = html.replace(/\n\n/g, '<br><br>');
    html = html.replace(/\n/g, '<br>');
    return html;
  },
};
