/* Embedding and reranking are global service roles, separate from chat models. */
(() => {
  'use strict';
  const labels = {
    embedding: '文本向量', visual_embedding: '视觉向量',
    reranker: '文本重排', visual_reranker: '视觉重排',
  };
  let loading = null;
  const cards = new Map();

  // Display prerequisites only. Saving still uses the existing server-side
  // connection check; the switch is not a live service-health indicator.
  function hasEndpointFields(row) {
    return !!row.base_url && (!row.role.includes('embedding') || (!!row.model_id && row.dim > 0));
  }

  function view(card, row) {
    card.dataset.editing = '';
    card.dataset.busy = '';
    const required = row.role === 'embedding';
    const configured = hasEndpointFields(row);
    const validated = required || row.role.includes('reranker') ? row.validated === true : configured;
    card.innerHTML = `<div class="llm-top"><span class="llm-title">${t(labels[row.role])}</span>
      <span class="llm-actions">${row.has_api_key ? `<span class="llm-chip key">${t('Key 已配置')}</span>` : ''}
      <button class="sm ghost endpoint-edit">${t('编辑')}</button>
      <span class="ep-toggle-slot"><label class="switch ep-switch${validated ? '' : ' unconfigured'}">
      <input type="checkbox" role="switch" class="ep-toggle" aria-label="${t(labels[row.role])}"
      ${required || row.enabled ? 'checked' : ''} ${required ? 'disabled' : ''}><i></i></label></span></span></div>
      <div class="llm-meta">${esc(row.model_id || t('自动发现'))}<br>
      ${esc(row.base_url || t('未配置'))}${row.dim ? `<br>${t('向量维度')}: ${row.dim}<br>${t('批量大小')}: ${row.batch_size} · ${t('请求并发')}: ${row.concurrency}` : ''}</div>`;
    card.querySelector('.endpoint-edit').addEventListener('click', () => form(card, row));
    card.querySelector('.ep-toggle').addEventListener('change', async event => {
      if (required || card.dataset.busy) return;
      const enabled = event.target.checked;
      if (!configured) {
        // Like other unsaved form values, this preference is held until the
        // missing endpoint is supplied and saved. Never activate an empty URL.
        card.dataset.draft = 'true';
        view(card, {...row, enabled});
        return;
      }
      card.dataset.busy = 'true';
      card.querySelectorAll('input,button').forEach(el => { el.disabled = true; });
      try {
        const result = await api(`/model-endpoints/${row.role}`, {
          method: 'PUT', body: {enabled, revision: row.revision},
        });
        card.dataset.draft = '';
        view(card, result);
        toast(t('已保存；新任务使用新配置，检索配置将在一分钟内更新'));
      } catch (error) {
        view(card, row);
        toast(t(error.message));
      }
    });
  }

  function form(card, row) {
    card.dataset.editing = 'true';
    card.innerHTML = `<div class="llm-top"><span class="llm-title">${t(labels[row.role])}</span>
      </div>
      <div class="fld"><label class="f" for="${row.role}-url">Base URL</label>
      <input id="${row.role}-url" class="ep-url mono" value="${esc(row.base_url)}" placeholder="http://127.0.0.1:8000/v1"></div>
      <div class="fld"><label class="f" for="${row.role}-model">Model ID</label>
      <input id="${row.role}-model" class="ep-model mono" value="${esc(row.model_id)}" placeholder="${row.dim ? 'Model ID' : t('自动发现')}"></div>
      ${row.dim ? `<div class="fld"><label class="f" for="${row.role}-dim">${t('向量维度')}</label>
      <input id="${row.role}-dim" type="number" class="ep-dim mono" min="1" max="65536" step="1" value="${row.dim}"></div>` : ''}
      ${row.dim ? `<div class="ep-throughput">
      <div class="fld"><label class="f" for="${row.role}-batch">${t('批量大小')}</label>
      <input id="${row.role}-batch" type="number" class="ep-batch mono" min="1" max="1024" step="1" value="${row.batch_size}" title="${t('每次请求的文本块或图片数量')}"></div>
      <div class="fld"><label class="f" for="${row.role}-concurrency">${t('请求并发')}</label>
      <input id="${row.role}-concurrency" type="number" class="ep-concurrency mono" min="1" max="32" step="1" value="${row.concurrency}" title="${t('单个任务同时发送的请求数')}"></div></div>` : ''}
      <div class="fld"><label class="f" for="${row.role}-key">API Key</label>
      <input id="${row.role}-key" class="ep-key mono" type="password" autocomplete="new-password"
      placeholder="${t(row.has_api_key ? '已配置,留空保持不变' : '本地服务可留空')}"></div>
      <div class="endpoint-feedback" role="status"></div>
      <div class="c-actions"><button class="ep-cancel">${t('取消')}</button>
      <button class="ep-save primary">${t('保存')}</button></div>`;
    const keyField = attachKeyVisibility(card.querySelector('.ep-key'), {
      hasKey: row.has_api_key,
      loadKey: () => api(`/model-endpoints/${row.role}/key`, {method: 'POST', body: {revision: row.revision}}),
    });
    const value = () => ({base_url: card.querySelector('.ep-url').value.trim(),
      model_id: card.querySelector('.ep-model').value.trim(), api_key: keyField.value(),
      enabled: row.role === 'embedding' || row.enabled,
      revision: row.revision, ...(row.dim ? {dim: Number(card.querySelector('.ep-dim').value),
        batch_size: Number(card.querySelector('.ep-batch').value),
        concurrency: Number(card.querySelector('.ep-concurrency').value)} : {})});
    let busy = false;
    async function submit() {
      if (busy) return;
      const payload = value();
      const feedback = card.querySelector('.endpoint-feedback');
      if (!payload.base_url || (row.dim && !payload.model_id)) {
        feedback.textContent = t('请填写接口地址和模型 ID'); return;
      }
      if (row.dim && (!Number.isInteger(payload.dim) || payload.dim < 1 || payload.dim > 65536)) {
        feedback.textContent = t('Embedding dimensions must be an integer from 1 to 65536'); return;
      }
      if (row.dim && (!Number.isInteger(payload.batch_size) || payload.batch_size < 1 || payload.batch_size > 1024)) {
        feedback.textContent = t('Batch size must be an integer from 1 to 1024'); return;
      }
      if (row.dim && (!Number.isInteger(payload.concurrency) || payload.concurrency < 1 || payload.concurrency > 32)) {
        feedback.textContent = t('Concurrency must be an integer from 1 to 32'); return;
      }
      busy = true;
      card.querySelectorAll('input,button').forEach(el => { el.disabled = true; });
      card.querySelector('.ep-save').textContent = t('测试连通性…');
      feedback.textContent = t('正在验证连接…');
      try {
        const result = await api(`/model-endpoints/${row.role}`, {method: 'PUT', body: payload});
        card.dataset.draft = '';
        view(card, result);
        toast(t('已保存；新任务使用新配置，检索配置将在一分钟内更新'));
      } catch (error) { feedback.textContent = t(error.message); }
      finally {
        busy = false;
        // A saved text-embedding card contains a permanently read-only switch.
        if (card.dataset.editing) {
          card.querySelectorAll('input,button').forEach(el => { el.disabled = false; });
          card.querySelector('.ep-save').textContent = t('保存');
        }
      }
    }
    card.querySelector('.ep-cancel').addEventListener('click', () => { if (!busy) view(card, row); });
    card.querySelector('.ep-save').addEventListener('click', submit);
  }

  window.refreshModelEndpoints = function () {
    if (loading) return loading;
    loading = (async () => {
      try {
        const data = await api('/model-endpoints');
        for (const row of data.endpoints) {
          if (!(row.role in labels)) continue;
          let card = cards.get(row.role);
          if (!card) {
            card = document.createElement('div');
            card.className = 'llm-card endpoint-card';
            card.dataset.role = row.role;
            document.querySelector(row.role.includes('embedding') ? '#embedding-cards' : '#reranker-cards').appendChild(card);
            cards.set(row.role, card);
          }
          if (!card.dataset.editing && !card.dataset.busy && !card.dataset.draft) view(card, row);
        }
      } catch (error) { toast(t('模型配置读取失败: ') + t(error.message)); }
    })().finally(() => { loading = null; });
    return loading;
  };

  const style = document.createElement('style');
  style.textContent = `
    #llm-win{width:560px;max-width:100vw;--settings-inset:15px;--settings-action-width:52px}
    #llm-win .pw-head{padding-left:31px}
    #llm-win .settings-section + .settings-section{margin-top:18px}
    #llm-win .settings-section{padding-bottom:18px;border-bottom:1px solid var(--line)}
    #llm-win .settings-section:last-child{padding-bottom:0;border-bottom:0}
    #llm-win .settings-section-head{display:flex;align-items:center;gap:8px;margin:0 0 12px;min-height:30px;padding:0 var(--settings-inset)}
    #llm-win .settings-section-head h2{margin:0;font-size:13px;font-weight:600;color:var(--text-2)}
    #llm-win #llm-add{margin:0 0 0 auto;width:auto;padding:4px 10px;font-size:12px;border-radius:7px}
    #llm-win .llm-card:last-child{margin-bottom:0}
    #llm-win .llm-top{min-height:26px}
    #llm-win .llm-title{flex:1 1 auto;width:auto;min-width:0;text-overflow:ellipsis}
    #llm-win .llm-actions{gap:8px}
    #llm-win .llm-actions .llm-chip{display:inline-flex;align-items:center;min-height:26px;margin-right:0;box-sizing:border-box}
    #llm-win .llm-actions button{height:26px;padding:0 10px}
    #llm-win .llm-actions .c-edit,#llm-win .endpoint-edit{width:46px;flex:0 0 46px}
    .endpoint-card .llm-meta{overflow-wrap:anywhere}
    #llm-cards .llm-actions > button:last-child,.ep-toggle-slot{width:var(--settings-action-width);flex:0 0 var(--settings-action-width);box-sizing:border-box}
    .ep-toggle-slot{display:flex;align-items:center;justify-content:center}
    .ep-switch.unconfigured input:checked + i{background:var(--line-strong)}
    .ep-switch:has(input:disabled){cursor:default}
    .ep-throughput{display:grid;grid-template-columns:1fr 1fr;gap:12px}
    .endpoint-feedback{font-size:12px;color:var(--text-2);margin:8px 0;overflow-wrap:anywhere}
  `;
  document.head.appendChild(style);
})();
