/* Shared, explicit key reveal for chat, embedding and reranking forms. */
function attachKeyVisibility(input, {hasKey = false, loadKey, isCleared = () => false} = {}) {
  const field = document.createElement('div');
  field.className = 'key-field';
  input.before(field);
  field.appendChild(input);
  input.type = 'password';
  input.spellcheck = false;
  input.setAttribute('aria-label', 'API Key');
  if (hasKey) {
    input.placeholder = '••••••••';
    input.title = t('已配置,留空保持不变');
  }
  const button = document.createElement('button');
  button.type = 'button';
  button.className = 'key-eye';
  button.innerHTML = '<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="1.7" aria-hidden="true"><path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7S2 12 2 12Z"/><circle cx="12" cy="12" r="3"/></svg>';
  field.appendChild(button);
  let edited = false, loadedValue = null, version = 0;
  function visible(on) {
    // Restore the same placeholder after viewing an unchanged saved key.
    if (!on && !edited && input.value === loadedValue) {
      input.value = '';
      loadedValue = null;
    }
    input.type = on ? 'text' : 'password';
    button.setAttribute('aria-pressed', String(on));
    button.title = t(on ? '隐藏密钥' : '显示密钥');
    button.setAttribute('aria-label', button.title);
  }
  visible(false);
  input.addEventListener('input', () => { edited = true; version++; });
  input.addEventListener('model-key-hide', () => { version++; visible(false); });
  button.addEventListener('click', async () => {
    if (button.disabled || input.disabled) return;
    if (input.type === 'text') { visible(false); return; }
    if (!input.value && !edited && hasKey && !isCleared()) {
      const started = version;
      button.disabled = true;
      button.setAttribute('aria-busy', 'true');
      try {
        const data = await loadKey();
        // A late response must not overwrite typed text or reveal a closed form.
        if (!input.isConnected || input.disabled || version !== started || isCleared()) return;
        loadedValue = String(data.api_key || '');
        input.value = loadedValue;
      } catch (error) {
        if (input.isConnected && version === started) toast(t('读取密钥失败: ') + t(error.message));
        return;
      } finally {
        button.disabled = input.disabled;
        button.removeAttribute('aria-busy');
      }
    }
    visible(true);
  });
  // Revealing a saved key is not an edit. Keep server-side preservation and
  // endpoint-change checks effective unless the user actually re-enters it.
  return {value: () => !edited && input.value === loadedValue ? '' : input.value};
}

function hideModelKeys() {
  document.querySelectorAll('.key-field input').forEach(input => {
    input.dispatchEvent(new Event('model-key-hide'));
  });
}
document.addEventListener('visibilitychange', () => { if (document.hidden) hideModelKeys(); });

const keyVisibilityStyle = document.createElement('style');
keyVisibilityStyle.textContent = `
  .key-field{position:relative;display:flex;align-items:center}
  .key-field input{width:100%;padding-right:42px}
  .key-field .key-eye{position:absolute;right:7px;top:50%;transform:translateY(-50%);width:28px;height:28px;padding:5px;display:flex;align-items:center;justify-content:center;border:0;border-radius:5px;background:transparent;color:#9ca3af;box-shadow:none}
  .key-field .key-eye[aria-pressed="true"]{color:#111111}
  .key-field .key-eye:hover{background:var(--bg)}
  .key-field .key-eye:focus-visible{outline:2px solid var(--accent);outline-offset:1px}
  .key-field .key-eye:disabled{opacity:.45;cursor:default}
`;
document.head.appendChild(keyVisibilityStyle);
