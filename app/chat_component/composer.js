(() => {
  'use strict';
  const input = document.getElementById('question');
  const list = document.getElementById('suggestions');
  const send = document.getElementById('send');
  const status = document.getElementById('status');
  let suggestions = [], visible = [], active = -1, focused = false, dismissed = false;
  let previousHeight = 0, sequence = 0;
  const sessionId = Date.now().toString(36) + Math.random().toString(36).slice(2);
  const post = (type, rest = {}) => window.parent.postMessage({ isStreamlitMessage: true, type, ...rest }, '*');
  const tokens = value => value.toLocaleLowerCase().match(/[\p{L}\p{M}\p{N}]+/gu) || [];

  function resize() {
    input.style.height = '44px';
    input.style.height = Math.min(140, Math.max(44, input.scrollHeight)) + 'px';
    requestAnimationFrame(() => {
      const height = Math.ceil(document.body.getBoundingClientRect().height) + 2;
      if (height !== previousHeight) {
        previousHeight = height;
        post('streamlit:setFrameHeight', { height });
      }
    });
  }

  function matches(query) {
    const words = tokens(query).filter(word => word.length >= 2);
    if (!words.length) return suggestions;
    return suggestions.map((item, order) => {
      const keys = tokens(item.keywords);
      const score = words.reduce((sum, word) => sum + (keys.some(key => key === word) ? 3 : keys.some(key => key.startsWith(word)) ? 1 : 0), 0);
      return { item, order, score };
    }).filter(row => row.score > 0).sort((a, b) => b.score - a.score || a.order - b.order).map(row => row.item);
  }

  function highlight() {
    for (let i = 0; i < list.children.length; i++) {
      list.children[i].setAttribute('aria-selected', String(i === active));
    }
    if (active >= 0) {
      input.setAttribute('aria-activedescendant', `suggestion-${active}`);
      list.children[active].scrollIntoView({ block: 'nearest' });
    } else input.removeAttribute('aria-activedescendant');
  }

  function refresh() {
    visible = focused && !dismissed ? matches(input.value) : [];
    active = -1;
    list.replaceChildren();
    visible.forEach((item, index) => {
      const option = document.createElement('li');
      option.id = `suggestion-${index}`;
      option.setAttribute('role', 'option');
      option.setAttribute('aria-selected', 'false');
      option.textContent = item.question;
      option.addEventListener('pointerdown', event => { event.preventDefault(); choose(index); });
      list.appendChild(option);
    });
    list.hidden = visible.length === 0;
    input.setAttribute('aria-expanded', String(visible.length > 0));
    input.removeAttribute('aria-activedescendant');
    send.disabled = !input.value.trim();
    status.textContent = visible.length ? `${visible.length} सवालों के सुझाव` : '';
    resize();
  }

  function choose(index) {
    if (!visible[index]) return;
    input.value = visible[index].question;
    dismissed = true;
    input.focus();
    refresh();
  }

  function submit(event) {
    event.preventDefault();
    const text = input.value.trim();
    if (!text) return;
    post('streamlit:setComponentValue', { value: { text, id: `${sessionId}-${++sequence}` }, dataType: 'json' });
    input.value = '';
    dismissed = true;
    refresh();
  }

  input.addEventListener('focus', () => { focused = true; dismissed = false; refresh(); });
  input.addEventListener('blur', () => { focused = false; refresh(); });
  input.addEventListener('input', () => { dismissed = false; refresh(); });
  input.addEventListener('keydown', event => {
    if (event.isComposing || event.keyCode === 229) return;
    if ((event.key === 'ArrowDown' || event.key === 'ArrowUp') && visible.length) {
      event.preventDefault();
      active = event.key === 'ArrowDown' ? (active + 1) % visible.length : (active <= 0 ? visible.length - 1 : active - 1);
      highlight();
    } else if (event.key === 'Escape') {
      dismissed = true;
      refresh();
    } else if (event.key === 'Enter' && !event.shiftKey) {
      if (active >= 0) { event.preventDefault(); choose(active); }
      else submit(event);
    }
  });
  send.addEventListener('pointerdown', event => event.preventDefault());
  document.getElementById('chat-form').addEventListener('submit', submit);
  window.addEventListener('message', event => {
    if (event.source !== window.parent || event.data?.type !== 'streamlit:render') return;
    const next = (event.data.args?.suggestions || []).filter(item => typeof item.question === 'string' && typeof item.keywords === 'string');
    const changed = JSON.stringify(next) !== JSON.stringify(suggestions);
    suggestions = next;
    const theme = event.data.theme;
    if (theme) {
      for (const [name, value] of Object.entries({ background: theme.backgroundColor, foreground: theme.textColor, secondary: theme.secondaryBackgroundColor, accent: theme.primaryColor })) {
        if (value) document.documentElement.style.setProperty(`--${name}`, value);
      }
    }
    if (changed) refresh();
    else resize();
  });
  window.addEventListener('resize', resize);
  post('streamlit:componentReady', { apiVersion: 1 });
  resize();
})();
