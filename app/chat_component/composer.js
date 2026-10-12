(() => {
  'use strict';
  const input = document.getElementById('question');
  const list = document.getElementById('suggestions');
  const send = document.getElementById('send');
  const status = document.getElementById('status');
  let crops = [], storageKey = '', restored = false;
  let suggestions = [], visible = [], active = -1, focused = false, dismissed = false;
  const translate = document.getElementById('translate');
  const preview = document.getElementById('hindi-preview');
  const translationStatus = document.getElementById('translation-status');
  let draftTimer, draftId = '', previewOriginal = '', previewReady = false, appliedPreview = '';
  function persist() {
    if (!storageKey) return;
    try { sessionStorage.setItem(storageKey, JSON.stringify({text: input.value, translate: translate.checked,
      draftId, previewOriginal, previewReady, appliedPreview, previewText: preview.value,
      previewHidden: preview.hidden, status: translationStatus.textContent})); } catch {}
  }
  function draft() {
    clearTimeout(draftTimer);
    draftId = ''; previewReady = false; previewOriginal = ''; preview.hidden = true;
    translationStatus.textContent = '';
    persist();
    if (!translate.checked || !/[a-z]/i.test(input.value)) { resize(); return; }
    translationStatus.textContent = 'हिंदी मसौदा तैयार हो रहा है…';
    draftTimer = setTimeout(() => {
      draftId = `${sessionId}-draft-${++sequence}`;
      persist();
      post('streamlit:setComponentValue', {value:{kind:'draft',text:input.value.trim(),id:draftId},dataType:'json'});
    }, 900);
    resize();
  }
  translate.addEventListener('change', draft);
  preview.addEventListener('input', () => { persist(); resize(); });
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
    const queryWords = tokens(query);
    const cropMatches = crops.flatMap(crop => crop.aliases.map(alias => ({crop, words: tokens(alias)})))
      .filter(row => row.words.length && queryWords.some((_, start) => row.words.every((word, j) => queryWords[start+j] === word)))
      .sort((a,b) => b.words.length-a.words.length);
    if (cropMatches.length) {
      const label = cropMatches[0].crop.label;
      const completions = [
        {question: `${label} की कीमत क्या है?`, keywords: 'price rate keemat kimat bhav daam कीमत भाव मूल्य दाम'},
        {question: `${label} की कौन सी किस्में उपयुक्त हैं?`, keywords: 'variety varieties kisam kism किस्म किस्में प्रकार'},
        {question: `${label} की खेती कैसे करें?`, keywords: 'grow cultivation kheti खेती उगाएं'},
        {question: `${label} में रोग और कीट की पहचान कैसे करें?`, keywords: 'pesticide disease pest rog keet dawa रोग कीट दवा'}
      ];
      const intentWords = words.filter(word => !['ki', 'ka', 'ke', 'की', 'का', 'के'].includes(word));
      const filtered = completions.filter(item => tokens(item.keywords).some(key => intentWords.some(word => key.startsWith(word))));
      return filtered.length ? filtered : completions;
    }
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
    draft();
  }

  function submit(event) {
    event.preventDefault();
    const original = input.value.trim();
    if (!original) return;
    if (translate.checked && /[a-z]/i.test(original) && !previewReady) {
      translationStatus.textContent = 'हिंदी मसौदे की प्रतीक्षा करें, या मूल सवाल भेजने के लिए हिंदी विकल्प बंद करें।';
      resize(); return;
    }
    const text = translate.checked && previewReady && previewOriginal === original && !preview.hidden ? preview.value.trim() : original;
    if (!text) return;
    clearTimeout(draftTimer); draftId = ''; previewReady = false; preview.hidden = true;
    translationStatus.textContent = '';
    post('streamlit:setComponentValue', { value: { kind: 'submit', text, id: `${sessionId}-${++sequence}` }, dataType: 'json' });
    input.value = '';
    persist();
    dismissed = true;
    refresh();
  }

  input.addEventListener('focus', () => { focused = true; dismissed = false; refresh(); });
  input.addEventListener('blur', () => { focused = false; refresh(); });
  input.addEventListener('input', () => { dismissed = false; refresh(); draft(); });
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
    if (!restored) {
      storageKey = 'kisaan-draft:' + event.data.args?.storage_key;
      restored = true;
      try {
        const saved = JSON.parse(sessionStorage.getItem(storageKey) || 'null');
        if (saved) {
          input.value = saved.text || ''; translate.checked = saved.translate !== false;
          draftId = saved.draftId || ''; previewOriginal = saved.previewOriginal || '';
          previewReady = !!saved.previewReady; appliedPreview = saved.appliedPreview || '';
          preview.value = saved.previewText || ''; preview.hidden = saved.previewHidden !== false;
          translationStatus.textContent = saved.status || '';
        }
      } catch {}
      refresh();
    }
    crops = event.data.args?.crops || [];
    const translated = event.data.args?.preview;
    if (draftId && translated?.id === draftId && appliedPreview !== draftId && translated.original === input.value.trim()) {
      appliedPreview = draftId;
      previewOriginal = translated.original; previewReady = true;
      preview.hidden = !['ok', 'unchanged'].includes(translated.status);
      preview.value = translated.text || '';
      translationStatus.textContent = preview.hidden
        ? ({
            not_configured: 'Sarvam की API कुंजी उपलब्ध नहीं है। Streamlit Secrets में SARVAM_API_KEY जोड़ें।',
            authentication_failed: 'Sarvam ने API कुंजी या अनुमति स्वीकार नहीं की। अपनी कुंजी और API पहुँच जाँचें।',
            invalid_request: 'Sarvam ने अनुवाद अनुरोध स्वीकार नहीं किया। ऐप अपडेट या अनुरोध की जाँच ज़रूरी है।',
            rate_limited: 'Sarvam की अनुरोध सीमा पूरी हुई है। थोड़ी देर बाद फिर लिखें।',
            timeout: 'Sarvam से जवाब मिलने में समय लगा। थोड़ी देर बाद फिर कोशिश करें।',
            network_error: 'Sarvam से कनेक्शन नहीं बन सका।',
            service_error: 'Sarvam सेवा अभी जवाब नहीं दे सकी।',
            too_long: 'लाइव हिंदी मसौदे के लिए सवाल 1,000 अक्षरों से छोटा रखें।',
            protected_values_changed: 'अनुवाद में स्थान या संख्या बदल गई थी, इसलिए मूल सवाल रखा गया है।',
            invalid_output: 'Sarvam से उपयोग योग्य हिंदी अनुवाद नहीं मिला।',
          }[translated.status] || 'हिंदी अनुवाद उपलब्ध नहीं है।') + ' मूल सवाल भेज सकते हैं।'
        : 'भेजा जाने वाला हिंदी सवाल — नीचे सुधार सकते हैं:';
    }
    persist();
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
