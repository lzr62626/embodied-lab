(() => {
  const root = document.documentElement;
  const themeButton = document.querySelector('[data-theme-toggle]');
  function updateThemeLabel() {
    const dark = root.dataset.theme ? root.dataset.theme === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches;
    themeButton?.setAttribute('aria-label', dark ? '切换为浅色主题' : '切换为深色主题');
  }
  updateThemeLabel();
  themeButton?.addEventListener('click', () => {
    const dark = root.dataset.theme ? root.dataset.theme === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches;
    root.dataset.theme = dark ? 'light' : 'dark';
    try { localStorage.setItem('embodied-lab-theme',root.dataset.theme); } catch {}
    updateThemeLabel();
  });
  const dialog = document.querySelector('.search-dialog');
  const input = document.querySelector('#site-search');
  const results = document.querySelector('.search-results');
  const clearButton = document.querySelector('[data-search-clear]');
  const articles = JSON.parse(document.querySelector('#search-data').textContent);
  let composing = false;
  let searchTrigger = null;
  function search() {
    clearButton.hidden = !input.value;
    const words = input.value.trim().toLocaleLowerCase().split(/\s+/).filter(Boolean);
    const matches = articles.filter(a => words.every(w => (a.title+' '+a.description+' '+a.tags.join(' ')).toLocaleLowerCase().includes(w)));
    results.replaceChildren();
    if (!matches.length) {
      const empty = document.createElement('p');
      empty.className = 'search-empty';
      empty.textContent = '暂时没有找到相关文章，可以试试“观测”“动作”或“仿真”。';
      results.append(empty);
      return;
    }
    for (const a of matches) {
      const link = document.createElement('a');
      link.href = a.url;
      const heading = document.createElement('h3');
      heading.textContent = a.title;
      const description = document.createElement('p');
      description.textContent = a.description;
      link.append(heading,description);
      results.append(link);
    }
  }
  function openSearch() {
    if (dialog.open) return;
    searchTrigger = document.activeElement;
    search();
    dialog.showModal();
    root.setAttribute('data-modal-open','');
    input.focus();
  }
  document.querySelector('[data-search-open]')?.addEventListener('click',openSearch);
  input.addEventListener('compositionstart',() => { composing = true; });
  input.addEventListener('compositionend',() => { composing = false; search(); });
  input.addEventListener('input',event => { if (!composing && !event.isComposing) search(); });
  clearButton.addEventListener('click',() => {
    input.value = '';
    composing = false;
    search();
    input.focus();
  });
  dialog.addEventListener('close',() => {
    root.removeAttribute('data-modal-open');
    if (searchTrigger instanceof HTMLElement && searchTrigger.isConnected) searchTrigger.focus({preventScroll:true});
  });
  dialog.addEventListener('click',event => {
    const r = dialog.getBoundingClientRect();
    if (event.target === dialog && (event.clientX < r.left || event.clientX > r.right || event.clientY < r.top || event.clientY > r.bottom)) dialog.close();
  });
  document.addEventListener('keydown',event => {
    if (event.isComposing || composing || event.keyCode === 229) return;
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'k') { event.preventDefault(); openSearch(); }
  });
  document.querySelectorAll('.prose pre').forEach(pre => {
    const button = document.createElement('button');
    button.className = 'copy-code';
    button.type = 'button';
    button.textContent = '复制';
    button.setAttribute('aria-label','复制这段代码或任务卡');
    button.setAttribute('aria-live','polite');
    button.addEventListener('click',async () => {
      const code = pre.querySelector('code');
      try {
        if (!navigator.clipboard?.writeText) throw new Error('clipboard unavailable');
        await navigator.clipboard.writeText(code.textContent);
        button.textContent = '已复制';
        button.setAttribute('aria-label','内容已复制');
      } catch {
        const range = document.createRange();
        range.selectNodeContents(code);
        const selection = window.getSelection();
        selection.removeAllRanges();
        selection.addRange(range);
        button.textContent = '已选中';
        button.setAttribute('aria-label','内容已选中，请手动复制');
      }
      setTimeout(() => { button.textContent = '复制'; button.setAttribute('aria-label','复制这段代码或任务卡'); },2000);
    });
    pre.append(button);
  });
  document.querySelectorAll('[data-trajectory-replay]').forEach(figure => {
    const image = figure.querySelector('img[data-replay-src]');
    const button = figure.querySelector('[data-replay-toggle]');
    const status = figure.querySelector('[data-replay-status]');
    if (!image || !button || !status) return;
    const poster = image.getAttribute('src');
    const source = new URL(image.dataset.replaySrc,document.baseURI);
    const seconds = (Number(image.dataset.durationMs)/1000).toFixed(2);
    let playing = false;
    let failed = false;
    let version = 0;
    function updateButton() {
      button.textContent = playing?'停止回放':'播放回放';
      button.setAttribute('aria-pressed',String(playing));
    }
    image.addEventListener('load',() => {
      if (failed) return;
      status.textContent = playing?`循环回放中 · 每轮 ${seconds} 秒。`:'当前显示原始第0帧。';
    });
    image.addEventListener('error',() => {
      if (!playing) return;
      playing = false;
      failed = true;
      updateButton();
      status.textContent = '回放未能加载，请使用下方下载链接。';
      image.src = poster;
    });
    button.addEventListener('click',() => {
      playing = !playing;
      failed = false;
      updateButton();
      if (playing) {
        source.searchParams.set('play',String(++version));
        status.textContent = '正在加载真实回放…';
        image.src = source.href;
      } else {
        image.src = poster;
        status.textContent = '当前显示原始第0帧。';
      }
    });
    button.hidden = false;
  });
  const links = Array.from(document.querySelectorAll('.article-toc a[href^="#"]'));
  if (links.length && 'IntersectionObserver' in window) {
    const observer = new IntersectionObserver(entries => {
      for (const entry of entries) if (entry.isIntersecting) {
        links.forEach(link => link.classList.toggle('is-active',link.hash === '#'+entry.target.id));
      }
    },{rootMargin:'-15% 0px -60% 0px'});
    document.querySelectorAll('.prose h2[id]').forEach(heading => observer.observe(heading));
  }
})();
