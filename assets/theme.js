(() => {
  try {
    const theme = localStorage.getItem('embodied-lab-theme');
    if (theme === 'light' || theme === 'dark') document.documentElement.dataset.theme = theme;
  } catch {}
})();
