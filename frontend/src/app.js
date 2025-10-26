const f = document.getElementById('f');
const out = document.getElementById('out');
const api = document.getElementById('api');
const modeSel = document.getElementById('mode');
const offenseFields = document.getElementById('offense-fields');
const defenseFields = document.getElementById('defense-fields');

modeSel.addEventListener('change', () => {
  const m = modeSel.value;
  offenseFields.style.display = m === 'offense' ? '' : 'none';
  defenseFields.style.display = m === 'defense' ? '' : 'none';
  // Toggle required attrs
  [...offenseFields.querySelectorAll('input')].forEach(i => i.required = (m==='offense'));
  [...defenseFields.querySelectorAll('input')].forEach(i => i.required = (m==='defense'));
});

f.addEventListener('submit', async (e) => {
  e.preventDefault();
  const data = Object.fromEntries(new FormData(f).entries());
  const mode = data.mode || 'offense';
  delete data.mode;
  for (const k of Object.keys(data)) {
    if (k === 'k') data[k] = parseInt(data[k], 10);
    else if (k !== 'api') data[k] = parseFloat(data[k]);
  }
  const urlBase = (api.value || '').replace(/\/$/, '');
  const url = `${urlBase}/similarity/${mode}`;
  const res = await fetch(url, {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(data)});
  out.textContent = JSON.stringify(await res.json(), null, 2);
});