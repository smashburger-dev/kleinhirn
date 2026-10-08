// Result channel for a browser without a debugger connection (R8 Festlegung 9, driver firefox-reg
// in runner-lib.mjs): with ?khpost=<url> the page posts its stage once a second and the whole
// result once it is done, as text/plain (no CORS preflight). navigator.webdriver goes with the
// result, so the runner can refuse a run under automation.

export function startPosting(get: () => { stage?: string; done?: boolean } | undefined): void {
  const url = new URLSearchParams(location.search).get('khpost');
  if (!url) return;
  let busy = false;
  const send = async (body: unknown): Promise<boolean> => {
    try {
      const r = await fetch(url, { method: 'POST', body: JSON.stringify(body), headers: { 'Content-Type': 'text/plain' } });
      return r.ok;
    } catch {
      return false;
    }
  };
  const id = setInterval(() => {
    if (busy) return;
    const r = get();
    if (!r) return;
    busy = true;
    const done = r.done === true;
    void send(done ? { ...r, webdriver: navigator.webdriver === true } : { stage: r.stage, done: false })
      .then((ok) => { if (ok && done) clearInterval(id); })
      .finally(() => { busy = false; });
  }, 1000);
}
