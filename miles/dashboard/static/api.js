const RETRIES_503 = 3;

export async function api(path, params = {}) {
  const url = new URL(path, location.origin);
  for (const [k, v] of Object.entries(params)) {
    if (v !== undefined && v !== null) url.searchParams.set(k, v);
  }
  for (let attempt = 0; ; attempt++) {
    const res = await fetch(url);
    if (res.status === 503 && attempt < RETRIES_503) {
      await new Promise((r) => setTimeout(r, 1500));
      continue;
    }
    if (!res.ok) {
      let detail = "";
      try {
        detail = (await res.json()).detail ?? "";
      } catch {
        /* non-json error body */
      }
      throw new Error(`HTTP ${res.status}: ${detail || url.pathname}`);
    }
    return res.json();
  }
}

let metaCache = null;

export async function getMeta() {
  if (metaCache === null) metaCache = await api("/api/meta");
  return metaCache;
}
