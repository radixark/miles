const RETRIES_503 = 3;

async function fetchOk(path, params) {
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
    return res;
  }
}

export async function api(path, params = {}) {
  return (await fetchOk(path, params)).json();
}

// framed binary endpoints (/api/timeline/heatmap):
// [4-byte LE header length][header JSON][uint8 matrix, row-major]
export async function apiBinary(path, params = {}) {
  const buf = await (await fetchOk(path, params)).arrayBuffer();
  const headerLen = new DataView(buf).getUint32(0, true);
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, headerLen)));
  return { ...header, values: new Uint8Array(buf, 4 + headerLen) };
}

let metaCache = null;

export async function getMeta() {
  if (metaCache === null) metaCache = await api("/api/meta");
  return metaCache;
}
