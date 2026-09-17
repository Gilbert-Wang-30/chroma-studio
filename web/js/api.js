/**
 * HTTP client for the Recolor API (docs/ARCHITECTURE.md §3.7): typed fetch wrappers,
 * a job SSE subscription and an abortable render queue.
 *
 * Every wrapper resolves with parsed JSON (or a Blob for images) and rejects with an
 * ApiError carrying the server's `{error, detail}` and the HTTP status.
 */

export class ApiError extends Error {
  constructor(message, { status = 0, detail = '', cause = null } = {}) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.detail = detail;
    this.cause = cause;
  }
  /** Human sentence for a toast. */
  get userMessage() {
    if (this.status === 0) return 'Cannot reach the server. Is it running?';
    return this.detail ? `${this.message}: ${this.detail}` : this.message;
  }
}

const BASE = '';

async function request(method, path, { json = null, form = null, signal = null, raw = false } = {}) {
  const init = { method, signal, headers: {} };
  if (json !== null) {
    init.headers['Content-Type'] = 'application/json';
    init.body = JSON.stringify(json);
  } else if (form) {
    init.body = form;
  }
  let res;
  try {
    res = await fetch(BASE + path, init);
  } catch (err) {
    if (err?.name === 'AbortError') throw err;
    throw new ApiError('Network error', { status: 0, cause: err });
  }
  if (!res.ok) {
    let payload = null;
    try { payload = await res.json(); } catch { /* not JSON */ }
    throw new ApiError(payload?.error || `${res.status} ${res.statusText}`,
      { status: res.status, detail: payload?.detail || '' });
  }
  if (raw) return res;
  if (res.status === 204) return null;
  return res.json();
}

/** URL of a job layer image; `bust` forces a fresh fetch after an edit. */
export function layerUrl(jobId, layer, bust = 0) {
  return `${BASE}/api/jobs/${jobId}/layers/${layer}${bust ? `?v=${bust}` : ''}`;
}

/** URL of an id-encoded PNG (`regions` or `groups`). */
export function idsUrl(jobId, kind, bust = 0) {
  return `${BASE}/api/jobs/${jobId}/ids/${kind}${bust ? `?v=${bust}` : ''}`;
}

export const api = {
  health: () => request('GET', '/api/health'),
  samples: () => request('GET', '/api/samples'),
  jobs: () => request('GET', '/api/jobs'),
  job: (id) => request('GET', `/api/jobs/${id}`),
  deleteJob: (id) => request('DELETE', `/api/jobs/${id}`),

  /**
   * Create a job from a File (multipart) or a sample name (JSON).
   * @param {{file?: File, sample?: string, detail?: string, intrinsic?: string, max_groups?: number|null, delta_e?: number}} p
   */
  createJob(p) {
    if (p.file) {
      const form = new FormData();
      form.append('file', p.file, p.file.name);
      for (const k of ['detail', 'intrinsic', 'max_groups', 'delta_e']) {
        if (p[k] !== undefined && p[k] !== null) form.append(k, String(p[k]));
      }
      return request('POST', '/api/jobs', { form });
    }
    const { file, ...json } = p;
    return request('POST', '/api/jobs', { json });
  },

  mergeGroups: (id, group_ids) => request('POST', `/api/jobs/${id}/groups/merge`, { json: { group_ids } }),
  splitGroup: (id, group_id, k = 2) => request('POST', `/api/jobs/${id}/groups/split`, { json: { group_id, k } }),
  moveRegions: (id, region_ids, group_id) => request('POST', `/api/jobs/${id}/groups/move`, { json: { region_ids, group_id } }),
  patchGroup: (id, gid, patch) => request('PATCH', `/api/jobs/${id}/groups/${gid}`, { json: patch }),
  regroup: (id, opts) => request('POST', `/api/jobs/${id}/regroup`, { json: opts }),

  createPalette: (prompt, n_colors) => request('POST', '/api/palettes', { json: { prompt, n_colors } }),
  palette: (pid) => request('GET', `/api/palettes/${pid}`),

  suggestMapping: (id, colors, strategy) =>
    request('POST', `/api/jobs/${id}/mapping/suggest`, { json: { colors, strategy } }).then((r) => r.mapping || {}),

  exportImage: (id, { mapping, options, quality, format }) =>
    request('POST', `/api/jobs/${id}/export`, { json: { mapping, options, quality, format } }),

  saveState: (id, state) => request('PUT', `/api/jobs/${id}/state`, { json: state }),

  /** Loads an image URL as an ImageBitmap (no colour management so id PNGs decode exactly). */
  async loadBitmap(url, { exact = false, signal = null } = {}) {
    const res = await request('GET', url, { raw: true, signal });
    const blob = await res.blob();
    return createImageBitmap(blob, exact
      ? { colorSpaceConversion: 'none', premultiplyAlpha: 'none' }
      : { premultiplyAlpha: 'none' });
  },
};

/**
 * Subscribe to a job's SSE stream. `handlers` may contain `stage`, `status`, `groups`,
 * `done`, `error` (server events) and `open` / `disconnect` (transport). Returns a
 * function that closes the stream.
 */
export function subscribeJob(jobId, handlers) {
  const es = new EventSource(`${BASE}/api/jobs/${jobId}/events`);
  const parse = (e) => { try { return JSON.parse(e.data); } catch { return null; } };
  for (const type of ['stage', 'status', 'groups', 'done', 'error']) {
    es.addEventListener(type, (e) => {
      const data = parse(e);
      if (data) handlers[type]?.(data);
    });
  }
  es.addEventListener('message', (e) => {
    const data = parse(e);
    if (data?.type) handlers[data.type]?.(data);
  });
  es.onopen = () => handlers.open?.();
  es.onerror = () => {
    if (es.readyState === EventSource.CLOSED) handlers.disconnect?.({ final: true });
    else handlers.disconnect?.({ final: false });
  };
  return () => es.close();
}

/**
 * Abortable preview renders. Only the newest request matters: calling `render` again
 * aborts the in-flight one, whose promise then resolves to `null`.
 */
export function createRenderQueue(jobId) {
  let controller = null;
  let seq = 0;
  return {
    /** @returns {Promise<{blob: Blob, ms: number, url: string} | null>} null when superseded. */
    async render(mapping, options) {
      controller?.abort();
      controller = new AbortController();
      const mine = ++seq;
      const t0 = performance.now();
      try {
        const res = await request('POST', `/api/jobs/${jobId}/render`,
          { json: { mapping, options }, signal: controller.signal, raw: true });
        const blob = await res.blob();
        if (mine !== seq) return null;
        const header = parseFloat(res.headers.get('X-Render-Ms'));
        const ms = Number.isFinite(header) ? header : performance.now() - t0;
        return { blob, ms, roundTripMs: performance.now() - t0 };
      } catch (err) {
        if (err?.name === 'AbortError') return null;
        throw err;
      }
    },
    cancel() { controller?.abort(); controller = null; seq++; },
  };
}
