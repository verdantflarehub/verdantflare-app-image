import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import vm from 'node:vm';
import {test} from 'node:test';

const html = await readFile(new URL('../src/static/index.html', import.meta.url), 'utf8');
const source = html.slice(html.indexOf('    function mergeTaskPage('), html.indexOf("    applyImageRoute('/dashboard#"));
const ratingSource = html.slice(html.indexOf('    const pendingRatingTasks'), html.indexOf('    async function handleTaskCardAction'));

function setup(extra = {}) {
  const calls = [], replies = [];
  const nodes = new Map();
  const getNode = id => {
    if (!nodes.has(id)) {
      nodes.set(id, {
        id,
        hidden: true,
        disabled: false,
        textContent: '',
        innerHTML: '',
        dataset: {},
        style: {},
        open: false,
        returnValue: '',
        checked: true,
        classList: { toggle() {}, add() {}, remove() {}, contains: () => false },
        focus() {},
        setAttribute(k, v) { this[k] = v; },
        addEventListener(type, fn) { this[`on_${type}`] = fn; },
        showModal() { this.open = true; },
        close(val) {
          if (val !== undefined) this.returnValue = val;
          this.open = false;
          if (this.on_close) this.on_close();
        }
      });
    }
    return nodes.get(id);
  };
  const context = vm.createContext({
    console,
    BigInt,
    Date,
    Math,
    Set,
    Map,
    URLSearchParams,
    encodeURIComponent,
    decodeURIComponent,
    setTimeout,
    clearTimeout,
    document: {
      hidden: false,
      body: { classList: { add() {}, remove() {}, contains: () => false } },
      getElementById: getNode,
      querySelector: () => null,
      querySelectorAll: () => []
    },
    window: { scrollY: 0, scrollBy() {}, notifyHostHeight() {} },
    parent: { postMessage() {} },
    isEmbedded: false,
    hostOrigin: '*',
    viewEpoch: '1',
    hostIncludeNsfw: false,
    taskView: 'gallery',
    currentFilter: 'all',
    galleryContainer: { hidden: false, querySelectorAll: () => [] },
    taskTableContainer: { hidden: true, querySelectorAll: () => [] },
    visibleContentTop: async () => 0,
    pendingDeleteTasks: new Set(),
    galleryTaskKeys: new Map(),
    lastTaskStructure: '',
    detailEpoch: 0,
    taskSearchQuery: '',
    taskQueryEpoch: 0,
    allTasks: [],
    taskLoadState: 'loading',
    taskLoadError: '',
    taskPageError: '',
    taskNextOffset: 0,
    taskPageExhausted: false,
    taskPageSize: 50,
    taskTotal: 0,
    syncingTasks: false,
    disposed: false,
    pendingDetailId: '',
    activeTask: null,
    artifactBlobMap: new Map(),
    artifactMediaMap: new Map(),
    requestApi: async (url, options) => {
      calls.push(options?.method ? `${options.method} ${url}` : url);
      const reply = typeof replies[0] === 'function' ? await replies.shift()(url, options) : await replies.shift();
      if (reply instanceof Error) throw reply;
      return reply;
    },
    loadCapabilities() {},
    actionModeCode: () => 'T2I',
    channelDisplayName: () => 'channel',
    outputMediaLabels: () => ({}),
    updateTaskFilters() {},
    renderGallery() {},
    loadThumbnails() {},
    renderChannelCatalog() {},
    ...extra
  });
  vm.runInContext(source, context);
  vm.runInContext(ratingSource, context);
  return { context, calls, replies, nodes, getNode };
}

const page = (start, count, total = 123) => ({
  total,
  tasks: Array.from({ length: count }, (_, i) => ({ task_id: `task-${start + i}`, status: 'completed' }))
});

function createKeysetBackend(initialCount = 0, { baseEpochSec = 1763160000 } = {}) {
  let nextRowId = 0;
  const rows = [];
  const addTask = ({ id, createdAt, status = 'completed', phase = null, prompt = '', isNsfw = false } = {}) => {
    const rowid = ++nextRowId;
    const ts = createdAt || new Date((baseEpochSec + rowid) * 1000).toISOString();
    const task = {
      rowid,
      task_id: id || `task-${rowid}`,
      created_at: ts,
      status,
      phase,
      prompt_preview: prompt || `prompt-${rowid}`,
      request_params: { prompt: prompt || `prompt-${rowid}` },
      content_rating: { rating: isNsfw ? 'nsfw' : 'general' }
    };
    rows.push(task);
    return task;
  };
  for (let i = 0; i < initialCount; i++) addTask();

  const sortedVisible = ({ q = '', includeNsfw = false } = {}) => {
    const needle = q.trim().toLowerCase();
    return rows
      .filter(r => includeNsfw || r.content_rating?.rating !== 'nsfw')
      .filter(r => !needle || r.task_id.toLowerCase().includes(needle) || (r.prompt_preview || '').toLowerCase().includes(needle))
      .sort((a, b) => {
        if (a.created_at !== b.created_at) return a.created_at > b.created_at ? -1 : 1;
        return b.rowid - a.rowid;
      });
  };

  const toWire = r => ({
    ...r,
    sort_cursor: `${r.rowid}:${r.created_at}`
  });

  const handle = async (url, options = {}, { includeNsfw = false } = {}) => {
    const method = options?.method || 'GET';
    const parsed = new URL(url, 'http://localhost');
    if (method === 'GET' && parsed.pathname === '/api/tasks') {
      const limit = Number(parsed.searchParams.get('limit') || '50');
      const offset = Number(parsed.searchParams.get('offset') || '0');
      const q = parsed.searchParams.get('q') || '';
      const beforeCursor = parsed.searchParams.get('before_cursor') || '';
      const all = sortedVisible({ q, includeNsfw });
      let slice;
      let hasMore;
      if (beforeCursor) {
        const sep = beforeCursor.indexOf(':');
        const cRow = BigInt(beforeCursor.slice(0, sep));
        const cTs = beforeCursor.slice(sep + 1);
        const after = all.filter(r => r.created_at < cTs || (r.created_at === cTs && BigInt(r.rowid) < cRow));
        hasMore = after.length > limit;
        slice = after.slice(0, limit);
      } else {
        slice = all.slice(offset, offset + limit);
        hasMore = offset + slice.length < all.length;
      }
      const wire = slice.map(toWire);
      return {
        tasks: wire,
        capabilities: {},
        total: all.length,
        limit,
        offset,
        has_more: hasMore,
        next_cursor: hasMore && wire.length ? wire.at(-1).sort_cursor : null
      };
    }
    const singleMatch = /^\/api\/tasks\/([^/]+)$/.exec(parsed.pathname);
    if (method === 'GET' && singleMatch) {
      const id = decodeURIComponent(singleMatch[1]);
      const found = rows.find(r => r.task_id === id);
      if (!found || (!includeNsfw && found.content_rating?.rating === 'nsfw')) {
        throw new Error('API 404');
      }
      return toWire(found);
    }
    const delMatch = /^\/api\/tasks\/([^/]+)\/delete$/.exec(parsed.pathname);
    if (method === 'POST' && delMatch) {
      const id = decodeURIComponent(delMatch[1]);
      if (id === '_cleanup') {
        const cleanUnknown = Boolean(options?.body?.clean_unknown);
        const cleanFailed = Boolean(options?.body?.clean_failed);
        const cleanedUnknown = [], cleanedFailed = [], recovered = [];
        for (let i = rows.length - 1; i >= 0; i--) {
          const r = rows[i];
          if (cleanUnknown && (r.phase === 'submission_unknown' || r.phase === 'collect_retry')) {
            if (r.recoverOnCleanup) {
              r.phase = null;
              r.status = 'completed';
              recovered.push(r.task_id);
            } else {
              cleanedUnknown.push(r.task_id);
              rows.splice(i, 1);
            }
          } else if (cleanFailed && r.status === 'failed') {
            cleanedFailed.push(r.task_id);
            rows.splice(i, 1);
          }
        }
        return {
          ok: true,
          recovered_count: recovered.length,
          cleaned_unknown_count: cleanedUnknown.length,
          cleaned_failed_count: cleanedFailed.length,
          recovered_task_ids: recovered,
          cleaned_unknown_task_ids: cleanedUnknown,
          cleaned_failed_task_ids: cleanedFailed
        };
      }
      const idx = rows.findIndex(r => r.task_id === id);
      if (idx < 0) throw new Error('API 404');
      rows.splice(idx, 1);
      return { ok: true, deleted: true, task_id: id };
    }
    const rateMatch = /^\/api\/tasks\/([^/]+)\/content-rating$/.exec(parsed.pathname);
    if (method === 'POST' && rateMatch) {
      const id = decodeURIComponent(rateMatch[1]);
      const found = rows.find(r => r.task_id === id);
      if (!found) throw new Error('API 404');
      const nextRating = options?.body?.action === 'mark' ? 'nsfw' : 'general';
      found.content_rating = { rating: nextRating };
      return { ok: true, task_id: id, content_rating: { rating: nextRating } };
    }
    throw new Error(`Unhandled mock route: ${method} ${url}`);
  };

  return {
    rows,
    addTask,
    deleteById(id) {
      const idx = rows.findIndex(r => r.task_id === id);
      if (idx >= 0) rows.splice(idx, 1);
    },
    getById(id) {
      return rows.find(r => r.task_id === id);
    },
    handle
  };
}

const flushMicrotasks = async (rounds = 6) => {
  for (let i = 0; i < rounds; i++) {
    await new Promise(setImmediate);
  }
};

test('legacy cursorless mocks deduplicate pages, preserve overlapping tail on refresh, and safely reset when disjoint', async () => {
  const { context: c, calls, replies } = setup();
  replies.push(page(0, 50, 123)); await c.syncBackendTasks();
  replies.push(page(49, 50, 123)); await c.loadMoreTasks();
  assert.equal(c.allTasks.length, 99);
  replies.push(page(0, 50, 123)); await c.syncBackendTasks();
  assert.equal(calls[2], '/api/tasks?limit=50&offset=0');
  assert.equal(c.allTasks.length, 99, 'overlapping head poll must preserve loaded tail');
  replies.push(page(99, 24, 123)); await c.loadMoreTasks();
  assert.equal(c.allTasks.length, 123);
  await c.loadMoreTasks();
  assert.equal(calls.length, 4);

  // Disjoint cursorless response resets safely to headItems
  replies.push(page(500, 50, 200)); await c.syncBackendTasks();
  assert.equal(c.allTasks.length, 50);
  assert.equal(c.allTasks[0].id, 'task-500');
  assert.equal(c.taskPageExhausted, false);
});

test('search is sent to backend; changed query discards stale response', async () => {
  const { context: c, calls } = setup();
  let finish;
  c.requestApi = url => { calls.push(url); return new Promise(resolve => { finish = resolve; }); };
  c.taskSearchQuery = 'old';
  const old = c.syncBackendTasks();
  c.taskSearchQuery = 'imgtask-target'; c.taskQueryEpoch++;
  finish(page(0, 50)); await old;
  assert.equal(c.allTasks.length, 0);
  c.requestApi = async url => { calls.push(url); return page(99, 1, 1); };
  await c.syncBackendTasks();
  assert.equal(calls[1], '/api/tasks?limit=50&offset=0&q=imgtask-target');
  assert.equal(c.allTasks.length, 1);
  await c.loadMoreTasks(); assert.equal(calls.length, 2);
  c.taskSearchQuery = ''; c.taskQueryEpoch++;
  await c.syncBackendTasks();
  assert.equal(calls[2], '/api/tasks?limit=50&offset=0');
});

test('page failure retains tasks and cursor; manual retry succeeds; concurrent requests are blocked', async () => {
  const { context: c, calls, replies } = setup();
  replies.push(page(0, 50)); await c.syncBackendTasks();
  const first = c.allTasks[0];
  replies.push(new Error('offline')); await c.loadMoreTasks();
  assert.equal(c.allTasks.length, 50); assert.equal(c.allTasks[0], first);
  assert.equal(c.taskNextOffset, 50); assert.equal(c.taskPageError, 'offline');
  replies.push(page(50, 50)); await Promise.all([c.loadMoreTasks(), c.loadMoreTasks(), c.syncBackendTasks()]);
  assert.equal(calls.length, 3); assert.equal(c.allTasks.length, 100); assert.equal(c.taskPageError, '');
  c.document.hidden = true; await c.loadMoreTasks(); assert.equal(calls.length, 3);
});

test('syncBackendTasks populates artifactMediaMap from server-measured specs and priority pool limits concurrency to 4 with P0 preemption', async () => {
  const { context: c, replies } = setup();
  replies.push({
    total: 1,
    tasks: [{
      task_id: 'task-measured',
      status: 'completed',
      artifact_id: 'art-measured',
      outputWidth: 2048,
      outputHeight: 1152,
      outputSizeBytes: 3145728,
      outputFormat: 'PNG'
    }]
  });
  await c.syncBackendTasks();
  assert.deepEqual({ ...c.artifactMediaMap.get('art-measured') }, {
    width: 2048,
    height: 1152,
    sizeBytes: 3145728,
    format: 'PNG'
  });

  const thumbSource = html.slice(
    html.indexOf('    let thumbnailLoadInProgress = false;'),
    html.indexOf('    // DOM Elements')
  );
  let active = 0, maxActive = 0;
  const order = [];
  const gates = [];
  const thumbCtx = vm.createContext({
    disposed: false,
    allTasks: Array.from({ length: 6 }, (_, i) => ({ id: `t-${i}`, title: `T${i}`, artifactId: `art-${i}` })),
    artifactBlobMap: new Map(),
    artifactLoadingMap: new Map(),
    previewBlobs: new Map(),
    galleryContainer: null,
    window: {},
    console,
    URL: { createObjectURL: () => 'blob:thumb' },
    updateCardThumb: () => {},
    getArtifactBlobUrl: async () => 'blob:fallback',
    requestApi: async url => {
      active++;
      maxActive = Math.max(maxActive, active);
      order.push(url);
      await new Promise(resolve => gates.push(resolve));
      active--;
      return { size: 128 };
    }
  });
  vm.runInContext(thumbSource, thumbCtx);

  const p1Promises = [
    thumbCtx.getArtifactThumbnailUrl('art-0', 320, 'p1'),
    thumbCtx.getArtifactThumbnailUrl('art-0', 320, 'p1'),
    thumbCtx.getArtifactThumbnailUrl('art-1', 320, 'p1'),
    thumbCtx.getArtifactThumbnailUrl('art-2', 320, 'p1'),
    thumbCtx.getArtifactThumbnailUrl('art-3', 320, 'p1'),
    thumbCtx.getArtifactThumbnailUrl('art-4', 320, 'p1'),
    thumbCtx.getArtifactThumbnailUrl('art-5', 320, 'p1')
  ];
  const p0Promise = thumbCtx.getArtifactThumbnailUrl('art-ref', 640, 'p0');

  await new Promise(setImmediate);
  assert.equal(maxActive, 4, 'concurrency pool must cap in-flight requests at 4');
  assert.equal(order.length, 4);

  gates.shift()();
  await new Promise(setImmediate);
  assert.equal(order[4], '/artifacts/art-ref/thumbnail?edge=640');

  while (gates.length) {
    gates.shift()();
    await new Promise(setImmediate);
  }
  const results = await Promise.all([...p1Promises, p0Promise]);
  assert.ok(results.every(r => r === 'blob:thumb'));
  assert.equal(maxActive, 4);
  assert.equal(order.length, 7, 'duplicate art-0 request must share single in-flight promise');
});

test('forced refresh waits for the in-flight list and then reads new state', async () => {
  const { context: c, replies, calls } = setup();
  let finish; replies.push(new Promise(r => { finish = r; }));
  const old = c.syncBackendTasks();
  replies.push(page(1, 1, 1));
  const fresh = c.syncBackendTasks(false, true);
  assert.equal(calls.length, 1);
  finish(page(0, 1, 1)); await old; await fresh;
  assert.equal(calls.length, 2); assert.equal(c.allTasks[0].id, 'task-1');
});

test('typing debounces, resets query pages, and empty search restores normal browsing', async () => {
  const listeners = {}, timers = new Map(); let next = 0; const calls = [];
  const c = vm.createContext({
    inputSearch: { value: '', addEventListener: (name, fn) => { listeners[name] = fn; } },
    taskSearchQuery: '', taskQueryEpoch: 0, taskSearchTimer: null, allTasks: [{ id: 'old' }], taskNextOffset: 100, taskTotal: 100,
    taskPageExhausted: true, taskPageError: 'old', taskLoadState: 'ready', renderGallery() {},
    setTimeout: fn => { timers.set(++next, fn); return next; }, clearTimeout: id => timers.delete(id),
    syncBackendTasks: (...args) => calls.push(args)
  });
  const start = html.indexOf('    inputSearch.addEventListener("input", () => {');
  const end = html.indexOf('    });', start) + 7;
  vm.runInContext(html.slice(start, end), c);
  c.inputSearch.value = 'first'; listeners.input();
  c.inputSearch.value = 'target'; listeners.input();
  assert.equal(timers.size, 1); assert.equal(c.allTasks.length, 0); assert.equal(c.taskNextOffset, 0);
  assert.equal(c.taskQueryEpoch, 2);
  [...timers.values()][0](); assert.equal(calls.length, 1);
  c.inputSearch.value = ''; listeners.input(); assert.equal(c.taskSearchQuery, '');
});

test('search footer visibility never fetches the next page automatically', async () => {
  let callback; const calls = [];
  class Observer { constructor(fn) { callback = fn; } observe() {} }
  const c = vm.createContext({
    window: { IntersectionObserver: Observer }, IntersectionObserver: Observer,
    taskSearchQuery: 'needle', taskPageError: '', loadMoreTasks: () => calls.push('next'),
    document: { getElementById: () => ({}) }
  });
  const start = html.indexOf('    const taskPageObserver =');
  const end = html.indexOf('    // A rating write', start);
  vm.runInContext(html.slice(start, end), c);
  callback([{ isIntersecting: true }]); assert.equal(calls.length, 0);
  c.taskSearchQuery = ''; callback([{ isIntersecting: true }]); assert.equal(calls.length, 1);
});

test('keyset pagination handles H1 counterexamples (onepage_delete_last, insert1_delete49, delete75, concurrent head mutations) with zero skips or duplicates', async () => {
  // 1. onepage_delete_last: 50 loaded out of 100, server deletes the 50th item (task-51)
  {
    const backend = createKeysetBackend(100);
    const { context: c } = setup({ requestApi: (u, o) => backend.handle(u, o) });
    await c.syncBackendTasks();
    assert.equal(c.allTasks.length, 50);
    assert.equal(c.allTasks.at(-1).id, 'task-51');
    backend.deleteById('task-51');
    await c.syncBackendTasks(); // 5s head poll
    assert.equal(c.allTasks.length, 50);
    assert.ok(!c.allTasks.some(t => t.id === 'task-51'), 'deleted 50th item must be purged');
    assert.equal(c.allTasks.at(-1).id, 'task-50', '51st item moves cleanly into head window');
    await c.loadMoreTasks();
    assert.equal(c.allTasks.length, 99);
    assert.equal(new Set(c.allTasks.map(t => t.id)).size, 99);
    assert.equal(c.allTasks.at(-1).id, 'task-1');
  }

  // 2. insert1_delete49: 100 loaded out of 200, server inserts 1 head task and deletes 49 head tasks
  {
    const backend = createKeysetBackend(200);
    const { context: c } = setup({ requestApi: (u, o) => backend.handle(u, o) });
    await c.syncBackendTasks();
    await c.loadMoreTasks();
    assert.equal(c.allTasks.length, 100); // task-200 .. task-101
    backend.addTask({ id: 'task-201' });
    for (let idNum = 200; idNum >= 152; idNum--) backend.deleteById(`task-${idNum}`);
    await c.syncBackendTasks();
    assert.equal(c.allTasks.length, 52); // task-201 + task-151..task-101
    assert.equal(c.allTasks[0].id, 'task-201');
    assert.equal(c.allTasks[1].id, 'task-151');
    assert.equal(c.allTasks.at(-1).id, 'task-101');
    await c.loadMoreTasks();
    assert.equal(c.allTasks.length, 102); // + task-100..task-51
    assert.equal(c.allTasks.at(-1).id, 'task-51');
    assert.equal(new Set(c.allTasks.map(t => t.id)).size, 102);
  }

  // 3. delete75 / shrink-below-loaded: 100 loaded out of 200, server deletes 110 middle items so total=90 < allTasks.length=100 while 40 older unloaded items remain
  {
    const backend = createKeysetBackend(200);
    const { context: c } = setup({ requestApi: (u, o) => backend.handle(u, o) });
    await c.syncBackendTasks();
    await c.loadMoreTasks();
    assert.equal(c.allTasks.length, 100); // task-200 .. task-101
    for (let idNum = 150; idNum >= 41; idNum--) backend.deleteById(`task-${idNum}`); // total becomes 90, unloaded task-40..task-1 remain!
    await c.syncBackendTasks();
    assert.equal(c.taskPageExhausted, false, 'must NOT falsely mark exhausted merely because allTasks.length >= taskTotal');
    await c.loadMoreTasks();
    assert.equal(c.allTasks.at(-1).id, 'task-1', 'must load all 40 remaining older items via keyset cursor');
    assert.equal(c.taskPageExhausted, true, 'append has_more=false marks pagination exhausted without wiping loaded cache');
    assert.ok(c.allTasks.length > 90, 'append has_more=false preserves optimistic loaded list');

    // Now delete enough so from-head window covers full visible dataset (has_more=false from head) -> full convergence!
    for (let idNum = 200; idNum >= 161; idNum--) backend.deleteById(`task-${idNum}`); // 50 items left total (task-160..151 + task-40..1)
    await c.syncBackendTasks();
    assert.equal(c.allTasks.length, 50, 'from-head has_more=false converges optimistic cache to exact server set');
    assert.ok(!c.allTasks.some(t => t.id === 'task-120'));
  }

  // 4. Concurrent head insert + delete right before loadMoreTasks
  {
    const backend = createKeysetBackend(120);
    const { context: c } = setup({ requestApi: (u, o) => backend.handle(u, o) });
    await c.syncBackendTasks();
    assert.equal(c.allTasks.length, 50); // task-120 .. task-71
    backend.addTask({ id: 'task-121' });
    backend.addTask({ id: 'task-122' });
    backend.deleteById('task-119');
    await c.loadMoreTasks();
    assert.equal(c.allTasks.length, 100);
    assert.equal(c.allTasks[50].id, 'task-70', 'before_cursor must not skip or duplicate task-70');
    assert.equal(new Set(c.allTasks.map(t => t.id)).size, 100);
  }
});

test('burst gap-bridging covers 199/200/201 new tasks + head deletions in browse mode, resets at >=250, and never auto-bridges in search mode', async () => {
  for (const burstCount of [199, 200, 201]) {
    const backend = createKeysetBackend(120);
    const calls = [];
    const { context: c } = setup({
      requestApi: (u, o) => { calls.push(u); return backend.handle(u, o); }
    });
    await c.syncBackendTasks();
    await c.loadMoreTasks();
    assert.equal(c.allTasks.length, 100); // task-120 .. task-21
    // Delete 15 old head items AND insert burstCount new items
    for (let idNum = 120; idNum >= 106; idNum--) backend.deleteById(`task-${idNum}`);
    for (let k = 1; k <= burstCount; k++) backend.addTask({ id: `burst-${burstCount}-${k}` });
    calls.length = 0;
    await c.syncBackendTasks();
    assert.ok(calls.length <= 5, `burst=${burstCount} must bridge within 5 pages (got ${calls.length})`);
    assert.equal(c.allTasks.length, burstCount + 85, `burst=${burstCount} must preserve surviving 85-item tail without gap`);
    assert.equal(c.allTasks.at(-1).id, 'task-21');
    assert.equal(new Set(c.allTasks.map(t => t.id)).size, burstCount + 85);
  }

  // >= 250 new tasks in browse mode exhausts 5-page budget -> resets to 250 contiguous items and skips stale anchor
  {
    const backend = createKeysetBackend(120);
    const calls = [];
    let scrollCompensationCalls = 0;
    const { context: c } = setup({
      visibleContentTop: async () => 1800,
      galleryContainer: {
        hidden: false,
        querySelectorAll: () => [{ dataset: { taskId: 'task-80' }, getBoundingClientRect: () => ({ top: 100, height: 300 }) }]
      },
      window: { scrollY: 1800, scrollBy: () => { scrollCompensationCalls++; }, notifyHostHeight() {} },
      requestApi: (u, o) => { calls.push(u); return backend.handle(u, o); }
    });
    await c.syncBackendTasks();
    await c.loadMoreTasks();
    for (let k = 1; k <= 260; k++) backend.addTask({ id: `huge-${k}` });
    calls.length = 0;
    scrollCompensationCalls = 0;
    await c.syncBackendTasks();
    assert.equal(calls.length, 5, 'browse budget must stop at 5 pages (250 items)');
    assert.equal(c.allTasks.length, 250, 'must reset to 250 contiguous head items without gap');
    assert.equal(c.allTasks[0].id, 'huge-260');
    assert.equal(c.allTasks.at(-1).id, 'huge-11');
    assert.equal(c.taskPageExhausted, false);
    assert.equal(scrollCompensationCalls, 0, 'budget-exhausted reset must skip stale anchor compensation');
    await c.loadMoreTasks();
    assert.equal(c.allTasks[250].id, 'huge-10', 'next loadMoreTasks continues contiguously from huge-11 cursor');
  }

  // Search mode: multi-page loaded + >= 50 new matching tasks -> 0 auto-bridge pages, resets to 50-item contiguous prefix
  {
    const backend = createKeysetBackend(0);
    for (let i = 1; i <= 120; i++) backend.addTask({ id: `hero-${i}`, prompt: `hero shot ${i}` });
    const calls = [];
    const { context: c } = setup({
      requestApi: (u, o) => { calls.push(u); return backend.handle(u, o); }
    });
    c.taskSearchQuery = 'hero';
    c.taskQueryEpoch = 1;
    await c.syncBackendTasks();
    await c.loadMoreTasks(true); // manual load page 2 in search mode
    assert.equal(c.allTasks.length, 100);
    for (let k = 121; k <= 180; k++) backend.addTask({ id: `hero-${k}`, prompt: `hero shot ${k}` });
    calls.length = 0;
    await c.syncBackendTasks();
    assert.equal(calls.length, 1, 'search mode must NEVER auto-bridge multiple pages');
    assert.equal(c.allTasks.length, 50, 'search mode disjoint head must reset to 50-item contiguous prefix');
    assert.equal(c.allTasks[0].id, 'hero-180');
    assert.equal(c.allTasks.at(-1).id, 'hero-131');
    assert.equal(c.taskPageExhausted, false);
    await c.loadMoreTasks(true);
    assert.equal(c.allTasks.length, 100);
    assert.equal(c.allTasks[50].id, 'hero-130');
  }
});

test('tombstones block stale list resurrection on delete/NSFW hide, and forced head sync runs before queued append', async () => {
  const backend = createKeysetBackend(80);
  let staleHeadGate;
  let interceptFirstPoll = true;
  const order = [];
  const { context: c, getNode } = setup({
    hostIncludeNsfw: false,
    requestApi: async (u, o) => {
      order.push(`${o?.method || 'GET'} ${u}`);
      if (interceptFirstPoll && (!o?.method || o.method === 'GET') && u.startsWith('/api/tasks?limit=50&offset=0')) {
        interceptFirstPoll = false;
        const staleSnapshot = await backend.handle(u, o, { includeNsfw: false });
        await new Promise(resolve => { staleHeadGate = resolve; });
        return staleSnapshot;
      }
      return backend.handle(u, o, { includeNsfw: false });
    }
  });

  interceptFirstPoll = false;
  await c.syncBackendTasks();
  assert.equal(c.allTasks.length, 50);

  // Start a slow 5s head poll that captured task-80 & task-79 before mutations
  interceptFirstPoll = true;
  const inFlightPoll = c.syncBackendTasks();
  await flushMicrotasks(2);

  // While poll is in flight:
  // 1) User triggers loadMoreTasks(true) -> queued append
  const queuedAppend = c.loadMoreTasks(true);
  // 2) User deletes task-80 (confirm dialog immediately) and marks task-79 as NSFW (hidden since !hostIncludeNsfw)
  const delDialog = getNode('taskDeleteDialog');
  const deleting = c.deleteTask('task-80');
  await flushMicrotasks(2);
  delDialog.close('confirm');
  const ratingTarget = c.allTasks.find(t => t.id === 'task-79');
  const ratingPromise = c.changeTaskRating(ratingTarget);
  await flushMicrotasks(3);

  // Immediate UI trim already removed task-80 & task-79 before stale poll even finishes!
  assert.ok(!c.allTasks.some(t => t.id === 'task-80' || t.id === 'task-79'));

  // Now release the stale head poll: tombstones must block resurrection, AND forced head sync must execute BEFORE queued append!
  order.length = 0;
  staleHeadGate();
  await Promise.all([inFlightPoll, deleting, ratingPromise, queuedAppend]);

  assert.ok(!c.allTasks.some(t => t.id === 'task-80' || t.id === 'task-79'), 'tombstones must block stale poll from resurrecting deleted/NSFW tasks');
  const forcedHeadIdx = order.findIndex(x => x === 'GET /api/tasks?limit=50&offset=0');
  const appendIdx = order.findIndex(x => x.includes('before_cursor='));
  assert.ok(forcedHeadIdx >= 0 && appendIdx > forcedHeadIdx, `forced head sync (${forcedHeadIdx}) must run before queued append (${appendIdx})`);
  assert.equal(c.allTasks.length, 78);

  // Also verify automatic (non-manual) queued append is suppressed if search query becomes non-empty
  let holdPoll;
  const backend2 = createKeysetBackend(80);
  const calls2 = [];
  const { context: c2 } = setup({
    requestApi: async (u, o) => {
      calls2.push(u);
      if (calls2.length === 2) await new Promise(r => { holdPoll = r; });
      return backend2.handle(u, o);
    }
  });
  await c2.syncBackendTasks();
  const slowPoll = c2.syncBackendTasks();
  const autoAppend = c2.loadMoreTasks(false); // observer-triggered (manual=false)
  c2.taskSearchQuery = 'prompt';
  c2.taskQueryEpoch++;
  holdPoll();
  await Promise.all([slowPoll, autoAppend]);
  assert.ok(!calls2.some(u => u.includes('before_cursor=')), 'epoch-mismatched / search auto-append must be suppressed');
});

test('executeCleanupTasks deterministically trims cleaned IDs, bounds >32 recovered reconciles to <=4 concurrency, and retries 503 with backoff', async () => {
  const backend = createKeysetBackend(100);
  // Mark 5 tail tasks as failed, 4 tail zombies to delete, and 40 tail zombies to recover (> 32 host limit!)
  for (let i = 1; i <= 5; i++) backend.getById(`task-${i}`).status = 'failed';
  for (let i = 6; i <= 9; i++) {
    const t = backend.getById(`task-${i}`);
    t.status = 'running';
    t.phase = 'submission_unknown';
  }
  for (let i = 10; i <= 49; i++) {
    const t = backend.getById(`task-${i}`);
    t.status = 'running';
    t.phase = 'collect_retry';
    t.recoverOnCleanup = true;
  }

  let activeSingleGets = 0;
  let maxConcurrentSingleGets = 0;
  let failOnceTask10 = true;
  let nowOffsetMs = 0;
  class ClockDate extends Date {
    static now() { return Date.now() + nowOffsetMs; }
  }
  const { context: c, getNode } = setup({
    Date: ClockDate,
    requestApi: async (u, o) => {
      if ((!o?.method || o.method === 'GET') && /^\/api\/tasks\/task-\d+$/.test(u)) {
        activeSingleGets++;
        maxConcurrentSingleGets = Math.max(maxConcurrentSingleGets, activeSingleGets);
        await new Promise(setImmediate);
        activeSingleGets--;
        if (u.endsWith('/task-10') && failOnceTask10) {
          failOnceTask10 = false;
          throw new Error('API 503');
        }
      }
      return backend.handle(u, o);
    }
  });

  await c.syncBackendTasks();
  await c.loadMoreTasks();
  assert.equal(c.allTasks.length, 100);

  getNode('chkCleanUnknown').checked = true;
  getNode('chkCleanFailed').checked = true;
  await c.executeCleanupTasks();
  await flushMicrotasks(12);

  // 9 items (5 failed + 4 unrecoverable unknown) removed while loaded tail preserved (91 items)!
  assert.equal(c.allTasks.length, 91);
  for (let i = 1; i <= 9; i++) {
    assert.ok(!c.allTasks.some(t => t.id === `task-${i}`), `cleaned task-${i} must be removed`);
  }
  assert.ok(maxConcurrentSingleGets <= 4, `single-task reconcile concurrency (${maxConcurrentSingleGets}) must stay <= 4`);

  // Run remaining 5s poll cycles to drain all 40 recovered items (8 per cycle)
  for (let cycle = 0; cycle < 5; cycle++) {
    await c.syncBackendTasks();
    await flushMicrotasks(12);
  }
  assert.ok(maxConcurrentSingleGets <= 4);
  // 39 of 40 recovered items are now succeeded; task-10 hit 503 once and is waiting on 10s backoff
  const succeededRecovered = c.allTasks.filter(t => {
    const n = Number(t.id.split('-')[1]);
    return n >= 11 && n <= 49 && t.status === 'succeeded';
  });
  assert.equal(succeededRecovered.length, 39);
  assert.equal(c.allTasks.find(t => t.id === 'task-10').status, 'running', '503 item stays preserved in pendingForcedReconcileIds during backoff');

  // Advance ClockDate.now past 10s backoff and run one more poll -> task-10 recovers to succeeded!
  nowOffsetMs = 15000;
  await c.syncBackendTasks();
  await flushMicrotasks(8);
  assert.equal(c.allTasks.find(t => t.id === 'task-10').status, 'succeeded', 'task-10 must recover after backoff expires');
});

test('unified viewport-relative anchor compensates head prepend, first-visible delete fallback, batch cleanup, and async filter shrink; non-empty poll error hides banner', async () => {
  const backend = createKeysetBackend(100);
  const wheelDeltas = [];
  const viewportHeight = 900;
  let contentHeight = 0;
  let hostScrollY = 2000;
  const clampHostScroll = y => {
    const maxScrollY = Math.max(0, contentHeight - viewportHeight);
    return Math.max(0, Math.min(y, maxScrollY));
  };
  // Simulate vertical layout with real browser [0, maxScrollY] scroll clamping
  const cardTops = new Map();
  let cRef;
  const recomputeLayout = () => {
    cardTops.clear();
    if (!cRef) return;
    const visible = cRef.allTasks.filter(t => cRef.currentFilter === 'all' || t.status === cRef.currentFilter);
    visible.forEach((t, idx) => {
      cardTops.set(t.id, idx * 320);
    });
    contentHeight = visible.length * 320;
    hostScrollY = clampHostScroll(hostScrollY);
  };

  const { context: c, getNode } = setup({
    isEmbedded: true,
    viewEpoch: '7',
    visibleContentTop: async () => hostScrollY,
    window: {
      scrollY: 0,
      scrollBy() {},
      notifyHostHeight() { hostScrollY = clampHostScroll(hostScrollY); }
    },
    parent: {
      postMessage: msg => {
        if (msg?.type === 'wheel' && Number.isFinite(msg.deltaY) && !msg.captureId) {
          wheelDeltas.push(msg.deltaY);
          hostScrollY = clampHostScroll(hostScrollY + msg.deltaY);
        }
      }
    },
    galleryContainer: {
      hidden: false,
      querySelectorAll: () => {
        if (!cRef) return [];
        const visible = cRef.allTasks.filter(t => cRef.currentFilter === 'all' || t.status === cRef.currentFilter);
        return visible.map(t => ({
          dataset: { taskId: t.id },
          getBoundingClientRect: () => ({ top: cardTops.get(t.id) ?? 0, height: 300 })
        }));
      }
    },
    renderGallery() { recomputeLayout(); },
    requestApi: (u, o) => backend.handle(u, o)
  });
  cRef = c;

  hostScrollY = 0;
  await c.syncBackendTasks();
  await c.loadMoreTasks();
  recomputeLayout();

  // Scroll user to hostScrollY = 2000 (within [0, 32000 - 900]): first card with (top - 2000 + 300 > 0) is index 6 (top = 1920, preRelTop = -80)
  hostScrollY = clampHostScroll(2000);
  assert.equal(hostScrollY, 2000);
  const anchorCardId = c.allTasks[6].id;
  assert.equal(cardTops.get(anchorCardId) - hostScrollY, -80);

  // 1) Prepend 4 new tasks (+1280px shift > 1000px segmented wheel!)
  for (let k = 1; k <= 4; k++) backend.addTask({ id: `prep-${k}` });
  wheelDeltas.length = 0;
  await c.syncBackendTasks();
  assert.deepEqual(wheelDeltas, [1000, 280], 'embedded anchor must segment >1000px delta into <=1000px wheel steps');
  assert.equal(hostScrollY, 3280);
  assert.equal(cardTops.get(anchorCardId) - hostScrollY, -80, 'anchor card must stay at exact same viewport-relative Y');

  // 2) Immediate single delete of the FIRST visible card -> falls through to next surviving visible card!
  const firstVisibleNow = c.allTasks.find(t => (cardTops.get(t.id) - hostScrollY) + 300 > 0);
  const secondVisibleNow = c.allTasks[c.allTasks.findIndex(t => t.id === firstVisibleNow.id) + 1];
  const secondPreRelTop = cardTops.get(secondVisibleNow.id) - hostScrollY;
  wheelDeltas.length = 0;
  const delDialog = getNode('taskDeleteDialog');
  const delPromise = c.deleteTask(firstVisibleNow.id);
  await flushMicrotasks(2);
  delDialog.close('confirm');
  await delPromise;
  assert.ok(hostScrollY >= 0 && hostScrollY <= contentHeight - viewportHeight);
  assert.equal(cardTops.get(secondVisibleNow.id) - hostScrollY, secondPreRelTop, 'deleting first visible card must anchor next surviving visible card');

  // 3a) Async single-task reconcile filter shrink WITH sufficient scroll headroom ([0, maxScrollY])
  for (let n = 1; n <= 12; n++) {
    const id = `task-${n}`;
    backend.getById(id).status = 'running';
    c.allTasks.find(t => t.id === id).status = 'running';
  }
  c.currentFilter = 'running';
  recomputeLayout(); // 12 * 320 = 3840px, maxScrollY = 2940px
  hostScrollY = clampHostScroll(1000); // task-9 (index 3, top=960, relTop=-40), task-8 (index 4, top=1280, relTop=+280)
  assert.equal(hostScrollY, 1000);
  assert.equal(cardTops.get('task-8') - hostScrollY, 280);
  // Backend completes task-12 (top=0) -> remaining 11 items = 3520px (maxScrollY = 2620px), hostScrollY adjusts 1000 -> 680!
  backend.getById('task-12').status = 'completed';
  await c.syncBackendTasks();
  await flushMicrotasks(10);
  assert.equal(hostScrollY, 680, 'scroll position stays strictly non-negative within [0, maxScrollY]');
  assert.equal(cardTops.get('task-8') - hostScrollY, 280, 'with sufficient scroll headroom, anchor relative Y is preserved');

  // 3b) Boundary best-effort clamping: when filtered list shrinks below viewport height, hostScrollY clamps cleanly to 0 (never negative)
  for (let n = 1; n <= 11; n++) {
    const id = `task-${n}`;
    const keepRunning = n === 11 || n === 10 || n === 9 || n === 8; // 4 items = 1280px, maxScrollY = 380px
    backend.getById(id).status = keepRunning ? 'running' : 'completed';
    c.allTasks.find(t => t.id === id).status = keepRunning ? 'running' : 'succeeded';
  }
  recomputeLayout();
  hostScrollY = clampHostScroll(50); // task-11 at top=0 (relTop=-50), task-10 at top=320 (relTop=270)
  assert.equal(hostScrollY, 50);
  backend.getById('task-11').status = 'completed'; // task-10 moves to top=0; scroll cannot go to -270, must clamp at 0!
  await c.syncBackendTasks();
  await flushMicrotasks(10);
  assert.equal(hostScrollY, 0, 'scroll position must clamp at 0 at top boundary and never go negative');
  assert.equal(cardTops.get('task-10') - hostScrollY, 0);

  // 4) Non-empty gallery encountering background 5s poll error keeps taskLoadState === 'ready'
  c.currentFilter = 'all';
  c.requestApi = async () => { throw new Error('API 502'); };
  await c.syncBackendTasks();
  assert.equal(c.taskLoadState, 'ready', 'background poll error on non-empty gallery must not flip taskLoadState to error');
  assert.equal(getNode('connText').textContent, '连接异常');
  assert.equal(getNode('embedStatusPill').dataset.state, 'error');
});

test('isolated deep-link activeTask, >8 tail round-robin, slow in-flight dedup, out-of-order reqSeq, and stale-epoch 404 safety', async () => {
  const backend = createKeysetBackend(70);
  // Make 12 tail tasks (task-1..task-12) active 'running' and 1 zombie (task-13)
  for (let i = 1; i <= 12; i++) backend.getById(`task-${i}`).status = 'running';
  backend.getById('task-13').status = 'running';
  backend.getById('task-13').phase = 'submission_unknown';

  const singleCalls = [];
  const gates = new Map();
  const { context: c } = setup({
    requestApi: async (u, o) => {
      if ((!o?.method || o.method === 'GET') && /^\/api\/tasks\/[^?]+$/.test(u)) {
        singleCalls.push(u);
        if (gates.has(u)) {
          const handler = gates.get(u);
          gates.delete(u);
          return handler();
        }
      }
      return backend.handle(u, o);
    }
  });

  await c.syncBackendTasks();
  await c.loadMoreTasks();
  assert.equal(c.allTasks.length, 70);

  // Attach an isolated activeTask (deep-linked remote task not in allTasks)
  const remoteRow = backend.addTask({ id: 'remote-deep-1', createdAt: '2020-01-01T00:00:00.000Z', status: 'running' });
  c.activeTask = c.mapBackendTask({ ...remoteRow, sort_cursor: `${remoteRow.rowid}:${remoteRow.created_at}` });
  c.detailEpoch = 5;

  // Hold task-12 (first tail candidate in allTasks order) slow across cycles to prove inFlightTaskReconciles deduplication AND out-of-order reqSeq protection
  let releaseSlowTask12;
  gates.set('/api/tasks/task-12', () => new Promise(resolve => { releaseSlowTask12 = resolve; }));

  singleCalls.length = 0;
  await c.syncBackendTasks();
  await flushMicrotasks(10);

  // Max 8 per cycle: isolated remote-deep-1 + 7 tail tasks (task-12..task-6), and zombie task-13 NEVER polled!
  assert.equal(singleCalls.length, 8);
  assert.ok(singleCalls.includes('/api/tasks/remote-deep-1'), 'isolated activeTask must be reconciled');
  assert.ok(singleCalls.includes('/api/tasks/task-12'), 'task-12 must be started in cycle 1');
  assert.ok(!c.allTasks.some(t => t.id === 'remote-deep-1'), 'isolated activeTask must NEVER be mixed into allTasks');
  assert.ok(!singleCalls.includes('/api/tasks/task-13'), 'submission_unknown zombie must not be polled by regular active predicate');

  // Next poll cycle: remaining 5 tail tasks must rotate in via tailReconcileAt, while slow task-12 is NOT dispatched again!
  remoteRow.status = 'completed';
  const cycle1Calls = [...singleCalls];
  singleCalls.length = 0;
  await c.syncBackendTasks();
  await flushMicrotasks(10);
  assert.equal(c.activeTask.status, 'succeeded', 'isolated activeTask updates in place');
  assert.ok(!singleCalls.includes('/api/tasks/task-12'), 'slow in-flight task-12 must not be dispatched twice');
  const combinedPolled = new Set([...cycle1Calls, ...singleCalls]);
  for (let i = 1; i <= 12; i++) {
    assert.ok(combinedPolled.has(`/api/tasks/task-${i}`), `tail active task-${i} must be covered by round-robin across 2 cycles`);
  }

  // While slow task-12 single GET (older reqSeq) is STILL in flight, bump task-12 via newer authoritative state -> stale response must be ignored!
  c.recordLocalTaskPatch('task-12', { status: 'succeeded', title: 'Newer Title' });
  const t12InList = c.allTasks.find(t => t.id === 'task-12');
  t12InList.status = 'succeeded';
  t12InList.title = 'Newer Title';
  releaseSlowTask12({ ...backend.getById('task-12'), status: 'running', title: 'Stale Old Title', sort_cursor: t12InList.sortCursor });
  await flushMicrotasks(6);
  assert.equal(c.allTasks.find(t => t.id === 'task-12').status, 'succeeded', 'stale lower-reqSeq single response must be discarded');
  assert.equal(c.allTasks.find(t => t.id === 'task-12').title, 'Newer Title');

  // Stale-epoch 404 must NOT poison locallyHiddenTaskIds or remove tasks in a new query epoch
  for (let i = 1; i <= 11; i++) {
    if (i !== 2) {
      backend.getById(`task-${i}`).status = 'completed';
      c.allTasks.find(t => t.id === `task-${i}`).status = 'succeeded';
    }
  }
  let rejectStale404;
  gates.set('/api/tasks/task-2', () => new Promise((_, reject) => { rejectStale404 = reject; }));
  await c.syncBackendTasks();
  await flushMicrotasks(3);
  c.taskQueryEpoch++; // user switched query epoch while single-task GET was in flight
  const freshTask2 = { ...c.allTasks.find(t => t.id === 'task-2') };
  c.allTasks = [freshTask2];
  rejectStale404(new Error('API 404'));
  await flushMicrotasks(6);
  assert.equal(c.allTasks.length, 1, 'stale-epoch 404 must be discarded without removing task from new query epoch');
});

test('openInspector never pollutes allTasks, contentPreferences syncs hostIncludeNsfw, and table-view row anchoring works', async () => {
  // 1. Verify host contentPreferences message listener + NSFW keep-visible when hostIncludeNsfw === true
  const msgListeners = [];
  const parentWin = {};
  const msgSlice = html.slice(
    html.indexOf('    // Listen to Parent Studio OS Messages'),
    html.indexOf('    // Notify Parent Ready')
  );
  const backend = createKeysetBackend(10);
  const { context: c } = setup({
    isEmbedded: true,
    parent: { ...parentWin, postMessage() {} },
    hostOrigin: 'https://studio.invalid',
    viewportRequests: new Map(),
    pendingReqs: new Map(),
    applyImageRoute() {},
    window: {
      scrollY: 0,
      scrollTo() {},
      scrollBy() {},
      notifyHostHeight() {},
      studioNavigate() {},
      addEventListener: (type, fn) => { if (type === 'message') msgListeners.push(fn); }
    },
    requestApi: (u, o) => backend.handle(u, o, { includeNsfw: c.hostIncludeNsfw })
  });
  vm.runInContext(msgSlice, c);
  assert.equal(msgListeners.length, 1);
  msgListeners[0]({
    source: c.parent,
    origin: 'https://studio.invalid',
    data: { channel: 'vf-studio', type: 'contentPreferences', includeNsfw: true }
  });
  assert.equal(vm.runInContext('hostIncludeNsfw', c), true);
  c.hostIncludeNsfw = true;

  await c.syncBackendTasks();
  assert.equal(c.allTasks.length, 10);
  const t10 = c.allTasks.find(t => t.id === 'task-10');
  await c.changeTaskRating(t10);
  assert.ok(c.allTasks.some(t => t.id === 'task-10' && t.isNsfw === true), 'when hostIncludeNsfw=true, marking NSFW keeps task visible');

  // 2. Verify real openInspector slice does not mix remote task into allTasks
  const inspectorSlice = html.slice(
    html.indexOf('    const openInspector = async function('),
    html.indexOf('    const closeInspector = async function(')
  );
  c.closeHostPreview = () => {};
  c.captureViewport = async () => 320;
  c.CSS = { escape: s => s };
  c.inspectorDrawer = { hidden: true, inert: true, classList: { add() {}, remove() {} } };
  c.channelLogo = () => '';
  c.escapeHtml = s => String(s ?? '');
  c.updateInspectorImages = () => {};
  c.getCachedThumbUrl = () => '';
  c.scheduleMediaLoad = fn => Promise.resolve().then(fn);
  c.getArtifactBlobUrl = async () => '';
  vm.runInContext(inspectorSlice, c);

  backend.addTask({ id: 'deep-link-only', createdAt: '2019-05-01T00:00:00.000Z', status: 'completed' });
  const openFn = vm.runInContext('openInspector', c);
  await openFn('deep-link-only', false);
  assert.equal(c.activeTask?.id, 'deep-link-only');
  assert.equal(c.allTasks.some(t => t.id === 'deep-link-only'), false, 'openInspector must not append remote task into allTasks');

  // 3. Verify table-view tr[data-row-task-id] viewport-relative anchor compensation with [0, maxScrollY] clamping
  for (let s = 1; s <= 20; s++) backend.addTask({ id: `tbl-seed-${s}` });
  await c.syncBackendTasks(); // 31 rows * 48px = 1488px
  c.activeTask = null;
  c.taskView = 'table';
  const tableViewportHeight = 600;
  let tableContentHeight = 0;
  let tableScrollY = 400;
  const clampTableScroll = y => Math.max(0, Math.min(y, Math.max(0, tableContentHeight - tableViewportHeight)));
  const tableDeltas = [];
  const rowTops = new Map();
  const recomputeRows = () => {
    rowTops.clear();
    c.allTasks.forEach((t, idx) => rowTops.set(t.id, idx * 48));
    tableContentHeight = c.allTasks.length * 48;
    tableScrollY = clampTableScroll(tableScrollY);
  };
  c.visibleContentTop = async () => tableScrollY;
  c.window.notifyHostHeight = () => { tableScrollY = clampTableScroll(tableScrollY); };
  c.parent.postMessage = msg => {
    if (msg?.type === 'wheel' && Number.isFinite(msg.deltaY) && !msg.captureId) {
      tableDeltas.push(msg.deltaY);
      tableScrollY = clampTableScroll(tableScrollY + msg.deltaY);
    }
  };
  c.renderGallery = recomputeRows;
  c.taskTableContainer = {
    hidden: false,
    querySelectorAll: () => c.allTasks.map(t => ({
      dataset: { rowTaskId: t.id },
      getBoundingClientRect: () => ({ top: rowTops.get(t.id) ?? 0, height: 44 })
    }))
  };
  recomputeRows();
  assert.equal(tableScrollY, 400);
  const anchorRowId = c.allTasks[9].id; // top = 432, relTop = +32
  backend.addTask({ id: 'table-new-1' });
  backend.addTask({ id: 'table-new-2' });
  await c.syncBackendTasks();
  assert.deepEqual(tableDeltas, [96], 'table view must compensate 2 inserted rows (96px)');
  assert.equal(tableScrollY, 496);
  assert.equal(rowTops.get(anchorRowId) - tableScrollY, 32);
});

test('serialized commit queue closes async anchor window races (H1), concurrent double scroll compensation (M1), and 64-bit BigInt cursor bounds (M2)', async () => {
  // 1) H1 Case A: Search switch while head poll response has ALREADY returned and is awaiting visibleContentTop()
  {
    const backend = createKeysetBackend(100);
    let releaseAnchorWait = null;
    let interceptAnchorOnce = false;
    const { context: c } = setup({
      isEmbedded: true,
      visibleContentTop: async () => {
        if (interceptAnchorOnce) {
          interceptAnchorOnce = false;
          await new Promise(resolve => { releaseAnchorWait = resolve; });
        }
        return 1200;
      },
      requestApi: (u, o) => backend.handle(u, o)
    });
    await c.syncBackendTasks();
    await c.loadMoreTasks();
    assert.equal(c.allTasks.length, 100);

    // Trigger head poll: let requestApi finish completely, then pause inside visibleContentTop() BEFORE mutator!
    interceptAnchorOnce = true;
    const pendingHeadPoll = c.syncBackendTasks();
    await flushMicrotasks(6);
    assert.equal(typeof releaseAnchorWait, 'function', 'poll must be paused inside visibleContentTop after network response');

    // Now simulate inputSearch switching to a new query ('no-match') during the anchor wait window!
    c.taskSearchQuery = 'no-match';
    c.taskQueryEpoch++;
    c.allTasks = [];
    c.taskNextOffset = 0;
    c.taskTotal = 0;
    c.taskPageExhausted = false;
    c.taskPageError = '';
    c.resetTaskQueryTransientState();
    c.taskLoadState = 'loading';

    // Release the paused visibleContentTop(): old poll mutator MUST see stale queryEpoch and abort!
    releaseAnchorWait();
    await pendingHeadPoll;
    assert.equal(c.allTasks.length, 0, 'old poll paused in visibleContentTop must NOT write stale 100 tasks into new search epoch');
    assert.equal(c.taskLoadState, 'loading', 'old poll must NOT flip new search epoch taskLoadState to ready');
  }

  // 2) H1 Case B: Delete, NSFW hide, and rename confirmed while old head poll is paused in visibleContentTop()
  {
    const backend = createKeysetBackend(80);
    let releaseAnchorWait = null;
    let pauseNextAnchor = false;
    const { context: c, getNode } = setup({
      isEmbedded: true,
      hostIncludeNsfw: false,
      visibleContentTop: async () => {
        if (pauseNextAnchor) {
          pauseNextAnchor = false;
          await new Promise(resolve => { releaseAnchorWait = resolve; });
        }
        return 960;
      },
      requestApi: (u, o) => backend.handle(u, o, { includeNsfw: false })
    });
    await c.syncBackendTasks();
    assert.equal(c.allTasks.length, 50); // task-80 .. task-31

    // Start head poll; network returns snapshot containing task-80, task-79, task-78, then pauses in visibleContentTop()!
    pauseNextAnchor = true;
    const stalePoll = c.syncBackendTasks();
    await flushMicrotasks(6);
    assert.equal(typeof releaseAnchorWait, 'function');

    // While stalePoll is suspended in visibleContentTop():
    // - Confirm delete of task-80
    const delDialog = getNode('taskDeleteDialog');
    const delOp = c.deleteTask('task-80');
    await flushMicrotasks(2);
    delDialog.close('confirm');
    // - Confirm NSFW mark on task-79
    const t79 = c.allTasks.find(t => t.id === 'task-79');
    const rateOp = c.changeTaskRating(t79);
    // - Confirm server rename on task-78 (and also patch task-77 locally without backend change to prove older page1Seq never overwrites newer seq!)
    backend.getById('task-78').title = 'Renamed In Anchor Window';
    const titleOp = c.applyConfirmedTaskTitle('task-78', 'Renamed In Anchor Window');
    await flushMicrotasks(4);

    // Release visibleContentTop() so stalePoll commits first, followed by serialized write commits & forced head sync
    releaseAnchorWait();
    await Promise.all([stalePoll, delOp, rateOp, titleOp]);

    assert.ok(!c.allTasks.some(t => t.id === 'task-80'), 'deleted task-80 must NOT be resurrected from anchor-paused head poll');
    assert.ok(!c.allTasks.some(t => t.id === 'task-79'), 'NSFW-hidden task-79 must NOT be resurrected from anchor-paused head poll');
    assert.equal(c.allTasks.find(t => t.id === 'task-78')?.title, 'Renamed In Anchor Window', 'newer title patch must survive anchor-paused head poll');

    // Also verify pure stalePoll vs local title patch WITHOUT any subsequent forced poll:
    pauseNextAnchor = true;
    const stalePollOnly = c.syncBackendTasks();
    await flushMicrotasks(6);
    const localOnlyTitle = c.applyConfirmedTaskTitle('task-77', 'Local Patch Newer Than Stale Poll');
    releaseAnchorWait();
    await Promise.all([stalePollOnly, localOnlyTitle]);
    assert.equal(c.allTasks.find(t => t.id === 'task-77')?.title, 'Local Patch Newer Than Stale Poll');
  }

  // 3) H1 Case C: Single-task reconcile paused in visibleContentTop() overtaken by newer local patch
  {
    const backend = createKeysetBackend(60);
    backend.getById('task-5').status = 'running';
    let releaseSingleAnchor = null;
    let pauseSingleAnchor = false;
    const { context: c } = setup({
      isEmbedded: true,
      visibleContentTop: async () => {
        if (pauseSingleAnchor) {
          pauseSingleAnchor = false;
          await new Promise(resolve => { releaseSingleAnchor = resolve; });
        }
        return 800;
      },
      requestApi: async (u, o) => {
        const res = await backend.handle(u, o);
        if (u === '/api/tasks/task-5') {
          pauseSingleAnchor = true;
        }
        return res;
      }
    });
    await c.syncBackendTasks();
    await c.loadMoreTasks();
    // Trigger tail reconcile for task-5, let GET /api/tasks/task-5 finish, then pause inside visibleContentTop()!
    backend.getById('task-5').status = 'completed';
    await c.syncBackendTasks();
    await flushMicrotasks(8);
    assert.equal(typeof releaseSingleAnchor, 'function');

    // User applies newer title patch to task-5 while single-task reconcile is suspended in visibleContentTop()!
    const renamePromise = c.applyConfirmedTaskTitle('task-5', 'User Fresh Title');
    releaseSingleAnchor();
    await renamePromise;
    await flushMicrotasks(8);
    assert.equal(c.allTasks.find(t => t.id === 'task-5')?.title, 'User Fresh Title', 'stale single-task reconcile in anchor wait must not overwrite newer user patch');
  }

  // 4) M1 Counterexample: Two concurrent single-task reconciles removing two 320px items above viewport at scrollY=2000px
  {
    const backend = createKeysetBackend(70);
    // Put 15 tail tasks (task-1..task-15) into 'running' so filtered view has 15 * 320 = 4800px height (maxScrollY = 3900px)
    for (let i = 1; i <= 15; i++) backend.getById(`task-${i}`).status = 'running';
    const viewportHeight = 900;
    let contentHeight = 0;
    let hostScrollY = 0;
    const clampScroll = y => Math.max(0, Math.min(y, Math.max(0, contentHeight - viewportHeight)));
    const cardTops = new Map();
    let cRef;
    const recompute = () => {
      cardTops.clear();
      if (!cRef) return;
      const visible = cRef.allTasks.filter(t => cRef.currentFilter === 'all' || t.status === cRef.currentFilter);
      visible.forEach((t, idx) => cardTops.set(t.id, idx * 320));
      contentHeight = visible.length * 320;
      hostScrollY = clampScroll(hostScrollY);
    };
    const { context: c } = setup({
      isEmbedded: true,
      visibleContentTop: async () => {
        await new Promise(setImmediate); // real async message round-trip!
        return hostScrollY;
      },
      window: {
        scrollY: 0,
        scrollBy() {},
        notifyHostHeight() { hostScrollY = clampScroll(hostScrollY); }
      },
      parent: {
        postMessage: msg => {
          if (msg?.type === 'wheel' && Number.isFinite(msg.deltaY) && !msg.captureId) {
            hostScrollY = clampScroll(hostScrollY + msg.deltaY);
          }
        }
      },
      galleryContainer: {
        hidden: false,
        querySelectorAll: () => {
          const visible = cRef.allTasks.filter(t => cRef.currentFilter === 'all' || t.status === cRef.currentFilter);
          return visible.map(t => ({
            dataset: { taskId: t.id },
            getBoundingClientRect: () => ({ top: cardTops.get(t.id) ?? 0, height: 300 })
          }));
        }
      },
      renderGallery: recompute,
      requestApi: (u, o) => backend.handle(u, o)
    });
    cRef = c;
    await c.syncBackendTasks();
    await c.loadMoreTasks();
    c.currentFilter = 'running';
    recompute(); // 15 items (task-15..task-1), height = 4800px
    hostScrollY = clampScroll(2000);
    assert.equal(hostScrollY, 2000);
    const anchorTask = c.allTasks.filter(t => t.status === 'running')[7]; // index 7, top = 2240, preRelTop = +240
    assert.equal(cardTops.get(anchorTask.id) - hostScrollY, 240);

    // Complete TWO running tasks above viewport simultaneously (task-15 at 0px and task-14 at 320px, total -640px)
    backend.getById('task-15').status = 'completed';
    backend.getById('task-14').status = 'completed';
    await c.syncBackendTasks();
    await flushMicrotasks(20);

    assert.equal(hostScrollY, 1360, 'serialized commit queue must adjust 2000 -> 1360 (never double-compensate to 1040)');
    assert.equal(cardTops.get(anchorTask.id) - hostScrollY, 240, 'anchor card stays at exact +240px relative Y');
  }

  // 5) M2 Counterexample: 64-bit BigInt cursor bounds [1, 9223372036854775807]
  {
    const { context: c } = setup();
    const maxValid = c.parseSortCursor('9223372036854775807:2026-11-15T00:00:00Z');
    assert.equal(maxValid?.rowid, 9223372036854775807n);
    assert.equal(
      c.compareSortCursor('9007199254740993:2026-11-15T00:00:00Z', '9007199254740992:2026-11-15T00:00:00Z'),
      1,
      'BigInt cursor comparison must preserve exact precision above Number.MAX_SAFE_INTEGER'
    );
    for (const invalid of [
      '9223372036854775808:2026-11-15T00:00:00Z',
      '0:2026-11-15T00:00:00Z',
      '010:2026-11-15T00:00:00Z',
      '-5:2026-11-15T00:00:00Z',
      '9999999999999999999999999999:2026-11-15T00:00:00Z'
    ]) {
      assert.equal(c.parseSortCursor(invalid), null, `invalid cursor must be rejected: ${invalid}`);
    }
  }

  // 6) Stage-2 R2 Regression: Queued local write mutators (title/delete/rating/cleanup) must check mutEpoch FIRST
  //    without polluting new epoch state or re-bumping seq inside mutator()
  {
    const backend = createKeysetBackend(75);
    backend.getById('task-5').status = 'running';
    let pauseAnchor = false;
    let releaseAnchor = null;
    const { context: c } = setup({
      isEmbedded: true,
      visibleContentTop: async () => {
        if (pauseAnchor) {
          pauseAnchor = false;
          await new Promise(resolve => { releaseAnchor = resolve; });
        }
        return 600;
      },
      requestApi: async (u, o) => {
        const res = await backend.handle(u, o);
        if (u === '/api/tasks/task-5') {
          pauseAnchor = true;
        }
        return res;
      }
    });
    await c.syncBackendTasks();
    await c.loadMoreTasks();
    // Trigger single-task reconcile on task-5 and pause it inside visibleContentTop()
    backend.getById('task-5').status = 'completed';
    await c.syncBackendTasks();
    await flushMicrotasks(8);
    assert.equal(typeof releaseAnchor, 'function');

    // Queue all 4 local write mutators in epoch = 0 while taskCommitChain is paused in visibleContentTop()
    const staleTitleP = c.applyConfirmedTaskTitle('task-70', 'Old A');
    const staleDeleteP = c.applyConfirmedTaskDelete('task-69');
    const task68InEpoch0 = c.allTasks.find(t => t.id === 'task-68');
    const staleRatingP = c.applyConfirmedTaskRating(task68InEpoch0, 'nsfw');
    const staleCleanupP = c.applyCleanupReceipt({
      cleaned_unknown_task_ids: ['task-67'],
      cleaned_failed_task_ids: [],
      recovered_task_ids: ['task-66']
    });

    // Server concurrently advances task-70 to 'New B' while task-69, task-68 (general), task-67 remain live on backend
    backend.getById('task-70').request_params.title = 'New B';

    // User switches search query (epoch 0 -> 1) BEFORE queued mutators execute!
    c.taskSearchQuery = 'prompt-';
    c.taskQueryEpoch += 1;
    c.allTasks = [];
    c.taskLoadState = 'loading';
    const newEpochSyncP = c.syncBackendTasks(false, true, true);
    await flushMicrotasks(8);

    // Release visibleContentTop(): stale epoch=0 mutators run first, followed by epoch=1 head sync mutator
    releaseAnchor();
    await Promise.all([staleTitleP, staleDeleteP, staleRatingP, staleCleanupP, newEpochSyncP]);
    await flushMicrotasks(10);

    assert.equal(c.allTasks.find(t => t.id === 'task-70')?.title, 'New B', 'stale epoch title mutator must NOT leak localTaskOverrides into epoch 1');
    assert.ok(c.allTasks.some(t => t.id === 'task-69'), 'stale epoch delete mutator must NOT tombstone task-69 in epoch 1');
    assert.equal(c.allTasks.find(t => t.id === 'task-68')?.isNsfw, false, 'stale epoch rating mutator must NOT hide or mark task-68 in epoch 1');
    assert.ok(c.allTasks.some(t => t.id === 'task-67'), 'stale epoch cleanup mutator must NOT tombstone task-67 in epoch 1');
  }
});

