import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import vm from 'node:vm';
import {test} from 'node:test';

const html = await readFile(new URL('../src/static/index.html', import.meta.url), 'utf8');
const source = html.slice(html.indexOf('    function mergeTaskPage('), html.indexOf("    applyImageRoute('/dashboard#"));
function setup() {
  const calls = [], replies = [];
  const nodes = new Map();
  const context = vm.createContext({
    document: {hidden:false, getElementById: id => {if (!nodes.has(id)) nodes.set(id,{dataset:{},addEventListener(){}}); return nodes.get(id);}},
    window:{}, allTasks:[], taskLoadState:'loading', taskLoadError:'', taskPageError:'', taskNextOffset:0,
    taskPageExhausted:false, taskPageSize:50, taskTotal:0, syncingTasks:false, disposed:false,
    pendingDetailId:'', activeTask:null, artifactBlobMap:new Map(), artifactMediaMap:new Map(),
    requestApi:async url => {calls.push(url); const reply=replies.shift(); if(reply instanceof Error) throw reply; return reply;},
    loadCapabilities(){}, actionModeCode:()=> 'T2I', channelDisplayName:()=> 'channel', outputMediaLabels:()=> ({}),
    updateTaskFilters(){},renderGallery(){},loadThumbnails(){},renderChannelCatalog(){}
  });
  vm.runInContext(source,context);
  return {context,calls,replies,nodes};
}
const page = (start,count,total=123) => ({total,tasks:Array.from({length:count},(_,i)=>({task_id:`task-${start+i}`,status:'completed'}))});

test('continues beyond first page, retains history on refresh, deduplicates and stops at end',async()=>{
  const {context:c,calls,replies,nodes}=setup();
  replies.push(page(0,50)); await c.syncBackendTasks();
  replies.push(page(49,50)); await c.loadMoreTasks();
  assert.equal(c.allTasks.length,99);
  assert.equal(c.taskNextOffset,100);
  replies.push(page(0,100)); await c.syncBackendTasks();
  assert.equal(c.allTasks.length,100);
  assert.equal(calls[2],'/api/tasks?limit=100&offset=0');
  replies.push(page(100,23)); await c.loadMoreTasks();
  assert.equal(c.allTasks.length,123);
  assert.equal(calls[3],'/api/tasks?limit=50&offset=100');
  await c.loadMoreTasks(); assert.equal(calls.length,4);
  assert.equal(nodes.get('btnLoadMoreTasks').hidden,true);
});

test('page failure retains tasks and cursor; manual retry succeeds; concurrent requests are blocked',async()=>{
  const {context:c,calls,replies}=setup();
  replies.push(page(0,50)); await c.syncBackendTasks();
  const first=c.allTasks[0];
  replies.push(new Error('offline')); await c.loadMoreTasks();
  assert.equal(c.allTasks.length,50); assert.equal(c.allTasks[0],first);
  assert.equal(c.taskNextOffset,50); assert.equal(c.taskPageError,'offline');
  replies.push(page(50,50)); await Promise.all([c.loadMoreTasks(),c.loadMoreTasks(),c.syncBackendTasks()]);
  assert.equal(calls.length,3); assert.equal(c.allTasks.length,100); assert.equal(c.taskPageError,'');
  c.document.hidden=true; await c.loadMoreTasks(); assert.equal(calls.length,3);
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

  // Queue 6 P1 thumbnail loads and 1 P0 reference thumbnail load; also test deduplication
  const p1Promises = [
    thumbCtx.getArtifactThumbnailUrl('art-0', 320, 'p1'),
    thumbCtx.getArtifactThumbnailUrl('art-0', 320, 'p1'), // duplicate in-flight
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

  // Release first slot: P0 ('art-ref' edge=640) must preempt queued P1 ('art-4', 'art-5')
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
