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
