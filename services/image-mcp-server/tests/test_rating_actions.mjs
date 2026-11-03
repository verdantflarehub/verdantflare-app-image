import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import vm from 'node:vm';
import {test} from 'node:test';

const html = await readFile(new URL('../src/static/index.html', import.meta.url), 'utf8');
const source = html.slice(html.indexOf('    const pendingRatingTasks'), html.indexOf('    async function handleTaskCardAction'));
function setup(requestApi) {
  const status = {setAttribute(k,v){this[k]=v;}};
  const dialog = {open:false, returnValue:'', addEventListener(_,f){this.closeHandler=f;}, showModal(){this.open=true;}, close(value){this.returnValue=value;this.open=false;this.closeHandler();}};
  const task = {id:'task-1',isNsfw:false};
  const c = vm.createContext({document:{getElementById:id=>['ratingClearDialog','taskDeleteDialog'].includes(id)?dialog:status},requestApi,
    setTimeout:()=>1,clearTimeout(){},pendingDeleteTasks:new Set(),taskTotal:1,galleryTaskKeys:new Map([['task-1','old']]),allTasks:[task],taskSyncPromise:null,lastTaskStructure:'old',taskLoadState:'ready',renderGallery(){},async syncBackendTasks(){}});
  vm.runInContext(source,c);
  return {c,task,status,dialog};
}
test('pending write rejects duplicate click and updates only after receipt',async()=>{
  let finish,calls=0;
  const {c,task,status}=setup(()=>{calls++;return new Promise(r=>finish=r);});
  const first=c.changeTaskRating(task);
  await c.changeTaskRating(task);
  assert.equal(calls,1);assert.equal(task.isNsfw,false);
  assert.equal(c.galleryTaskKeys.has(task.id),false);
  assert.equal(vm.runInContext('pendingRatingTasks.has("task-1")',c),true);
  finish({ok:true,content_rating:{rating:'nsfw'}});await first;
  assert.equal(task.isNsfw,true);assert.match(status.textContent,/已设为 NSFW/);
  assert.equal(vm.runInContext('pendingRatingTasks.size',c),0);
});
test('403 and malformed receipts produce visible error without false success',async()=>{
  for(const response of [new Error('API 403'),{ok:true,content_rating:{rating:'general'}}]){
    const {c,task,status}=setup(async()=>{if(response instanceof Error)throw response;return response;});
    await c.changeTaskRating(task);
    assert.equal(task.isNsfw,false);assert.equal(status.hidden,false);assert.equal(status.role,'alert');
    assert.match(status.textContent,/未确认保存/);
  }
});
test('clear uses page dialog and cancellation sends no write',async()=>{
  let calls=0;
  const {c,task,dialog,status}=setup(async()=>{calls++;return {ok:true,content_rating:{rating:'general'}};});
  task.isNsfw=true;
  const cancel=c.changeTaskRating(task);assert.equal(dialog.open,true);dialog.close('cancel');await cancel;
  assert.equal(calls,0);assert.equal(task.isNsfw,true);
  const clear=c.changeTaskRating(task);dialog.close('confirm');await clear;
  assert.equal(calls,1);assert.equal(task.isNsfw,false);
  assert.equal(c.galleryTaskKeys.has(task.id),false);assert.match(status.textContent,/已设为 SFW/);
});
test('saved rating survives an older list and refresh failure is distinguished',async()=>{
  const {c,task,status}=setup(async()=>({ok:true,content_rating:{rating:'nsfw'}}));
  let finish;
  c.taskSyncPromise=new Promise(r=>finish=r);
  const save=c.changeTaskRating(task);
  const replacement={...task};c.allTasks=[replacement];finish();
  c.syncBackendTasks=async(append,force)=>{assert.equal(force,true);c.taskLoadState='error';};
  await save;
  assert.equal(replacement.isNsfw,true);assert.match(status.textContent,/已设为 NSFW.*列表刷新失败/);
});


test('deletion confirms in-page, cancels without write, and rejects duplicate submissions',async()=>{
  let calls=0,finish;
  const {c,task,dialog,status}=setup(()=>{calls++;return new Promise(resolve=>finish=resolve);});
  const canceled=c.deleteTask(task.id);dialog.close('cancel');await canceled;
  assert.equal(calls,0);assert.equal(c.allTasks.length,1);
  const deleting=c.deleteTask(task.id);dialog.close('confirm');await Promise.resolve();
  await c.deleteTask(task.id);assert.equal(calls,1);
  finish({ok:true,deleted:true,task_id:task.id});await deleting;
  assert.equal(c.allTasks.length,0);assert.equal(c.taskTotal,0);assert.equal(status.textContent,'已删除任务');
});

test('failed or malformed delete retains task and shows failure',async()=>{
  for(const result of [new Error('API 403'),{ok:true,deleted:false,task_id:'task-1'}]){
    const {c,task,dialog,status}=setup(async()=>{if(result instanceof Error)throw result;return result;});
    const deleting=c.deleteTask(task.id);dialog.close('confirm');await deleting;
    assert.equal(c.allTasks.length,1);assert.match(status.textContent,/删除失败/);
  }
});

test('a saved NSFW task next submits clear, never another mark',async()=>{
  const actions=[];
  const {c,task,dialog}=setup(async(_path,options)=>{actions.push(options.body.action);return {ok:true,content_rating:{rating:options.body.action==='mark'?'nsfw':'general'}};});
  await c.changeTaskRating(task);
  const next=c.changeTaskRating(task);dialog.close('confirm');await next;
  assert.deepEqual(actions,['mark','clear']);assert.equal(task.isNsfw,false);
  assert.match(html,/isNsfw \? 'SFW' : 'NSFW'/);
});
