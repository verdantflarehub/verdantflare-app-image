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
  const c = vm.createContext({document:{getElementById:id=>id==='ratingClearDialog'?dialog:status},requestApi,
    allTasks:[task],taskSyncPromise:null,lastTaskStructure:'old',taskLoadState:'ready',renderGallery(){},async syncBackendTasks(){}});
  vm.runInContext(source,c);
  return {c,task,status,dialog};
}
test('pending write rejects duplicate click and updates only after receipt',async()=>{
  let finish,calls=0;
  const {c,task,status}=setup(()=>{calls++;return new Promise(r=>finish=r);});
  const first=c.changeTaskRating(task);
  await c.changeTaskRating(task);
  assert.equal(calls,1);assert.equal(task.isNsfw,false);
  assert.equal(vm.runInContext('pendingRatingTasks.has("task-1")',c),true);
  finish({ok:true,content_rating:{rating:'nsfw'}});await first;
  assert.equal(task.isNsfw,true);assert.match(status.textContent,/已标记/);
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
  assert.equal(calls,1);assert.equal(task.isNsfw,false);assert.match(status.textContent,/已移除/);
});
test('saved rating survives an older list and refresh failure is distinguished',async()=>{
  const {c,task,status}=setup(async()=>({ok:true,content_rating:{rating:'nsfw'}}));
  let finish;
  c.taskSyncPromise=new Promise(r=>finish=r);
  const save=c.changeTaskRating(task);
  const replacement={...task};c.allTasks=[replacement];finish();
  c.syncBackendTasks=async(append,force)=>{assert.equal(force,true);c.taskLoadState='error';};
  await save;
  assert.equal(replacement.isNsfw,true);assert.match(status.textContent,/已标记.*列表刷新失败/);
});
