import test from 'node:test';import assert from 'node:assert/strict';import vm from 'node:vm';import fs from 'node:fs';
const html=fs.readFileSync(new URL('../src/static/index.html',import.meta.url),'utf8');
test('global preview sends authorized bytes and closes on navigation; stale reads stay closed',async()=>{
 let message,listener,focused=0;
 const focus={isConnected:true,focus(){focused++}};
 const parent={postMessage(data){message=data}};
 const context={isEmbedded:true,parent,viewEpoch:'1',hostOrigin:'https://studio.example',disposed:false,activeTask:{id:'task'},previewBlobs:new Map(),setTimeout,clearTimeout,document:{activeElement:focus,hidden:false,addEventListener(){},getElementById(){return {addEventListener(){}}}},window:{addEventListener(type,fn){if(type==='message')listener=fn}}};
 vm.createContext(context);vm.runInContext(html.slice(html.indexOf('    let previewRevision ='),html.indexOf('    /* Cursor-Invariant Affine Zoom Lightbox')),context);
 context.previewBlobs.set('blob:img',new Blob(['image'],{type:'image/png'}));const pending=context.openHostPreview('blob:img','Result');await new Promise(setImmediate);
 assert.equal(message.type,'media-preview');assert.equal(new TextDecoder().decode(message.bytes),'image');
 listener({source:parent,origin:context.hostOrigin,data:{channel:'vf-studio',type:'media-result',id:message.id,ok:true}});await pending;
 listener({source:parent,origin:context.hostOrigin,data:{channel:'vf-studio',type:'media-closed',id:message.id}});assert.equal(focused,1);
 let release;context.previewBlobs.set('blob:late',{type:'image/png',arrayBuffer:()=>new Promise(r=>release=r)});const late=context.openHostPreview('blob:late','Late');context.closeHostPreview();release(new ArrayBuffer(1));await late;assert.equal(message.type,'media-close');
});
