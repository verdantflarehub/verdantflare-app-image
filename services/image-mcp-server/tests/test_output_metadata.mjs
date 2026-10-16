import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import vm from 'node:vm';
import { Blob } from 'node:buffer';
import { test } from 'node:test';

// Lightweight unit checks; no browser, server, network or image generation.
export async function runOutputMetadataTests() {
  const html = await readFile(new URL('../src/static/index.html', import.meta.url), 'utf8');
  for (const match of html.matchAll(/<script\b[^>]*>([\s\S]*?)<\/script>/g)) {
    new vm.Script(match[1]);
  }
  const source = html.slice(html.indexOf('    function outputMediaLabels('), html.indexOf('    function updateCardThumb('));
  let requests = 0, closes = 0, renders = 0, updates = 0;
  let release;
  const gate = new Promise(resolve => { release = resolve; });
  const png = new Blob([new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10])]);
  const context = vm.createContext({
    Blob, Uint8Array, console,
    disposed: false,
    allTasks: [{ artifactId: 'output' }], activeTask: { artifactId: 'output' },
    artifactBlobMap: new Map(), artifactLoadingMap: new Map(),
    artifactMediaMap: new Map(), previewBlobs: new Map(),
    URL: { createObjectURL: () => 'blob:original' },
    requestApi: async () => { requests++; await gate; return png; },
    createImageBitmap: async () => ({ width: 2048, height: 1152, close: () => { closes++; } }),
    renderGallery: () => { renders++; },
    updateInspectorImages: () => { updates++; }
  });
  vm.runInContext(source, context);
  const labels = context.outputMediaLabels;
  assert.equal(labels({ width: 2048, height: 1152, sizeBytes: 12000000, format: 'PNG' }).resShort, '2048×1152 px');
  assert.equal(labels({ width: 2048, height: 1152 }).aspectRatio, '16:9');
  assert.equal(labels({ width: 1152, height: 2048 }).aspectRatio, '9:16');
  assert.equal(labels({ width: 1254, height: 1254 }).aspectRatio, '1:1');
  assert.equal(labels({ width: 1536, height: 1024 }).aspectRatio, '3:2');
  assert.equal(labels({ width: 1001, height: 777 }).aspectRatio, '143:111');
  for (const value of [undefined, {}, { width: 2048 }, { width: 0, height: 10 }, { width: -1, height: 1 }, { width: NaN, height: 1 }, { width: '2048', height: 1152 }]) {
    assert.equal(labels(value).resShort, '实际尺寸未测得');
    assert.equal(labels(value).aspectRatio, '实际比例未测得');
  }
  assert.equal(labels({ resolution: '2k', aspect_ratio: '1:1' }).resShort, '实际尺寸未测得');
  assert.equal(labels({ sizeBytes: 12000000 }).sizeLabel, '12MB');
  assert.equal(labels({ sizeBytes: 1234567 }).sizeLabel, '1.23MB');
  assert.equal(labels({ sizeBytes: 1500 }).sizeLabel, '1.5KB');
  assert.equal(labels({ sizeBytes: 100 }).sizeLabel, '100B');
  assert.equal(labels().formatLabel, '格式未测得');
  for (const [bytes, format] of [
    [[137, 80, 78, 71], 'PNG'], [[255, 216, 255], 'JPEG'],
    ['RIFFxxxxWEBP', 'WebP'], ['GIF89a', 'GIF'], ['xxxxftypavif', 'AVIF'], ['unknown', '']
  ]) {
    const blob = new Blob([typeof bytes === 'string' ? bytes : new Uint8Array(bytes)], { type: 'image/png' });
    const measured = await context.measureOutputBlob(blob);
    assert.equal(measured.format, format);
    assert.equal(measured.sizeBytes, blob.size);
    assert.equal(measured.width, 2048);
  }
  assert.equal(closes, 6);
  const first = context.getArtifactBlobUrl('output');
  const second = context.getArtifactBlobUrl('output');
  // Polling can replace the task objects while the original image is loading.
  context.allTasks = [{ artifactId: 'output', requestedResolution: '2k', requestedAspectRatio: '1:1' }];
  context.activeTask = context.allTasks[0];
  release();
  assert.equal(await first, 'blob:original');
  assert.equal(await second, 'blob:original');
  assert.equal(requests, 1);
  assert.equal(context.activeTask.aspectRatio, '16:9');
  assert.equal(context.activeTask.resShort, '2048×1152 px');
  assert.equal(context.activeTask.img, 'blob:original');
  assert.equal(renders, 1);
  assert.equal(updates, 1);
  assert.equal(await context.getArtifactBlobUrl('output'), 'blob:original');
  assert.equal(requests, 1);
  assert.equal(labels(context.artifactMediaMap.get('output')).aspectRatio, '16:9');
  context.createImageBitmap = async () => { throw new Error('invalid image'); };
  const invalid = labels(await context.measureOutputBlob(new Blob(['broken'])));
  assert.equal(invalid.resShort, '实际尺寸未测得');
  assert.equal(invalid.formatLabel, '格式未测得');
  assert.equal(invalid.sizeLabel, '6B');
  context.disposed = true;
  assert.equal(await context.getArtifactBlobUrl('late-output'), '');
  assert.equal(context.artifactMediaMap.has('late-output'), false);
  return 'Passed: script syntax, exact dimensions/ratios, file sizes/formats, unknown output, decode failure, shared loading, polling race, cache and disposal.';
}

test('Image output metadata uses original bytes, not requested settings', runOutputMetadataTests);
