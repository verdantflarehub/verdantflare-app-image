import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';

const html = readFileSync(new URL('../src/static/index.html', import.meta.url), 'utf8');
const context = vm.createContext({ googleChannelLogo: 'google', localChannelLogo: 'local' });
vm.runInContext(html.slice(html.indexOf('    function channelLogo('), html.indexOf('    const PNG_MODEL_LOGOS')), context);
vm.runInContext(html.slice(html.indexOf('    function getModelQuotas('), html.indexOf('    function isZombieTask(')), context);

test('Google Flow has its own task filter label, icon and reference quota', () => {
  for (const label of ['google-flow', 'Google Flow']) {
    assert.equal(context.channelDisplayName(label), 'Google Flow');
    assert.match(context.channelLogo(label), /Google Flow/);
  }
  assert.equal(context.channelDisplayName('gemini'), 'Google');
  assert.equal(context.getModelQuotas('nano21', 'edit', 'google-flow').maxImages, 3);
});
