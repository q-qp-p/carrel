"""Execute the enrollment form's render/save flow without a browser or live services."""
import shutil
import subprocess
from pathlib import Path

import pytest


def test_service_controls_follow_deployment_edition():
    node = shutil.which('node')
    if not node:
        pytest.skip('Node.js is needed for console behavior tests')
    static = Path(__file__).resolve().parents[1] / 'kb_server/static'
    script = r'''
const assert = require('node:assert/strict');
const fs = require('node:fs');
const source = fs.readFileSync(process.argv[1] + '/app.js', 'utf8');
const html = fs.readFileSync(process.argv[1] + '/index.html', 'utf8');
assert.match(html, /id="btn-health" hidden/);
assert.match(html, /\.tabs button\[hidden\]\{display:none\}/);
const start = source.indexOf('async function _refreshHealth()');
const definition = source.slice(start, source.indexOf('\n}', start) + 2);
const elements = new Map();
const $ = id => {
  if (!elements.has(id)) elements.set(id, {hidden: true, innerHTML: '',
    replaceChildren() { this.innerHTML = ''; }, addEventListener() {}});
  return elements.get(id);
};
const $$ = () => [];
const state = {}, SERVICES = [['database', 'qdrant', h => h.qdrant]];
const svcBulk = null, svcRestarting = new Map();
const esc = s => s;
let language = 'zh', health;
const t = s => language === 'en' && s === '系统服务' ? 'System services' : s;
const api = async () => health;
const onRestartService = () => { throw new Error('unexpected action'); };
eval(definition);
(async () => {
  for (language of ['zh', 'en']) {
    health = {system_services: true, qdrant: true, managed: []};
    await _refreshHealth();
    assert.equal($('#btn-health').hidden, false);
    assert.match($('#health-body').innerHTML, /database/);
    health = {system_services: false};
    await _refreshHealth();
    assert.equal($('#btn-health').hidden, true);
    assert.equal($('#health-body').innerHTML, '');
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
'''
    result = subprocess.run([node, '-e', script, str(static)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_enrollment_draft_readiness_and_directory_switching():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is needed for console behavior tests")
    source = Path(__file__).resolve().parents[1] / "kb_server/static/app.js"
    script = r'''
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync(process.argv[1], 'utf8');
function definition(name) {
  const start = source.search(new RegExp('^(?:async )?function ' + name + '\\(', 'm'));
  assert.ok(start >= 0, name);
  return source.slice(start, source.indexOf('\n}', start) + 2);
}
const definitions = ['renderConfig', 'parseFormValues', 'requireInt', 'collectParseConfig',
  'collectGraphConfig', 'collectInterval', 'collectNewChunk', 'splitInterval', 'splitNewChunk',
  'rebuildProgressParts', 'vlmOptions', 'parseAction', 'graphAction', 'renderPipelineActions'].map(definition).join('\n');
const saveStart = source.indexOf('$("#cfg-save").addEventListener');
const saveHandler = source.slice(saveStart, source.indexOf('\n});', saveStart) + 4);
const harness = `
let formBase = null, schemaResultApplied = '', buildStarting = false, graphPendingAction = '';
const draftCache = new Map(), elements = new Map(), calls = [], notices = [];
const parseActionPending = new Map();
const models = [
  {name: 'vision-a', protocol: 'openai', multimodal: true},
  {name: 'text-only', protocol: 'openai', multimodal: false},
  {name: 'messages-only', protocol: 'anthropic', multimodal: true}
];
let selected = {dir: 'Library A', state: 'unenrolled'};
const state = {draft: selected.dir, cfgKey: null, llms: [], graphSchema: {}, tabByKb: {}};
function $(id) {
  if (!elements.has(id)) elements.set(id, {
    style: {}, checked: false, open: false, _value: '',
    get value() { return this._value; }, set value(v) { this._value = String(v); },
    addEventListener(event, fn) { this[event] = fn; }
  });
  return elements.get(id);
}
const selectedEntry = () => selected, t = s => s, kbDot = () => '', kbStateText = () => '';
const esc = s => s;
const renderAdoptRow = () => {}, updateTabs = () => {}, syncTuneButton = () => {};
const setSampleCount = () => {}, setChunkHint = () => {}, setOverlapHint = () => {};
const refreshSchemaTaskResult = () => {}, refreshTab = () => {}, remember = () => {};
const refreshOverview = async () => {}, llmOptions = () => '', restoreDraft = () => false;
const setSchemaView = () => { state.graphSchema = {dirty: false, active: ''}; };
const toast = msg => notices.push(msg);
let read = async path => path === '/llms' ? models : {};
async function api(path, options) {
  if (options) { calls.push({path, ...options}); return {}; }
  return read(path);
}
`;
const checks = `
(async () => {
  // Saving before defaults arrive must not submit values from another library.
  let release;
  const gate = new Promise(resolve => release = resolve);
  read = async path => { await gate; return path === '/llms' ? models : {}; };
  const pending = renderConfig();
  $('#cfg-max').value = 512;
  $('#cfg-ov').value = 64;
  $('#cfg-vlm').value = 'vision-a';
  await $('#cfg-save').click();
  assert.equal(calls.length, 0);
  assert.match(notices.at(-1), /载入中/);
  release(); await pending;
  $('#cfg-max').value = 512;
  $('#cfg-ov').value = 64;
  const visionOptions = vlmOptions('vision-a');
  assert.match(visionOptions, /vision-a/);
  assert.doesNotMatch(visionOptions, /text-only|messages-only/);
  assert.match(vlmOptions('removed'), /removed.*不可用/);
  // A ready form still needs an explicit vision model before enrollment.
  $('#cfg-vlm').value = '';
  await $('#cfg-save').click();
  assert.equal(calls.length, 0);
  assert.match(notices.at(-1), /请选择多模态模型/);
  $('#cfg-vlm').value = 'vision-a';
  await $('#cfg-save').click();
  assert.equal(calls.length, 1, 'a ready new-library form must reach enrollment');
  assert.equal(calls[0].path, '/enroll');
  assert.equal(calls[0].body.dir, 'Library A');
  assert.equal(calls[0].body.config.max_tokens, 512);
  assert.equal(calls[0].body.config.overlap_tokens, 64);
  assert.equal(calls[0].body.config.vlm_model, 'vision-a');
  assert.equal(calls[0].body.config.graph_enabled, false);
  assert.equal(state.draft, null);

  // A late response for A must not overwrite B's defaults or readiness.
  state.draft = selected.dir; state.cfgKey = null;
  const late = new Promise(resolve => release = resolve);
  read = async path => { await late; return path === '/llms' ? models : {}; };
  const oldRender = renderConfig();
  selected = {dir: 'Library B', state: 'unenrolled', dir_defaults: {max_tokens: 600}};
  state.draft = selected.dir; state.cfgKey = null;
  read = async path => path === '/llms' ? models : {};
  await renderConfig();
  release(); await oldRender;
  assert.equal(formBase.dir, 'Library B');
  assert.equal($('#cfg-max').value, '600');
  assert.equal(collectGraphConfig().graph_enabled, false);

  // Cancelling the draft clears readiness; unavailable optional hints still allow defaults.
  state.draft = null; state.cfgKey = null;
  await renderConfig();
  assert.throws(() => collectGraphConfig(), /载入中/);
  state.draft = selected.dir; state.cfgKey = null;
  read = async () => { throw new Error('temporary hint failure'); };
  await renderConfig();
  assert.equal(formBase.dir, 'Library B');
  $('#cfg-vlm').value = 'vision-a';
  await $('#cfg-save').click();
  assert.equal(calls.length, 2);
  assert.equal(calls[1].body.dir, 'Library B');
  assert.equal(calls[1].body.config.vlm_model, 'vision-a');

  const config = enabled => ({
    effective: {graph_enabled: enabled, max_tokens: 400, overlap_tokens: 80, vlm_model: 'vision-a'},
    config: {}, limits: {}, schema: null,
  });
  // A slow config/model request must not briefly display the opposite saved state.
  // Failed or stopped builds still have their graph policy enabled.
  for (const status of ['pending', 'running', 'ok', 'failed', 'stopped', 'disabled']) {
    const enabled = status !== 'disabled';
    selected = {dir: 'Library C', kb_id: 'synthetic-c', state: 'active', graph_status: status};
    state.draft = null; state.cfgKey = null; formBase = null;
    $('#cfg-graph').checked = !enabled;
    const waiting = new Promise(resolve => release = resolve);
    read = async path => { await waiting; return path === '/llms' ? models : config(enabled); };
    const loading = renderConfig();
    assert.equal($('#sel-toggle').checked, true);
    assert.equal($('#cfg-graph').checked, enabled, 'first render: ' + status);
    assert.equal($('#graph-fields').disabled, !enabled);
    assert.equal($('#cfg-gbuild').disabled, status !== 'running');
    assert.equal($('#cfg-grebuild').disabled, !enabled || status === 'running');
    assert.equal($('#cfg-gbuild').textContent, status === 'running' ? '暂停建图' : status === 'stopped' ? '继续建图' : '立即建图');
    release(); await loading;
    assert.equal($('#cfg-graph').checked, enabled, 'after config: ' + status);
  }

  // Ordinary overview polling must keep the user's unsaved switch selection.
  $('#cfg-graph').checked = true;
  let reads = 0;
  read = async () => { reads++; throw new Error('unexpected re-fetch'); };
  await renderConfig();
  assert.equal($('#cfg-graph').checked, true);
  assert.equal(reads, 0);

  // On refresh of the same loaded form, keep its displayed value while waiting.
  state.cfgKey = null;
  const reloading = new Promise(resolve => release = resolve);
  read = async path => { await reloading; return path === '/llms' ? models : config(false); };
  const reload = renderConfig();
  assert.equal($('#cfg-graph').checked, true);
  release(); await reload;

  // Switching from an enabled to a disabled graph must use the new library's overview
  // immediately; the former library's later response cannot switch it back on.
  selected = {dir: 'Library D', kb_id: 'synthetic-d', state: 'active', graph_status: 'ok'};
  const delayed = new Promise(resolve => release = resolve);
  read = async path => { await delayed; return path === '/llms' ? models : config(true); };
  const oldLibrary = renderConfig();
  assert.equal($('#cfg-graph').checked, true);
  selected = {dir: 'Library E', kb_id: 'synthetic-e', state: 'active', graph_status: 'disabled'};
  read = async path => path === '/llms' ? models : config(false);
  await renderConfig();
  assert.equal($('#cfg-graph').checked, false);
  release(); await oldLibrary;
  assert.equal($('#cfg-graph').checked, false);
  assert.equal(calls.length, 2, 'rendering switches must not write configuration');
})()
`;
vm.runInNewContext(harness + definitions + '\n' + saveHandler + '\n' + checks, {assert})
  .catch(error => { console.error(error); process.exitCode = 1; });
'''
    result = subprocess.run([node, "-e", script, str(source)], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stdout + result.stderr


def test_pipeline_action_states_and_requests():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is needed for console behavior tests")
    source = Path(__file__).resolve().parents[1] / "kb_server/static/app.js"
    script = r'''
const assert = require('node:assert/strict'), fs = require('node:fs'), vm = require('node:vm');
const source = fs.readFileSync(process.argv[1], 'utf8');
const between = (a,b) => source.slice(source.indexOf(a), source.indexOf(b,source.indexOf(a)));
const code = between('const parseActionPending =', 'async function renderConfig()')
 + between('$("#cfg-parse-now").addEventListener', '// Changing the version only changes')
 + between('let buildStarting = false;', '$("#cfg-gvdel").addEventListener');
const harness = `
const elements = new Map(), calls = [], notices = [];
let kb = {kb_id:'synthetic',dir:'Synthetic library',state:'active',graph_status:'ok'};
let confirmed = true, fail = false, wait = null;
function $(id) { if(!elements.has(id)) elements.set(id,{checked:true,addEventListener(e,fn){this[e]=fn;}}); return elements.get(id); }
const selectedEntry=()=>kb, t=x=>x, toast=x=>notices.push(x), noticeDialog=async x=>notices.push(x);
const PARSE_BUSY_MSG='parse busy', collectGraphConfig=()=>({graph_enabled:true}), markGraphConfigSaved=()=>{};
const setTab=()=>{}, deltaText=()=>'', confirmDialog=async()=>confirmed, refreshOverview=async()=>renderPipelineActions(kb);
async function api(path,options) {calls.push({path,...options}); if(wait) await wait; if(fail) throw new Error('offline'); return {};}
`;
const checks = `
(async()=>{
 const base = {state:'active',kb_id:'synthetic',graph_status:'ok'};
 const parseCases = [
  [{},'start',false], [{files_pending:1},'start',true], [{jobs_queued:1},'start',true],
  [{files_total:0,dir_files:2},'start',true], [{jobs_running:1,jobs_queued:2},'pause',true],
  [{parse_paused:true},'resume',true], [{parse_paused:true,jobs_running:1,parse_pausing:true},'pause',false],
  [{files_failed:1},'start',false], [{state:'inactive',jobs_queued:1},'start',false]
 ];
 for(const [extra,kind,enabled] of parseCases) {const a=parseAction({...base,...extra});assert.equal(a.kind,kind);assert.equal(a.enabled,enabled);}
 const graphCases = [
  [{},'start',false], [{graph_status:'pending',rebuild_check:{manual_run:{kind:'build'}}},'start',true],
  [{rebuild_check:{manual_run:{kind:'append'}}},'start',true],
  [{graph_status:'running'},'pause',true],
  [{graph_status:'stopped',graph_build:{cache_reusable:true}},'resume',true],
  [{graph_status:'stopped',graph_build:{cache_reusable:false}},'resume',false],
  [{graph_paused:true},'resume',true],
  [{graph_status:'failed',graph_build:{cache_reusable:true}},'start',true],
  [{graph_status:'failed',graph_build:{cache_reusable:false}},'start',false],
  [{graph_status:'disabled',rebuild_check:{manual_run:{kind:'build'}}},'start',false],
  [{jobs_active:1,rebuild_check:{manual_run:{kind:'append'}}},'start',false],
  [{schema_task:{status:'running'},rebuild_check:{manual_run:{kind:'append'}}},'start',false]
 ];
 for(const [extra,kind,enabled] of graphCases) {const a=graphAction({...base,...extra});assert.equal(a.kind,kind);assert.equal(a.enabled,enabled);}
 renderPipelineActions(kb);
 assert.equal($('#cfg-parse-now').textContent,'立即解析');assert.equal($('#cfg-parse-now').disabled,true);
 assert.equal($('#cfg-gbuild').textContent,'立即建图');assert.equal($('#cfg-gbuild').disabled,true);
 await $('#cfg-parse-now').click();await $('#cfg-gbuild').click();assert.equal(calls.length,0);
 for(const [extra,endpoint] of [[{jobs_queued:1},'parse_now'],[{jobs_running:1},'parse_pause'],[{parse_paused:true},'parse_now']]) {
  kb={...base,...extra};calls.length=0;await $('#cfg-parse-now').click();assert.ok(calls[0].path.endsWith(endpoint));
 }
 kb={...base,jobs_queued:1};calls.length=0;
 let finish;wait=new Promise(resolve=>finish=resolve);
 const pending=$('#cfg-parse-now').click();await $('#cfg-parse-now').click();
 assert.equal(calls.length,1);renderPipelineActions(kb);assert.equal($('#cfg-parse-now').disabled,true);
 finish();await pending;wait=null;
 kb={...base,rebuild_check:{manual_run:{kind:'append'}}};calls.length=0;
 await $('#cfg-gbuild').click();assert.equal(calls[1].body.mode,'run');
 kb={...base,graph_status:'running'};calls.length=0;
 await $('#cfg-gbuild').click();assert.equal(calls.length,1);assert.ok(calls[0].path.endsWith('graph_pause'));
 kb={...base,graph_status:'stopped',graph_build:{cache_reusable:true}};calls.length=0;
 await $('#cfg-gbuild').click();assert.equal(calls[1].body.mode,'run');
 confirmed=false;calls.length=0;await $('#cfg-grebuild').click();assert.equal(calls.length,0);
 confirmed=true;await $('#cfg-grebuild').click();assert.equal(calls[1].body.mode,'rebuild');
 calls.length=0;wait=new Promise(resolve=>finish=resolve);
 const building=$('#cfg-gbuild').click();await $('#cfg-grebuild').click();await $('#cfg-gbuild').click();
 assert.equal(calls.length,1);assert.equal($('#cfg-gbuild').disabled,true);assert.equal($('#cfg-grebuild').disabled,true);
 finish();await building;wait=null;
 fail=true;await $('#cfg-gbuild').click();assert.equal(buildStarting,false);assert.equal($('#cfg-gbuild').disabled,false);
 assert.match(notices.at(-1),/offline/);
})()
`;
vm.runInNewContext(harness+code+checks,{assert}).catch(e=>{console.error(e);process.exitCode=1;});
'''
    result = subprocess.run([node, "-e", script, str(source)], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stdout + result.stderr
