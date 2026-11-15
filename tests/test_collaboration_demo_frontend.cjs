// Offline browser-state regression tests (no DOM, network or provider calls).
"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const {createPoller, createLauncher, createPlayback} = require("../aos/static/collaboration.js");

// Control promise completion explicitly so selection races are deterministic.
function deferred() {
  let resolve, reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return {promise, resolve, reject};
}

// Give each queued asynchronous continuation a chance to complete without real waiting.
async function settle() {
  for (let i = 0; i < 8; i++) await Promise.resolve();
}

// Replace wall-clock timers with an inspectable single-threaded queue.
function scheduler() {
  let sequence = 0;
  const tasks = new Map();
  return {
    setTimer(fn) { const id = ++sequence; tasks.set(id, fn); return id; },
    clearTimer(id) { tasks.delete(id); },
    fire() { const next = tasks.entries().next().value; assert.ok(next, "Expected a scheduled read"); tasks.delete(next[0]); next[1](); },
    count() { return tasks.size; }
  };
}

// Idle reads continue so externally imported conversations can appear.
test("idle polling continues with no selected conversation", async () => {
  const timer = scheduler(); let calls = 0;
  const poll = createPoller({...timer, read: async () => ++calls, onData() {}});
  poll.start(); timer.fire(); await settle();
  assert.equal(calls, 1); assert.equal(timer.count(), 1);
  timer.fire(); await settle(); assert.equal(calls, 2);
  poll.stop(); assert.equal(timer.count(), 0);
});

// A slow old response cannot overwrite a newly selected record.
test("selection changes discard old reads and never overlap requests", async () => {
  const timer = scheduler(), first = deferred(); const accepted = [];
  let calls = 0, concurrent = 0, peak = 0;
  const poll = createPoller({...timer,
    read: async id => { concurrent++; peak = Math.max(peak, concurrent); calls++; const result = calls === 1 ? await first.promise : id; concurrent--; return result; },
    onData: value => accepted.push(value)});
  poll.select("old"); poll.start(); timer.fire();
  poll.select("new"); assert.equal(calls, 1); assert.equal(timer.count(), 0);
  first.resolve("old result"); await settle();
  assert.deepEqual(accepted, []); assert.equal(timer.count(), 1);
  timer.fire(); await settle();
  assert.deepEqual(accepted, ["new"]); assert.equal(peak, 1); poll.stop();
});

// Stopping observation ignores an already pending read.
test("stop ignores the outstanding response and schedules no read", async () => {
  const timer = scheduler(), pending = deferred(); let accepted = 0;
  const poll = createPoller({...timer, read: () => pending.promise, onData: () => accepted++});
  poll.start(); timer.fire(); poll.stop(); pending.resolve({});
  await settle(); assert.equal(accepted, 0); assert.equal(timer.count(), 0);
});

// Transient read errors preserve prior data and schedule a read-only retry.
test("refresh failure retains the last snapshot and retries the read", async () => {
  const timer = scheduler(), values = []; let calls = 0, errors = 0;
  const poll = createPoller({...timer, read: async () => { if (++calls === 2) throw Error("offline"); return calls; },
    onData: value => values.push(value), onError: () => errors++});
  poll.start(); timer.fire(); await settle(); timer.fire(); await settle();
  assert.deepEqual(values, [1]); assert.equal(errors, 1); assert.equal(timer.count(), 1);
  poll.stop();
});

// A double click while the receipt is outstanding sends one POST.
test("double submit sends one start request", async () => {
  const pending = deferred(); let calls = 0;
  const launcher = createLauncher({newId: () => "launch-1", post: () => { calls++; return pending.promise; }});
  const first = launcher.submit({case_id:"case", version:"governed-v1"});
  assert.equal(await launcher.submit({case_id:"case", version:"governed-v1"}), null);
  assert.equal(calls, 1);
  pending.resolve({run_id:"run-1", status:"queued", duplicate:false});
  assert.equal((await first).run_id, "run-1"); assert.equal(launcher.state().pending, null);
});

// An uncertain receipt is never retried automatically; explicit rechecks retain the original ID.
test("uncertain start keeps its identity without an automatic retry", async () => {
  let calls = 0; const requests = [];
  const launcher = createLauncher({newId: () => "launch-fixed",
    post: async payload => { calls++; requests.push({...payload}); if (calls === 1) throw Error("connection dropped"); return {run_id:"run-1", status:"running", duplicate:true}; }});
  await assert.rejects(launcher.submit({case_id:"first", version:"governed-v1"}));
  await settle(); assert.equal(calls, 1); assert.equal(launcher.state().pending.launch_id, "launch-fixed");
  await launcher.submit({case_id:"different", version:"observed-v2"});
  assert.equal(calls, 2); assert.deepEqual(requests[0], requests[1]);
});

// Malformed receipts do not imply acceptance and retain the original launch identity.
test("invalid receipt is not displayed as an accepted conversation", async () => {
  const launcher = createLauncher({newId: () => "launch-1", post: async () => ({run_id:"bad"})});
  await assert.rejects(launcher.submit({case_id:"case", version:"observed-v2"}));
  assert.equal(launcher.state().pending.launch_id, "launch-1");
});

// A confirmed preflight rejection can be corrected as a new launch.
test("confirmed rejection clears the pending launch", async () => {
  const launcher = createLauncher({newId: () => "launch-1", post: async () => {
    const error = Error("disabled"); error.outcome_unknown = false; throw error;
  }});
  await assert.rejects(launcher.submit({case_id:"case", version:"governed-v1"}));
  assert.equal(launcher.state().pending, null);
});

// Replay affects only visible event count and never fabricates an extra event.
test("replay steps and pauses without changing the recorded event count", () => {
  const playback = createPlayback();
  assert.deepEqual(playback.state(3), {count:3, playing:false});
  playback.play(); assert.deepEqual(playback.state(3), {count:0, playing:true});
  playback.step(3); assert.deepEqual(playback.state(3), {count:1, playing:true});
  playback.pause(); assert.deepEqual(playback.state(3), {count:1, playing:false});
  playback.step(3); playback.step(3); assert.deepEqual(playback.state(3), {count:3, playing:false});
  playback.all(); assert.equal(playback.state(3).count, 3);
});

// A refreshed page checks the saved launch instead of generating a second conversation.
test("restored pending launch retains its identity after refresh", async () => {
  const prior = {case_id:"saved-case", version:"observed-v2", launch_id:"launch-saved"};
  let received;
  const launcher = createLauncher({restored: prior, newId: () => { throw Error("must reuse saved identity"); },
    post: async payload => { received = {...payload}; return {run_id:"run-saved", status:"completed", duplicate:true}; }});
  await launcher.submit({case_id:"different", version:"governed-v1"});
  assert.deepEqual(received, prior); assert.equal(launcher.state().pending, null);
});
