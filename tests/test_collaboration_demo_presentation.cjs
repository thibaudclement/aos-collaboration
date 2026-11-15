// Offline presentation checks: exact saved answers, scoped empty responses, and unchanged peer counts.
"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const {conversationSummary, createPlayback} = require("../aos/static/collaboration.js");

// Build a saved conversation whose final peer response contains no claims.
function example() {
  return {
    metadata: {run_id: "run-example", title: "Saved example", case_id: "example", status: "completed",
      version: "observed-v2", mode: "recorded_replay", model: "scripted", backend: "scripted"},
    view: {snapshot_status: "completed", scenario: {task: "Use the supplied evidence."},
      executor_id: "agent-001", participants: [{id: "agent-001", name: "Requester"}, {id: "agent-002", name: "Owner"}],
      timeline: [{kind: "request", sender_id: "agent-001", recipient_id: "agent-002", request_id: "request-1", delivery: "confirmed", body: "What is supported?"},
        {kind: "response", sender_id: "agent-002", recipient_id: "agent-001", request_id: "request-1", delivery: "confirmed", response: {claims: []}}],
      local_evidence: [], evidence: [],
      answer: {state: "completed", text: "Recorded conclusion\n<keep this text exactly>", used_evidence_refs: ["ref-existing"], insufficient_evidence: false},
      conflict_detection: {status: "not_recorded"}},
    evaluation: {status: "not_loaded"}, warnings: []
  };
}

// Rewinding affects visibility only and never changes the saved answer or peer count.
test("the saved outcome appears only at replay end and hides again on rewind", () => {
  const {metadata, view} = example(), playback = createPlayback();
  const before = JSON.stringify(view);
  playback.play();
  assert.equal(conversationSummary(metadata, view, playback.state(2).count), null);
  playback.step(2);
  assert.equal(conversationSummary(metadata, view, playback.state(2).count), null);
  playback.step(2);
  const summary = conversationSummary(metadata, view, playback.state(2).count);
  assert.equal(summary.text, view.answer.text);
  assert.deepEqual(summary.refs, view.answer.used_evidence_refs);
  assert.equal(summary.accepted, true);
  summary.refs.push("display-only");
  assert.equal(JSON.stringify(view), before);
  playback.play();
  assert.equal(conversationSummary(metadata, view, playback.state(2).count), null);
});

// All currently visible messages do not imply that a live or uncertain execution finished.
test("active and uncertain executions have no terminal summary", () => {
  const {metadata, view} = example();
  for (const status of ["queued", "preparing", "running", "unknown"]) {
    assert.equal(conversationSummary({...metadata, status}, view, 2), null);
  }
  assert.equal(conversationSummary(metadata, null, 0), null);
  assert.equal(conversationSummary(null, view, 2), null);
});

// A failure must not acquire an accepted answer from an empty or partial saved result.
test("failed and interrupted executions preserve their distinction from accepted answers", () => {
  const {metadata, view} = example();
  const empty = {...view, snapshot_status: "failed", answer: {state: "not_available", text: "", used_evidence_refs: []}};
  const failed = conversationSummary({...metadata, status: "failed"}, empty, 2);
  assert.equal(failed.heading, "Execution failed");
  assert.equal(failed.text, "No accepted final answer was recorded.");
  assert.equal(failed.accepted, false);
  const interrupted = conversationSummary({...metadata, status: "interrupted"}, view, 2);
  assert.equal(interrupted.heading, "Execution interrupted");
  assert.equal(interrupted.text, view.answer.text);
  assert.equal(interrupted.accepted, false);
  assert.match(interrupted.note, /not an accepted final answer/);
});

// A source-final answer remains exact when its outer generation health check failed.
test("a failed generation stage retains native final text without accepting completion", () => {
  const {metadata, view} = example();
  const summary = conversationSummary({...metadata, mode: "native_replay", version: "native-decentralized",
    status: "failed", generation_stage_status: "failed"}, view, view.timeline.length);
  assert.equal(summary.heading, "Execution failed");
  assert.equal(summary.accepted, false);
  assert.equal(summary.text, view.answer.text);
  assert.match(summary.note, /not an accepted final answer/);
});

// A stale incomplete projection must not become accepted merely because the registry completed.
test("acceptance requires the completed saved answer and preserves insufficient evidence", () => {
  const {metadata, view} = example();
  const stale = {...view, snapshot_status: "running", answer: {...view.answer, state: "recorded_before_run_finished"}};
  assert.equal(conversationSummary(metadata, stale, 2).accepted, false);
  const insufficient = {...view, answer: {...view.answer, insufficient_evidence: true}};
  const summary = conversationSummary(metadata, insufficient, 2);
  assert.equal(summary.accepted, true);
  assert.match(summary.note, /insufficient evidence/);
  assert.equal(summary.text, view.answer.text);
});

// Provide only the DOM methods used by this view (reject any attempt to inject HTML).
function browserHarness(snapshot) {
  const elements = new Map(), timers = new Map();
  let sequence = 0;
  // Implement only the browser element behavior exercised by this presentation check.
  class Element {
    // Store simple DOM state while keeping rendered text separate from HTML execution.
    constructor(tag) {
      this.tagName = tag; this.children = []; this.dataset = {}; this.listeners = {}; this.attributes = {};
      this.hidden = false; this.value = ""; this._text = ""; this.parentNode = null;
      this.classList = {add() {}, remove() {}};
    }
    // Register dynamically created element identifiers for later resets.
    set id(value) { this._id = value; elements.set(value, this); }
    // Return the recorded identifier without inventing a DOM node.
    get id() { return this._id; }
    // Model textContent assignment without parsing markup.
    set textContent(value) { this._text = String(value); this.children = []; }
    // Include child text for assertions over an entire rendered card.
    get textContent() { return this._text + this.children.map(child => child.textContent).join(""); }
    // Fail the test if agent-controlled text is ever sent through HTML interpolation.
    set innerHTML(_) { throw Error("HTML injection is forbidden"); }
    // Expose select options through the children already created by the frontend.
    get options() { return this.children; }
    // Preserve parent relationships so the summary's placement can be checked.
    appendChild(child) { child.parentNode = this; this.children.push(child); return child; }
    // Reset children without detaching the element itself from its parent.
    replaceChildren() { this.children = []; this._text = ""; }
    // Record attributes without evaluating them.
    setAttribute(key, value) { this.attributes[key] = value; }
    // Retain explicit user-action callbacks for deterministic browser tests.
    addEventListener(name, callback) { this.listeners[name] = callback; }
    // Place the summary after local evidence and outside the ordered peer list.
    insertAdjacentElement(where, child) {
      assert.equal(where, "afterend");
      child.parentNode = this.parentNode;
      this.parentNode.children.splice(this.parentNode.children.indexOf(this) + 1, 0, child);
    }
    // Find the one unordered reference list used by the existing outcome panel.
    querySelector(selector) {
      if (selector === "ul") {
        let child = this.children.find(item => item.tagName === "ul");
        if (!child) child = this.appendChild(new Element("ul"));
        return child;
      }
      return null;
    }
    // Return evidence disclosures recursively for the existing expansion-state code.
    querySelectorAll(selector) {
      const found = [];
      for (const child of this.children) {
        if (child.tagName === "details" && child.dataset.evidenceKey && (!selector.includes("[open]") || child.open)) found.push(child);
        found.push(...child.querySelectorAll(selector));
      }
      return found;
    }
    // Keep selection-only browser actions inert in the local harness.
    scrollIntoView() {}
    // Keep keyboard focus changes inert in the local harness.
    focus() {}
  }
  const html = fs.readFileSync(path.join(__dirname, "../aos/static/collaboration.html"), "utf8");
  const panel = new Element("section");
  for (const match of html.matchAll(/id="([^"]+)"/g)) {
    const element = new Element(match[1] === "timeline" ? "ol" : "div");
    element.id = match[1]; panel.appendChild(element);
  }
  const storage = {getItem() { return null; }, setItem() {}, removeItem() {}};
  const window = {
    document: {getElementById: id => elements.get(id) || null, createElement: tag => new Element(tag), documentElement: {dataset: {}}},
    location: {origin: "http://127.0.0.1:8770"}, localStorage: storage, sessionStorage: storage,
    crypto: {randomUUID() { throw Error("No start expected"); }}, addEventListener() {},
    setTimeout(callback) { const id = ++sequence; timers.set(id, callback); return id; },
    clearTimeout(id) { timers.delete(id); },
    async fetch(url, options) {
      assert.equal(options.method, undefined, "Presentation must not submit work");
      const data = url.endsWith("/catalog") ? {cases: []} : url.endsWith("/runs") ?
        {runs: [snapshot.metadata], active_run_id: null} : snapshot;
      return {ok: true, async json() { return data; }};
    }
  };
  const source = fs.readFileSync(path.join(__dirname, "../aos/static/collaboration.js"), "utf8");
  vm.runInNewContext(source, {window, setTimeout: window.setTimeout, clearTimeout: window.clearTimeout});
  return {elements, panel,
    // Complete a queued read with no network and no wall-clock waiting.
    async tick() { const [id, callback] = timers.entries().next().value; timers.delete(id); await callback(); },
    // Trigger a saved UI action without manufacturing model responses.
    click(id) { elements.get(id).listeners.click(); }};
}

// Exercise actual browser rendering rather than merely comparing helper strings.
test("the DOM scopes empty replies and places an exact outcome outside the peer list", async () => {
  const snapshot = example(), before = JSON.stringify(snapshot);
  const browser = browserHarness(snapshot);
  await browser.tick();
  await browser.tick();
  const timeline = browser.elements.get("timeline"), summary = browser.elements.get("conversation-end-summary");
  assert.equal(timeline.children.length, 2);
  assert.match(timeline.children[1].textContent, /This response contained no released claims\./);
  assert.equal(summary.parentNode, browser.elements.get("local-evidence").parentNode);
  assert.equal(timeline.children.includes(summary), false);
  assert.equal(summary.hidden, false);
  assert.match(summary.textContent, /not a peer message/);
  assert.ok(summary.textContent.includes(snapshot.view.answer.text));
  assert.equal(browser.elements.get("event-count").textContent, "2 recorded peer events");
  browser.click("play");
  assert.equal(summary.hidden, true);
  browser.click("show-all");
  assert.equal(summary.hidden, false);
  browser.click("new-session");
  assert.equal(summary.hidden, true);
  assert.equal(summary.children.length, 0);
  assert.equal(JSON.stringify(snapshot), before);
});

// Native replay displays recorded text without manufacturing release evidence or delivery receipts.
test("native messages and gathered summary retain their separate source meanings", async () => {
  const snapshot = example();
  snapshot.metadata.mode = "native_replay";
  snapshot.metadata.version = "native-decentralized";
  snapshot.view.replay_kind = "pisas_decentralized";
  snapshot.view.source_note = "Native source log order; timestamps were not recorded.";
  snapshot.view.timeline = [{kind: "native_message", sender_id: "agent-002", recipient_id: "agent-001",
    body: "The room seats 24. <literal text>", timestamp: null, request_id: null,
    delivery: "recorded_in_source", source_type: "received", source_log_index: 1}];
  snapshot.view.native_gathering = {text: "The room seats 24.",
    note: "Native gathered summary; no AOS claim-release audit was recorded."};
  snapshot.view.answer.used_evidence_refs = [];
  const before = JSON.stringify(snapshot), browser = browserHarness(snapshot);
  await browser.tick(); await browser.tick();
  const timeline = browser.elements.get("timeline");
  assert.equal(timeline.children.length, 1);
  assert.ok(timeline.textContent.includes(snapshot.view.timeline[0].body));
  assert.match(timeline.textContent, /Time not recorded/);
  assert.match(timeline.textContent, /no AOS delivery receipt/);
  assert.doesNotMatch(timeline.textContent, /Delivery confirmed|no released claims|Delivery attempted/);
  assert.match(browser.elements.get("run-labels").textContent, /Recorded replay.*Native PiSAs/);
  const gathered = browser.elements.get("native-gathering");
  assert.equal(gathered.hidden, false);
  assert.ok(gathered.textContent.includes(snapshot.view.native_gathering.text));
  assert.equal(timeline.children.includes(gathered), false);
  browser.click("play"); assert.equal(gathered.hidden, true);
  browser.click("show-all"); assert.equal(gathered.hidden, false);
  browser.click("new-session"); assert.equal(gathered.hidden, true);
  assert.equal(JSON.stringify(snapshot), before);
});

// A shared wire protocol does not authorize relabelling the new integrated treatment as governed v1.
test("integrated treatment has an explicit display label", async () => {
  const snapshot = example(); snapshot.metadata.version = "integrated-v3";
  const browser = browserHarness(snapshot);
  await browser.tick(); await browser.tick();
  assert.match(browser.elements.get("run-labels").textContent, /Integrated v3/);
  assert.doesNotMatch(browser.elements.get("run-labels").textContent, /Governed v1/);
});

// The serializer profile appears beside either component version throughout the real rendered record.
test("response contract is visible in history, labels and record details", async () => {
  for (const [version, label] of [["governed-v1", "Governed v1"], ["integrated-v3", "Integrated v3"]]) {
    const snapshot = example();
    snapshot.metadata.version = version;
    snapshot.metadata.response_contract_id = "aos-response-contract-v1";
    const before = JSON.stringify(snapshot), browser = browserHarness(snapshot);
    await browser.tick(); await browser.tick();
    for (const id of ["history", "run-labels", "record-details"]) {
      assert.ok(browser.elements.get(id).textContent.includes(label + " + response contract v1"), id);
    }
    assert.equal(snapshot.metadata.version, version);
    assert.equal(JSON.stringify(snapshot), before);
  }
});

// Missing, unknown, malformed or inapplicable profile metadata cannot retroactively tag another treatment.
test("old records and unsupported response profiles retain their original labels", async () => {
  for (const [version, profile] of [["governed-v1", undefined], ["integrated-v3", undefined],
    ["integrated-v3", "aos-response-contract-v99"], ["integrated-v3", {id: "aos-response-contract-v1"}],
    ["native-decentralized", "aos-response-contract-v1"], ["observed-v2", "aos-response-contract-v1"]]) {
    const snapshot = example(); snapshot.metadata.version = version;
    if (profile !== undefined) snapshot.metadata.response_contract_id = profile;
    const browser = browserHarness(snapshot);
    await browser.tick(); await browser.tick();
    for (const id of ["history", "run-labels", "record-details"]) {
      assert.doesNotMatch(browser.elements.get(id).textContent, /response contract v1/);
    }
  }
});

// A configured serializer label reports treatment identity and never supplies an accepted outcome.
test("response contract label does not promote a failed generation", async () => {
  const snapshot = example();
  Object.assign(snapshot.metadata, {version: "integrated-v3", response_contract_id: "aos-response-contract-v1",
    status: "failed", generation_stage_status: "failed", generation_error_type: "ModelFailure"});
  const browser = browserHarness(snapshot);
  await browser.tick(); await browser.tick();
  assert.match(browser.elements.get("run-labels").textContent, /Integrated v3 \+ response contract v1/);
  const outcome = browser.elements.get("conversation-end-summary").textContent;
  assert.match(outcome, /Execution failed/);
  assert.match(outcome, /not an accepted final answer/);
  assert.doesNotMatch(outcome, /Recorded outcome/);
  assert.match(browser.elements.get("record-details").textContent, /Recorded errorModelFailure/);
});

// Recorded cohort, actor model, and status remain searchable without changing the source records.
test("history search includes cohort, model, and status", async () => {
  const snapshot = example();
  snapshot.metadata.cohort = "external-baseline";
  snapshot.metadata.model = "test-provider/model";
  const browser = browserHarness(snapshot);
  await browser.tick(); await browser.tick();
  const search = browser.elements.get("search"), history = browser.elements.get("history");
  for (const query of ["external-baseline", "test-provider/model", "completed"]) {
    search.value = query;
    search.listeners.input();
    assert.equal(history.children[0].tagName, "button");
  }
  search.value = "missing-cohort";
  search.listeners.input();
  assert.match(history.textContent, /No matching conversations/);
});

// A failed saved judgment remains visible without being mistaken for a completed evaluation.
test("a recorded evaluation failure shows its error and no-judge disclaimer", async () => {
  const snapshot = example();
  snapshot.evaluation = {status: "failed", error_type: "ProviderError",
    message: "The recorded judge attempts failed; no official evaluation was completed for this conversation."};
  const browser = browserHarness(snapshot);
  await browser.tick();
  await browser.tick();
  assert.match(browser.elements.get("evaluation-status").textContent, /recorded failure/);
  const rendered = browser.elements.get("evaluation-body").textContent;
  assert.match(rendered, /ProviderError/);
  assert.match(rendered, /recorded judge attempts failed/);
  assert.match(rendered, /No official evaluation is attached\. Replay and live collaboration do not invoke a judge\./);
});
