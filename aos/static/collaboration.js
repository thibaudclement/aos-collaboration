// AOS collaboration (browser state uses only filtered local records).
(function (root) {
  "use strict";

  // Poll serially, discarding responses from older selections and preserving the last good view.
  function createPoller(options) {
    var selection = null, revision = 0, active = false, busy = false, timer = null;
    var later = options.setTimer || setTimeout, cancel = options.clearTimer || clearTimeout;
    // Schedule at most one future read.
    function schedule(delay) {
      if (timer !== null) cancel(timer);
      timer = later(tick, delay);
    }
    // Complete one read before scheduling another (stale results never replace a new selection).
    async function tick() {
      timer = null;
      if (!active || busy) return;
      busy = true;
      var mine = revision, selected = selection;
      try {
        var data = await options.read(selected);
        if (active && mine === revision) options.onData(data, selected);
      } catch (error) {
        if (active && mine === revision && options.onError) options.onError(error);
      } finally {
        busy = false;
        if (active) schedule(mine === revision ? (options.interval || 1500) : 0);
      }
    }
    return {
      // Start local-file observation (including periods with no active conversation).
      start: function () { if (!active) { active = true; schedule(0); } },
      // Change selection without launching an overlapping read.
      select: function (id) { selection = id; revision += 1; if (active && !busy) schedule(0); },
      // Refresh read-only records without submitting a task.
      refresh: function () { revision += 1; if (active && !busy) schedule(0); },
      // Ignore outstanding reads after the observer is stopped.
      stop: function () { active = false; revision += 1; if (timer !== null) cancel(timer); timer = null; }
    };
  }

  // Preserve one launch identity across an uncertain receipt (never retrying a POST automatically).
  function createLauncher(options) {
    var busy = false, pending = options.restored || null;
    return {
      // Expose only enough submission state for the controls to prevent duplicate starts.
      state: function () { return {busy: busy, pending: pending}; },
      // Submit once; a deliberate recheck retains the original payload and launch identifier.
      submit: async function (payload) {
        if (busy) return null;
        busy = true;
        if (!pending) {
          pending = {case_id: payload.case_id, version: payload.version, launch_id: options.newId()};
          if (options.save) options.save(pending);
        }
        try {
          var result = await options.post(pending);
          if (!result || typeof result.run_id !== "string" || !result.run_id ||
              typeof result.status !== "string" || typeof result.duplicate !== "boolean") {
            var malformed = new Error("The server returned an invalid launch receipt.");
            malformed.outcome_unknown = true;
            throw malformed;
          }
          pending = null;
          if (options.save) options.save(null);
          return result;
        } catch (error) {
          if (error.outcome_unknown === false) {
            pending = null;
            if (options.save) options.save(null);
          }
          throw error;
        } finally {
          busy = false;
        }
      }
    };
  }

  // Keep replay position independent of both live execution and recorded timestamps.
  function createPlayback() {
    var position = null, playing = false;
    return {
      // Start from the first recorded event without changing the underlying record.
      play: function () { position = 0; playing = true; },
      // Pause animation at the current visible event count.
      pause: function () { playing = false; },
      // Advance one recorded event (reaching the end stops playback).
      step: function (length) {
        if (position === null || position >= length) position = 0;
        position = Math.min(length, position + 1);
        if (position >= length) playing = false;
        return position;
      },
      // Restore the complete snapshot for ordinary inspection.
      all: function () { position = null; playing = false; },
      // Return the display count without inventing events.
      state: function (length) { return {count: position === null ? length : Math.min(position, length), playing: playing}; }
    };
  }

  // Derive a terminal display summary without adding or changing recorded peer events.
  function conversationSummary(metadata, view, visibleCount) {
    if (!metadata || !view || !Array.isArray(view.timeline) ||
        visibleCount !== view.timeline.length ||
        !["completed", "completed_with_warning", "failed", "interrupted"].includes(metadata.status)) return null;
    var answer = view.answer || {}, text = typeof answer.text === "string" ? answer.text : "";
    var accepted = ["completed", "completed_with_warning"].includes(metadata.status) &&
      view.snapshot_status === "completed" && answer.state === "completed" && !!text.trim();
    var heading = accepted ? "Recorded outcome" : metadata.status === "failed" ? "Execution failed" :
      metadata.status === "interrupted" ? "Execution interrupted" : "No accepted outcome";
    var note = "Run summary · not a peer message.";
    if (accepted && answer.insufficient_evidence) note += " The run reported insufficient evidence.";
    if (!accepted && text) note += " This text was recorded before termination; it is not an accepted final answer.";
    return {heading: heading, note: note, accepted: accepted,
      text: text || "No accepted final answer was recorded.",
      refs: text && Array.isArray(answer.used_evidence_refs) ? answer.used_evidence_refs.slice() : []};
  }

  // Export pure state controls for dependency-free Node regression tests.
  var exported = {createPoller: createPoller, createLauncher: createLauncher, createPlayback: createPlayback,
    conversationSummary: conversationSummary};
  if (typeof module !== "undefined" && module.exports) module.exports = exported;
  if (!root.document) return;

  var doc = root.document;
  var selected = null, current = null, catalog = null, runList = [], activeRun = null;
  var historySignature = "", catalogueSignature = "", snapshotSignature = "", playbackTimer = null;
  var playback = createPlayback();
  var startStorageKey = "aos-collaboration-launch:" + root.location.origin;
  var selectStorageKey = "aos-collaboration-selected:" + root.location.origin;

  // Look up a fixed UI element by ID.
  function byId(id) { return doc.getElementById(id); }
  // Create text nodes rather than interpolating agent text into HTML.
  function node(tag, className, text) {
    var value = doc.createElement(tag);
    if (className) value.className = className;
    if (text !== undefined && text !== null) value.textContent = String(text);
    return value;
  }
  // Display status text with a consistent severity.
  function notice(id, text, severity) {
    var target = byId(id);
    target.textContent = text || "";
    target.className = "notice " + (severity || "");
    target.hidden = !text;
  }
  // Read small persisted browser state without relying on storage being available.
  function storageRead(store, key) {
    try { return store.getItem(key); } catch (_) { return null; }
  }
  // Persist only local display or launch IDs, never owner credentials or model inputs.
  function storageWrite(store, key, value) {
    try { if (value === null) store.removeItem(key); else store.setItem(key, value); } catch (_) {}
  }
  // Fetch a same-origin JSON endpoint and retain the server's receipt certainty.
  async function request(path, payload) {
    var options = {credentials: "same-origin", cache: "no-store"};
    if (payload) {
      options.method = "POST";
      options.headers = {"Content-Type": "application/json"};
      options.body = JSON.stringify(payload);
    }
    var response, value;
    try {
      response = await root.fetch(path, options);
      value = await response.json();
    } catch (_) {
      var uncertain = new Error(payload ? "The launch receipt could not be read. Check the conversation list." : "The local server could not be reached. Retrying the read.");
      uncertain.outcome_unknown = !!payload;
      throw uncertain;
    }
    if (!response.ok) {
      var error = new Error(value.error || "The local request could not be completed.");
      error.outcome_unknown = value.outcome_unknown !== false;
      throw error;
    }
    return value;
  }
  // Return a readable timestamp while preserving invalid historic values as text.
  function stamp(value) {
    if (!value) return "Time not recorded";
    var date = new Date(value);
    return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleString();
  }
  // Name execution states without treating a model answer as proof of correctness.
  function statusText(value) {
    return ({completed: "Completed", running: "Running", queued: "Queued", preparing: "Preparing",
      failed: "Execution failed", interrupted: "Interrupted", unknown: "Status uncertain",
      completed_with_warning: "Completed with warning"})[value] || String(value || "Not started");
  }
  // Map execution status to visual emphasis.
  function statusClass(value) {
    if (value === "completed") return "good";
    if (["running", "queued", "preparing"].indexOf(value) >= 0) return "busy";
    if (value === "failed") return "bad";
    return "warning";
  }
  // Display the tested baseline and observed decision version as distinct treatments.
  function versionName(value) {
    return value === "observed-v2" ? "Observed v2" : value === "governed-v1" ? "Governed v1" :
      value === "integrated-v3" ? "Integrated v3" : value === "native-decentralized" ? "Native PiSAs · decentralized" :
      String(value || "Version not recorded");
  }
  // Show the known serializer separately from the unchanged owner-component version.
  function systemName(metadata) {
    var label = versionName(metadata.version);
    if (["governed-v1", "integrated-v3"].includes(metadata.version) &&
        metadata.response_contract_id === "aos-response-contract-v1") label += " + response contract v1";
    return label;
  }
  // Treat both faithful import schemas as replays without implying that native messages were AOS releases.
  function isReplay(metadata) {
    return ["recorded_replay", "native_replay"].includes(metadata.mode);
  }
  // Find a participant's public name from this selected record only.
  function person(view, id) {
    var entry = (view.participants || []).find(function (p) { return p.id === id; });
    return entry ? entry.name : id;
  }
  // Add a compact label without relying on model-produced CSS or HTML.
  function tag(parent, text, accent) { parent.appendChild(node("span", "tag" + (accent ? " accent" : ""), text)); }
  // Expand evidence references without exposing the underlying owner workspace.
  function refs(parent, values, heading) {
    if (!values || !values.length) return;
    var details = node("details"), list = node("ul");
    details.dataset.evidenceKey = values.join("|");
    details.appendChild(node("summary", "", heading || "Evidence references"));
    values.forEach(function (ref) { list.appendChild(node("li", "mono", ref)); });
    details.appendChild(list); parent.appendChild(details);
  }
  // Render one actual released claim and its references.
  function claimCard(claim) {
    var card = node("div", "claim");
    if (claim.claim_id) card.appendChild(node("div", "claim-id", claim.claim_id));
    card.appendChild(node("p", "", claim.text));
    refs(card, claim.evidence_refs);
    return card;
  }

  // Restore a pending start receipt without repeating its request automatically.
  var restored = null;
  try {
    restored = JSON.parse(storageRead(root.sessionStorage, startStorageKey) || "null");
    if (!restored || typeof restored.case_id !== "string" || typeof restored.version !== "string" || typeof restored.launch_id !== "string") restored = null;
  } catch (_) { restored = null; }
  var launcher = createLauncher({
    restored: restored,
    // Generate a browser-local idempotency identity for one deliberate user start.
    newId: function () { return "launch-" + root.crypto.randomUUID(); },
    // Persist uncertainty across a page refresh so the next click checks the same start.
    save: function (value) { storageWrite(root.sessionStorage, startStorageKey, value ? JSON.stringify(value) : null); },
    // Submit exactly one same-origin launch request.
    post: function (payload) { return request("/api/collaboration/start", payload); }
  });

  // Keep the catalogue stable while the user is choosing a scenario.
  function renderCatalogue() {
    if (!catalog) return;
    var signature = JSON.stringify(catalog.cases || []);
    if (signature !== catalogueSignature) {
      catalogueSignature = signature;
      var old = byId("case").value, select = byId("case");
      select.replaceChildren();
      (catalog.cases || []).forEach(function (item) {
        var option = node("option", "", item.title || item.case_id);
        option.value = item.case_id;
        select.appendChild(option);
      });
      if (!select.options.length) {
        var placeholder = node("option", "", "No scenarios registered");
        placeholder.value = ""; select.appendChild(placeholder);
      }
      if (Array.from(select.options).some(function (item) { return item.value === old; })) select.value = old;
    }
    var state = launcher.state();
    if (state.pending) {
      byId("case").value = state.pending.case_id;
      byId("version").value = state.pending.version;
    }
    updateStart();
  }
  // Explain spending and availability before any live model call.
  function updateStart() {
    var state = launcher.state();
    var item = catalog && (catalog.cases || []).find(function (c) { return c.case_id === byId("case").value; });
    var supported = !item || item.kind !== "scripted" || byId("version").value !== "integrated-v3";
    var allowed = item && supported && (item.kind === "scripted" || item.live_enabled);
    byId("start").disabled = state.busy || (!state.pending && (!allowed || !!activeRun));
    byId("case").disabled = state.busy || !!state.pending;
    byId("version").disabled = state.busy || !!state.pending;
    byId("start").textContent = state.busy ? "Waiting for receipt…" : state.pending ? "Check previous start" : "Start collaboration →";
    var text;
    if (state.pending) text = "A start receipt is pending. Checking again reuses that request; no automatic retry is made.";
    else if (activeRun) text = "A collaboration is running. Open it in the sidebar; one live run is allowed at a time.";
    else if (!item) text = "Choose a PiSAs scenario to start a collaboration.";
    else if (!supported) text = "Integrated v3 requires a PiSAs scenario.";
    else if (item.kind === "scripted") text = "Authored scripted example · no model or official judge calls. This tests the mechanism, not benchmark performance.";
    else if (!item.live_enabled) text = "Live model execution is disabled for this session. " +
      (catalog.live_disabled_reason || "Add API_KEY to .env and restart to enable new collaborations.") +
      " Saved conversations remain available for replay.";
    else text = "Uses " + catalog.model + " for actor calls. No official evaluator is launched. Each start creates a new conversation.";
    byId("launch-hint").textContent = text;
  }
  // Render persistent conversations without changing their recorded order or version.
  function renderHistory() {
    var query = byId("search").value.trim().toLowerCase();
    var signature = JSON.stringify([runList, selected, query]);
    if (signature === historySignature) return;
    historySignature = signature;
    var target = byId("history"); target.replaceChildren();
    var filtered = runList.filter(function (item) {
      return [item.title, item.case_id, item.version, systemName(item), item.run_id,
        item.cohort, item.model, item.status].join(" ").toLowerCase().includes(query);
    });
    filtered.forEach(function (item) {
      var button = node("button", "history-row" + (item.run_id === selected ? " selected" : ""));
      button.type = "button";
      button.setAttribute("aria-current", item.run_id === selected ? "page" : "false");
      button.appendChild(node("span", "history-title", item.title || item.case_id));
      var meta = node("span", "history-meta");
      meta.appendChild(node("span", "state-dot " + statusClass(item.status)));
      meta.appendChild(node("span", "", statusText(item.status)));
      meta.appendChild(node("span", "", "· " + systemName(item)));
      var detail = node("span", "history-meta", (isReplay(item) ? "Imported replay" : "Session") + " · " + stamp(item.created));
      button.appendChild(meta); button.appendChild(detail);
      button.addEventListener("click", function () { choose(item.run_id); });
      target.appendChild(button);
    });
    if (!filtered.length) target.appendChild(node("p", "history-empty", query ? "No matching conversations." : "No saved conversations yet."));
  }
  // Switch conversation identity before accepting another asynchronous snapshot.
  function choose(id) {
    selected = id; current = null; snapshotSignature = ""; playback.all();
    if (playbackTimer !== null) root.clearTimeout(playbackTimer);
    playbackTimer = null;
    storageWrite(root.localStorage, selectStorageKey, id);
    renderHistory(); renderConversation(); poller.select(id);
    if (id) byId("launch").classList.add("compact");
  }
  // Render exact canonical message text and associated released evidence.
  function renderTimeline(view, count) {
    var events = view.timeline || [], target = byId("timeline");
    var opened = new Set(Array.from(target.querySelectorAll("details[open][data-evidence-key]")).map(function (item) { return item.dataset.evidenceKey; }));
    target.replaceChildren();
    events.slice(0, count).forEach(function (event) {
      var card = node("li", "event " + event.kind);
      var head = node("div", "event-head");
      head.appendChild(node("span", "event-route", person(view, event.sender_id) + " → " + person(view, event.recipient_id)));
      head.appendChild(node("span", "event-kind", event.kind === "native_message" ? "Native message" : event.kind));
      var time = node("time", "", stamp(event.timestamp));
      if (event.timestamp) time.dateTime = event.timestamp;
      head.appendChild(time); card.appendChild(head);
      if (event.kind === "request" || event.kind === "native_message") card.appendChild(node("p", "event-text", event.body));
      else if (event.response) {
        if (!event.response.claims.length) card.appendChild(node("p", "event-text muted", "This response contained no released claims."));
        event.response.claims.forEach(function (claim) {
          var known = (view.evidence || []).find(function (item) {
            return item.owner_id === event.sender_id && item.request_id === event.request_id && item.text === claim.text &&
                   JSON.stringify(item.evidence_refs) === JSON.stringify(claim.evidence_refs);
          });
          card.appendChild(claimCard(known || claim));
        });
      } else card.appendChild(node("p", "event-text muted", event.visibility_note || "Delivery is unconfirmed; response contents are not displayed as received evidence."));
      var footer = node("div", "event-footer");
      footer.appendChild(node("span", event.kind === "native_message" ? "muted" : event.delivery === "confirmed" ? "good" : "warning",
        event.kind === "native_message" ? "Recorded native message · no AOS delivery receipt" :
        event.delivery === "confirmed" ? "Delivery confirmed" : "Delivery attempted"));
      if (event.request_id) footer.appendChild(node("span", "mono", event.request_id));
      else if (event.kind === "native_message") footer.appendChild(node("span", "mono", "Source log #" + (event.source_log_index + 1)));
      card.appendChild(footer); target.appendChild(card);
    });
    target.querySelectorAll("details[data-evidence-key]").forEach(function (item) { item.open = opened.has(item.dataset.evidenceKey); });
    byId("timeline-empty").hidden = count > 0;
    byId("timeline-empty").textContent = events.length && count === 0 ? "Replay is at the beginning." : "No peer messages have been recorded yet.";
    byId("event-count").textContent = events.length + " recorded peer events";
    byId("playback-count").textContent = count + " / " + events.length;
  }
  // Preserve local evidence as a separate section rather than inventing another peer transmission.
  function renderLocal(view) {
    var target = byId("local-body"), local = view.local_evidence || [];
    target.replaceChildren(); byId("local-evidence").hidden = !local.length;
    var count = 0;
    local.forEach(function (item) {
      var claims = item.response ? item.response.claims : [];
      claims.forEach(function (claim) { target.appendChild(claimCard(claim)); count += 1; });
    });
    byId("local-count").textContent = "· " + count + " released claims";
    if (!count) target.appendChild(node("p", "small muted", "No local claims were released."));
  }
  // Keep the native gathered summary separate from peer messages and AOS released-claim records.
  function renderNativeGathering(view, ready) {
    var target = byId("native-gathering"), value = view && view.native_gathering;
    if (!target && !value) return;
    if (!target) {
      target = node("section", "result-card notice"); target.id = "native-gathering";
      byId("local-evidence").insertAdjacentElement("afterend", target);
    }
    target.replaceChildren(); target.hidden = !value || !ready;
    if (target.hidden) return;
    target.appendChild(node("h2", "", "Native gathered summary"));
    target.appendChild(node("p", "small muted", value.note));
    target.appendChild(node("p", "preserve", value.text));
  }
  // Close the visible transcript with the saved outcome, outside the canonical peer-event list.
  function renderEndSummary(metadata, view, visibleCount) {
    var target = byId("conversation-end-summary"), summary = conversationSummary(metadata, view, visibleCount);
    if (!target) {
      target = node("section", "result-card notice");
      target.id = "conversation-end-summary";
      target.setAttribute("aria-labelledby", "conversation-end-heading");
      byId("local-evidence").insertAdjacentElement("afterend", target);
    }
    target.hidden = !summary;
    target.replaceChildren();
    if (!summary) return;
    var heading = node("h2", "", summary.heading);
    heading.id = "conversation-end-heading";
    target.appendChild(heading);
    target.appendChild(node("p", "small muted", summary.note));
    target.appendChild(node("p", "preserve", summary.text));
    refs(target, summary.refs, "Evidence used");
  }
  // Render a model's explicit conflict assessment without promoting it to verified truth.
  function renderDisagreements(view, ready) {
    var target = byId("disagreements"); target.replaceChildren();
    var record = view.conflict_detection || {status: "not_recorded"};
    if (!ready) {
      byId("disagreement-note").textContent = "The recorded assessment appears after the replay reaches the end.";
      return;
    }
    if (["not_started", "started", "failed"].includes(record.status)) {
      byId("disagreement-note").textContent = record.status === "started" ? "The decision stage is assessing the released evidence." : record.status === "failed" ? "The disagreement assessment did not complete. No resolution is inferred." : "The decision assessment has not started.";
      return;
    }
    if (record.status !== "recorded") {
      byId("disagreement-note").textContent = (view.replay_kind === "pisas_decentralized" ? record.note + " " : "") +
        "No explicit disagreement assessment was recorded in this run. This does not mean that the evidence was conflict-free.";
      return;
    }
    byId("disagreement-note").textContent = record.decision_note || record.note || "Recorded assessment based on the requester’s released evidence.";
    var disagreements = record.disagreements || [];
    if (!disagreements.length) target.appendChild(node("p", "small muted", "The recorded assessment found no disagreement. This is not an independent verification."));
    disagreements.forEach(function (item) {
      var card = node("div", "disagreement-item");
      card.appendChild(node("span", "badge " + (item.action === "resolved" ? "good" : "warning"), String(item.action || "recorded").replaceAll("_", " ")));
      card.appendChild(node("h3", "", String(item.kind || "disagreement").replaceAll("_", " ")));
      if (item.scope) card.appendChild(node("p", "small muted", item.scope));
      (item.claim_ids || []).forEach(function (id) {
        var claim = (view.evidence || []).find(function (entry) { return entry.claim_id === id; });
        if (claim) {
          var rendered = claimCard(claim);
          if ((item.selected_claim_ids || []).includes(id)) rendered.prepend(node("div", "small good", "Retained for this decision"));
          card.appendChild(rendered);
        } else card.appendChild(node("p", "small mono", id));
      });
      card.appendChild(node("p", "", item.reason || ""));
      card.appendChild(node("p", "small muted", item.material ? "Could affect the requested decision." : "Recorded as nonblocking for this decision."));
      target.appendChild(card);
    });
  }
  // Display saved official metrics outside the actors' inputs, with their explicit attachment provenance.
  function renderEvaluation(evaluation) {
    var value = evaluation || {status: "not_loaded"}, target = byId("evaluation-body");
    target.replaceChildren();
    byId("evaluation-status").textContent = value.status === "attached" ? "· attached" :
      value.status === "failed" ? "· recorded failure" : "· not attached";
    if (value.status !== "attached") {
      if (value.status === "failed") {
        target.appendChild(node("p", "small warning", value.message || "The recorded evaluation failed."));
        if (value.error_type) target.appendChild(node("p", "explanation warning", "Recorded error: " + value.error_type));
      }
      target.appendChild(node("p", "small muted", "No official evaluation is attached. Replay and live collaboration do not invoke a judge."));
      return;
    }
    target.appendChild(node("p", "small muted", value.display_note || "Saved official judgment; not a new evaluation of this playback."));
    if (value.judge) target.appendChild(node("p", "explanation", "Judge: " + value.judge + (value.judge_mode ? " · " + value.judge_mode : "")));
    var table = node("table");
    (value.rows || []).forEach(function (row) {
      var tr = node("tr"); tr.appendChild(node("td", "", row.label)); tr.appendChild(node("td", "", row.value)); table.appendChild(tr);
    });
    target.appendChild(table);
    var verification = value.verification || {};
    target.appendChild(node("p", "explanation " + (verification.degraded ? "warning" : "muted"),
      "Verification: " + (verification.degraded ? "degraded" : "not marked degraded") +
      " · missing votes: " + (verification.votes_missing === undefined ? "not recorded" : verification.votes_missing)));
    if (value.binding_note) target.appendChild(node("p", "small muted", value.binding_note));
    var flagged = node("details"); flagged.appendChild(node("summary", "", "Flagged attribute IDs by surface"));
    var list = node("ul");
    Object.keys(value.metrics || {}).forEach(function (surface) {
      var metric = value.metrics[surface];
      if (metric && Array.isArray(metric.flagged)) list.appendChild(node("li", "small mono", surface + ": " + (metric.flagged.join(", ") || "none")));
    });
    flagged.appendChild(list); target.appendChild(flagged);
  }
  // Describe the record itself separately from claims about the correctness of its answer.
  function recordDetails(metadata, view) {
    var target = byId("record-details"); target.replaceChildren();
    var items = [
      ["Conversation", metadata.run_id],
      ["System", systemName(metadata)],
      ["Model", metadata.model || "not recorded"],
      ["Collection", metadata.cohort || "local collaboration"],
      ["Run index", metadata.run_index === undefined ? "not recorded" : metadata.run_index +
        (metadata.run_index_origin === "operator_selected" ? " · selected by importing operator" : "")],
      ["Mode", metadata.mode],
      ["Saved", stamp(metadata.created)],
      ["Date source", metadata.date_origin || "local record"],
      ["Run started", stamp(view.recorded_started)],
      ["Run finished", stamp(view.recorded_finished)],
      ["Pipeline export saved", stamp(view.recorded_exported)],
      ["Original duration", typeof view.recorded_seconds === "number" ? Math.round(view.recorded_seconds) + " seconds" : "not recorded"],
      ["Source hash", view.source_sha256 || "not available"]
    ];
    if (metadata.generation_error_type) items.push(["Recorded error", metadata.generation_error_type]);
    items.forEach(function (pair) { target.appendChild(node("dt", "", pair[0])); target.appendChild(node("dd", "", pair[1])); });
  }
  // Display the benchmark reference independently of recorded answers and replay progress.
  function renderBenchmark(reference) {
    var available = reference && reference.status === "available" && reference.answer != null;
    byId("benchmark-answer").textContent = available ?
      (typeof reference.answer === "string" ? reference.answer : JSON.stringify(reference.answer, null, 2)) :
      "No benchmark expected outcome is available for this record.";
    byId("benchmark-note").textContent = available ?
      "Reference for the bundled PiSAs scenario. Not supplied to the agents." : "";
  }
  // Update the selected conversation without reading private artifacts or changing the run.
  function renderConversation() {
    renderBenchmark(selected && current ? current.benchmark_reference : null);
    byId("empty").hidden = !!selected;
    byId("conversation").hidden = !selected;
    if (!selected) { renderEndSummary(null, null, 0); renderNativeGathering(null, false); return; }
    if (!current) {
      renderEndSummary(null, null, 0);
      renderNativeGathering(null, false);
      byId("run-title").textContent = "Loading conversation…";
      byId("run-subtitle").textContent = "Reading saved local records.";
      byId("run-status").textContent = "Loading"; byId("run-status").className = "badge";
      byId("event-count").textContent = ""; byId("playback-note").textContent = "";
      byId("answer-status").textContent = ""; byId("answer-refs").hidden = true;
      byId("record-details").replaceChildren();
      byId("run-labels").replaceChildren(); byId("participants").replaceChildren();
      byId("task").textContent = ""; byId("timeline").replaceChildren();
      byId("timeline-empty").hidden = false; byId("answer").textContent = "";
      byId("disagreements").replaceChildren(); byId("disagreement-note").textContent = "";
      byId("playback").hidden = true; byId("local-evidence").hidden = true;
      byId("progress").hidden = true; renderEvaluation(null);
      return;
    }
    var renderState = JSON.stringify([selected, current, playback.state(current.view ? current.view.timeline.length : 0)]);
    if ((current.warnings || []).length) notice("view-error", current.warnings.join("\n"), "warning");
    if (renderState === snapshotSignature) return;
    snapshotSignature = renderState;
    var metadata = current.metadata, view = current.view;
    var labels = byId("run-labels"); labels.replaceChildren();
    tag(labels, isReplay(metadata) ? "Recorded replay" : "Live session", true);
    tag(labels, systemName(metadata));
    if (metadata.backend === "scripted" || (view && view.source_kind === "synthetic")) tag(labels, "Authored example");
    byId("run-title").textContent = metadata.title || metadata.case_id;
    byId("run-subtitle").textContent = metadata.case_id + " · " + (metadata.model || "model not recorded") + " · " + stamp(metadata.created);
    byId("run-status").textContent = statusText(metadata.status);
    byId("run-status").className = "badge " + statusClass(metadata.status);
    var ongoing = ["queued", "preparing", "running"].includes(metadata.status);
    var progress = current.progress || {};
    byId("progress").hidden = !ongoing;
    byId("progress").textContent = progress.message || "Waiting for the next persisted execution record…";
    if (!view) {
      renderEndSummary(null, null, 0);
      renderNativeGathering(null, false);
      byId("task").textContent = ongoing ? "The collaboration is preparing its isolated owner workspaces." : "No validated requester view was recorded for this attempt.";
      byId("participants").replaceChildren(); byId("record-details").replaceChildren();
      byId("event-count").textContent = ""; byId("playback-note").textContent = "";
      byId("answer-refs").hidden = true; byId("disagreements").replaceChildren();
      byId("timeline").replaceChildren(); byId("timeline-empty").hidden = false;
      byId("timeline-empty").textContent = "No peer messages have been recorded yet.";
      byId("answer").textContent = ongoing ? "The final answer will appear here." : "No accepted answer was recorded.";
      byId("answer-status").textContent = "";
      byId("playback").hidden = true; byId("local-evidence").hidden = true;
      byId("disagreement-note").textContent = "No assessment is available.";
      renderEvaluation(current.evaluation);
      return;
    }
    byId("task").textContent = view.scenario.task;
    var people = byId("participants"); people.replaceChildren();
    (view.participants || []).forEach(function (p) {
      var chip = node("span", "person" + (p.id === view.executor_id ? " executor" : ""));
      chip.appendChild(node("span", "initial", p.id.replace("agent-", "")));
      chip.appendChild(node("span", "", p.name + (p.id === view.executor_id ? " · requester" : "")));
      people.appendChild(chip);
    });
    var events = view.timeline || [], state = playback.state(events.length);
    var ready = state.count === events.length;
    byId("playback").hidden = ongoing || !events.length;
    byId("play").textContent = state.playing ? "Pause replay" : state.count < events.length ? "Replay from start" : "Replay from start";
    byId("playback-note").textContent = ongoing ? "Live view · refreshes persisted records automatically." :
      "Playback reveals one recorded event at a time. The original timestamps remain visible; playback speed is not execution time.";
    if (view.replay_kind === "pisas_decentralized") byId("playback-note").textContent = view.source_note;
    renderTimeline(view, state.count); renderLocal(view);
    renderNativeGathering(view, ready);
    renderEndSummary(metadata, view, state.count);
    var answer = view.answer || {};
    byId("answer-status").textContent = ready ? (answer.insufficient_evidence ? "Evidence insufficient" : answer.state === "not_available" ? "Not available" : "Recorded answer") : "Replay in progress";
    byId("answer").textContent = !ready ? "The recorded answer appears after the final event." :
      answer.text || (ongoing ? "The final answer will appear here." : "No accepted final answer was recorded.");
    var answerRefs = byId("answer-refs"); answerRefs.hidden = !ready || !(answer.used_evidence_refs || []).length;
    var refsList = answerRefs.querySelector("ul"); refsList.replaceChildren();
    (answer.used_evidence_refs || []).forEach(function (ref) { refsList.appendChild(node("li", "mono", ref)); });
    renderDisagreements(view, ready);
    renderEvaluation(current.evaluation);
    recordDetails(metadata, view);
    if ((current.warnings || []).length) notice("view-error", current.warnings.join("\n"), "warning");
  }
  // Advance replay on an explicit visual timer without invoking any server mutation.
  function playbackTick() {
    if (!current || !current.view) return;
    var count = current.view.timeline.length;
    if (!playback.state(count).playing) return;
    playback.step(count); renderConversation();
    if (playback.state(count).playing) playbackTimer = root.setTimeout(playbackTick, Number(byId("speed").value));
  }

  // Read catalogue, sidebar and selection through the filtered same-origin API.
  var poller = createPoller({
    interval: 1500,
    read: async function (id) {
      var results = await Promise.all([
        request("/api/collaboration/catalog"),
        request("/api/collaboration/runs"),
        id ? request("/api/collaboration/run?id=" + encodeURIComponent(id)) : Promise.resolve(null)
      ]);
      return {catalog: results[0], list: results[1], snapshot: results[2]};
    },
    // Accept a snapshot only for the selection that initiated this read.
    onData: function (data, id) {
      catalog = data.catalog;
      runList = data.list.runs || [];
      activeRun = data.list.active_run_id || null;
      if (id === selected && data.snapshot) current = data.snapshot;
      byId("connection").textContent = "Local records · " + new Date().toLocaleTimeString();
      notice("view-error", "", "");
      renderCatalogue(); renderHistory(); renderConversation();
      if (!selected && runList.length && !root.__aosExplicitNew) {
        var saved = storageRead(root.localStorage, selectStorageKey);
        choose(runList.some(function (item) { return item.run_id === saved; }) ? saved : runList[0].run_id);
      }
    },
    // Keep the last good display and continue polling after a read failure.
    onError: function (error) {
      byId("connection").textContent = "Connection interrupted · retrying read";
      notice("view-error", error.message, "warning");
    }
  });

  // Wire explicit user actions (periodic observation never submits model work).
  byId("launch-form").addEventListener("submit", async function (event) {
    event.preventDefault();
    notice("launch-result", "Waiting for the start receipt…", "");
    var task = launcher.submit({case_id: byId("case").value, version: byId("version").value});
    updateStart();
    try {
      var receipt = await task;
      if (receipt) {
        root.__aosExplicitNew = false;
        notice("launch-result", receipt.duplicate ? "The existing collaboration was found." : "Collaboration started. Its record is saved in the sidebar.", "success");
        choose(receipt.run_id);
      }
    } catch (error) {
      notice("launch-result", error.message + (error.outcome_unknown !== false ? " No automatic start retry was made." : ""), "warning");
    } finally {
      updateStart(); poller.refresh();
    }
  });
  byId("case").addEventListener("change", updateStart);
  byId("version").addEventListener("change", updateStart);
  byId("refresh").addEventListener("click", function () { poller.refresh(); });
  byId("search").addEventListener("input", renderHistory);
  byId("new-session").addEventListener("click", function () {
    root.__aosExplicitNew = true; choose(null);
    byId("launch").classList.remove("compact");
    byId("main").scrollIntoView({behavior: "smooth", block: "start"});
    byId("case").focus();
  });
  byId("play").addEventListener("click", function () {
    if (playbackTimer !== null) root.clearTimeout(playbackTimer);
    if (playback.state(current.view.timeline.length).playing) playback.pause();
    else { playback.play(); playbackTimer = root.setTimeout(playbackTick, Number(byId("speed").value)); }
    renderConversation();
  });
  byId("step").addEventListener("click", function () {
    if (playbackTimer !== null) root.clearTimeout(playbackTimer);
    playback.pause(); playback.step(current.view.timeline.length); renderConversation();
  });
  byId("show-all").addEventListener("click", function () {
    if (playbackTimer !== null) root.clearTimeout(playbackTimer);
    playback.all(); renderConversation();
  });
  byId("theme").addEventListener("click", function () {
    var light = doc.documentElement.dataset.theme !== "light";
    doc.documentElement.dataset.theme = light ? "light" : "dark";
    byId("theme").textContent = light ? "Dark theme" : "Light theme";
    storageWrite(root.localStorage, "aos-collaboration-theme", light ? "light" : "dark");
  });
  root.addEventListener("beforeunload", function () { poller.stop(); if (playbackTimer !== null) root.clearTimeout(playbackTimer); });

  // Restore cosmetic preferences and begin read-only observation
  var theme = storageRead(root.localStorage, "aos-collaboration-theme");
  if (theme === "light") { doc.documentElement.dataset.theme = "light"; byId("theme").textContent = "Dark theme"; }
  if (restored) notice("launch-result", "An earlier launch receipt is pending. Inspect history or check the previous start using the same request.", "warning");
  poller.start();
})(typeof window !== "undefined" ? window : globalThis);
