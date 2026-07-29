"use strict";

const STORAGE_KEY = "videoquant-b1-anonymous-review-v1";
const catastropheTags = [
  ["identity_switch", "Identity switch"],
  ["background_jump", "Background jump"],
  ["texture_repetition", "Texture repetition"],
  ["motion_freeze", "Motion freeze"],
  ["color_drift", "Color drift"],
  ["black_or_nan", "Black / invalid frames"],
];
const actionLabels = {
  turn_left: "Turn left",
  turn_right: "Turn right",
  forward: "Forward",
  backward: "Backward",
};

const state = { manifest: null, index: 0, reviewerId: "", answers: {} };
const byId = (id) => document.getElementById(id);

function escapeHtml(value) {
  return String(value).replace(/[&<>'"]/g, (char) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;",
  })[char]);
}

function save() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify({
    manifest_sha256: state.manifest.manifest_sha256,
    reviewer_id: state.reviewerId,
    answers: state.answers,
  }));
}

function loadSaved() {
  const raw = localStorage.getItem(STORAGE_KEY);
  if (!raw) return;
  try {
    const saved = JSON.parse(raw);
    if (saved.manifest_sha256 === state.manifest.manifest_sha256) {
      state.reviewerId = saved.reviewer_id || "";
      state.answers = saved.answers || {};
    }
  } catch (_) {
    localStorage.removeItem(STORAGE_KEY);
  }
}

function answerFor(task) {
  if (!state.answers[task.id]) state.answers[task.id] = { complete: false, sides: {} };
  return state.answers[task.id];
}

function catastropheForm(task, side) {
  const answer = answerFor(task).sides[side] || {};
  const checks = catastropheTags.map(([key, label]) => `
    <label><input type="checkbox" data-field="${key}" ${answer[key] ? "checked" : ""}>${label}</label>
  `).join("");
  return `<div class="form-body" data-side="${side}">
    <fieldset><legend>Observed failures</legend><div class="check-grid">${checks}</div></fieldset>
    <div class="field-grid">
      ${selectField("severity", "Overall severity", ["", "0 - none", "1 - mild", "2 - material", "3 - severe"], answer.severity)}
      ${selectField("catastrophe", "Catastrophic failure?", ["", "no", "yes"], answer.catastrophe)}
      ${selectField("onset", "First visible onset", ["", "none", "early", "middle", "late"], answer.onset)}
      <label class="field">Notes<textarea data-field="notes">${escapeHtml(answer.notes || "")}</textarea></label>
    </div>
  </div>`;
}

function selectField(key, label, options, selected) {
  const rendered = options.map((value) => {
    const actual = value.includes(" - ") ? value.split(" - ")[0] : value;
    return `<option value="${escapeHtml(actual)}" ${String(selected ?? "") === actual ? "selected" : ""}>${escapeHtml(value || "Select")}</option>`;
  }).join("");
  return `<label class="field">${label}<select data-field="${key}">${rendered}</select></label>`;
}

function renderCatastrophe(task) {
  byId("review-root").innerHTML = `<div class="comparison">${["A", "B"].map((side) => `
    <article class="side"><h2>Video ${side}</h2>
      <video controls preload="metadata" src="${task.media[side]}"></video>
      ${catastropheForm(task, side)}
    </article>`).join("")}</div>`;
}

function actionForm(task, side) {
  const sideAnswer = answerFor(task).sides[side] || {};
  const actions = task.actions.map((action) => {
    const answer = (sideAnswer.actions || {})[action] || {};
    return `<fieldset data-action="${action}"><legend>${actionLabels[action]}</legend><div class="field-grid">
      ${selectField("action_executed", "Action executed", ["", "yes", "partial", "no"], answer.action_executed)}
      ${selectField("direction_correct", "Direction correct", ["", "yes", "no", "unclear"], answer.direction_correct)}
      ${selectField("onset", "Response onset", ["", "early", "middle", "late", "never"], answer.onset)}
      ${selectField("catastrophe", "Catastrophic failure?", ["", "no", "yes"], answer.catastrophe)}
      ${selectField("identity_preserved", "Identity preserved", ["", "yes", "no", "unclear"], answer.identity_preserved)}
      ${selectField("background_preserved", "Background preserved", ["", "yes", "no", "unclear"], answer.background_preserved)}
      ${selectField("motion_freeze", "Motion freeze", ["", "no", "yes"], answer.motion_freeze)}
    </div></fieldset>`;
  }).join("");
  const pair = sideAnswer.pair || {};
  return `<div class="form-body" data-side="${side}">${actions}
    <fieldset data-pair="true"><legend>Counterfactual separation</legend><div class="field-grid">
      ${selectField("steering_separation", "Left vs right separation", ["", "0 - absent", "1 - weak", "2 - clear"], pair.steering_separation)}
      ${selectField("longitudinal_separation", "Forward vs backward separation", ["", "0 - absent", "1 - weak", "2 - clear"], pair.longitudinal_separation)}
      <label class="field">Notes<textarea data-field="notes">${escapeHtml(pair.notes || "")}</textarea></label>
    </div></fieldset>
  </div>`;
}

function renderAction(task) {
  byId("review-root").innerHTML = `<div class="comparison">${["A", "B"].map((side) => `
    <article class="side"><h2>Group ${side}</h2><div class="action-grid">
      ${task.actions.map((action) => `<div class="action-video"><h3>${actionLabels[action]}</h3><video controls preload="metadata" src="${task.media[side][action]}"></video></div>`).join("")}
    </div>${actionForm(task, side)}</article>`).join("")}</div>`;
}

function collectCurrent() {
  const task = state.manifest.tasks[state.index];
  if (!task) return;
  const answer = answerFor(task);
  document.querySelectorAll("[data-side]").forEach((sideRoot) => {
    const side = sideRoot.dataset.side;
    if (task.kind === "catastrophe") {
      const values = {};
      sideRoot.querySelectorAll("[data-field]").forEach((input) => {
        values[input.dataset.field] = input.type === "checkbox" ? input.checked : input.value;
      });
      answer.sides[side] = values;
    } else {
      const sideValue = { actions: {}, pair: {} };
      sideRoot.querySelectorAll("[data-action]").forEach((root) => {
        const values = {};
        root.querySelectorAll("[data-field]").forEach((input) => {
          values[input.dataset.field] = input.type === "checkbox" ? input.checked : input.value;
        });
        sideValue.actions[root.dataset.action] = values;
      });
      sideRoot.querySelectorAll("[data-pair] [data-field]").forEach((input) => {
        sideValue.pair[input.dataset.field] = input.value;
      });
      answer.sides[side] = sideValue;
    }
  });
  answer.complete = false;
  save();
}

function missingFields(task, answer) {
  const missing = [];
  for (const side of ["A", "B"]) {
    const sideAnswer = answer.sides[side] || {};
    if (task.kind === "catastrophe") {
      for (const field of ["severity", "catastrophe", "onset"]) if (sideAnswer[field] === undefined || sideAnswer[field] === "") missing.push(`${side}.${field}`);
    } else {
      for (const action of task.actions) {
        const actionAnswer = (sideAnswer.actions || {})[action] || {};
        for (const field of ["action_executed", "direction_correct", "onset", "catastrophe", "identity_preserved", "background_preserved", "motion_freeze"]) if (!actionAnswer[field]) missing.push(`${side}.${action}.${field}`);
      }
      for (const field of ["steering_separation", "longitudinal_separation"]) if ((sideAnswer.pair || {})[field] === undefined || (sideAnswer.pair || {})[field] === "") missing.push(`${side}.${field}`);
    }
  }
  return missing;
}

function render() {
  const task = state.manifest.tasks[state.index];
  if (!task) return;
  byId("task-meta").textContent = `Task ${state.index + 1} of ${state.manifest.task_count}`;
  byId("kind-badge").textContent = task.kind === "catastrophe" ? "VIDEO QUALITY" : "ACTION CONTROL";
  byId("baseline").textContent = task.baseline;
  byId("prompt").textContent = task.prompt ? `Prompt ${task.prompt_index + 1}: ${task.prompt}` : task.scene_label;
  const note = byId("conditioning-note");
  note.hidden = !task.conditioning_frames;
  note.textContent = task.conditioning_frames ? `Judge generated continuation only; ignore the first ${task.conditioning_frames} shared conditioning frames.` : "";
  task.kind === "catastrophe" ? renderCatastrophe(task) : renderAction(task);
  byId("validation").textContent = answerFor(task).complete ? "Saved as complete." : "";
  updateProgress();
}

function updateProgress() {
  const complete = state.manifest.tasks.filter((task) => state.answers[task.id]?.complete).length;
  byId("progress-text").textContent = `${complete} / ${state.manifest.task_count}`;
  byId("progress").max = state.manifest.task_count;
  byId("progress").value = complete;
}

function navigate(delta) {
  collectCurrent();
  state.index = Math.max(0, Math.min(state.manifest.tasks.length - 1, state.index + delta));
  render();
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function nextIncomplete() {
  collectCurrent();
  const tasks = state.manifest.tasks;
  for (let offset = 1; offset <= tasks.length; offset += 1) {
    const index = (state.index + offset) % tasks.length;
    if (!state.answers[tasks[index].id]?.complete) { state.index = index; render(); return; }
  }
}

function markComplete() {
  collectCurrent();
  if (!state.reviewerId.trim()) { byId("validation").textContent = "Reviewer ID is required."; return; }
  const task = state.manifest.tasks[state.index];
  const missing = missingFields(task, answerFor(task));
  if (missing.length) { byId("validation").textContent = `Complete required fields: ${missing.slice(0, 4).join(", ")}${missing.length > 4 ? "..." : ""}`; return; }
  answerFor(task).complete = true;
  save();
  nextIncomplete();
}

function exportJson() {
  collectCurrent();
  const payload = {
    schema_version: 1,
    manifest_sha256: state.manifest.manifest_sha256,
    reviewer_id: state.reviewerId.trim(),
    exported_at: new Date().toISOString(),
    answers: state.answers,
  };
  const blob = new Blob([`${JSON.stringify(payload, null, 2)}\n`], { type: "application/json" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = `b1-review-${state.reviewerId.trim() || "anonymous"}.json`;
  link.click();
  URL.revokeObjectURL(link.href);
}

async function importJson(file) {
  const payload = JSON.parse(await file.text());
  if (payload.manifest_sha256 !== state.manifest.manifest_sha256) throw new Error("Manifest hash does not match this review package.");
  state.reviewerId = payload.reviewer_id || "";
  state.answers = payload.answers || {};
  byId("reviewer-id").value = state.reviewerId;
  save(); render();
}

async function init() {
  state.manifest = await fetch("review_manifest.json", { cache: "no-store" }).then((response) => {
    if (!response.ok) throw new Error(`Manifest load failed: ${response.status}`);
    return response.json();
  });
  loadSaved();
  byId("reviewer-id").value = state.reviewerId;
  byId("reviewer-id").addEventListener("input", (event) => { state.reviewerId = event.target.value; save(); });
  byId("prev").addEventListener("click", () => navigate(-1));
  byId("previous-bottom").addEventListener("click", () => navigate(-1));
  byId("next-incomplete").addEventListener("click", nextIncomplete);
  byId("complete-next").addEventListener("click", markComplete);
  byId("export").addEventListener("click", exportJson);
  byId("import").addEventListener("change", async (event) => {
    try { await importJson(event.target.files[0]); } catch (error) { byId("validation").textContent = error.message; }
  });
  byId("task-filter").addEventListener("change", (event) => {
    collectCurrent();
    const value = event.target.value;
    const index = state.manifest.tasks.findIndex((task) => value === "all" || task.kind === value || (value === "incomplete" && !state.answers[task.id]?.complete));
    if (index >= 0) { state.index = index; render(); }
  });
  document.addEventListener("change", collectCurrent);
  render();
}

init().catch((error) => { byId("review-root").textContent = error.message; });
