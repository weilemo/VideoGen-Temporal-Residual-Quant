"use strict";

const STORAGE_KEY = "videoquant-b1-grouped-review-v2";
const labels = ["A", "B", "C", "D", "E"];
const catastropheTags = [
  ["identity_switch", "身份切换 / Identity switch"],
  ["background_jump", "背景跳变 / Background jump"],
  ["texture_repetition", "纹理重复 / Texture repetition"],
  ["motion_freeze", "运动冻结 / Motion freeze"],
  ["color_drift", "颜色漂移 / Color drift"],
  ["black_or_nan", "黑帧或无效帧 / Black or invalid frames"],
];
const actionLabels = {
  turn_left: "左转 / Turn left",
  turn_right: "右转 / Turn right",
  forward: "前进 / Forward",
  backward: "后退 / Backward",
};
const baselineLabels = {
  "Causal Forcing": "因果强制 / Causal Forcing",
  "LongCat Video": "LongCat 视频 / LongCat Video",
  "HY-WorldPlay": "HY-WorldPlay 世界模型 / World Model",
};
const rubric = {
  catastrophe: `<details class="rubric"><summary>异常定义 / Failure definitions</summary><p>只标记可见异常。轻微清晰度差异不算灾难；身份或场景连续性被根本破坏、黑屏或长期冻结才算 catastrophe。</p></details>`,
  action: `<details class="rubric"><summary>动作判定 / Action definitions</summary><p>通过：方向明确且及时；部分：方向可见但幅度弱或迟到；失败：未响应或方向相反。不确定时不要猜。</p></details>`,
};

const state = {
  manifest: null,
  index: 0,
  reviewerId: "",
  answers: {},
  action: "turn_left",
  speed: 1,
  syncing: false,
};
const byId = (id) => document.getElementById(id);

function assetUrl(path) {
  const url = new URL(path, window.location.href);
  const token = new URL(window.location.href).searchParams.get("token");
  if (token) url.searchParams.set("token", token);
  return url.toString();
}

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
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || "null");
    if (saved?.manifest_sha256 === state.manifest.manifest_sha256) {
      state.reviewerId = saved.reviewer_id || "";
      state.answers = saved.answers || {};
    }
  } catch (_) {
    // Preserve the bad payload for manual recovery instead of deleting it.
  }
}

function answerFor(task) {
  if (!state.answers[task.id]) {
    state.answers[task.id] = { complete: false, verdict: "", items: {} };
  }
  return state.answers[task.id];
}

function itemFor(task, label) {
  const answer = answerFor(task);
  if (!answer.items[label]) answer.items[label] = {};
  return answer.items[label];
}

function selectField(key, label, options, selected) {
  const rendered = options.map(([value, text]) =>
    `<option value="${escapeHtml(value)}" ${String(selected ?? "") === value ? "selected" : ""}>${escapeHtml(text)}</option>`
  ).join("");
  return `<label class="field">${label}<select data-field="${key}">${rendered}</select></label>`;
}

function catastropheDetails(task, label) {
  const item = itemFor(task, label);
  const checks = catastropheTags.map(([key, text]) =>
    `<label><input type="checkbox" data-field="${key}" ${item[key] ? "checked" : ""}>${text}</label>`
  ).join("");
  return `<div class="details ${item.flagged ? "" : "is-hidden"}" data-details>
    <div class="check-grid">${checks}</div>
    <div class="field-grid">
      ${selectField("severity", "严重度 / Severity", [["", "请选择"], ["1", "1 轻微"], ["2", "2 明显"], ["3", "3 严重"]], item.severity)}
      ${selectField("catastrophe", "灾难失败 / Catastrophe", [["", "请选择"], ["no", "否"], ["yes", "是"]], item.catastrophe)}
      ${selectField("onset", "首次出现 / Onset", [["", "请选择"], ["early", "前段"], ["middle", "中段"], ["late", "后段"]], item.onset)}
      <label class="field">备注 / Notes<textarea data-field="notes">${escapeHtml(item.notes || "")}</textarea></label>
    </div>
  </div>`;
}

function renderCatastrophe(task) {
  byId("action-tabs").hidden = true;
  byId("review-root").innerHTML = `<div class="video-panel">${task.labels.map((label) => {
    const item = itemFor(task, label);
    return `<article class="video-card" data-label="${label}">
      <div class="card-title"><h2>${label}</h2><label class="flag-control"><input type="checkbox" data-flag ${item.flagged ? "checked" : ""}>可疑 / Flag</label></div>
      <video controls preload="metadata" src="${escapeHtml(assetUrl(task.media[label]))}"></video>
      ${catastropheDetails(task, label)}
    </article>`;
  }).join("")}</div>`;
}

function actionDetails(item) {
  return `<details class="compact-details"><summary>异常细节 / Details</summary><div class="field-grid">
    ${selectField("catastrophe", "灾难失败", [["no", "否"], ["yes", "是"]], item.catastrophe || "no")}
    ${selectField("identity_preserved", "身份保持", [["yes", "是"], ["no", "否"], ["unclear", "不确定"]], item.identity_preserved || "yes")}
    ${selectField("background_preserved", "背景保持", [["yes", "是"], ["no", "否"], ["unclear", "不确定"]], item.background_preserved || "yes")}
    ${selectField("motion_freeze", "运动冻结", [["no", "否"], ["yes", "是"]], item.motion_freeze || "no")}
  </div></details>`;
}

function renderAction(task) {
  byId("action-tabs").hidden = false;
  byId("action-tabs").innerHTML = task.actions.map((action) =>
    `<button type="button" data-action-tab="${action}" class="${state.action === action ? "active" : ""}">${actionLabels[action]}</button>`
  ).join("");
  byId("review-root").innerHTML = `<div class="video-panel">${task.labels.map((label) => {
    const parent = itemFor(task, label);
    parent.actions ||= {};
    parent.pair ||= {};
    const item = parent.actions[state.action] || {};
    return `<article class="video-card" data-label="${label}" data-action="${state.action}">
      <div class="card-title"><h2>${label}</h2><span>${actionLabels[state.action]}</span></div>
      <video controls preload="metadata" src="${escapeHtml(assetUrl(task.media[label][state.action]))}"></video>
      <div class="field-grid compact">
        ${selectField("rating", "动作结果 / Result", [["", "请选择"], ["pass", "通过"], ["partial", "部分或迟到"], ["fail", "失败"], ["unclear", "不确定"]], item.rating)}
        ${selectField("onset", "响应开始 / Onset", [["", "请选择"], ["early", "前段"], ["middle", "中段"], ["late", "后段"], ["never", "未响应"]], item.onset)}
      </div>
      ${actionDetails(item)}
      <div class="pair-fields ${state.action === task.actions.at(-1) ? "" : "is-hidden"}" data-pair>
        ${selectField("steering_separation", "左右分离", [["", "请选择"], ["0", "0 无"], ["1", "1 弱"], ["2", "2 清楚"]], parent.pair.steering_separation)}
        ${selectField("longitudinal_separation", "前后分离", [["", "请选择"], ["0", "0 无"], ["1", "1 弱"], ["2", "2 清楚"]], parent.pair.longitudinal_separation)}
      </div>
    </article>`;
  }).join("")}</div>`;
  byId("action-tabs").querySelectorAll("button").forEach((button) => button.addEventListener("click", () => {
    collectCurrent();
    state.action = button.dataset.actionTab;
    render();
  }));
}

function collectCurrent() {
  const task = state.manifest.tasks[state.index];
  if (!task) return;
  document.querySelectorAll(".video-card").forEach((card) => {
    const label = card.dataset.label;
    const parent = itemFor(task, label);
    if (task.kind === "catastrophe") {
      parent.flagged = card.querySelector("[data-flag]").checked;
      card.querySelectorAll("[data-field]").forEach((input) => {
        parent[input.dataset.field] = input.type === "checkbox" ? input.checked : input.value;
      });
    } else {
      parent.actions ||= {};
      parent.pair ||= {};
      const item = parent.actions[card.dataset.action] || {};
      card.querySelectorAll(":scope > .field-grid [data-field], :scope > details [data-field]").forEach((input) => {
        item[input.dataset.field] = input.value;
      });
      parent.actions[card.dataset.action] = item;
      card.querySelectorAll("[data-pair] [data-field]").forEach((input) => {
        parent.pair[input.dataset.field] = input.value;
      });
    }
  });
  answerFor(task).complete = false;
  save();
}

function clearCatastrophe(task) {
  const answer = answerFor(task);
  answer.verdict = "clear";
  answer.items = Object.fromEntries(task.labels.map((label) => [label, {
    flagged: false, severity: "0", catastrophe: "no", onset: "none",
  }]));
}

function clearAction(task) {
  const answer = answerFor(task);
  answer.verdict = "clear";
  answer.items = Object.fromEntries(task.labels.map((label) => [label, {
    actions: Object.fromEntries(task.actions.map((action) => [action, {
      rating: "pass", onset: "early", catastrophe: "no", identity_preserved: "yes",
      background_preserved: "yes", motion_freeze: "no",
    }])),
    pair: { steering_separation: "2", longitudinal_separation: "2" },
  }]));
}

function validateDetailed(task) {
  const answer = answerFor(task);
  if (task.kind === "catastrophe") {
    const flagged = task.labels.filter((label) => answer.items[label]?.flagged);
    if (!flagged.length) return "未发现异常时请使用绿色快速通过按钮。";
    for (const label of flagged) {
      const item = answer.items[label];
      if (!item.severity || !item.catastrophe || !item.onset) return `请补齐 ${label} 的异常细节。`;
    }
  } else {
    for (const label of task.labels) {
      const item = answer.items[label] || {};
      for (const action of task.actions) {
        if (!item.actions?.[action]?.rating || !item.actions?.[action]?.onset) return `请完成 ${label} 的四个动作判断。`;
      }
      if (!item.pair?.steering_separation || !item.pair?.longitudinal_separation) return `请在“后退”页填写 ${label} 的反事实分离。`;
    }
  }
  return "";
}

function requireReviewer() {
  if (state.reviewerId.trim()) return true;
  byId("validation").textContent = "请先填写审阅者 ID / Reviewer ID is required.";
  byId("reviewer-id").focus();
  return false;
}

function requireMediaReady() {
  const clips = videos();
  if (clips.length && clips.every((video) => video.readyState >= 1 && !video.error)) return true;
  byId("validation").textContent = "视频尚未加载成功，不能提交本面板。请等待或刷新页面。";
  return false;
}

function completeQuick() {
  if (!requireReviewer() || !requireMediaReady()) return;
  const task = state.manifest.tasks[state.index];
  task.kind === "catastrophe" ? clearCatastrophe(task) : clearAction(task);
  answerFor(task).complete = true;
  save();
  nextIncomplete(false);
}

function completeDetailed() {
  if (!requireReviewer() || !requireMediaReady()) return;
  collectCurrent();
  const task = state.manifest.tasks[state.index];
  const error = validateDetailed(task);
  if (error) { byId("validation").textContent = error; return; }
  if (task.kind === "catastrophe") {
    task.labels.forEach((label) => {
      const item = itemFor(task, label);
      if (!item.flagged) Object.assign(item, { severity: "0", catastrophe: "no", onset: "none" });
    });
  }
  answerFor(task).verdict = "flagged";
  answerFor(task).complete = true;
  save();
  nextIncomplete(false);
}

function videos() { return [...document.querySelectorAll("video")]; }

function applyVideoSettings() {
  const task = state.manifest.tasks[state.index];
  videos().forEach((video) => {
    video.playbackRate = state.speed;
    video.addEventListener("error", () => {
      const answer = answerFor(task);
      if (answer.complete) {
        answer.complete = false;
        save();
        updateProgress();
      }
      byId("validation").textContent = "视频加载失败，本面板已恢复为未完成。请刷新后重试。";
    });
  });
  videos().forEach((video) => video.addEventListener("seeking", () => {
    if (state.syncing) return;
    state.syncing = true;
    videos().forEach((other) => { if (other !== video) other.currentTime = video.currentTime; });
    state.syncing = false;
  }));
}

async function togglePlayback() {
  const clips = videos();
  if (!clips.length) return;
  if (clips.some((video) => !video.paused)) {
    clips.forEach((video) => video.pause());
    return;
  }
  const time = Math.min(...clips.map((video) => Number.isFinite(video.currentTime) ? video.currentTime : 0));
  clips.forEach((video) => { video.currentTime = time; video.playbackRate = state.speed; });
  await Promise.allSettled(clips.map((video) => video.play()));
}

function render() {
  const task = state.manifest.tasks[state.index];
  if (!task) return;
  byId("task-meta").textContent = `面板 ${state.index + 1} / ${state.manifest.task_count}`;
  byId("kind-badge").textContent = task.kind === "catastrophe" ? "视频质量" : "动作控制";
  byId("baseline").textContent = baselineLabels[task.baseline] || task.baseline;
  byId("prompt").textContent = task.prompt ? `Prompt ${task.prompt_index + 1}: ${task.prompt}` : `场景 ${task.scene_index + 1} / ${task.scene_label}`;
  byId("conditioning-note").hidden = !task.conditioning_frames;
  byId("conditioning-note").textContent = task.conditioning_frames ? `只评判生成续写，忽略前 ${task.conditioning_frames} 个共享条件帧。` : "";
  byId("rubric").innerHTML = rubric[task.kind];
  byId("quick-complete").textContent = task.kind === "catastrophe" ? "未发现明显异常，下一项 (N)" : "全部动作清楚且及时，下一项 (N)";
  task.kind === "catastrophe" ? renderCatastrophe(task) : renderAction(task);
  byId("validation").textContent = answerFor(task).complete ? "已完成 / Completed" : "";
  updateProgress();
  applyVideoSettings();
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
  state.action = "turn_left";
  render();
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function nextIncomplete(collect = true) {
  if (collect) collectCurrent();
  const tasks = state.manifest.tasks;
  for (let offset = 1; offset <= tasks.length; offset += 1) {
    const index = (state.index + offset) % tasks.length;
    if (!state.answers[tasks[index].id]?.complete) {
      state.index = index;
      state.action = "turn_left";
      render();
      return;
    }
  }
  render();
  byId("validation").textContent = "全部面板已完成，请导出 JSON。";
}

function exportJson() {
  collectCurrent();
  const payload = {
    schema_version: 2,
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
  if (payload.manifest_sha256 !== state.manifest.manifest_sha256) throw new Error("Manifest hash 与当前审阅包不匹配。");
  state.reviewerId = payload.reviewer_id || "";
  state.answers = payload.answers || {};
  byId("reviewer-id").value = state.reviewerId;
  save();
  render();
}

function toggleFlag(index) {
  const task = state.manifest.tasks[state.index];
  if (task.kind !== "catastrophe" || index >= task.labels.length) return;
  collectCurrent();
  const item = itemFor(task, task.labels[index]);
  item.flagged = !item.flagged;
  render();
}

async function init() {
  state.manifest = await fetch(assetUrl("review_manifest.json"), { cache: "no-store" }).then((response) => {
    if (!response.ok) throw new Error(`Manifest load failed: ${response.status}`);
    return response.json();
  });
  if (state.manifest.schema_version !== 2) throw new Error("此页面需要 schema v2 审阅包，请重新生成 package。");
  loadSaved();
  byId("reviewer-id").value = state.reviewerId;
  byId("reviewer-id").addEventListener("input", (event) => { state.reviewerId = event.target.value; save(); });
  byId("prev").addEventListener("click", () => navigate(-1));
  byId("next-incomplete").addEventListener("click", () => nextIncomplete());
  byId("quick-complete").addEventListener("click", completeQuick);
  byId("detailed-complete").addEventListener("click", completeDetailed);
  byId("sync-play").addEventListener("click", togglePlayback);
  byId("speed").addEventListener("change", (event) => {
    state.speed = Number(event.target.value);
    videos().forEach((video) => { video.playbackRate = state.speed; });
  });
  byId("export").addEventListener("click", exportJson);
  byId("import").addEventListener("change", async (event) => {
    try { await importJson(event.target.files[0]); } catch (error) { byId("validation").textContent = error.message; }
  });
  byId("task-filter").addEventListener("change", (event) => {
    collectCurrent();
    const value = event.target.value;
    const index = state.manifest.tasks.findIndex((task) =>
      value === "all" || task.kind === value || (value === "incomplete" && !state.answers[task.id]?.complete)
    );
    if (index >= 0) { state.index = index; state.action = "turn_left"; render(); }
  });
  document.addEventListener("change", (event) => {
    if (event.target.matches("[data-flag]")) { collectCurrent(); render(); }
    else collectCurrent();
  });
  document.addEventListener("keydown", (event) => {
    if (event.target.matches("input, textarea, select")) return;
    if (event.code === "Space") { event.preventDefault(); togglePlayback(); }
    else if (event.key.toLowerCase() === "n") completeQuick();
    else if (event.key === "ArrowLeft") navigate(-1);
    else if (event.key === "ArrowRight") nextIncomplete();
    else if (/^[1-5]$/.test(event.key)) toggleFlag(Number(event.key) - 1);
  });
  render();
}

function renderStartupError(error) {
  const localFile = window.location.protocol === "file:";
  const title = localFile ? "此源文件不能直接打开" : "审阅包未完整加载";
  const detail = localFile
    ? "该目录只包含界面源码，没有 review_manifest.json 和视频。请从生成后的 public 目录启动 HTTP 服务。"
    : `请确认当前 public 目录包含 review_manifest.json、media/ 和页面文件。错误：${error.message}`;
  byId("review-root").innerHTML = `<section class="startup-error" role="alert">
    <h2>${title}</h2>
    <p>${detail}</p>
    <code>cd /path/to/generated/review/public<br>python -m http.server 8766</code>
  </section>`;
}

init().catch(renderStartupError);
