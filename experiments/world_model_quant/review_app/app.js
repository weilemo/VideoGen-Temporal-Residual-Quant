"use strict";

const STORAGE_KEY = "videoquant-b1-anonymous-review-v1";
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
const catastropheRubric = `
  <details class="rubric" open>
    <summary>判定说明与例子 / Definitions and examples</summary>
    <dl class="rubric-grid">
      <div><dt>身份切换 / Identity switch</dt><dd>主体身份、外观或关键物体突然变成另一个；例如人物脸、车辆型号或服装无因改变。</dd></div>
      <div><dt>背景跳变 / Background jump</dt><dd>相机运动无法解释的场景突变；例如道路、房间布局或地平线在相邻帧瞬移。</dd></div>
      <div><dt>纹理重复 / Texture repetition</dt><dd>局部纹理出现明显复制、周期性条带或不断累积的图案。</dd></div>
      <div><dt>运动冻结 / Motion freeze</dt><dd>视频仍在播放，但主体或整幅画面异常静止；短暂停顿后恢复也应记录。</dd></div>
      <div><dt>颜色漂移 / Color drift</dt><dd>整体色调或局部颜色持续偏移，且不是光照或场景变化造成。</dd></div>
      <div><dt>黑帧或无效帧 / Black or invalid frames</dt><dd>黑屏、纯色帧、严重花屏、NaN 式噪声或无法辨认的解码异常。</dd></div>
      <div><dt>严重度 / Severity</dt><dd>0 无；1 轻微且不影响理解；2 明显影响内容；3 严重破坏主体、场景或连续性。</dd></div>
      <div><dt>灾难失败 / Catastrophic failure</dt><dd>主体或场景连续性被根本破坏，视频无法再按原 prompt 正常理解；轻微模糊不算灾难。</dd></div>
    </dl>
  </details>`;
const actionRubric = `
  <details class="rubric" open>
    <summary>动作判定说明与例子 / Action review definitions</summary>
    <dl class="rubric-grid">
      <div><dt>动作执行 / Action executed</dt><dd>Yes：清楚完成目标动作；Partial：方向可见但幅度弱、迟到或只完成一部分；No：没有响应或执行了相反动作。</dd></div>
      <div><dt>方向正确 / Direction correct</dt><dd>只判断运动方向是否符合标签；看不清或镜头运动造成歧义时选 Unclear，不要猜测。</dd></div>
      <div><dt>响应时机 / Response onset</dt><dd>记录首次可确认动作出现于视频前段、中段、后段；始终没有则选 Never。</dd></div>
      <div><dt>身份与背景保持 / Identity and background preserved</dt><dd>动作过程中主体外观和环境布局应保持连续；遮挡导致暂时看不清可选 Unclear。</dd></div>
      <div><dt>反事实分离 / Counterfactual separation</dt><dd>0：两种相反动作几乎一样；1：存在差别但较弱；2：左右或前后轨迹清楚分开。</dd></div>
      <div><dt>灾难失败 / Catastrophic failure</dt><dd>黑屏、主体消失、场景崩坏、长期冻结，或动作完全失控到无法判断原任务。</dd></div>
    </dl>
  </details>`;

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
    <fieldset><legend>观察到的失败 / Observed failures</legend><div class="check-grid">${checks}</div></fieldset>
    <div class="field-grid">
      ${selectField("severity", "总体严重度 / Overall severity", [["", "请选择 / Select"], ["0", "0 - 无 / None"], ["1", "1 - 轻微 / Mild"], ["2", "2 - 明显 / Material"], ["3", "3 - 严重 / Severe"]], answer.severity)}
      ${selectField("catastrophe", "是否为灾难失败？/ Catastrophic failure?", [["", "请选择 / Select"], ["no", "否 / No"], ["yes", "是 / Yes"]], answer.catastrophe)}
      ${selectField("onset", "首次出现位置 / First visible onset", [["", "请选择 / Select"], ["none", "未出现 / None"], ["early", "前段 / Early"], ["middle", "中段 / Middle"], ["late", "后段 / Late"]], answer.onset)}
      <label class="field">备注 / Notes<textarea data-field="notes">${escapeHtml(answer.notes || "")}</textarea></label>
    </div>
  </div>`;
}

function selectField(key, label, options, selected) {
  const rendered = options.map((option) => {
    const [actual, display] = Array.isArray(option) ? option : [option, option || "请选择 / Select"];
    return `<option value="${escapeHtml(actual)}" ${String(selected ?? "") === actual ? "selected" : ""}>${escapeHtml(display)}</option>`;
  }).join("");
  return `<label class="field">${label}<select data-field="${key}">${rendered}</select></label>`;
}

function renderCatastrophe(task) {
  byId("review-root").innerHTML = `<div class="comparison">${["A", "B"].map((side) => `
    <article class="side"><h2>视频 ${side} / Video ${side}</h2>
      <video controls preload="metadata" src="${task.media[side]}"></video>
      ${catastropheForm(task, side)}
    </article>`).join("")}</div>`;
}

function actionForm(task, side) {
  const sideAnswer = answerFor(task).sides[side] || {};
  const actions = task.actions.map((action) => {
    const answer = (sideAnswer.actions || {})[action] || {};
    return `<fieldset data-action="${action}"><legend>${actionLabels[action]}</legend><div class="field-grid">
      ${selectField("action_executed", "动作是否执行 / Action executed", [["", "请选择 / Select"], ["yes", "是 / Yes"], ["partial", "部分 / Partial"], ["no", "否 / No"]], answer.action_executed)}
      ${selectField("direction_correct", "方向是否正确 / Direction correct", [["", "请选择 / Select"], ["yes", "是 / Yes"], ["no", "否 / No"], ["unclear", "不确定 / Unclear"]], answer.direction_correct)}
      ${selectField("onset", "响应开始位置 / Response onset", [["", "请选择 / Select"], ["early", "前段 / Early"], ["middle", "中段 / Middle"], ["late", "后段 / Late"], ["never", "未响应 / Never"]], answer.onset)}
      ${selectField("catastrophe", "是否为灾难失败？/ Catastrophic failure?", [["", "请选择 / Select"], ["no", "否 / No"], ["yes", "是 / Yes"]], answer.catastrophe)}
      ${selectField("identity_preserved", "身份是否保持 / Identity preserved", [["", "请选择 / Select"], ["yes", "是 / Yes"], ["no", "否 / No"], ["unclear", "不确定 / Unclear"]], answer.identity_preserved)}
      ${selectField("background_preserved", "背景是否保持 / Background preserved", [["", "请选择 / Select"], ["yes", "是 / Yes"], ["no", "否 / No"], ["unclear", "不确定 / Unclear"]], answer.background_preserved)}
      ${selectField("motion_freeze", "是否运动冻结 / Motion freeze", [["", "请选择 / Select"], ["no", "否 / No"], ["yes", "是 / Yes"]], answer.motion_freeze)}
    </div></fieldset>`;
  }).join("");
  const pair = sideAnswer.pair || {};
  return `<div class="form-body" data-side="${side}">${actions}
    <fieldset data-pair="true"><legend>反事实分离 / Counterfactual separation</legend><div class="field-grid">
      ${selectField("steering_separation", "左右分离 / Left vs right separation", [["", "请选择 / Select"], ["0", "0 - 无 / Absent"], ["1", "1 - 弱 / Weak"], ["2", "2 - 清楚 / Clear"]], pair.steering_separation)}
      ${selectField("longitudinal_separation", "前后分离 / Forward vs backward separation", [["", "请选择 / Select"], ["0", "0 - 无 / Absent"], ["1", "1 - 弱 / Weak"], ["2", "2 - 清楚 / Clear"]], pair.longitudinal_separation)}
      <label class="field">备注 / Notes<textarea data-field="notes">${escapeHtml(pair.notes || "")}</textarea></label>
    </div></fieldset>
  </div>`;
}

function renderAction(task) {
  byId("review-root").innerHTML = `<div class="comparison">${["A", "B"].map((side) => `
    <article class="side"><h2>组 ${side} / Group ${side}</h2><div class="action-grid">
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
  byId("task-meta").textContent = `任务 ${state.index + 1} / ${state.manifest.task_count} · Task ${state.index + 1} of ${state.manifest.task_count}`;
  byId("kind-badge").textContent = task.kind === "catastrophe" ? "视频质量 / VIDEO QUALITY" : "动作控制 / ACTION CONTROL";
  byId("baseline").textContent = baselineLabels[task.baseline] || task.baseline;
  byId("prompt").textContent = task.prompt ? `提示词 ${task.prompt_index + 1} / Prompt ${task.prompt_index + 1}: ${task.prompt}` : `场景 ${task.scene_index + 1} / ${task.scene_label}`;
  const note = byId("conditioning-note");
  note.hidden = !task.conditioning_frames;
  note.textContent = task.conditioning_frames ? `仅评判生成的续写部分；忽略前 ${task.conditioning_frames} 个共享条件帧。/ Judge the generated continuation only; ignore the first ${task.conditioning_frames} shared conditioning frames.` : "";
  byId("rubric").innerHTML = task.kind === "catastrophe" ? catastropheRubric : actionRubric;
  task.kind === "catastrophe" ? renderCatastrophe(task) : renderAction(task);
  byId("validation").textContent = answerFor(task).complete ? "已保存为完成 / Saved as complete." : "";
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
  if (!state.reviewerId.trim()) { byId("validation").textContent = "请填写审阅者 ID / Reviewer ID is required."; return; }
  const task = state.manifest.tasks[state.index];
  const missing = missingFields(task, answerFor(task));
  if (missing.length) { byId("validation").textContent = `请补齐必填项 / Complete required fields: ${missing.slice(0, 4).join(", ")}${missing.length > 4 ? "..." : ""}`; return; }
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
  if (payload.manifest_sha256 !== state.manifest.manifest_sha256) throw new Error("Manifest hash 与当前审阅包不匹配 / Manifest hash does not match this review package.");
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
