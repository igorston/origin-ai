import { api, ApiError, pullModels, streamChat } from "./api.js";
import { renderMarkdown } from "./markdown.js";
import { brand, formatDateTime, formatNumber, languages, locale, setLanguage, t, translatePage } from "./i18n.js";

const $ = (selector) => document.querySelector(selector);

const el = {
  app: $("#app"),
  sessionList: $("#session-list"),
  newSession: $("#new-session"),
  chatTitle: $("#chat-title"),
  messages: $("#messages"),
  empty: $("#empty-state"),
  composer: $("#composer"),
  input: $("#input"),
  send: $("#send"),
  stop: $("#stop"),
  health: $("#health"),
  memoryCount: $("#memory-count"),
  settingsHint: $("#settings-hint"),
  // memory manager
  memoryDialog: $("#memory-dialog"),
  memoryStats: $("#memory-stats"),
  memorySearch: $("#memory-search"),
  memorySource: $("#memory-source"),
  memoryList: $("#memory-list"),
  addMemory: $("#add-memory"),
  memoryText: $("#memory-text"),
  // settings
  settingsDialog: $("#settings-dialog"),
  useMemory: $("#use-memory"),
  useTools: $("#use-tools"),
  toolList: $("#tool-list"),
  systemInfo: $("#system-info"),
  toast: $("#toast"),
  // context
  contextMeter: $("#context-meter"),
  contextFill: $("#context-fill"),
  contextReserve: $("#context-reserve"),
  contextLabel: $("#context-label"),
  contextPanel: $("#context-panel"),
  contextNotice: $("#context-notice"),
  contextNoticeText: $("#context-notice-text"),
  continueSession: $("#continue-session"),
};

const state = {
  sessionId: null,
  context: null, // ContextUsage of the open session
  closed: null, // reason, when the open session hit its operational limit
  streaming: null, // AbortController of the in-flight turn
  memory: { status: "active", editing: null, highlight: new Set() },
};

// "Otimizar com IA" preference, shared by the add form and the inline editor.
const OPTIMIZE_KEY = "origin:optimize-memory";
function optimizeEnabled() {
  try {
    return localStorage.getItem(OPTIMIZE_KEY) !== "false";
  } catch {
    return true;
  }
}
function setOptimize(value) {
  try { localStorage.setItem(OPTIMIZE_KEY, value); } catch {}
  document.querySelectorAll(".optimize-toggle").forEach((box) => (box.checked = value));
}

// Tools that change stored memories; after they run, memory views are refreshed.
const MEMORY_TOOLS = new Set(["remember", "forget"]);

// ---------------------------------------------------------------- helpers

function h(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (key === "class") node.className = value;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else if (value !== false && value != null) node.setAttribute(key, value === true ? "" : value);
  }
  for (const child of children.flat(Infinity)) {
    if (child != null && child !== false) node.append(child);
  }
  return node;
}

/** replaceChildren that skips null/false (the DOM would render them as the text "null"). */
function setChildren(node, ...children) {
  node.replaceChildren(...children.flat(Infinity).filter((child) => child != null && child !== false));
}

function toast(message, kind = "error") {
  el.toast.textContent = message;
  el.toast.className = `toast ${kind}`;
  el.toast.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => (el.toast.hidden = true), 4000);
}

function formatDate(iso) {
  if (!iso) return "";
  const date = new Date(iso);
  const sameDay = date.toDateString() === new Date().toDateString();
  return sameDay
    ? date.toLocaleTimeString(locale, { hour: "2-digit", minute: "2-digit" })
    : date.toLocaleDateString(locale, { day: "2-digit", month: "short" });
}

const seconds = (ms) => `${(ms / 1000).toFixed(1)} s`;

function scrollToBottom(force = false) {
  const m = el.messages;
  const nearBottom = m.scrollHeight - m.scrollTop - m.clientHeight < 120;
  if (force || nearBottom) m.scrollTop = m.scrollHeight;
}

function describeError(error) {
  if (error instanceof ApiError) return error.message;
  if (error.name === "TypeError") return t("error.server");
  return error.message || String(error);
}

function openDialog(dialog) {
  closeDrawers();
  if (!dialog.open) dialog.showModal();
}

// ---------------------------------------------------------------- health

function setHealth(kind, label, title) {
  el.health.classList.remove("ok", "warn", "bad");
  el.health.classList.add(kind);
  el.health.querySelector(".health-label").textContent = label;
  el.health.title = title;
}

let healthTimer;
async function refreshHealth() {
  clearTimeout(healthTimer);
  let health = null;
  try {
    health = await api.health();
    if (health.status !== "ok") {
      setHealth("bad", t("health.degraded"), t("health.degraded_hint"));
      maybeOpenSetup(health.ollama);
    }
    else if (!health.ready) setHealth("warn", t("health.loading"), t("health.loading_hint"));
    else setHealth("ok", t("health.online"), t("health.online_hint"));
  } catch {
    setHealth("bad", t("health.offline"), t("health.offline_hint"));
  }
  renderSystem(health);
  // Poll fast while models load, so the indicator turns green as soon as they are ready.
  healthTimer = setTimeout(refreshHealth, health?.status === "ok" && !health.ready ? 3000 : 30000);
}

function renderSystem(health) {
  if (!health) {
    el.systemInfo.replaceChildren(h("p", { class: "error-text" }, t("system.unreachable")));
    return;
  }
  const { ollama } = health;
  const models = Object.entries(ollama.models || {}).map(([name, installed]) => {
    const loaded = ollama.loaded?.[name];
    const [kind, label] = !installed ? ["bad", t("system.model_missing")] : loaded ? ["ok", t("system.model_loaded")] : ["warn", t("system.model_idle")];
    return h("li", {}, h("span", { class: `badge ${kind}` }, label), " ", h("code", {}, name));
  });
  setChildren(el.systemInfo,
    h("dl", { class: "facts" },
      h("dt", {}, t("system.status")), h("dd", {}, h("span", { class: health.status === "ok" ? "badge ok" : "badge bad" }, health.status)),
      h("dt", {}, t("system.version")), h("dd", {}, health.version),
      h("dt", {}, "Ollama"), h("dd", {}, h("code", {}, ollama.url), " ", ollama.reachable ? "✓" : t("system.ollama_down")),
    ),
    models.length ? h("div", {}, h("h3", {}, t("system.models")), h("ul", { class: "plain" }, models)) : null,
  );
}

// ---------------------------------------------------------------- sessions

async function refreshSessions() {
  let sessions;
  try {
    sessions = await api.sessions();
  } catch (error) {
    toast(describeError(error));
    return;
  }
  el.sessionList.replaceChildren(
    ...sessions.map((s) =>
      h("div", { class: `session${s.id === state.sessionId ? " active" : ""}`, "data-id": s.id },
        h("button", { class: "session-open", type: "button", onclick: () => openSession(s.id) },
          h("span", { class: "session-title" }, s.title || t("session.new")),
          h("span", { class: "session-meta" },
            t("session.meta", { date: formatDate(s.updated_at), n: s.message_count }) + (s.status === "closed" ? t("session.closed_tag") : "")),
        ),
        h("button", { class: "session-delete", type: "button", title: t("session.delete"), "aria-label": t("session.delete"),
          onclick: (event) => { event.stopPropagation(); removeSession(s.id, s.title); } }, "✕"),
      ),
    ),
  );
  if (!sessions.length) el.sessionList.append(h("p", { class: "muted small" }, t("session.none")));
}

function resetChat(title = t("session.new")) {
  el.chatTitle.textContent = title;
  el.messages.replaceChildren(el.empty);
  el.empty.hidden = false;
  renderContext(null);
  setClosed(null);
}

async function openSession(id) {
  if (state.streaming) return;
  closeDrawers();
  try {
    const session = await api.session(id);
    state.sessionId = id;
    resetChat(session.title || t("session.new"));
    el.empty.hidden = session.messages.length > 0 || Boolean(session.summary);
    // A continued conversation starts from its predecessor's summary.
    if (session.summary && session.summarized_upto === 0) el.messages.append(contextDivider({ inherited: true }));
    for (const message of session.messages) {
      if (message.role === "user") addUserMessage(message.content, message.created_at);
      else addAssistantMessage({ text: message.content, toolCalls: message.tool_calls, at: message.created_at });
      if (message.id === session.summarized_upto) {
        el.messages.append(contextDivider({ summarized: session.context.summarized_messages }));
      }
    }
    renderContext(session.context);
    setClosed(session.status === "closed" ? session.closed_reason || t("session.closed_default") : null);
    scrollToBottom(true);
    refreshSessions();
  } catch (error) {
    toast(describeError(error));
  }
}

function newChat() {
  if (state.streaming) return;
  state.sessionId = null;
  resetChat();
  refreshSessions();
  closeDrawers();
  el.input.focus();
}

async function removeSession(id, title) {
  if (!confirm(t("session.confirm_delete", { title: title || t("session.new") }))) return;
  try {
    await api.deleteSession(id);
    if (id === state.sessionId) newChat();
    else refreshSessions();
  } catch (error) {
    toast(describeError(error));
  }
}

// ---------------------------------------------------------------- messages

function addUserMessage(text, at) {
  el.empty.hidden = true;
  const article = h("article", { class: "message user" },
    h("div", { class: "bubble" }, text),
    h("div", { class: "meta" }, formatDate(at || new Date().toISOString())),
  );
  el.messages.append(article);
  return article;
}

function summarizeArgs(args = {}) {
  const text = Object.values(args).map((v) => (typeof v === "string" ? v : JSON.stringify(v))).join(", ");
  return text.length > 60 ? `${text.slice(0, 57)}…` : text;
}

function toolCard(call) {
  const args = Object.keys(call.args || {}).length ? JSON.stringify(call.args, null, 2) : t("tool.no_args");
  return h("details", { class: "tool-call" },
    h("summary", {}, h("span", { class: "tool-icon" }, "🔧"), h("code", {}, call.name), h("span", { class: "tool-args" }, summarizeArgs(call.args))),
    h("div", { class: "tool-body" },
      h("div", { class: "label" }, t("tool.args")), h("pre", {}, args),
      h("div", { class: "label" }, t("tool.result")), h("pre", {}, call.output),
    ),
  );
}

/** Create an assistant message; returns handles to update it while streaming. */
function addAssistantMessage({ text = "", toolCalls = [], at = null, pending = false } = {}) {
  el.empty.hidden = true;
  const tools = h("div", { class: "tool-calls" }, toolCalls.map(toolCard));
  const content = h("div", { class: "content" });
  const meta = h("div", { class: "meta" }, at ? formatDate(at) : "");
  const article = h("article", { class: `message assistant${pending ? " pending" : ""}` }, h("div", { class: "bubble" }, tools, content), meta);
  el.messages.append(article);

  let raw = text;
  const render = () => {
    content.innerHTML = raw ? renderMarkdown(raw) : pending ? '<span class="typing"><i></i><i></i><i></i></span>' : "";
  };
  render();
  return {
    appendText(chunk) { raw += chunk; render(); },
    addTool(call) { tools.append(toolCard(call)); },
    setMeta(value) { meta.textContent = value; },
    finish() { pending = false; article.classList.remove("pending"); render(); },
    remove() { article.remove(); },
    fail(message) {
      pending = false;
      article.classList.remove("pending");
      article.classList.add("failed");
      render();
      content.append(h("p", { class: "error-text" }, `⚠ ${message}`));
    },
  };
}

// ---------------------------------------------------------------- chat turn

async function send(message) {
  message = message.trim();
  if (!message || state.streaming || state.closed) return;

  // Busy from here on: a second Enter while the session is being created would
  // otherwise start another conversation.
  setStreaming(new AbortController());
  if (!state.sessionId) {
    try {
      state.sessionId = (await api.createSession()).id;
    } catch (error) {
      toast(describeError(error));
      setStreaming(null);
      el.input.value = message;
      return;
    }
  }

  const userArticle = addUserMessage(message);
  const reply = addAssistantMessage({ pending: true });
  scrollToBottom(true);

  const started = performance.now();
  let firstToken = null;
  let toolCount = 0;
  let touchedMemory = false;
  try {
    const body = {
      message,
      session_id: state.sessionId,
      use_memory: el.useMemory.checked,
      use_tools: el.useTools.checked,
    };
    for await (const { event, data } of streamChat(body, state.streaming.signal)) {
      if (event === "token") {
        firstToken ??= performance.now() - started;
        reply.appendText(data.text);
      } else if (event === "tool_call") {
        toolCount += 1;
        touchedMemory ||= MEMORY_TOOLS.has(data.name);
        reply.addTool(data);
      } else if (event === "context") {
        // Old messages were folded into the summary before this turn.
        if (data.compactions.length) {
          const folded = data.compactions.reduce((sum, c) => sum + c.folded, 0);
          userArticle.before(contextDivider({ folded, summarized: data.usage?.summarized_messages }));
        }
        if (data.usage) renderContext(data.usage);
      } else if (event === "done") {
        if (data.context) renderContext(data.context);
      } else if (event === "error") {
        throw new Error(data.detail);
      }
      scrollToBottom();
    }
    reply.finish();
  } catch (error) {
    if (error instanceof ApiError && error.detail?.closed) {
      // Refused, not stored: give the text back so it can go to the next conversation.
      userArticle.remove();
      reply.remove();
      el.input.value = message;
      if (error.detail.context) renderContext({ ...error.detail.context, state: "closed" });
      setClosed(error.detail.reason);
    } else {
      reply.fail(error.name === "AbortError" ? t("chat.stopped") : describeError(error));
    }
  } finally {
    const parts = [formatDate(new Date().toISOString())];
    if (firstToken != null) parts.push(t("chat.first_token", { s: seconds(firstToken) }));
    parts.push(t("chat.total", { s: seconds(performance.now() - started) }));
    if (toolCount) parts.push(t("chat.tools", { n: toolCount }));
    reply.setMeta(parts.join(" · "));
    setStreaming(null);
    scrollToBottom();
    if (touchedMemory) refreshMemory();
    // The first user message titles the session on the server.
    await refreshSessions();
    const current = el.sessionList.querySelector(".session.active .session-title");
    if (current) el.chatTitle.textContent = current.textContent;
  }
}

function setStreaming(controller) {
  state.streaming = controller;
  el.send.hidden = Boolean(controller);
  el.stop.hidden = !controller;
  el.input.disabled = Boolean(controller) || Boolean(state.closed);
  el.send.disabled = Boolean(state.closed);
  if (!controller && !state.closed) el.input.focus();
}

// ---------------------------------------------------------------- context window

const tokens = (n) => formatNumber(n);
const windowSource = (source) => t(`ctx.source.${source}`);
const contextState = (state) => t(`ctx.state.${state}`);
// Close to the operational limit (share of the room taken by content that is never
// condensed): say so before the chat actually closes.
const PROTECTED_WARNING = 60;
const protectedShare = (u) => (u.protected_room > 0 ? Math.round((u.protected / u.protected_room) * 100) : 0);

function renderContext(usage) {
  state.context = usage;
  el.contextMeter.hidden = !usage;
  if (!usage) {
    el.contextPanel.hidden = true;
    renderNotice();
    return;
  }
  // Shown as a share of the model's whole window, with the reply reserve marked at its
  // end; the thresholds and state still come from the server (a share of the rest).
  const share = usage.used / usage.window;
  el.contextMeter.className = `context-meter ${usage.state}`;
  el.contextFill.style.width = `${Math.min(share, 1) * 100}%`;
  el.contextReserve.style.width = `${((usage.window - usage.usable) / usage.window) * 100}%`;
  el.contextLabel.textContent = share > 1 ? t("ctx.full") : `${usage.measured ? "" : "~"}${Math.round(share * 100)}%`;
  el.contextMeter.title =
    t("ctx.meter_title", {
      state: contextState(usage.state),
      used: tokens(usage.used),
      window: tokens(usage.window),
      source: windowSource(usage.window_source),
    }) + (usage.compactions ? t("ctx.meter_optimized", { n: usage.compactions }) : "");
  if (!el.contextPanel.hidden) renderContextPanel();
  renderNotice();
}

function renderContextPanel() {
  const u = state.context;
  if (!u) return;
  // Before a turn the parts are estimates; after one, Ollama reports the real total.
  const parts = [
    ["base", t("ctx.part.base"), u.base],
    ["summary", t("ctx.part.summary"), u.summary],
    ["history", t("ctx.part.history"), u.history],
    ["memory", t("ctx.part.memory"), u.memory],
    ["message", t("ctx.part.message"), u.message],
  ].filter(([, , value]) => value > 0);
  const scale = Math.max(u.window, u.used, 1);
  const reserve = u.window - u.usable;
  const share = u.used / u.window;
  setChildren(el.contextPanel,
    h("h3", {}, `${contextState(u.state)} · ${share > 1 ? t("ctx.heading_over") : t("ctx.heading_share", { p: Math.round(share * 100) })}`),
    h("div", { class: "context-stack", title: t("ctx.composition") },
      parts.map(([key, label, value]) =>
        h("span", { class: `seg-${key}`, style: `width:${(value / scale) * 100}%`, title: `${label}: ${tokens(value)}` })),
      h("span", { class: "seg-reserve", style: `width:${(reserve / scale) * 100}%;margin-left:auto`, title: `${t("ctx.reserve")}: ${tokens(reserve)}` })),
    u.used > u.usable
      ? h("p", { class: "context-over" },
          u.used > u.window
            ? t("ctx.over_window")
            : t("ctx.over_reserve"))
      : null,
    h("dl", { class: "context-rows" },
      h("dt", {}, u.measured ? t("ctx.used_measured") : t("ctx.used_estimated")), h("dd", {}, `${tokens(u.used)} / ${tokens(u.window)}`),
      h("dt", { title: t("ctx.free_hint") }, t("ctx.free")),
      h("dd", {}, tokens(Math.max(0, u.usable - u.used))),
      parts.map(([key, label, value]) => [
        h("dt", {}, h("span", { class: `swatch seg-${key}` }), label), h("dd", {}, `~${tokens(value)}`),
      ]),
      h("dt", {}, h("span", { class: "swatch seg-reserve" }), t("ctx.reserve")), h("dd", {}, tokens(reserve)),
      h("dt", {}, t("ctx.summarized")), h("dd", {}, String(u.summarized_messages)),
      h("dt", {}, t("ctx.optimizations")), h("dd", {}, String(u.compactions)),
      h("dt", { title: t("ctx.condensations_hint") }, t("ctx.condensations")),
      h("dd", {}, String(u.compressions)),
      h("dt", { title: t("ctx.protected_hint") }, t("ctx.protected")),
      h("dd", {}, t("ctx.protected_value", { n: tokens(u.protected), p: protectedShare(u) })),
    ),
    h("h4", {}, t("ctx.capacity")),
    h("dl", { class: "context-rows" },
      h("dt", {}, t("ctx.window")), h("dd", {}, `${tokens(u.window)} (${windowSource(u.window_source)})`),
      u.model_limit ? [h("dt", {}, t("ctx.model_limit")), h("dd", { title: u.model }, tokens(u.model_limit))] : null,
      u.vram_limit != null
        ? [h("dt", { title: t("ctx.vram_hint") }, t("ctx.vram")),
           h("dd", {}, `~${tokens(u.vram_limit)}`)]
        : null,
      u.gpu
        ? [h("dt", {}, t("ctx.gpu")),
           h("dd", { title: u.gpu }, `${u.gpu.replace(/^NVIDIA\s+(GeForce\s+)?/i, "")}, ${String(u.vram_total_gb).replace(".", ",")} GB`)]
        : null,
    ),
    h("p", {}, t("ctx.explain")),
  );
}

function renderNotice() {
  const u = state.context;
  el.contextNotice.classList.toggle("closed", Boolean(state.closed));
  if (state.closed) {
    el.contextNoticeText.textContent =
      t("notice.closed", { reason: state.closed });
    el.continueSession.hidden = false;
    el.contextNotice.hidden = false;
    return;
  }
  // The real limit: what is never condensed (facts, written pieces) filling the room.
  const nearLimit = u && protectedShare(u) >= PROTECTED_WARNING;
  if (nearLimit) {
    el.contextNoticeText.textContent =
      t("notice.near_limit", { p: protectedShare(u) });
  } else if (u?.state === "critical") {
    el.contextNoticeText.textContent =
      t("notice.critical");
  }
  el.continueSession.hidden = !nearLimit;
  el.contextNotice.hidden = !(nearLimit || u?.state === "critical");
}

function setClosed(reason) {
  state.closed = reason;
  el.input.disabled = Boolean(reason) || Boolean(state.streaming);
  el.send.disabled = Boolean(reason);
  el.input.placeholder = reason ? t("chat.placeholder_closed") : t("chat.placeholder");
  renderNotice();
}

/** Divider marking where the model's view switches from the summary to real messages. */
function contextDivider({ folded = 0, summarized = 0, inherited = false } = {}) {
  const label = inherited
    ? t("divider.inherited")
    : folded
      ? t("divider.folded", { n: folded }) + (summarized > folded ? t("divider.total", { n: summarized }) : "")
      : t("divider.summarized", { n: summarized });
  const text = h("div", { class: "summary-text" }, t("divider.loading"));
  const details = h("details", { class: "context-divider" }, h("summary", { title: t("divider.hint") }, label), text);
  const sessionId = state.sessionId;
  details.addEventListener("toggle", async () => {
    if (!details.open) return;
    try {
      text.textContent = (await api.session(sessionId)).summary || t("divider.empty");
    } catch (error) {
      text.textContent = describeError(error);
    }
  });
  return details;
}

async function continueConversation() {
  if (!state.sessionId || state.streaming) return;
  el.continueSession.disabled = true;
  el.continueSession.textContent = t("continue.busy");
  const draft = el.input.value;
  try {
    const next = await api.continueSession(state.sessionId);
    await openSession(next.id);
    el.input.value = draft;
    autoResize();
    el.input.focus();
  } catch (error) {
    toast(describeError(error));
  } finally {
    el.continueSession.disabled = false;
    el.continueSession.textContent = t("continue.button");
  }
}

// ---------------------------------------------------------------- memory manager

let records = new Map(); // id -> record, from the last full listing (resolves "superseded by")

async function refreshMemory() {
  const query = el.memorySearch.value.trim();
  const { status } = state.memory;
  const source = el.memorySource.value;
  try {
    const [stats, all] = await Promise.all([api.memoryStats(), api.memories(1000)]);
    records = new Map(all.map((r) => [r.id, r]));
    el.memoryCount.textContent = stats.active || "";
    el.memoryStats.textContent =
      `${t("mem.active", { n: stats.active })} · ${t("mem.archived_count", { n: stats.archived })}`;
    if (!el.memoryDialog.open) return;

    // Semantic search only covers active memories; archived ones are matched by text.
    let items = query ? await api.searchMemory(query, 50) : all;
    if (query && status !== "active") {
      const needle = query.toLowerCase();
      items = [...items, ...all.filter((r) => r.metadata.archived && r.content.toLowerCase().includes(needle))];
    }
    items = items.filter((m) => {
      const archived = Boolean(m.metadata.archived);
      if (status === "active" && archived) return false;
      if (status === "archived" && !archived) return false;
      return !source || (m.metadata.source || "") === source;
    });

    el.memoryList.replaceChildren(...items.map(memoryRow));
    if (!items.length) {
      el.memoryList.append(
        h("li", { class: "memory-empty" },
          query ? t("mem.none_match") : status === "archived" ? t("mem.none_archived") : t("mem.none")),
      );
    }
  } catch (error) {
    el.memoryStats.textContent = describeError(error);
  }
}

function memoryRow(memory) {
  const meta = memory.metadata || {};
  const editing = state.memory.editing === memory.id;
  const supersededBy = meta.superseded_by && records.get(meta.superseded_by);

  const chips = [
    meta.source ? h("span", { class: "badge" }, t(`mem.source.${meta.source}`)) : null,
    meta.created_at ? h("span", { class: "muted small", title: formatDateTime(new Date(meta.created_at)) }, formatDate(meta.created_at)) : null,
    meta.edited_at ? h("span", { class: "badge", title: t("mem.edited_at", { date: formatDateTime(new Date(meta.edited_at)) }) }, t("mem.edited")) : null,
    memory.score != null ? h("span", { class: "badge", title: t("mem.score_hint") }, `score ${memory.score.toFixed(2)}`) : null,
    meta.archived ? h("span", { class: "badge warn" }, t("mem.archived")) : null,
    supersededBy ? h("span", { class: "muted small" }, t("mem.superseded", { text: supersededBy.content })) : null,
  ];

  const body = editing ? memoryEditor(memory) : h("p", { class: "memory-content", title: t("mem.dblclick"), ondblclick: () => startEdit(memory.id) }, memory.content);

  const actions = editing
    ? null
    : h("div", { class: "memory-actions" },
        h("button", { class: "link", type: "button", onclick: () => startEdit(memory.id) }, t("mem.edit")),
        h("button", { class: "link", type: "button",
          onclick: () => mutateMemory(() => api.updateMemory(memory.id, { archived: !meta.archived }), meta.archived ? t("mem.restored") : t("mem.archived_toast")) },
          meta.archived ? t("mem.restore") : t("mem.archive")),
        h("button", { class: "link danger", type: "button",
          onclick: () => confirm(t("mem.confirm_delete", { text: memory.content })) && mutateMemory(() => api.deleteMemory(memory.id), t("mem.deleted")) },
          t("mem.delete")),
      );

  const flash = state.memory.highlight.has(memory.id) ? " flash" : "";
  return h("li", { class: `memory-row${meta.archived ? " archived" : ""}${editing ? " editing" : ""}${flash}`, "data-id": memory.id },
    h("div", { class: "memory-main" }, body, h("div", { class: "memory-meta" }, chips)),
    actions,
  );
}

function memoryEditor(memory) {
  const textarea = h("textarea", { class: "field", rows: 2 });
  textarea.value = memory.content;
  const optimize = h("input", { type: "checkbox", class: "optimize-toggle" });
  optimize.checked = optimizeEnabled();
  optimize.addEventListener("change", () => setOptimize(optimize.checked));
  const saveButton = h("button", { class: "button primary", type: "button" }, t("common.save"));
  const cancelButton = h("button", { class: "button", type: "button" }, t("common.cancel"));

  let busy = false;
  const save = async () => {
    const content = textarea.value.trim();
    if (busy) return;
    if (!content) return toast(t("mem.empty_error"));
    if (content === memory.content && !optimize.checked) return cancel();
    busy = true;
    textarea.disabled = saveButton.disabled = cancelButton.disabled = true;
    saveButton.textContent = optimize.checked ? t("common.optimizing") : t("common.saving");
    try {
      const result = await api.updateMemory(memory.id, { content, optimize: optimize.checked });
      state.memory.editing = null;
      showCuration(result, t("mem.updated"));
    } catch (error) {
      toast(describeError(error));
      busy = false;
      textarea.disabled = saveButton.disabled = cancelButton.disabled = false;
      saveButton.textContent = t("common.save");
      return;
    }
    await refreshMemory();
  };
  const cancel = () => { state.memory.editing = null; refreshMemory(); };
  saveButton.addEventListener("click", save);
  cancelButton.addEventListener("click", cancel);
  textarea.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); save(); }
    if (event.key === "Escape") { event.preventDefault(); event.stopPropagation(); if (!busy) cancel(); }
  });
  queueMicrotask(() => { textarea.focus(); textarea.setSelectionRange(textarea.value.length, textarea.value.length); });
  return h("div", { class: "memory-editor" },
    textarea,
    h("div", { class: "row-actions" },
      h("span", { class: "muted small" }, t("mem.editor_hint")),
      h("label", { class: "check optimize", title: t("mem.optimize_hint") },
        optimize, " ", t("mem.optimize")),
      cancelButton,
      saveButton,
    ),
  );
}

/** Tell the user what curation did, and flash the memories it touched. */
function showCuration(result, fallback) {
  const saved = result.saved || [];
  const duplicates = result.duplicates || [];
  const editedId = result.memory?.id;
  const superseded = (result.archived || []).filter((a) => a.id !== editedId || !duplicates.length);
  const parts = [];
  if (result.normalized && saved.length > 1) parts.push(t("cur.split", { n: saved.length }));
  else if (result.normalized && saved.length === 1) parts.push(t("cur.optimized", { text: saved[0].content }));
  if (duplicates.length === 1) parts.push(t("cur.duplicate", { text: duplicates[0].content }));
  else if (duplicates.length > 1) parts.push(t("cur.duplicates", { n: duplicates.length }));
  if (superseded.length) {
    parts.push(t("cur.superseded", { n: superseded.length }));
  }
  const message = parts.length ? parts.join(" · ") : fallback;
  toast(message.charAt(0).toUpperCase() + message.slice(1), "info");

  state.memory.highlight = new Set([...saved, ...duplicates].map((r) => r.id));
  clearTimeout(showCuration.timer);
  showCuration.timer = setTimeout(() => {
    state.memory.highlight.clear();
    document.querySelectorAll(".memory-row.flash").forEach((row) => row.classList.remove("flash"));
  }, 2500);
}

function startEdit(id) {
  state.memory.editing = id;
  refreshMemory();
}

async function mutateMemory(action, success) {
  try {
    await action();
    if (success) toast(success, "info");
  } catch (error) {
    toast(describeError(error));
  }
  await refreshMemory();
}

function openMemory() {
  openDialog(el.memoryDialog);
  refreshMemory();
}

// ---------------------------------------------------------------- settings

async function refreshTools() {
  try {
    const tools = await api.tools();
    el.toolList.replaceChildren(
      ...tools.map((tool) =>
        h("li", { class: "tool" },
          h("code", { class: "tool-name" }, tool.name),
          h("p", { class: "tool-description" }, tool.description.trim()),
          Object.keys(tool.args).length
            ? h("div", { class: "muted small" }, t("tools.args"), Object.keys(tool.args).join(", "))
            : h("div", { class: "muted small" }, t("tools.no_args")),
        ),
      ),
    );
    if (!tools.length) el.toolList.append(h("li", { class: "muted" }, t("tools.none")));
  } catch (error) {
    el.toolList.replaceChildren(h("li", { class: "error-text" }, describeError(error)));
  }
}

function selectTab(name) {
  el.settingsDialog.querySelectorAll(".tab").forEach((tab) => tab.classList.toggle("active", tab.dataset.tab === name));
  el.settingsDialog.querySelectorAll(".tab-panel").forEach((panel) => (panel.hidden = panel.dataset.panel !== name));
  if (name === "tools") refreshTools();
  if (name === "system") refreshHealth();
  if (name === "calibration") refreshCalibration();
}

function openSettings(tab = "general") {
  openDialog(el.settingsDialog);
  selectTab(tab);
}

function renderSettingsHint() {
  if (state.calibration?.needs_reindex) {
    el.settingsHint.textContent = t("settings.hint_reindex");
    return;
  }
  const off = [!el.useMemory.checked && t("settings.hint_memory"), !el.useTools.checked && t("settings.hint_tools")].filter(Boolean);
  el.settingsHint.textContent = off.length ? t("settings.hint_off", { list: off.join(t("settings.hint_and")) }) : "";
}

// ---------------------------------------------------------------- memory calibration

const THRESHOLD_INFO = Object.fromEntries(
  ["dedup", "conflict", "min_score"].map((key) => [key, [t(`cal.threshold.${key}`), t(`cal.threshold.${key}_hint`)]]),
);
const scoreLabel = (key) => t(`cal.score.${key}`);

async function refreshCalibration() {
  try {
    state.calibration = await api.calibrationStatus();
  } catch (error) {
    $("#calibration-status").replaceChildren(h("p", { class: "error-text" }, describeError(error)));
    return;
  }
  renderSettingsHint();
  const s = state.calibration;
  $("#reindex-banner").hidden = !s.needs_reindex;
  $("#reindex-banner-text").textContent = s.needs_reindex ? t("mem.reindex_needed", { reason: s.reason }) : "";
  if (!el.settingsDialog.open) return;

  const dims = (dim) => (dim ? t("cal.dims", { n: dim }) : "");
  const thresholds = Object.entries(THRESHOLD_INFO).map(([key, [label, help]]) =>
    h("tr", {},
      h("th", { title: help }, label),
      h("td", {}, h("code", {}, s.thresholds[key].toFixed(3))),
      h("td", { class: "muted small" }, s.thresholds[key] === s.defaults[key] ? t("cal.default") : t("cal.default_value", { v: s.defaults[key].toFixed(3) })),
    ),
  );
  setChildren($("#calibration-status"),
    s.needs_reindex
      ? h("div", { class: "banner warn" },
          h("span", {}, t("cal.needs_reindex", { n: s.count, reason: s.reason })),
          h("button", { class: "button primary", type: "button", onclick: reindex }, t("cal.reindex_now")),
        )
      : null,
    h("dl", { class: "facts" },
      h("dt", {}, t("cal.embeddings")), h("dd", {}, h("code", {}, s.embed_model), dims(s.current_dim)),
      h("dt", {}, t("cal.indexed_with")), h("dd", {}, s.indexed_model ? h("code", {}, s.indexed_model) : "—", dims(s.stored_dim), t("cal.count", { n: s.count })),
      h("dt", {}, t("cal.chat_model")), h("dd", {}, h("code", {}, s.chat_model)),
      h("dt", {}, t("cal.calibration")), h("dd", {},
        s.calibrated
          ? h("span", { class: "badge ok" }, t("cal.calibrated_at", { date: formatDateTime(new Date(s.calibrated_at)) }))
          : h("span", { class: "badge warn" }, t("cal.uncalibrated"))),
    ),
    h("table", { class: "thresholds" }, h("tbody", {}, thresholds)),
    h("div", { class: "row-actions start" },
      h("button", { class: "button primary", type: "button", id: "run-calibration", onclick: calibrate }, t("cal.calibrate")),
      !s.needs_reindex && s.count ? h("button", { class: "button", type: "button", onclick: reindex }, t("cal.reindex")) : null,
      s.calibrated ? h("button", { class: "button", type: "button", onclick: resetThresholds }, t("cal.reset")) : null,
    ),
  );
}

async function reindex() {
  const s = state.calibration;
  if (!confirm(t("cal.confirm_reindex", { n: s.count, model: s.embed_model }))) return;
  toast(t("cal.reindexing"), "info");
  try {
    const result = await api.reindexMemory();
    toast(t("cal.reindexed", { n: result.reindexed, s: result.duration_s }), "info");
  } catch (error) {
    toast(describeError(error));
  }
  await refreshCalibration();
  refreshMemory();
}

async function calibrate() {
  const button = $("#run-calibration");
  button.disabled = true;
  button.textContent = t("cal.calibrating");
  try {
    renderReport(await api.runCalibration(false));
  } catch (error) {
    toast(describeError(error));
  } finally {
    button.disabled = false;
    button.textContent = t("cal.calibrate");
  }
}

function renderReport(report) {
  const rows = Object.entries(THRESHOLD_INFO).map(([key, [label, help]]) => {
    const [now, next] = [report.current[key], report.suggested[key]];
    return h("tr", {},
      h("th", { title: help }, label),
      h("td", {}, h("code", {}, now.toFixed(3))),
      h("td", {}, h("code", { class: Math.abs(next - now) >= 0.01 ? "changed" : "" }, next.toFixed(3))),
    );
  });
  const scores = Object.entries(report.scores).map(([key, st]) =>
    h("tr", {}, h("th", {}, scoreLabel(key)), h("td", {}, st.min.toFixed(2)), h("td", {}, st.mean.toFixed(2)), h("td", {}, st.max.toFixed(2))),
  );
  const judge = report.judge;
  $("#calibration-report").replaceChildren(
    h("div", { class: "report" },
      h("h3", {}, t("cal.result", { embed: report.embed_model, chat: report.chat_model, s: report.duration_s })),
      h("table", { class: "thresholds" },
        h("thead", {}, h("tr", {}, h("th", {}, ""), h("th", {}, t("cal.in_use")), h("th", {}, t("cal.suggested")))),
        h("tbody", {}, rows),
      ),
      h("p", { class: "small" },
        t("cal.recall", { hits: report.recall_hits, total: report.recall_total }),
        judge ? t("cal.judge", { replaced: judge.replaced_contradictions, contradictions: judge.contradictions, wrong: judge.false_replacements.length, compatible: judge.compatible }) : "",
      ),
      judge?.false_replacements.length
        ? h("ul", { class: "small" }, judge.false_replacements.map((pair) => h("li", {}, pair)))
        : null,
      report.warnings.length
        ? h("ul", { class: "warnings" }, report.warnings.map((w) => h("li", {}, w)))
        : null,
      h("details", {},
        h("summary", { class: "small" }, t("cal.distribution")),
        h("table", { class: "thresholds scores" },
          h("thead", {}, h("tr", {}, h("th", {}, ""), h("th", {}, t("cal.min")), h("th", {}, t("cal.mean")), h("th", {}, t("cal.max")))),
          h("tbody", {}, scores),
        ),
      ),
      h("div", { class: "row-actions start" },
        h("button", { class: "button primary", type: "button", onclick: () => applySuggested(report.suggested) }, t("cal.apply")),
        h("button", { class: "button", type: "button", onclick: () => $("#calibration-report").replaceChildren() }, t("cal.discard")),
      ),
    ),
  );
}

async function applySuggested(thresholds) {
  try {
    await api.setThresholds(thresholds);
    toast(t("cal.applied"), "info");
    $("#calibration-report").replaceChildren();
  } catch (error) {
    toast(describeError(error));
  }
  await refreshCalibration();
}

async function resetThresholds() {
  if (!confirm(t("cal.confirm_reset"))) return;
  try {
    await api.resetThresholds();
    toast(t("cal.reset_done"), "info");
  } catch (error) {
    toast(describeError(error));
  }
  await refreshCalibration();
}

// ---------------------------------------------------------------- layout

function closeDrawers() {
  el.app.classList.remove("sidebar-open");
}

const INPUT_MAX_HEIGHT = 200;

function autoResize() {
  el.input.style.height = "auto";
  el.input.style.height = `${Math.min(el.input.scrollHeight, INPUT_MAX_HEIGHT)}px`;
  el.input.style.overflowY = el.input.scrollHeight > INPUT_MAX_HEIGHT ? "auto" : "hidden";
}

// ---------------------------------------------------------------- first-run setup

let setupShown = false;

function maybeOpenSetup(ollama) {
  const missing = Object.values(ollama.models || {}).some((installed) => !installed);
  if (setupShown || !(missing || !ollama.reachable)) return;
  setupShown = true;
  renderSetup(ollama);
  openDialog($("#setup-dialog"));
}

function renderSetup(ollama) {
  $("#setup-text").textContent = ollama.reachable ? t("setup.intro") : t("setup.ollama_down", { url: ollama.url });
  $("#setup-pull").hidden = !ollama.reachable;
  $("#setup-error").hidden = true;
  setChildren($("#setup-models"),
    Object.entries(ollama.models || {}).map(([name, installed]) =>
      h("li", { class: "setup-model", "data-model": name },
        h("span", { class: `badge ${installed ? "ok" : "bad"}` }, installed ? t("setup.installed") : t("system.model_missing")),
        " ", h("code", {}, name),
        installed ? null : h("progress", { max: 100, hidden: true }),
        h("span", { class: "muted small setup-progress" }))));
}

async function runSetup() {
  const button = $("#setup-pull");
  const error = $("#setup-error");
  button.disabled = true;
  error.hidden = true;
  try {
    for await (const { event, data } of pullModels()) {
      const row = data.model && $(`#setup-models [data-model="${CSS.escape(data.model)}"]`);
      if (event === "progress" && row) {
        const bar = row.querySelector("progress");
        bar.hidden = data.percent == null;
        if (data.percent != null) bar.value = data.percent;
        row.querySelector(".setup-progress").textContent =
          data.percent != null ? t("setup.pulling", { model: data.model, p: data.percent }) : data.status;
      } else if (event === "model_done" && row) {
        row.querySelector(".badge").className = "badge ok";
        row.querySelector(".badge").textContent = t("setup.installed");
        row.querySelector("progress")?.remove();
        row.querySelector(".setup-progress").textContent = "";
      } else if (event === "error") {
        throw new Error(t("setup.failed", { model: data.model || "Ollama", error: data.detail }));
      } else if (event === "done") {
        toast(`${t("setup.done")} ${t("setup.restart_hint")}`, "info");
      }
    }
  } catch (err) {
    error.textContent = describeError(err);
    error.hidden = false;
  } finally {
    button.disabled = false;
    refreshHealth();
  }
}

async function recheckSetup() {
  try {
    renderSetup(await api.setupStatus());
  } catch (error) {
    toast(describeError(error));
  }
  refreshHealth();
}

// ---------------------------------------------------------------- brand

/** Names, logo, welcome screen, links and the language picker, from the brand. */
function applyBrand() {
  translatePage();
  const logo = $("#brand-logo");
  if (brand.logo.url) logo.replaceChildren(h("img", { src: brand.logo.url, alt: "" }), " ", brand.product_name);
  else logo.textContent = `${brand.logo.text} ${brand.product_name}`;

  const welcome = brand.welcome || {};
  $("#welcome-title").textContent = welcome.title ?? t("brand.welcome_title");
  $("#welcome-text").textContent = welcome.text ?? t("brand.welcome_text");
  setChildren($("#suggestions"),
    (welcome.suggestions ?? t("brand.suggestions")).map((text) =>
      h("button", { type: "button", class: "suggestion", onclick: () => send(text) }, text)));

  const links = Object.entries(brand.links || {});
  const nav = $("#brand-links");
  nav.hidden = !links.length;
  setChildren(nav, links.map(([label, url]) => h("a", { href: url, target: "_blank", rel: "noopener noreferrer" }, label)));

  const picker = $("#language");
  setChildren(picker, Object.entries(languages).map(([code, name]) =>
    h("option", { value: code, selected: code === locale }, name)));
  picker.addEventListener("change", () => setLanguage(picker.value));

  el.input.placeholder = t("chat.placeholder");
}

// ---------------------------------------------------------------- wiring

el.composer.addEventListener("submit", (event) => {
  event.preventDefault();
  const message = el.input.value;
  el.input.value = "";
  autoResize();
  send(message);
});
el.input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    el.composer.requestSubmit();
  }
});
el.input.addEventListener("input", autoResize);
el.stop.addEventListener("click", () => state.streaming?.abort());
el.newSession.addEventListener("click", newChat);

el.continueSession.addEventListener("click", continueConversation);
el.contextMeter.addEventListener("click", (event) => {
  event.stopPropagation();
  el.contextPanel.hidden = !el.contextPanel.hidden;
  el.contextMeter.setAttribute("aria-expanded", String(!el.contextPanel.hidden));
  if (!el.contextPanel.hidden) renderContextPanel();
});
document.addEventListener("click", (event) => {
  if (!el.contextPanel.hidden && !el.contextPanel.contains(event.target)) {
    el.contextPanel.hidden = true;
    el.contextMeter.setAttribute("aria-expanded", "false");
  }
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !el.contextPanel.hidden) el.contextPanel.hidden = true;
});

$("#open-sidebar").addEventListener("click", () => el.app.classList.add("sidebar-open"));
$("#scrim").addEventListener("click", closeDrawers);
$("#open-memory").addEventListener("click", openMemory);
$("#open-settings").addEventListener("click", () => openSettings());
el.health.addEventListener("click", () => openSettings("system"));
$("#refresh-health").addEventListener("click", refreshHealth);
el.settingsDialog.querySelectorAll(".tab").forEach((tab) => tab.addEventListener("click", () => selectTab(tab.dataset.tab)));

$("#setup-pull").addEventListener("click", runSetup);
$("#setup-recheck").addEventListener("click", recheckSetup);

for (const dialog of [el.memoryDialog, el.settingsDialog, $("#setup-dialog")]) {
  dialog.querySelectorAll("[data-close]").forEach((button) => button.addEventListener("click", () => dialog.close()));
  // Click on the backdrop (outside the dialog box) closes it.
  dialog.addEventListener("click", (event) => { if (event.target === dialog) dialog.close(); });
  dialog.addEventListener("close", () => {
    state.memory.editing = null;
    el.input.focus();
  });
}

// Memory manager controls
let searchTimer;
el.memorySearch.addEventListener("input", () => {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(refreshMemory, 300);
});
el.memorySource.addEventListener("change", refreshMemory);
el.memoryDialog.querySelectorAll("[data-status]").forEach((button) =>
  button.addEventListener("click", () => {
    state.memory.status = button.dataset.status;
    el.memoryDialog.querySelectorAll("[data-status]").forEach((b) => b.classList.toggle("active", b === button));
    refreshMemory();
  }),
);
$("#toggle-add").addEventListener("click", () => {
  el.addMemory.hidden = !el.addMemory.hidden;
  if (!el.addMemory.hidden) el.memoryText.focus();
});
$("#cancel-add").addEventListener("click", () => {
  el.addMemory.hidden = true;
  el.memoryText.value = "";
});
el.addMemory.addEventListener("submit", async (event) => {
  event.preventDefault();
  const optimize = optimizeEnabled();
  // With optimization the model splits facts itself, so free text is sent as one note;
  // without it, each line is stored as its own memory.
  const texts = optimize
    ? [el.memoryText.value.trim()].filter(Boolean)
    : el.memoryText.value.split("\n").map((t) => t.trim()).filter(Boolean);
  if (!texts.length) return;
  const submit = el.addMemory.querySelector("[type=submit]");
  submit.disabled = true;
  submit.textContent = optimize ? t("common.optimizing") : t("common.saving");
  try {
    const result = await api.addMemory(texts, optimize);
    showCuration(result, texts.length > 1 ? t("mem.saved_many", { n: texts.length }) : t("mem.saved"));
    el.memoryText.value = "";
    el.addMemory.hidden = true;
  } catch (error) {
    toast(describeError(error));
  } finally {
    submit.disabled = false;
    submit.textContent = t("common.save");
  }
  await refreshMemory();
});
el.addMemory.querySelector(".optimize-toggle").addEventListener("change", (event) => setOptimize(event.target.checked));
setOptimize(optimizeEnabled());

// Persist the preferences across reloads (a per-browser convenience only).
for (const input of [el.useMemory, el.useTools]) {
  try {
    const saved = localStorage.getItem(`origin:${input.id}`);
    if (saved !== null) input.checked = saved === "true";
  } catch {}
  input.addEventListener("change", () => {
    try { localStorage.setItem(`origin:${input.id}`, input.checked); } catch {}
    renderSettingsHint();
  });
}

$("#reindex-banner-open").addEventListener("click", () => {
  el.memoryDialog.close();
  openSettings("calibration");
});

applyBrand();
renderSettingsHint();
refreshCalibration();
refreshHealth();
refreshSessions();
refreshMemory();
el.input.focus();
