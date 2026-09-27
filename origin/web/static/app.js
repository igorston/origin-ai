import { api, ApiError, streamChat } from "./api.js";
import { renderMarkdown } from "./markdown.js";

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
  useMemory: $("#use-memory"),
  useTools: $("#use-tools"),
  health: $("#health"),
  memoryStats: $("#memory-stats"),
  memorySearch: $("#memory-search"),
  showArchived: $("#show-archived"),
  addMemory: $("#add-memory"),
  memoryText: $("#memory-text"),
  memoryList: $("#memory-list"),
  toolList: $("#tool-list"),
  systemInfo: $("#system-info"),
  toast: $("#toast"),
};

const state = {
  sessionId: null,
  streaming: null, // AbortController of the in-flight turn
};

// Tools that change stored memories; after they run, the memory panel is refreshed.
const MEMORY_TOOLS = new Set(["remember", "forget"]);

// ---------------------------------------------------------------- helpers

function h(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (key === "class") node.className = value;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else if (value !== false && value != null) node.setAttribute(key, value === true ? "" : value);
  }
  for (const child of children.flat()) {
    if (child != null && child !== false) node.append(child);
  }
  return node;
}

function toast(message, kind = "error") {
  el.toast.textContent = message;
  el.toast.className = `toast ${kind}`;
  el.toast.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => (el.toast.hidden = true), 5000);
}

function formatDate(iso) {
  if (!iso) return "";
  const date = new Date(iso);
  const sameDay = date.toDateString() === new Date().toDateString();
  return sameDay
    ? date.toLocaleTimeString("pt-BR", { hour: "2-digit", minute: "2-digit" })
    : date.toLocaleDateString("pt-BR", { day: "2-digit", month: "short" });
}

const seconds = (ms) => `${(ms / 1000).toFixed(1)} s`;

function scrollToBottom(force = false) {
  const m = el.messages;
  const nearBottom = m.scrollHeight - m.scrollTop - m.clientHeight < 120;
  if (force || nearBottom) m.scrollTop = m.scrollHeight;
}

function describeError(error) {
  if (error instanceof ApiError) return error.message;
  if (error.name === "TypeError") return "Não foi possível falar com o servidor do Origin.";
  return error.message || String(error);
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
    if (health.status !== "ok") setHealth("bad", "degradado", "Ollama inacessível ou modelo ausente");
    else if (!health.ready) setHealth("warn", "carregando", "Modelos fora da memória: a próxima resposta pode levar ~15 s");
    else setHealth("ok", "online", "Modelos carregados");
  } catch {
    setHealth("bad", "offline", "Servidor do Origin inacessível");
  }
  renderSystem(health);
  // Poll fast while models load, so the indicator turns green as soon as they are ready.
  healthTimer = setTimeout(refreshHealth, health?.status === "ok" && !health.ready ? 3000 : 30000);
}

function renderSystem(health) {
  if (!health) {
    el.systemInfo.replaceChildren(h("p", { class: "error-text" }, "Servidor do Origin inacessível."));
    return;
  }
  const { ollama } = health;
  const models = Object.entries(ollama.models || {}).map(([name, installed]) => {
    const loaded = ollama.loaded?.[name];
    const [kind, label] = !installed ? ["bad", "ausente"] : loaded ? ["ok", "carregado"] : ["warn", "em espera"];
    return h("li", {}, h("span", { class: `badge ${kind}` }, label), " ", h("code", {}, name));
  });
  el.systemInfo.replaceChildren(
    h("dl", { class: "facts" },
      h("dt", {}, "Status"), h("dd", {}, h("span", { class: health.status === "ok" ? "badge ok" : "badge bad" }, health.status)),
      h("dt", {}, "Versão"), h("dd", {}, health.version),
      h("dt", {}, "Ollama"), h("dd", {}, h("code", {}, ollama.url), " ", ollama.reachable ? "✓" : "✕ inacessível"),
    ),
    models.length ? h("div", {}, h("h3", {}, "Modelos"), h("ul", { class: "plain" }, models)) : null,
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
          h("span", { class: "session-title" }, s.title || "Nova conversa"),
          h("span", { class: "session-meta" }, `${formatDate(s.updated_at)} · ${s.message_count} msg`),
        ),
        h("button", { class: "session-delete", type: "button", title: "Apagar conversa", "aria-label": "Apagar conversa",
          onclick: (event) => { event.stopPropagation(); removeSession(s.id, s.title); } }, "✕"),
      ),
    ),
  );
  if (!sessions.length) el.sessionList.append(h("p", { class: "muted small" }, "Nenhuma conversa ainda."));
}

function resetChat(title = "Nova conversa") {
  el.chatTitle.textContent = title;
  el.messages.replaceChildren(el.empty);
  el.empty.hidden = false;
}

async function openSession(id) {
  if (state.streaming) return;
  closeDrawers();
  try {
    const session = await api.session(id);
    state.sessionId = id;
    resetChat(session.title || "Nova conversa");
    el.empty.hidden = session.messages.length > 0;
    for (const message of session.messages) {
      if (message.role === "user") addUserMessage(message.content, message.created_at);
      else addAssistantMessage({ text: message.content, toolCalls: message.tool_calls, at: message.created_at });
    }
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
  if (!confirm(`Apagar a conversa "${title || "Nova conversa"}"?`)) return;
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
  el.messages.append(
    h("article", { class: "message user" },
      h("div", { class: "bubble" }, text),
      h("div", { class: "meta" }, formatDate(at || new Date().toISOString())),
    ),
  );
}

function toolCard(call) {
  const args = Object.keys(call.args || {}).length ? JSON.stringify(call.args, null, 2) : "(sem argumentos)";
  return h("details", { class: "tool-call" },
    h("summary", {}, h("span", { class: "tool-icon" }, "🔧"), h("code", {}, call.name), h("span", { class: "tool-args" }, summarizeArgs(call.args))),
    h("div", { class: "tool-body" },
      h("div", { class: "label" }, "Argumentos"), h("pre", {}, args),
      h("div", { class: "label" }, "Resultado"), h("pre", {}, call.output),
    ),
  );
}

function summarizeArgs(args = {}) {
  const text = Object.values(args).map((v) => (typeof v === "string" ? v : JSON.stringify(v))).join(", ");
  return text.length > 60 ? `${text.slice(0, 57)}…` : text;
}

/** Create an assistant message; returns handles to update it while streaming. */
function addAssistantMessage({ text = "", toolCalls = [], at = null, pending = false } = {}) {
  el.empty.hidden = true;
  const tools = h("div", { class: "tool-calls" }, toolCalls.map(toolCard));
  const content = h("div", { class: "content" });
  const meta = h("div", { class: "meta" }, at ? formatDate(at) : "");
  const bubble = h("div", { class: "bubble" }, tools, content);
  const article = h("article", { class: `message assistant${pending ? " pending" : ""}` }, bubble, meta);
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
    fail(message) {
      pending = false;
      article.classList.remove("pending");
      article.classList.add("failed");
      render();
      content.append(h("p", { class: "error-text" }, `⚠ ${message}`));
    },
    get text() { return raw; },
  };
}

// ---------------------------------------------------------------- chat turn

async function send(message) {
  message = message.trim();
  if (!message || state.streaming) return;

  if (!state.sessionId) {
    try {
      state.sessionId = (await api.createSession()).id;
    } catch (error) {
      toast(describeError(error));
      return;
    }
  }

  addUserMessage(message);
  const reply = addAssistantMessage({ pending: true });
  scrollToBottom(true);
  setStreaming(new AbortController());

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
      } else if (event === "error") {
        throw new Error(data.detail);
      }
      scrollToBottom();
    }
    reply.finish();
  } catch (error) {
    if (error.name === "AbortError") reply.fail("Geração interrompida.");
    else reply.fail(describeError(error));
  } finally {
    const total = performance.now() - started;
    const parts = [formatDate(new Date().toISOString())];
    if (firstToken != null) parts.push(`1º token ${seconds(firstToken)}`);
    parts.push(`total ${seconds(total)}`);
    if (toolCount) parts.push(`${toolCount} tool${toolCount > 1 ? "s" : ""}`);
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
  el.input.disabled = Boolean(controller);
  if (!controller) el.input.focus();
}

// ---------------------------------------------------------------- memory panel

async function refreshMemory() {
  const query = el.memorySearch.value.trim();
  try {
    const [stats, items] = await Promise.all([
      api.memoryStats(),
      query ? api.searchMemory(query) : api.memories(),
    ]);
    el.memoryStats.textContent =
      `${stats.active} ativa${stats.active === 1 ? "" : "s"} · ${stats.archived} arquivada${stats.archived === 1 ? "" : "s"}`;
    const visible = items.filter((m) => el.showArchived.checked || !m.metadata.archived);
    el.memoryList.replaceChildren(...visible.map(memoryItem));
    if (!visible.length) {
      el.memoryList.append(h("li", { class: "muted small" }, query ? "Nenhuma memória relevante." : "Nenhuma memória salva."));
    }
  } catch (error) {
    el.memoryStats.textContent = describeError(error);
  }
}

function memoryItem(memory) {
  const meta = memory.metadata || {};
  const tags = [
    meta.source ? h("span", { class: "badge" }, meta.source) : null,
    memory.score != null ? h("span", { class: "badge" }, `score ${memory.score.toFixed(2)}`) : null,
    meta.archived ? h("span", { class: "badge warn", title: meta.superseded_by ? "Substituída por um fato mais novo" : "" }, "arquivada") : null,
    meta.created_at ? h("span", { class: "muted small" }, formatDate(meta.created_at)) : null,
  ];
  return h("li", { class: `memory${meta.archived ? " archived" : ""}` },
    h("p", {}, memory.content),
    h("div", { class: "memory-meta" }, tags,
      h("span", { class: "spacer" }),
      meta.archived
        ? h("button", { class: "link", type: "button", onclick: () => mutateMemory(() => api.restoreMemory(memory.id)) }, "restaurar")
        : null,
      h("button", { class: "link danger", type: "button",
        onclick: () => confirm(`Apagar "${memory.content}"?`) && mutateMemory(() => api.deleteMemory(memory.id)) }, "apagar"),
    ),
  );
}

async function mutateMemory(action) {
  try {
    await action();
    await refreshMemory();
  } catch (error) {
    toast(describeError(error));
  }
}

// ---------------------------------------------------------------- tools panel

async function refreshTools() {
  try {
    const tools = await api.tools();
    el.toolList.replaceChildren(
      ...tools.map((tool) =>
        h("li", { class: "tool" },
          h("code", { class: "tool-name" }, tool.name),
          h("p", { class: "tool-description" }, tool.description.trim()),
          Object.keys(tool.args).length
            ? h("div", { class: "muted small" }, "Argumentos: ", Object.keys(tool.args).map((a) => h("code", {}, a)).reduce((acc, c) => (acc.length ? [...acc, ", ", c] : [c]), []))
            : h("div", { class: "muted small" }, "Sem argumentos"),
        ),
      ),
    );
    if (!tools.length) el.toolList.append(h("li", { class: "muted" }, "Nenhuma tool carregada."));
  } catch (error) {
    el.toolList.replaceChildren(h("li", { class: "error-text" }, describeError(error)));
  }
}

// ---------------------------------------------------------------- layout

function closeDrawers() {
  el.app.classList.remove("sidebar-open", "panel-open");
}

function selectTab(name) {
  document.querySelectorAll(".tab").forEach((tab) => tab.classList.toggle("active", tab.dataset.tab === name));
  document.querySelectorAll(".tab-panel").forEach((panel) => (panel.hidden = panel.dataset.panel !== name));
  if (name === "memory") refreshMemory();
  if (name === "tools") refreshTools();
  if (name === "system") refreshHealth();
}

const INPUT_MAX_HEIGHT = 200;

function autoResize() {
  el.input.style.height = "auto";
  el.input.style.height = `${Math.min(el.input.scrollHeight, INPUT_MAX_HEIGHT)}px`;
  el.input.style.overflowY = el.input.scrollHeight > INPUT_MAX_HEIGHT ? "auto" : "hidden";
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
document.querySelectorAll(".suggestion").forEach((button) =>
  button.addEventListener("click", () => send(button.textContent)),
);

$("#open-sidebar").addEventListener("click", () => el.app.classList.add("sidebar-open"));
$("#toggle-panel").addEventListener("click", () => el.app.classList.toggle("panel-open"));
$("#close-panel").addEventListener("click", closeDrawers);
$("#scrim").addEventListener("click", closeDrawers);
el.health.addEventListener("click", () => {
  el.app.classList.add("panel-open");
  selectTab("system");
});
$("#refresh-health").addEventListener("click", refreshHealth);
document.querySelectorAll(".tab").forEach((tab) => tab.addEventListener("click", () => selectTab(tab.dataset.tab)));

let searchTimer;
el.memorySearch.addEventListener("input", () => {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(refreshMemory, 300);
});
el.showArchived.addEventListener("change", refreshMemory);
el.addMemory.addEventListener("submit", async (event) => {
  event.preventDefault();
  const texts = el.memoryText.value.split("\n").map((t) => t.trim()).filter(Boolean);
  if (!texts.length) return;
  await mutateMemory(() => api.addMemory(texts));
  el.memoryText.value = "";
});

// Persist the switches across reloads (a per-browser convenience only).
for (const input of [el.useMemory, el.useTools]) {
  try {
    const saved = localStorage.getItem(`origin:${input.id}`);
    if (saved !== null) input.checked = saved === "true";
  } catch {}
  input.addEventListener("change", () => {
    try { localStorage.setItem(`origin:${input.id}`, input.checked); } catch {}
  });
}

refreshHealth();
refreshSessions();
refreshMemory();
el.input.focus();
