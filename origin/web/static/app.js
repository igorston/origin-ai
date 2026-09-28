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
const SOURCE_LABELS = { agent: "agente", manual: "manual" };

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
  setChildren(el.systemInfo,
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
          h("span", { class: "session-meta" },
            `${formatDate(s.updated_at)} · ${s.message_count} msg${s.status === "closed" ? " · encerrada" : ""}`),
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
  renderContext(null);
  setClosed(null);
}

async function openSession(id) {
  if (state.streaming) return;
  closeDrawers();
  try {
    const session = await api.session(id);
    state.sessionId = id;
    resetChat(session.title || "Nova conversa");
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
    setClosed(session.status === "closed" ? session.closed_reason || "limite de contexto atingido" : null);
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
  const args = Object.keys(call.args || {}).length ? JSON.stringify(call.args, null, 2) : "(sem argumentos)";
  return h("details", { class: "tool-call" },
    h("summary", {}, h("span", { class: "tool-icon" }, "🔧"), h("code", {}, call.name), h("span", { class: "tool-args" }, summarizeArgs(call.args))),
    h("div", { class: "tool-body" },
      h("div", { class: "label" }, "Argumentos"), h("pre", {}, args),
      h("div", { class: "label" }, "Resultado"), h("pre", {}, call.output),
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
      reply.fail(error.name === "AbortError" ? "Geração interrompida." : describeError(error));
    }
  } finally {
    const parts = [formatDate(new Date().toISOString())];
    if (firstToken != null) parts.push(`1º token ${seconds(firstToken)}`);
    parts.push(`total ${seconds(performance.now() - started)}`);
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
  el.input.disabled = Boolean(controller) || Boolean(state.closed);
  el.send.disabled = Boolean(state.closed);
  if (!controller && !state.closed) el.input.focus();
}

// ---------------------------------------------------------------- context window

const tokens = (n) => n.toLocaleString("pt-BR");
const WINDOW_SOURCES = {
  vram: "auto, pela VRAM",
  model: "auto, limite do modelo",
  config: "fixa (OLLAMA_NUM_CTX)",
  fallback: "padrão, VRAM desconhecida",
};
const CONTEXT_STATES = {
  ok: "Contexto folgado",
  warning: "Contexto enchendo",
  critical: "Contexto quase cheio",
  closed: "Conversa encerrada",
};
// Close to the operational limit: say so before the chat actually closes.
const COMPRESSIONS_WARNING = 2;

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
  el.contextLabel.textContent = share > 1 ? "cheio" : `${usage.measured ? "" : "~"}${Math.round(share * 100)}%`;
  el.contextMeter.title =
    `${CONTEXT_STATES[usage.state]}: ${tokens(usage.used)} de ${tokens(usage.window)} tokens ` +
    `(janela ${WINDOW_SOURCES[usage.window_source] || ""})` +
    (usage.compactions ? ` · otimizado ${usage.compactions}×` : "");
  if (!el.contextPanel.hidden) renderContextPanel();
  renderNotice();
}

function renderContextPanel() {
  const u = state.context;
  if (!u) return;
  // Before a turn the parts are estimates; after one, Ollama reports the real total.
  const parts = [
    ["base", "Instruções e tools", u.base],
    ["summary", "Resumo da conversa", u.summary],
    ["history", "Mensagens recentes", u.history],
    ["memory", "Memórias recuperadas", u.memory],
    ["message", "Mensagem atual", u.message],
  ].filter(([, , value]) => value > 0);
  const scale = Math.max(u.window, u.used, 1);
  const reserve = u.window - u.usable;
  const share = u.used / u.window;
  setChildren(el.contextPanel,
    h("h3", {}, `${CONTEXT_STATES[u.state]} · ${share > 1 ? "acima da janela" : `${Math.round(share * 100)}% da janela`}`),
    h("div", { class: "context-stack", title: "Composição estimada" },
      parts.map(([key, label, value]) =>
        h("span", { class: `seg-${key}`, style: `width:${(value / scale) * 100}%`, title: `${label}: ${tokens(value)}` })),
      h("span", { class: "seg-reserve", style: `width:${(reserve / scale) * 100}%;margin-left:auto`, title: `Reservado para a resposta: ${tokens(reserve)}` })),
    u.used > u.usable
      ? h("p", { class: "context-over" },
          u.used > u.window
            ? "A conversa já não cabe na janela: na próxima mensagem, as mensagens mais antigas serão resumidas."
            : "A conversa entrou no espaço reservado para a resposta: na próxima mensagem, as mais antigas serão resumidas.")
      : null,
    h("dl", { class: "context-rows" },
      h("dt", {}, u.measured ? "Em uso (medido)" : "Em uso (estimado)"), h("dd", {}, `${tokens(u.used)} / ${tokens(u.window)}`),
      h("dt", { title: "Quando acaba, as mensagens mais antigas são resumidas na próxima mensagem" }, "Livre antes da reserva"),
      h("dd", {}, tokens(Math.max(0, u.usable - u.used))),
      parts.map(([key, label, value]) => [
        h("dt", {}, h("span", { class: `swatch seg-${key}` }), label), h("dd", {}, `~${tokens(value)}`),
      ]),
      h("dt", {}, h("span", { class: "swatch seg-reserve" }), "Reservado para a resposta"), h("dd", {}, tokens(reserve)),
      h("dt", {}, "Mensagens resumidas"), h("dd", {}, String(u.summarized_messages)),
      h("dt", {}, "Otimizações"), h("dd", {}, String(u.compactions)),
      h("dt", { title: "Cada condensação do resumo perde detalhes; no limite, a conversa é encerrada" }, "Condensações do resumo"),
      h("dd", {}, `${u.compressions} / ${u.max_compressions}`),
    ),
    h("h4", {}, "Capacidade"),
    h("dl", { class: "context-rows" },
      h("dt", {}, "Janela"), h("dd", {}, `${tokens(u.window)} (${WINDOW_SOURCES[u.window_source] || u.window_source})`),
      u.model_limit ? [h("dt", {}, "Limite do modelo"), h("dd", { title: u.model }, tokens(u.model_limit))] : null,
      u.vram_limit != null
        ? [h("dt", { title: "Estimativa: VRAM total, menos outros programas, os dois modelos e o cache de contexto de cada token" }, "Cabe na VRAM"),
           h("dd", {}, `~${tokens(u.vram_limit)}`)]
        : null,
      u.gpu
        ? [h("dt", {}, "GPU"),
           h("dd", { title: u.gpu }, `${u.gpu.replace(/^NVIDIA\s+(GeForce\s+)?/i, "")}, ${String(u.vram_total_gb).replace(".", ",")} GB`)]
        : null,
    ),
    h("p", {},
      "Perto do limite, as mensagens mais antigas são resumidas automaticamente para o modelo ",
      "(o histórico completo continua salvo aqui). Fatos, nomes, números e códigos que você escreveu são preservados. ",
      "A conversa só é encerrada quando nem o resumo condensado cabe mais."),
  );
}

function renderNotice() {
  const u = state.context;
  el.contextNotice.classList.toggle("closed", Boolean(state.closed));
  if (state.closed) {
    el.contextNoticeText.textContent =
      `Esta conversa foi encerrada: ${state.closed}. O histórico está salvo. Continue numa nova conversa, que começa com o resumo desta.`;
    el.continueSession.hidden = false;
    el.contextNotice.hidden = false;
    return;
  }
  const nearLimit = u && u.max_compressions - u.compressions <= COMPRESSIONS_WARNING && u.compressions > 0;
  if (nearLimit) {
    const left = u.max_compressions - u.compressions;
    el.contextNoticeText.textContent =
      (left > 0
        ? `Conversa perto do limite operacional: o resumo pode ser condensado só mais ${left} vez${left === 1 ? "" : "es"}. `
        : `O resumo não pode mais ser condensado: a conversa será encerrada quando o contexto encher (${Math.round(u.percent * 100)}% agora). `) +
      "Se preferir, continue numa nova conversa agora (ela leva o resumo).";
  } else if (u?.state === "critical") {
    el.contextNoticeText.textContent =
      "Contexto quase cheio. Na próxima mensagem, as mais antigas serão resumidas automaticamente.";
  }
  el.continueSession.hidden = !nearLimit;
  el.contextNotice.hidden = !(nearLimit || u?.state === "critical");
}

function setClosed(reason) {
  state.closed = reason;
  el.input.disabled = Boolean(reason) || Boolean(state.streaming);
  el.send.disabled = Boolean(reason);
  el.input.placeholder = reason ? "Conversa encerrada: continue numa nova conversa" : "Mensagem para o Origin…";
  renderNotice();
}

/** Divider marking where the model's view switches from the summary to real messages. */
function contextDivider({ folded = 0, summarized = 0, inherited = false } = {}) {
  const label = inherited
    ? "↪ Continuação: o modelo recebeu o resumo da conversa anterior"
    : folded
      ? `🗜 Contexto otimizado: ${folded} mensage${folded === 1 ? "m resumida" : "ns resumidas"}` +
        (summarized > folded ? ` (${summarized} no total)` : "")
      : `🗜 As ${summarized} mensagens acima estão resumidas para o modelo`;
  const text = h("div", { class: "summary-text" }, "Carregando resumo…");
  const details = h("details", { class: "context-divider" }, h("summary", { title: "Ver o resumo que o modelo recebe" }, label), text);
  const sessionId = state.sessionId;
  details.addEventListener("toggle", async () => {
    if (!details.open) return;
    try {
      text.textContent = (await api.session(sessionId)).summary || "(vazio)";
    } catch (error) {
      text.textContent = describeError(error);
    }
  });
  return details;
}

async function continueConversation() {
  if (!state.sessionId || state.streaming) return;
  el.continueSession.disabled = true;
  el.continueSession.textContent = "Resumindo…";
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
    el.continueSession.textContent = "Continuar em nova conversa";
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
      `${stats.active} ativa${stats.active === 1 ? "" : "s"} · ${stats.archived} arquivada${stats.archived === 1 ? "" : "s"}`;
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
          query ? "Nenhuma memória corresponde à busca." : status === "archived" ? "Nenhuma memória arquivada." : "Nenhuma memória ainda. Diga ao Origin \"lembra que…\" ou adicione uma aqui."),
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
    meta.source ? h("span", { class: "badge" }, SOURCE_LABELS[meta.source] || meta.source) : null,
    meta.created_at ? h("span", { class: "muted small", title: new Date(meta.created_at).toLocaleString("pt-BR") }, formatDate(meta.created_at)) : null,
    meta.edited_at ? h("span", { class: "badge", title: `Editada em ${new Date(meta.edited_at).toLocaleString("pt-BR")}` }, "editada") : null,
    memory.score != null ? h("span", { class: "badge", title: "Similaridade com a busca" }, `score ${memory.score.toFixed(2)}`) : null,
    meta.archived ? h("span", { class: "badge warn" }, "arquivada") : null,
    supersededBy ? h("span", { class: "muted small" }, `substituída por "${supersededBy.content}"`) : null,
  ];

  const body = editing ? memoryEditor(memory) : h("p", { class: "memory-content", title: "Clique duas vezes para editar", ondblclick: () => startEdit(memory.id) }, memory.content);

  const actions = editing
    ? null
    : h("div", { class: "memory-actions" },
        h("button", { class: "link", type: "button", onclick: () => startEdit(memory.id) }, "editar"),
        h("button", { class: "link", type: "button",
          onclick: () => mutateMemory(() => api.updateMemory(memory.id, { archived: !meta.archived }), meta.archived ? "Memória restaurada." : "Memória arquivada.") },
          meta.archived ? "restaurar" : "arquivar"),
        h("button", { class: "link danger", type: "button",
          onclick: () => confirm(`Apagar definitivamente "${memory.content}"?`) && mutateMemory(() => api.deleteMemory(memory.id), "Memória apagada.") },
          "apagar"),
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
  const saveButton = h("button", { class: "button primary", type: "button" }, "Salvar");
  const cancelButton = h("button", { class: "button", type: "button" }, "Cancelar");

  let busy = false;
  const save = async () => {
    const content = textarea.value.trim();
    if (busy) return;
    if (!content) return toast("O texto da memória não pode ficar vazio.");
    if (content === memory.content && !optimize.checked) return cancel();
    busy = true;
    textarea.disabled = saveButton.disabled = cancelButton.disabled = true;
    saveButton.textContent = optimize.checked ? "Otimizando…" : "Salvando…";
    try {
      const result = await api.updateMemory(memory.id, { content, optimize: optimize.checked });
      state.memory.editing = null;
      showCuration(result, "Memória atualizada.");
    } catch (error) {
      toast(describeError(error));
      busy = false;
      textarea.disabled = saveButton.disabled = cancelButton.disabled = false;
      saveButton.textContent = "Salvar";
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
      h("span", { class: "muted small" }, "Enter salva · Esc cancela"),
      h("label", { class: "check optimize", title: "O modelo reescreve em fatos curtos e separados, sem perder detalhes; evita duplicatas e arquiva o que ficou desatualizado" },
        optimize, " ✨ Otimizar com IA"),
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
  if (result.normalized && saved.length > 1) parts.push(`dividida em ${saved.length} memórias`);
  else if (result.normalized && saved.length === 1) parts.push(`otimizada: "${saved[0].content}"`);
  if (duplicates.length === 1) parts.push(`já existia ("${duplicates[0].content}"), mesclada`);
  else if (duplicates.length > 1) parts.push(`${duplicates.length} já existiam, mescladas`);
  if (superseded.length) {
    parts.push(`substituiu ${superseded.length} memória${superseded.length > 1 ? "s" : ""} desatualizada${superseded.length > 1 ? "s" : ""}`);
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
            ? h("div", { class: "muted small" }, "Argumentos: ", Object.keys(tool.args).join(", "))
            : h("div", { class: "muted small" }, "Sem argumentos"),
        ),
      ),
    );
    if (!tools.length) el.toolList.append(h("li", { class: "muted" }, "Nenhuma tool carregada."));
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
    el.settingsHint.textContent = "reindexar";
    return;
  }
  const off = [!el.useMemory.checked && "memória", !el.useTools.checked && "tools"].filter(Boolean);
  el.settingsHint.textContent = off.length ? `${off.join(" e ")} off` : "";
}

// ---------------------------------------------------------------- memory calibration

const THRESHOLD_INFO = {
  dedup: ["Duplicata", "Acima disso, dois fatos são considerados o mesmo e não são salvos de novo."],
  conflict: ["Candidatos a substituição", "Acima disso, o modelo de chat avalia se o fato novo substitui o antigo."],
  min_score: ["Relevância na busca", "Memórias abaixo disso não entram no contexto da conversa."],
};
const SCORE_LABELS = {
  paraphrase: "Paráfrases (mesmo fato)",
  contradiction: "Contradições",
  compatible: "Fatos compatíveis",
  unrelated: "Sem relação",
  relevant: "Perguntas relevantes",
  irrelevant: "Perguntas irrelevantes",
};

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
  $("#reindex-banner-text").textContent = s.needs_reindex ? `A memória precisa ser reindexada: ${s.reason}.` : "";
  if (!el.settingsDialog.open) return;

  const dims = (dim) => (dim ? ` · ${dim} dimensões` : "");
  const thresholds = Object.entries(THRESHOLD_INFO).map(([key, [label, help]]) =>
    h("tr", {},
      h("th", { title: help }, label),
      h("td", {}, h("code", {}, s.thresholds[key].toFixed(3))),
      h("td", { class: "muted small" }, s.thresholds[key] === s.defaults[key] ? "padrão" : `padrão ${s.defaults[key].toFixed(3)}`),
    ),
  );
  setChildren($("#calibration-status"),
    s.needs_reindex
      ? h("div", { class: "banner warn" },
          h("span", {}, `As ${s.count} memórias precisam ser reindexadas: ${s.reason}. Até lá, a busca na memória não funciona.`),
          h("button", { class: "button primary", type: "button", onclick: reindex }, "Reindexar agora"),
        )
      : null,
    h("dl", { class: "facts" },
      h("dt", {}, "Embeddings"), h("dd", {}, h("code", {}, s.embed_model), dims(s.current_dim)),
      h("dt", {}, "Indexadas com"), h("dd", {}, s.indexed_model ? h("code", {}, s.indexed_model) : "—", dims(s.stored_dim), ` · ${s.count} memória${s.count === 1 ? "" : "s"}`),
      h("dt", {}, "Modelo de chat"), h("dd", {}, h("code", {}, s.chat_model)),
      h("dt", {}, "Calibração"), h("dd", {},
        s.calibrated
          ? h("span", { class: "badge ok" }, `calibrado em ${new Date(s.calibrated_at).toLocaleString("pt-BR")}`)
          : h("span", { class: "badge warn" }, "limiares padrão, não calibrados para este modelo")),
    ),
    h("table", { class: "thresholds" }, h("tbody", {}, thresholds)),
    h("div", { class: "row-actions start" },
      h("button", { class: "button primary", type: "button", id: "run-calibration", onclick: calibrate }, "Calibrar"),
      !s.needs_reindex && s.count ? h("button", { class: "button", type: "button", onclick: reindex }, "Reindexar") : null,
      s.calibrated ? h("button", { class: "button", type: "button", onclick: resetThresholds }, "Restaurar padrões") : null,
    ),
  );
}

async function reindex() {
  const s = state.calibration;
  if (!confirm(`Recalcular os vetores de ${s.count} memória(s) com ${s.embed_model}? Um backup em JSON é salvo antes.`)) return;
  toast("Reindexando…", "info");
  try {
    const result = await api.reindexMemory();
    toast(`${result.reindexed} memória(s) reindexada(s) em ${result.duration_s} s. Agora calibre os limiares.`, "info");
  } catch (error) {
    toast(describeError(error));
  }
  await refreshCalibration();
  refreshMemory();
}

async function calibrate() {
  const button = $("#run-calibration");
  button.disabled = true;
  button.textContent = "Calibrando…";
  try {
    renderReport(await api.runCalibration(false));
  } catch (error) {
    toast(describeError(error));
  } finally {
    button.disabled = false;
    button.textContent = "Calibrar";
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
    h("tr", {}, h("th", {}, SCORE_LABELS[key] || key), h("td", {}, st.min.toFixed(2)), h("td", {}, st.mean.toFixed(2)), h("td", {}, st.max.toFixed(2))),
  );
  const judge = report.judge;
  $("#calibration-report").replaceChildren(
    h("div", { class: "report" },
      h("h3", {}, `Resultado · ${report.embed_model} + ${report.chat_model} · ${report.duration_s} s`),
      h("table", { class: "thresholds" },
        h("thead", {}, h("tr", {}, h("th", {}, ""), h("th", {}, "Em uso"), h("th", {}, "Sugerido"))),
        h("tbody", {}, rows),
      ),
      h("p", { class: "small" },
        `Busca: ${report.recall_hits}/${report.recall_total} perguntas encontram seu fato com o limiar sugerido.`,
        judge ? ` Substituição: ${judge.replaced_contradictions}/${judge.contradictions} contradições reconhecidas, ${judge.false_replacements.length}/${judge.compatible} fatos verdadeiros seriam arquivados.` : "",
      ),
      judge?.false_replacements.length
        ? h("ul", { class: "small" }, judge.false_replacements.map((pair) => h("li", {}, pair)))
        : null,
      report.warnings.length
        ? h("ul", { class: "warnings" }, report.warnings.map((w) => h("li", {}, w)))
        : null,
      h("details", {},
        h("summary", { class: "small" }, "Distribuição de similaridade"),
        h("table", { class: "thresholds scores" },
          h("thead", {}, h("tr", {}, h("th", {}, ""), h("th", {}, "mín"), h("th", {}, "média"), h("th", {}, "máx"))),
          h("tbody", {}, scores),
        ),
      ),
      h("div", { class: "row-actions start" },
        h("button", { class: "button primary", type: "button", onclick: () => applySuggested(report.suggested) }, "Aplicar sugeridos"),
        h("button", { class: "button", type: "button", onclick: () => $("#calibration-report").replaceChildren() }, "Descartar"),
      ),
    ),
  );
}

async function applySuggested(thresholds) {
  try {
    await api.setThresholds(thresholds);
    toast("Limiares calibrados aplicados.", "info");
    $("#calibration-report").replaceChildren();
  } catch (error) {
    toast(describeError(error));
  }
  await refreshCalibration();
}

async function resetThresholds() {
  if (!confirm("Voltar aos limiares padrão do .env?")) return;
  try {
    await api.resetThresholds();
    toast("Limiares padrão restaurados.", "info");
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
document.querySelectorAll(".suggestion").forEach((button) => button.addEventListener("click", () => send(button.textContent)));

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

for (const dialog of [el.memoryDialog, el.settingsDialog]) {
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
  submit.textContent = optimize ? "Otimizando…" : "Salvando…";
  try {
    const result = await api.addMemory(texts, optimize);
    showCuration(result, texts.length > 1 ? `${texts.length} memórias salvas.` : "Memória salva.");
    el.memoryText.value = "";
    el.addMemory.hidden = true;
  } catch (error) {
    toast(describeError(error));
  } finally {
    submit.disabled = false;
    submit.textContent = "Salvar";
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

renderSettingsHint();
refreshCalibration();
refreshHealth();
refreshSessions();
refreshMemory();
el.input.focus();
