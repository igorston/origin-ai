// Thin client for the Origin HTTP API.

export class ApiError extends Error {
  constructor(status, detail) {
    super(detail || `HTTP ${status}`);
    this.status = status;
  }
}

async function errorDetail(response) {
  try {
    const body = await response.json();
    if (typeof body.detail === "string") return body.detail;
    if (Array.isArray(body.detail)) return body.detail.map((d) => d.msg).join("; ");
    return JSON.stringify(body);
  } catch {
    return response.statusText;
  }
}

async function request(method, path, body) {
  const response = await fetch(path, {
    method,
    headers: body === undefined ? {} : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!response.ok) throw new ApiError(response.status, await errorDetail(response));
  return response.status === 204 ? null : response.json();
}

export const api = {
  // /health answers 503 with a JSON body when degraded, so read it either way.
  health: async () => (await fetch("/health")).json(),

  sessions: () => request("GET", "/sessions"),
  session: (id) => request("GET", `/sessions/${id}`),
  createSession: (title = "") => request("POST", "/sessions", { title }),
  deleteSession: (id) => request("DELETE", `/sessions/${id}`),

  memories: (limit = 500) => request("GET", `/memory?limit=${limit}`),
  searchMemory: (q, k = 10) =>
    request("GET", `/memory/search?${new URLSearchParams({ q, k })}`),
  addMemory: (texts, optimize = false) =>
    request("POST", "/memory", { texts, metadata: { source: "manual" }, optimize }),
  deleteMemory: (id) => request("DELETE", `/memory/${id}`),
  updateMemory: (id, patch) => request("PATCH", `/memory/${id}`, patch),
  restoreMemory: (id) => request("POST", `/memory/${id}/restore`),
  memoryStats: () => request("GET", "/memory/stats"),

  tools: () => request("GET", "/tools"),

  calibrationStatus: () => request("GET", "/memory/calibration"),
  runCalibration: (apply = false) => request("POST", "/memory/calibration/run", { apply }),
  setThresholds: (thresholds) => request("PUT", "/memory/calibration/thresholds", thresholds),
  resetThresholds: () => request("DELETE", "/memory/calibration/thresholds"),
  reindexMemory: () => request("POST", "/memory/reindex"),
};

/** Parse one SSE block ("event: x\ndata: {...}") into {event, data}. */
export function parseSseBlock(block) {
  let event = "message";
  const data = [];
  for (const line of block.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) data.push(line.slice(5).trimStart());
  }
  return data.length ? { event, data: JSON.parse(data.join("\n")) } : null;
}

/**
 * POST /chat/events and yield its Server-Sent Events. EventSource only supports GET,
 * so the stream is read and split manually.
 */
export async function* streamChat(body, signal) {
  const response = await fetch("/chat/events", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!response.ok) throw new ApiError(response.status, await errorDetail(response));

  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += value.replace(/\r\n/g, "\n");
    let boundary;
    while ((boundary = buffer.indexOf("\n\n")) !== -1) {
      const event = parseSseBlock(buffer.slice(0, boundary));
      buffer = buffer.slice(boundary + 2);
      if (event) yield event;
    }
  }
}
