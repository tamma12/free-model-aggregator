/** Dial the grader's gateway: every free model is reached on the Chat
 *  Completions route, POST $OPENAI_BASE_URL/chat/completions, with the
 *  trial's dummy key as a Bearer token. The gateway swaps it for the
 *  entrant's own OpenRouter key, which never enters the sandbox. */

export interface UpstreamFailure {
  status: number;
  code: string;
  model: string | null;
}

export interface GatewaySuccess {
  ok: true;
  /** The gateway's answer body, for a JSON call. */
  json: Record<string, unknown> | null;
  /** The gateway's raw SSE lines, for a streamed call, in arrival order. */
  sseLines: string[] | null;
}

export interface GatewayRefusal {
  ok: false;
  failure: UpstreamFailure;
}

export type GatewayResult = GatewaySuccess | GatewayRefusal;

/** The longest a single model call may take. The client waits at most 30
 *  seconds for an answer, so a call that runs longer is abandoned and the
 *  next model is tried instead. */
const CALL_TIMEOUT_MS = 25000;

/** Read a fetch Response body as text, whatever it holds. */
async function readBodyText(res: Response): Promise<string> {
  try {
    return await res.text();
  } catch {
    return "";
  }
}

/** The short error code the gateway's refusal carries, for the failures
 *  trail. A provider error body names its own code; anything else falls
 *  back to the HTTP status in words. */
function errorCode(status: number, bodyText: string): string {
  try {
    const parsed = JSON.parse(bodyText) as Record<string, unknown>;
    const err = parsed.error as Record<string, unknown> | undefined;
    if (err && typeof err === "object") {
      if (typeof err.code === "string" && err.code.length > 0) return err.code;
      if (typeof err.type === "string" && err.type.length > 0) return err.type;
    }
    if (typeof parsed.code === "string" && parsed.code.length > 0) return parsed.code;
  } catch {
    // Not JSON: fall through to the status.
  }
  if (status === 400) return "bad_request";
  if (status === 401) return "unauthorized";
  if (status === 403) return "forbidden";
  if (status === 404) return "not_found";
  if (status === 429) return "rate_limited";
  if (status >= 500) return "upstream_error";
  return "http_" + String(status);
}

/** One call to the gateway. `body` is the Chat Completions body; `stream`
 *  asks for server-sent events. A refusal is returned, never thrown, so the
 *  caller can fail over to the next model and report the trail. */
export async function dialModel(
  baseUrl: string,
  apiKey: string,
  body: Record<string, unknown>,
  stream: boolean,
): Promise<GatewayResult> {
  const model = typeof body.model === "string" ? body.model : null;
  const url = baseUrl.replace(/\/$/, "") + "/chat/completions";
  let res: Response;
  try {
    res = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: "Bearer " + apiKey,
        Accept: stream ? "text/event-stream" : "application/json",
      },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(CALL_TIMEOUT_MS),
    });
  } catch (err) {
    const timedOut = err instanceof Error && err.name === "TimeoutError";
    return {
      ok: false,
      failure: { status: timedOut ? 504 : 502, code: timedOut ? "timeout" : "network_error", model },
    };
  }
  if (!res.ok) {
    const text = await readBodyText(res);
    return { ok: false, failure: { status: res.status, code: errorCode(res.status, text), model } };
  }
  if (!stream) {
    const text = await readBodyText(res);
    let json: Record<string, unknown>;
    try {
      json = JSON.parse(text) as Record<string, unknown>;
    } catch {
      return { ok: false, failure: { status: 502, code: "bad_gateway_body", model } };
    }
    return { ok: true, json, sseLines: null };
  }
  const lines = await collectSseLines(res);
  return { ok: true, json: null, sseLines: lines };
}

/** Read a streamed response to its end and split it into its SSE data
 *  payloads, in arrival order. A line that is not a data line is kept out;
 *  the terminal "[DONE]" is kept as its own line so the caller can close
 *  the stream the same way the provider did. */
export async function collectSseLines(res: Response): Promise<string[]> {
  const lines: string[] = [];
  if (!res.body) return lines;
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const step = await reader.read();
    if (step.done) break;
    buffer += decoder.decode(step.value, { stream: true });
    let cut = buffer.indexOf("\n\n");
    while (cut >= 0) {
      const event = buffer.slice(0, cut);
      buffer = buffer.slice(cut + 2);
      for (const line of event.split("\n")) {
        if (line.startsWith("data:")) lines.push(line.slice(5).trimStart());
      }
      cut = buffer.indexOf("\n\n");
    }
  }
  buffer += decoder.decode();
  if (buffer.length > 0) {
    for (const line of buffer.split("\n")) {
      if (line.startsWith("data:")) lines.push(line.slice(5).trimStart());
    }
  }
  return lines;
}

/** The x-makefun-upstream-failures trail: at most 8 rows, in the order the
 *  calls refused, as a JSON string fit for a header value. */
export function failuresHeader(failures: UpstreamFailure[]): string {
  return JSON.stringify(failures.slice(0, 8));
}
