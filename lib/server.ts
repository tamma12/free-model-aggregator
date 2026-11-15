/** The aggregator's HTTP server: the two completion wires and the model
 *  list. One prompt is served by dialling the best model that meets its
 *  requirements, failing over down the rating order when a call is refused,
 *  and reporting every refusal in the x-makefun-upstream-failures header.
 *  The server never answers from its own text: every completion is a
 *  gateway answer, returned whole in the wire the client used. */

import http from "node:http";
import { candidatesFor, type FreeModel } from "./catalog.ts";
import { dialModel, failuresHeader, type UpstreamFailure } from "./gateway.ts";
import {
  anthropicStreamEvents,
  buildAnthropicAnswer,
  buildOpenAiAnswer,
  gatewayBody,
  parseAnthropic,
  parseOpenAi,
  rewriteOpenAiChunk,
  Stamps,
  type CallPlan,
} from "./wire.ts";

export interface ServerOptions {
  models: FreeModel[];
  stamps: Stamps;
  openAiBaseUrl: string;
  openAiApiKey: string;
}

/** The largest request body the server reads: the env's prompts are short,
 *  and the gateway refuses bodies over 8 MiB. */
const MAX_BODY_BYTES = 8 * 1024 * 1024;

/** Read one request body, whole, as text. */
function readBody(req: http.IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    let total = 0;
    req.on("data", (chunk: Buffer) => {
      total += chunk.length;
      if (total > MAX_BODY_BYTES) {
        reject(new Error("request too large"));
        req.destroy();
        return;
      }
      chunks.push(chunk);
    });
    req.on("end", () => resolve(Buffer.concat(chunks).toString("utf8")));
    req.on("error", reject);
  });
}

/** A JSON answer with the failures trail attached when there is one. */
function sendJson(
  res: http.ServerResponse,
  status: number,
  payload: Record<string, unknown>,
  failures: UpstreamFailure[],
): void {
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (failures.length > 0) headers["x-makefun-upstream-failures"] = failuresHeader(failures);
  res.writeHead(status, headers);
  res.end(JSON.stringify(payload));
}

/** The OpenAI-shaped error for a prompt no model would serve. */
function openAiError(failures: UpstreamFailure[]): Record<string, unknown> {
  return {
    error: {
      message: "all models refused the prompt",
      type: "server_error",
      code: "all_models_failed",
    },
  };
}

/** The Anthropic-shaped error for a prompt no model would serve. */
function anthropicError(): Record<string, unknown> {
  return {
    type: "error",
    error: { type: "api_error", message: "all models refused the prompt" },
  };
}

/** Serve one prompt: try the candidates in rating order until one answers,
 *  then return its answer in the client's wire. Returns the failures the
 *  attempts left behind, so the caller can report the trail. */
async function servePrompt(
  plan: CallPlan,
  wire: "openai" | "anthropic",
  options: ServerOptions,
  res: http.ServerResponse,
): Promise<void> {
  const candidates = candidatesFor(options.models, plan.needs);
  const failures: UpstreamFailure[] = [];
  for (const candidate of candidates) {
    const body = gatewayBody(plan, candidate.id, plan.stream);
    const result = await dialModel(options.openAiBaseUrl, options.openAiApiKey, body, plan.stream);
    if (!result.ok) {
      failures.push(result.failure);
      continue;
    }
    if (wire === "openai") {
      const responseId = options.stamps.chatId();
      if (!plan.stream) {
        const answer = buildOpenAiAnswer(
          result.json ?? {},
          candidate.id,
          options.stamps,
          responseId,
          plan.declaredTools,
        );
        sendJson(res, 200, answer, failures);
        return;
      }
      const headers: Record<string, string> = {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        Connection: "keep-alive",
      };
      if (failures.length > 0) headers["x-makefun-upstream-failures"] = failuresHeader(failures);
      res.writeHead(200, headers);
      const toolNamesByIndex = new Map<number, string>();
      const surviving = { count: 0 };
      let sawDone = false;
      for (const line of result.sseLines ?? []) {
        const rewritten = rewriteOpenAiChunk(
          line,
          candidate.id,
          options.stamps,
          responseId,
          plan.declaredTools,
          toolNamesByIndex,
          surviving,
        );
        if (rewritten === null) continue;
        if (rewritten === "[DONE]") sawDone = true;
        res.write("data: " + rewritten + "\n\n");
      }
      if (!sawDone) res.write("data: [DONE]\n\n");
      res.end();
      return;
    }
    const responseId = options.stamps.msgId();
    if (!plan.stream) {
      const answer = buildAnthropicAnswer(
        result.json ?? {},
        candidate.id,
        responseId,
        plan.declaredTools,
      );
      sendJson(res, 200, answer, failures);
      return;
    }
    const headers: Record<string, string> = {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
    };
    if (failures.length > 0) headers["x-makefun-upstream-failures"] = failuresHeader(failures);
    res.writeHead(200, headers);
    const events = anthropicStreamEvents(
      result.sseLines ?? [],
      candidate.id,
      options.stamps,
      responseId,
      plan.declaredTools,
    );
    for (const event of events) res.write(event + "\n\n");
    res.end();
    return;
  }
  if (wire === "openai") sendJson(res, 502, openAiError(failures), failures);
  else sendJson(res, 502, anthropicError(), failures);
}

/** The OpenAI model list: the fourteen dialable ids, once each. */
function openAiModels(options: ServerOptions): Record<string, unknown> {
  return {
    object: "list",
    data: options.models.map((m) => ({
      id: m.id,
      object: "model",
      created: options.stamps.epoch,
      owned_by: "openrouter",
    })),
  };
}

/** The Anthropic model list: the same fourteen ids, in its shape. */
function anthropicModels(options: ServerOptions): Record<string, unknown> {
  const data = options.models.map((m) => ({
    id: m.id,
    type: "model",
    display_name: m.displayName,
    created_at: options.stamps.iso,
  }));
  return {
    data,
    has_more: false,
    first_id: data.length > 0 ? data[0]?.id : null,
    last_id: data.length > 0 ? data[data.length - 1]?.id : null,
  };
}

/** The server. The wire is the auth header: Authorization: Bearer is
 *  OpenAI, x-api-key is Anthropic. A request that names neither still gets
 *  the OpenAI shape, which is the gateway's own wire. */
export function createServer(options: ServerOptions): http.Server {
  return http.createServer((req, res) => {
    void (async () => {
      const url = new URL(req.url ?? "/", "http://127.0.0.1");
      const path = url.pathname;
      try {
        if (req.method === "GET" && path === "/v1/models") {
          const auth = req.headers.authorization ?? "";
          const apiKey = req.headers["x-api-key"];
          if (typeof apiKey === "string" && apiKey.length > 0 && !auth.startsWith("Bearer ")) {
            sendJson(res, 200, anthropicModels(options), []);
          } else {
            sendJson(res, 200, openAiModels(options), []);
          }
          return;
        }
        if (req.method === "POST" && path === "/v1/chat/completions") {
          const text = await readBody(req);
          let body: Record<string, unknown>;
          try {
            body = JSON.parse(text) as Record<string, unknown>;
          } catch {
            sendJson(
              res,
              400,
              { error: { message: "request body is not JSON", type: "invalid_request_error", code: "invalid_json" } },
              [],
            );
            return;
          }
          await servePrompt(parseOpenAi(body), "openai", options, res);
          return;
        }
        if (req.method === "POST" && path === "/v1/messages") {
          const text = await readBody(req);
          let body: Record<string, unknown>;
          try {
            body = JSON.parse(text) as Record<string, unknown>;
          } catch {
            sendJson(res, 400, { type: "error", error: { type: "invalid_request_error", message: "request body is not JSON" } }, []);
            return;
          }
          await servePrompt(parseAnthropic(body), "anthropic", options, res);
          return;
        }
        sendJson(res, 404, { error: { message: "not found", type: "invalid_request_error", code: "not_found" } }, []);
      } catch (err) {
        if (!res.headersSent) {
          sendJson(
            res,
            500,
            { error: { message: err instanceof Error ? err.message : "internal error", type: "server_error", code: "internal_error" } },
            [],
          );
        } else {
          res.end();
        }
      }
    })();
  });
}
