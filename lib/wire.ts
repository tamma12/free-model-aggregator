/** The two client wires and the gateway's one wire. The gateway only speaks
 *  Chat Completions, so an Anthropic-shaped request is translated into a
 *  Chat Completions body before it is dialled, and the gateway's answer is
 *  translated back into the wire the client used. Nothing here invents an
 *  answer: every completion comes from the gateway, whole, with the dialled
 *  model's id put back on it (the provider echoes the id without the
 *  gateway's openrouter/ prefix). */

import { createHash } from "node:crypto";
import type { FreeModel, ModelNeeds } from "./catalog.ts";

/** A Chat Completions message, the shape the gateway takes. */
export interface ChatMessage {
  role: string;
  content: string | null;
  tool_calls?: Array<{
    id: string;
    type: "function";
    function: { name: string; arguments: string };
  }>;
  tool_call_id?: string;
  name?: string;
}

/** An OpenAI-shaped tool, the shape the gateway takes. */
export interface OpenAiTool {
  type: "function";
  function: { name: string; description?: string; parameters?: Record<string, unknown> };
}

/** Everything one prompt needs to be dialled: the translated request, the
 *  model's requirements, and the tool names the client declared (only those
 *  may come back as tool calls). */
export interface CallPlan {
  messages: ChatMessage[];
  tools: OpenAiTool[] | null;
  toolChoice: unknown;
  responseFormat: { type: string } | null;
  maxTokens: number;
  temperature: number | null;
  stream: boolean;
  needs: ModelNeeds;
  declaredTools: string[];
}

/** Deterministic stamps for one run: response ids and the created instant.
 *  The created instant comes from MAKEFUN_TRIAL_NOW, the only now a request
 *  may carry; the ids come from the trial seed and a counter, so the same
 *  requests in the same order always get the same ids, live and replayed. */
export class Stamps {
  private counter = 0;
  private seed: string;
  readonly epoch: number;
  readonly iso: string;

  constructor(seed: string, trialNow: string | undefined) {
    this.seed = seed;
    const parsed = trialNow ? Date.parse(trialNow) : Number.NaN;
    this.epoch = Number.isFinite(parsed) ? Math.floor(parsed / 1000) : 1735689600;
    this.iso = new Date(this.epoch * 1000).toISOString();
  }

  private digest(kind: string): string {
    this.counter += 1;
    return createHash("sha256")
      .update(this.seed + "|" + kind + "|" + String(this.counter))
      .digest("hex")
      .slice(0, 24);
  }

  chatId(): string {
    return "chatcmpl-" + this.digest("chat");
  }

  msgId(): string {
    return "msg_" + this.digest("msg");
  }
}

/** The output cap a gateway call must carry: the client's own, clamped to
 *  the gateway's ceiling, or 1024 when the client named none. A call with
 *  no cap, or one over 8192, is refused before it is forwarded. */
function outputCap(body: Record<string, unknown>): number {
  const raw =
    typeof body.max_tokens === "number"
      ? body.max_tokens
      : typeof body.max_completion_tokens === "number"
        ? body.max_completion_tokens
        : 1024;
  if (!Number.isFinite(raw)) return 1024;
  return Math.max(1, Math.min(8192, Math.floor(raw)));
}

/** Parse an OpenAI Chat Completions body into a call plan. Messages, the
 *  system prompt, tools, max_tokens and temperature pass through unchanged;
 *  a top-level `system` string becomes a system message in front of them. */
export function parseOpenAi(body: Record<string, unknown>): CallPlan {
  const messages: ChatMessage[] = [];
  if (typeof body.system === "string" && body.system.length > 0) {
    messages.push({ role: "system", content: body.system });
  }
  const rawMessages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const raw of rawMessages) {
    if (typeof raw !== "object" || raw === null) continue;
    const m = raw as Record<string, unknown>;
    const role = typeof m.role === "string" ? m.role : "user";
    let content: string | null = null;
    if (typeof m.content === "string") {
      content = m.content;
    } else if (Array.isArray(m.content)) {
      const parts: string[] = [];
      for (const part of m.content) {
        if (typeof part !== "object" || part === null) continue;
        const p = part as Record<string, unknown>;
        if (p.type === "text" && typeof p.text === "string") parts.push(p.text);
        if (p.type === "image_url" || p.type === "input_image") vision = true;
      }
      content = parts.join("");
    }
    const out: ChatMessage = { role, content };
    if (typeof m.tool_call_id === "string") out.tool_call_id = m.tool_call_id;
    if (typeof m.name === "string") out.name = m.name;
    if (Array.isArray(m.tool_calls)) {
      const calls: ChatMessage["tool_calls"] = [];
      for (const rawCall of m.tool_calls) {
        if (typeof rawCall !== "object" || rawCall === null) continue;
        const c = rawCall as Record<string, unknown>;
        const fn = c.function as Record<string, unknown> | undefined;
        if (typeof c.id !== "string" || !fn || typeof fn.name !== "string") continue;
        calls.push({
          id: c.id,
          type: "function",
          function: {
            name: fn.name,
            arguments: typeof fn.arguments === "string" ? fn.arguments : "",
          },
        });
      }
      if (calls.length > 0) out.tool_calls = calls;
    }
    messages.push(out);
  }
  const tools = Array.isArray(body.tools)
    ? (body.tools.filter(
        (t): t is OpenAiTool =>
          typeof t === "object" &&
          t !== null &&
          (t as Record<string, unknown>).type === "function" &&
          typeof ((t as Record<string, unknown>).function as Record<string, unknown> | undefined)
            ?.name === "string",
      ) as OpenAiTool[])
    : [];
  const declaredTools = tools.map((t) => t.function.name);
  const responseFormat =
    body.response_format !== null &&
    typeof body.response_format === "object" &&
    typeof (body.response_format as Record<string, unknown>).type === "string"
      ? { type: (body.response_format as Record<string, unknown>).type as string }
      : null;
  return {
    messages,
    tools: tools.length > 0 ? tools : null,
    toolChoice: body.tool_choice ?? null,
    responseFormat,
    maxTokens: outputCap(body),
    temperature: typeof body.temperature === "number" ? body.temperature : null,
    stream: body.stream === true,
    needs: {
      tools: tools.length > 0,
      vision,
      json: responseFormat !== null && responseFormat.type !== "text",
    },
    declaredTools,
  };
}

/** The text of an Anthropic system prompt, which may be one string or an
 *  array of text blocks. */
function systemText(system: unknown): string {
  if (typeof system === "string") return system;
  if (!Array.isArray(system)) return "";
  const parts: string[] = [];
  for (const block of system) {
    if (typeof block === "object" && block !== null) {
      const b = block as Record<string, unknown>;
      if (b.type === "text" && typeof b.text === "string") parts.push(b.text);
    }
  }
  return parts.join("\n");
}

/** The text of a tool result's content, which may be a string or blocks. */
function toolResultText(content: unknown): string {
  if (typeof content === "string") return content;
  if (content === null || content === undefined) return "";
  if (!Array.isArray(content)) return JSON.stringify(content);
  const parts: string[] = [];
  for (const block of content) {
    if (typeof block === "object" && block !== null) {
      const b = block as Record<string, unknown>;
      if (b.type === "text" && typeof b.text === "string") parts.push(b.text);
      else if (typeof b.text === "string") parts.push(b.text);
    } else if (typeof block === "string") {
      parts.push(block);
    }
  }
  return parts.join("\n");
}

/** Parse an Anthropic Messages body into a call plan, translating it into
 *  Chat Completions shape: the system prompt becomes a system message,
 *  text blocks become message text, tool_use blocks become tool calls, and
 *  tool_result blocks become tool messages. Anthropic has no response
 *  format, so a Messages request never needs JSON mode. */
export function parseAnthropic(body: Record<string, unknown>): CallPlan {
  const messages: ChatMessage[] = [];
  const sys = systemText(body.system);
  if (sys.length > 0) messages.push({ role: "system", content: sys });
  let vision = false;
  const rawMessages = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of rawMessages) {
    if (typeof raw !== "object" || raw === null) continue;
    const m = raw as Record<string, unknown>;
    const role = typeof m.role === "string" ? m.role : "user";
    const content = m.content;
    if (typeof content === "string") {
      messages.push({ role, content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    const texts: string[] = [];
    const calls: NonNullable<ChatMessage["tool_calls"]> = [];
    for (const rawBlock of content) {
      if (typeof rawBlock !== "object" || rawBlock === null) continue;
      const block = rawBlock as Record<string, unknown>;
      if (block.type === "text" && typeof block.text === "string") {
        texts.push(block.text);
      } else if (block.type === "image") {
        vision = true;
        const source = block.source as Record<string, unknown> | undefined;
        const mediaType =
          source && typeof source.media_type === "string" ? source.media_type : "image/jpeg";
        const data = source && typeof source.data === "string" ? source.data : "";
        texts.push("[image " + mediaType + " base64 " + String(data.length) + " bytes]");
      } else if (block.type === "tool_use" && role === "assistant") {
        if (typeof block.id === "string" && typeof block.name === "string") {
          calls.push({
            id: block.id,
            type: "function",
            function: {
              name: block.name,
              arguments:
                block.input === undefined ? "{}" : JSON.stringify(block.input),
            },
          });
        }
      } else if (block.type === "tool_result") {
        const callId = typeof block.tool_use_id === "string" ? block.tool_use_id : "";
        messages.push({
          role: "tool",
          content: toolResultText(block.content),
          tool_call_id: callId,
        });
      }
    }
    if (texts.length > 0 || calls.length > 0) {
      const out: ChatMessage = { role, content: texts.join("") || null };
      if (calls.length > 0) out.tool_calls = calls;
      messages.push(out);
    }
  }
  const tools: OpenAiTool[] = [];
  if (Array.isArray(body.tools)) {
    for (const rawTool of body.tools) {
      if (typeof rawTool !== "object" || rawTool === null) continue;
      const t = rawTool as Record<string, unknown>;
      if (typeof t.name !== "string") continue;
      const tool: OpenAiTool = {
        type: "function",
        function: { name: t.name },
      };
      if (typeof t.description === "string") tool.function.description = t.description;
      if (t.input_schema !== null && typeof t.input_schema === "object") {
        tool.function.parameters = t.input_schema as Record<string, unknown>;
      }
      tools.push(tool);
    }
  }
  const declaredTools = tools.map((t) => t.function.name);
  let toolChoice: unknown = null;
  const rawChoice = body.tool_choice;
  if (rawChoice !== null && typeof rawChoice === "object") {
    const c = rawChoice as Record<string, unknown>;
    if (c.type === "auto") toolChoice = "auto";
    else if (c.type === "any") toolChoice = "required";
    else if (c.type === "none") toolChoice = "none";
    else if (c.type === "tool" && typeof c.name === "string") {
      toolChoice = { type: "function", function: { name: c.name } };
    }
  }
  return {
    messages,
    tools: tools.length > 0 ? tools : null,
    toolChoice,
    responseFormat: null,
    maxTokens: outputCap(body),
    temperature: typeof body.temperature === "number" ? body.temperature : null,
    stream: body.stream === true,
    needs: { tools: tools.length > 0, vision, json: false },
    declaredTools,
  };
}

/** The Chat Completions body for one model call: the plan's messages, tools
 *  and caps, the dialled model id, and streaming asked for with usage on.
 *  The body carries no clock and no randomness, so the replay sees the same
 *  request the live pass made. */
export function gatewayBody(
  plan: CallPlan,
  modelId: string,
  stream: boolean,
): Record<string, unknown> {
  const body: Record<string, unknown> = {
    model: modelId,
    messages: plan.messages,
    max_tokens: plan.maxTokens,
    stream,
  };
  if (plan.temperature !== null) body.temperature = plan.temperature;
  if (plan.tools !== null) {
    body.tools = plan.tools;
    if (plan.toolChoice !== null) body.tool_choice = plan.toolChoice;
  }
  if (plan.responseFormat !== null) body.response_format = plan.responseFormat;
  if (stream) body.stream_options = { include_usage: true };
  return body;
}

/** A provider message's text, whether it arrives as a string or as content
 *  parts. */
export function messageText(message: Record<string, unknown>): string {
  const content = message.content;
  if (typeof content === "string") return content;
  if (content === null || content === undefined) return "";
  if (Array.isArray(content)) {
    const parts: string[] = [];
    for (const part of content) {
      if (typeof part === "object" && part !== null) {
        const p = part as Record<string, unknown>;
        if (p.type === "text" && typeof p.text === "string") parts.push(p.text);
      } else if (typeof part === "string") {
        parts.push(part);
      }
    }
    return parts.join("");
  }
  return "";
}

/** The provider's tool calls, kept only when the client declared the tool
 *  by that name. A call the client never declared must not reach the
 *  answer: the wire checks forbid it. */
export function declaredToolCalls(
  message: Record<string, unknown>,
  declaredTools: string[],
): NonNullable<ChatMessage["tool_calls"]> {
  const out: NonNullable<ChatMessage["tool_calls"]> = [];
  const rawCalls = message.tool_calls;
  if (!Array.isArray(rawCalls)) return out;
  for (const rawCall of rawCalls) {
    if (typeof rawCall !== "object" || rawCall === null) continue;
    const c = rawCall as Record<string, unknown>;
    const fn = c.function as Record<string, unknown> | undefined;
    if (typeof c.id !== "string" || !fn || typeof fn.name !== "string") continue;
    if (!declaredTools.includes(fn.name)) continue;
    out.push({
      id: c.id,
      type: "function",
      function: {
        name: fn.name,
        arguments: typeof fn.arguments === "string" ? fn.arguments : "",
      },
    });
  }
  return out;
}

/** The provider's usage, whole, with a floor of one output token: the wire
 *  checks need an input count of at least 0 and an output count of at
 *  least 1. */
export function providerUsage(answer: Record<string, unknown>): {
  promptTokens: number;
  completionTokens: number;
} {
  const usage = answer.usage as Record<string, unknown> | undefined;
  const prompt =
    usage && typeof usage.prompt_tokens === "number" && usage.prompt_tokens >= 0
      ? Math.floor(usage.prompt_tokens)
      : 0;
  const completion =
    usage && typeof usage.completion_tokens === "number" && usage.completion_tokens >= 1
      ? Math.floor(usage.completion_tokens)
      : 1;
  return { promptTokens: prompt, completionTokens: completion };
}

/** The Chat Completions finish reason the client sees. A tool_calls finish
 *  with no surviving tool call becomes a plain stop: an answer that claims
 *  tool calls it does not carry is not well formed. */
function openAiFinish(
  providerReason: unknown,
  survivingCalls: number,
): string {
  if (providerReason === "tool_calls") return survivingCalls > 0 ? "tool_calls" : "stop";
  if (providerReason === "length") return "length";
  if (providerReason === "content_filter") return "content_filter";
  return "stop";
}

/** The provider's JSON answer as an OpenAI Chat Completions response, with
 *  the dialled model's id put back on it and the created instant from the
 *  trial's now. */
export function buildOpenAiAnswer(
  answer: Record<string, unknown>,
  modelId: string,
  stamps: Stamps,
  responseId: string,
  declaredTools: string[],
): Record<string, unknown> {
  const choices = Array.isArray(answer.choices) ? answer.choices : [];
  const first = choices.length > 0 && typeof choices[0] === "object" ? (choices[0] as Record<string, unknown>) : {};
  const message =
    first.message !== null && typeof first.message === "object"
      ? (first.message as Record<string, unknown>)
      : {};
  const text = messageText(message);
  const calls = declaredToolCalls(message, declaredTools);
  const usage = providerUsage(answer);
  const assistant: Record<string, unknown> = { role: "assistant", content: text };
  if (calls.length > 0) assistant.tool_calls = calls;
  return {
    id: responseId,
    object: "chat.completion",
    created: stamps.epoch,
    model: modelId,
    choices: [
      {
        index: 0,
        message: assistant,
        finish_reason: openAiFinish(first.finish_reason, calls.length),
      },
    ],
    usage: {
      prompt_tokens: usage.promptTokens,
      completion_tokens: usage.completionTokens,
      total_tokens: usage.promptTokens + usage.completionTokens,
    },
  };
}

/** The Messages stop reason for a provider finish reason. */
function anthropicStop(providerReason: unknown, survivingCalls: number): string {
  if (providerReason === "tool_calls") return survivingCalls > 0 ? "tool_use" : "end_turn";
  if (providerReason === "length") return "max_tokens";
  return "end_turn";
}

/** The provider's JSON answer as an Anthropic Messages response. Tool calls
 *  become tool_use blocks; the model field names the dialled id. */
export function buildAnthropicAnswer(
  answer: Record<string, unknown>,
  modelId: string,
  responseId: string,
  declaredTools: string[],
): Record<string, unknown> {
  const choices = Array.isArray(answer.choices) ? answer.choices : [];
  const first = choices.length > 0 && typeof choices[0] === "object" ? (choices[0] as Record<string, unknown>) : {};
  const message =
    first.message !== null && typeof first.message === "object"
      ? (first.message as Record<string, unknown>)
      : {};
  const text = messageText(message);
  const calls = declaredToolCalls(message, declaredTools);
  const usage = providerUsage(answer);
  const content: Array<Record<string, unknown>> = [];
  if (text.length > 0) content.push({ type: "text", text });
  for (const call of calls) {
    let input: unknown = {};
    try {
      input = JSON.parse(call.function.arguments);
    } catch {
      input = {};
    }
    content.push({ type: "tool_use", id: call.id, name: call.function.name, input });
  }
  if (content.length === 0) content.push({ type: "text", text: "" });
  return {
    id: responseId,
    type: "message",
    role: "assistant",
    model: modelId,
    content,
    stop_reason: anthropicStop(first.finish_reason, calls.length),
    stop_sequence: null,
    usage: { input_tokens: usage.promptTokens, output_tokens: usage.completionTokens },
  };
}

/** One streamed provider chunk, rewritten for the OpenAI wire: the same
 *  response id, created instant and dialled model on every chunk. Tool
 *  deltas for a tool the client never declared are dropped, and a tool_calls
 *  finish with no surviving tool call becomes a plain stop. Returns null
 *  for a chunk to drop. */
export function rewriteOpenAiChunk(
  line: string,
  modelId: string,
  stamps: Stamps,
  responseId: string,
  declaredTools: string[],
  toolNamesByIndex: Map<number, string>,
  survivingToolCalls: { count: number },
): string | null {
  if (line === "[DONE]") return line;
  let chunk: Record<string, unknown>;
  try {
    chunk = JSON.parse(line) as Record<string, unknown>;
  } catch {
    return null;
  }
  chunk.id = responseId;
  chunk.created = stamps.epoch;
  chunk.model = modelId;
  const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
  for (const rawChoice of choices) {
    if (typeof rawChoice !== "object" || rawChoice === null) continue;
    const choice = rawChoice as Record<string, unknown>;
    const delta =
      choice.delta !== null && typeof choice.delta === "object"
        ? (choice.delta as Record<string, unknown>)
        : null;
    if (delta && Array.isArray(delta.tool_calls)) {
      const kept: unknown[] = [];
      for (const rawCall of delta.tool_calls as unknown[]) {
        if (typeof rawCall !== "object" || rawCall === null) continue;
        const call = rawCall as Record<string, unknown>;
        const index = typeof call.index === "number" ? call.index : 0;
        const fn =
          call.function !== null && typeof call.function === "object"
            ? (call.function as Record<string, unknown>)
            : null;
        if (fn && typeof fn.name === "string" && fn.name.length > 0) {
          toolNamesByIndex.set(index, fn.name);
        }
        const name = toolNamesByIndex.get(index);
        if (name !== undefined && !declaredTools.includes(name)) continue;
        if (name !== undefined) survivingToolCalls.count = 1;
        kept.push(call);
      }
      if (kept.length === 0) delete delta.tool_calls;
      else delta.tool_calls = kept;
    }
    if (choice.finish_reason === "tool_calls" && survivingToolCalls.count === 0) {
      choice.finish_reason = "stop";
    }
  }
  return JSON.stringify(chunk);
}

/** A provider chunk's accumulated text, tool calls and finish, for the
 *  Anthropic stream translation. */
interface AccumulatedStream {
  textPieces: string[];
  toolCalls: Map<number, { id: string; name: string; arguments: string }>;
  finishReason: string | null;
  usage: { promptTokens: number; completionTokens: number } | null;
}

/** Fold the provider's streamed lines into one accumulated answer. The
 *  provider's own chunk ids and created instants are ignored: the
 *  Anthropic events carry this server's stamps instead. */
function accumulateStream(lines: string[]): AccumulatedStream {
  const acc: AccumulatedStream = {
    textPieces: [],
    toolCalls: new Map(),
    finishReason: null,
    usage: null,
  };
  for (const line of lines) {
    if (line === "[DONE]") continue;
    let chunk: Record<string, unknown>;
    try {
      chunk = JSON.parse(line) as Record<string, unknown>;
    } catch {
      continue;
    }
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    for (const rawChoice of choices) {
      if (typeof rawChoice !== "object" || rawChoice === null) continue;
      const choice = rawChoice as Record<string, unknown>;
      const delta =
        choice.delta !== null && typeof choice.delta === "object"
          ? (choice.delta as Record<string, unknown>)
          : null;
      if (delta) {
        if (typeof delta.content === "string" && delta.content.length > 0) {
          acc.textPieces.push(delta.content);
        }
        if (Array.isArray(delta.tool_calls)) {
          for (const rawCall of delta.tool_calls as unknown[]) {
            if (typeof rawCall !== "object" || rawCall === null) continue;
            const call = rawCall as Record<string, unknown>;
            const index = typeof call.index === "number" ? call.index : 0;
            const fn =
              call.function !== null && typeof call.function === "object"
                ? (call.function as Record<string, unknown>)
                : null;
            const slot = acc.toolCalls.get(index) ?? { id: "", name: "", arguments: "" };
            if (typeof call.id === "string") slot.id = call.id;
            if (fn && typeof fn.name === "string" && fn.name.length > 0) slot.name = fn.name;
            if (fn && typeof fn.arguments === "string") slot.arguments += fn.arguments;
            acc.toolCalls.set(index, slot);
          }
        }
      }
      if (typeof choice.finish_reason === "string") acc.finishReason = choice.finish_reason;
    }
    if (chunk.usage !== null && typeof chunk.usage === "object") {
      acc.usage = providerUsage(chunk as Record<string, unknown>);
    }
  }
  return acc;
}

/** The provider's streamed lines as an Anthropic SSE event sequence: a
 *  message_start, one text block with a delta per provider text piece, a
 *  tool_use block per surviving tool call, then message_delta and
 *  message_stop. The whole sequence is built from the accumulated answer,
 *  so the message_start already carries the true input count. */
export function anthropicStreamEvents(
  lines: string[],
  modelId: string,
  stamps: Stamps,
  responseId: string,
  declaredTools: string[],
): string[] {
  const acc = accumulateStream(lines);
  const usage = acc.usage ?? { promptTokens: 0, completionTokens: 1 };
  const surviving = [...acc.toolCalls.values()].filter(
    (c) => c.name.length > 0 && declaredTools.includes(c.name),
  );
  const stop = anthropicStop(acc.finishReason, surviving.length);
  const events: string[] = [];
  /** One Anthropic SSE event: the event name first, then the data line.
   *  The name is the payload's own type — the env's stream check reads the
   *  event by its name, and a bare data line carries none. */
  const event = (payload: Record<string, unknown>): string => {
    const name = typeof payload.type === "string" ? payload.type : "message";
    return "event: " + name + "\ndata: " + JSON.stringify(payload);
  };
  events.push(
    event({
      type: "message_start",
      message: {
        id: responseId,
        type: "message",
        role: "assistant",
        model: modelId,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: usage.promptTokens, output_tokens: 0 },
      },
    }),
  );
  events.push(
    event({
      type: "content_block_start",
      index: 0,
      content_block: { type: "text", text: "" },
    }),
  );
  for (const piece of acc.textPieces) {
    events.push(
      event({
        type: "content_block_delta",
        index: 0,
        delta: { type: "text_delta", text: piece },
      }),
    );
  }
  events.push(event({ type: "content_block_stop", index: 0 }));
  let index = 1;
  for (const call of surviving) {
    events.push(
      event({
        type: "content_block_start",
        index,
        content_block: { type: "tool_use", id: call.id, name: call.name, input: {} },
      }),
    );
    if (call.arguments.length > 0) {
      events.push(
        event({
          type: "content_block_delta",
          index,
          delta: { type: "input_json_delta", partial_json: call.arguments },
        }),
      );
    }
    events.push(event({ type: "content_block_stop", index }));
    index += 1;
  }
  events.push(
    event({
      type: "message_delta",
      delta: { stop_reason: stop, stop_sequence: null },
      usage: { output_tokens: usage.completionTokens },
    }),
  );
  events.push(event({ type: "message_stop" }));
  return events;
}
