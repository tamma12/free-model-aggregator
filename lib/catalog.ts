/** The fourteen free models this competition may dial, with the house's
 *  quality rating from the brief. `id` is the dialable id the gateway routes;
 *  a completion echoes the same id back without the gateway prefix. The table
 *  is written out here so the catalog and the model choice never depend on a
 *  file the sandbox might not hand us; the brief file, when it is there, is
 *  parsed at startup and replaces this table with its own identical rows. */

export interface FreeModel {
  id: string;
  displayName: string;
  ratingMilli: number;
  contextLength: number;
  tools: boolean;
  vision: boolean;
  json: boolean;
}

export const FREE_MODELS: FreeModel[] = [
  { id: "openrouter/baidu/ernie-5.5-flash-preview:free", displayName: "Baidu: ERNIE-5.5-Flash-Preview (free)", ratingMilli: 1150, contextLength: 1024000, tools: true, vision: false, json: true },
  { id: "openrouter/cohere/north-mini-code:free", displayName: "Cohere: North Mini Code (free)", ratingMilli: 1200, contextLength: 256000, tools: true, vision: false, json: false },
  { id: "openrouter/dots-studio/dots-3-note-preview:free", displayName: "Dots Studio: Dots3-Note Preview (free)", ratingMilli: 1150, contextLength: 512000, tools: true, vision: true, json: true },
  { id: "openrouter/google/gemma-4-26b-a4b-it:free", displayName: "Google: Gemma 4 26B A4B (free)", ratingMilli: 1150, contextLength: 262144, tools: true, vision: true, json: true },
  { id: "openrouter/google/gemma-4-31b-it:free", displayName: "Google: Gemma 4 31B (free)", ratingMilli: 1210, contextLength: 262144, tools: true, vision: true, json: true },
  { id: "openrouter/inclusionai/ling-3.2-flash:free", displayName: "inclusionAI: Ling 3.2 Flash (free)", ratingMilli: 1150, contextLength: 1048576, tools: true, vision: true, json: true },
  { id: "openrouter/liquid/lfm-3-1.7b:free", displayName: "LiquidAI: LFM3-1.7B (free)", ratingMilli: 1150, contextLength: 262144, tools: true, vision: false, json: true },
  { id: "openrouter/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free", displayName: "NVIDIA: Nemotron 3 Nano Omni (free)", ratingMilli: 1150, contextLength: 256000, tools: true, vision: true, json: false },
  { id: "openrouter/nvidia/nemotron-3-ultra-550b-a55b:free", displayName: "NVIDIA: Nemotron 3 Ultra (free)", ratingMilli: 1285, contextLength: 1000000, tools: true, vision: false, json: false },
  { id: "openrouter/nvidia/nemotron-3.5-lightning:free", displayName: "NVIDIA: Nemotron 3.5 Lightning (free)", ratingMilli: 1221, contextLength: 1000000, tools: true, vision: false, json: false },
  { id: "openrouter/nvidia/nemotron-3.5-super-vl:free", displayName: "NVIDIA: Nemotron 3.5 Super VL (free)", ratingMilli: 1303, contextLength: 1000000, tools: true, vision: true, json: true },
  { id: "openrouter/particle-ai/particle-1:free", displayName: "Particle AI: Particle-1 (free)", ratingMilli: 1150, contextLength: 1048576, tools: true, vision: true, json: true },
  { id: "openrouter/recursive/flint-v1.0:free", displayName: "Recursive: Flint (free)", ratingMilli: 1150, contextLength: 262144, tools: true, vision: false, json: true },
  { id: "openrouter/thinkingmachines/inkling-2-small:free", displayName: "Thinking Machines: Inkling 2 Small (free)", ratingMilli: 1475, contextLength: 1048576, tools: true, vision: true, json: false },
];

/** What a prompt needs a model to do. A requirement left false asks nothing. */
export interface ModelNeeds {
  tools: boolean;
  vision: boolean;
  json: boolean;
}

/** The catalog document, exactly as the brief spells it. One provider,
 *  openrouter, and exactly the fourteen free ids, once each. The same object
 *  is what --list prints and what the register body carries, so the two can
 *  never drift apart. */
export function buildCatalog(models: FreeModel[]): Record<string, unknown> {
  return {
    providers: [
      {
        id: "openrouter",
        base_url: "https://openrouter.ai/api/v1",
        key_env: "OPENROUTER_API_KEY",
        wire: "openai",
        models: models.map((m) => ({ id: m.id, free: true, context_length: m.contextLength })),
      },
    ],
  };
}

/** The one JSON string the catalog is, for --list and for registration. */
export function catalogJson(models: FreeModel[]): string {
  return JSON.stringify(buildCatalog(models));
}

/** The models that meet a prompt's requirements, best rating first. Ties on
 *  rating break by id, so the order is the same on every run. The first row
 *  is the model that earns the prompt its best quality: for a JSON prompt
 *  that is nemotron-3.5-super-vl (1303, the best JSON-capable model), for
 *  every other prompt inkling-2-small (1475, the best model outright). The
 *  rest of the list is the failover order, tried in turn when a call is
 *  refused. */
export function candidatesFor(models: FreeModel[], needs: ModelNeeds): FreeModel[] {
  const fitting = models.filter(
    (m) =>
      (!needs.tools || m.tools) && (!needs.vision || m.vision) && (!needs.json || m.json),
  );
  fitting.sort((a, b) => b.ratingMilli - a.ratingMilli || (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
  return fitting;
}

/** Read the brief file's model table when it is there. The brief carries the
 *  same fourteen rows this file embeds; parsing it keeps the server honest to
 *  the instance it runs on, and a brief that cannot be read falls back to
 *  the embedded table instead of failing the trial. Returns null when the
 *  file holds no usable model list. */
export function parseBriefModels(briefText: string): FreeModel[] | null {
  const marker = '"schema":"free-model-aggregator-02/models/v1"';
  const at = briefText.indexOf(marker);
  if (at < 0) return null;
  const start = briefText.indexOf('"models":', at);
  if (start < 0) return null;
  const open = briefText.indexOf("[", start);
  if (open < 0) return null;
  let depth = 0;
  let end = -1;
  for (let i = open; i < briefText.length; i++) {
    const ch = briefText[i];
    if (ch === "[") depth++;
    else if (ch === "]") {
      depth--;
      if (depth === 0) {
        end = i + 1;
        break;
      }
    }
  }
  if (end < 0) return null;
  let rows: unknown;
  try {
    rows = JSON.parse(briefText.slice(open, end));
  } catch {
    return null;
  }
  if (!Array.isArray(rows) || rows.length === 0) return null;
  const models: FreeModel[] = [];
  for (const row of rows) {
    if (typeof row !== "object" || row === null) return null;
    const r = row as Record<string, unknown>;
    if (
      typeof r.id !== "string" ||
      typeof r.rating_milli !== "number" ||
      typeof r.context_length !== "number"
    ) {
      return null;
    }
    models.push({
      id: r.id,
      displayName: typeof r.display_name === "string" ? r.display_name : r.id,
      ratingMilli: r.rating_milli,
      contextLength: r.context_length,
      tools: r.tools === true,
      vision: r.vision === true,
      json: r.json === true,
    });
  }
  return models;
}
