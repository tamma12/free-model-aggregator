/** The best live free-model aggregator: one server that answers both
 *  completion wires with real completions from the fourteen free models,
 *  dialled through the grader's gateway.
 *
 *  node main.ts --list   prints the catalog document and exits 0.
 *  node main.ts          listens on 0.0.0.0 at a free port, registers that
 *                        port and its catalog with the env, serves the two
 *                        wires, polls GET $MAKEFUN_ENV_URL/work and exits 0
 *                        when it says done.
 *
 *  Every choice comes from the brief (MAKEFUN_PROMPT_FILE) and the requests
 *  themselves: no wall clock in any request (the created instant is
 *  MAKEFUN_TRIAL_NOW), no unseeded randomness (ids come from
 *  MAKEFUN_TRIAL_SEED and a counter), and one env request at a time. */

import { readFileSync } from "node:fs";
import {
  FREE_MODELS,
  buildCatalog,
  catalogJson,
  parseBriefModels,
  type FreeModel,
} from "./lib/catalog.ts";
import { createServer } from "./lib/server.ts";
import { Stamps } from "./lib/wire.ts";

const sleep = (ms: number): Promise<void> =>
  new Promise((resolve) => {
    setTimeout(resolve, ms);
  });

/** The brief's model table when the brief file is there, else the table
 *  this entry embeds. The two are the same fourteen rows; the brief is
 *  authoritative for the instance this trial runs on. */
function loadModels(): FreeModel[] {
  const briefPath = process.env.MAKEFUN_PROMPT_FILE;
  if (briefPath) {
    try {
      const text = readFileSync(briefPath, "utf8");
      const parsed = parseBriefModels(text);
      if (parsed !== null && parsed.length > 0) return parsed;
    } catch {
      // A brief that cannot be read falls back to the embedded table.
    }
  }
  return FREE_MODELS;
}

/** Register this server's port and catalog with the env, retrying a few
 *  times: the env must know the port before it sends the first prompt. */
async function register(
  envUrl: string,
  port: number,
  catalog: Record<string, unknown>,
): Promise<boolean> {
  const url = envUrl.replace(/\/$/, "") + "/register";
  for (let attempt = 0; attempt < 5; attempt++) {
    try {
      const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ protocol_version: 1, port, catalog }),
        signal: AbortSignal.timeout(5000),
      });
      if (res.ok) return true;
    } catch {
      // The env may not be listening yet; wait and try again.
    }
    await sleep(500);
  }
  return false;
}

/** Poll GET $MAKEFUN_ENV_URL/work until it says done, one request at a
 *  time. A poll that fails is retried: only the done answer ends the
 *  trial. */
async function waitForDone(envUrl: string): Promise<void> {
  const url = envUrl.replace(/\/$/, "") + "/work";
  for (;;) {
    try {
      const res = await fetch(url, { signal: AbortSignal.timeout(5000) });
      if (res.ok) {
        const body = (await res.json()) as Record<string, unknown>;
        if (body.done === true) return;
      }
    } catch {
      // The env may be busy; poll again.
    }
    await sleep(500);
  }
}

async function main(): Promise<number> {
  const models = loadModels();
  if (process.argv.includes("--list")) {
    process.stdout.write(catalogJson(models));
    return 0;
  }
  const stamps = new Stamps(
    process.env.MAKEFUN_TRIAL_SEED ?? "0",
    process.env.MAKEFUN_TRIAL_NOW,
  );
  const server = createServer({
    models,
    stamps,
    openAiBaseUrl: process.env.OPENAI_BASE_URL ?? "https://openrouter.ai/api/v1",
    openAiApiKey: process.env.OPENAI_API_KEY ?? "",
  });
  const port = await new Promise<number>((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "0.0.0.0", () => {
      const address = server.address();
      if (address !== null && typeof address === "object") resolve(address.port);
      else reject(new Error("server did not bind a port"));
    });
  });
  const envUrl = process.env.MAKEFUN_ENV_URL;
  if (envUrl) {
    const registered = await register(envUrl, port, buildCatalog(models));
    if (!registered) {
      console.error("free-model-aggregator: the env would not take the registration");
      server.close();
      return 1;
    }
    await waitForDone(envUrl);
    server.close();
    return 0;
  }
  // No env to serve: keep listening until stopped (local testing).
  await new Promise<void>((resolve) => {
    const stop = (): void => {
      server.close();
      resolve();
    };
    process.once("SIGTERM", stop);
    process.once("SIGINT", stop);
  });
  return 0;
}

main()
  .then((code) => process.exit(code))
  .catch((err: unknown) => {
    console.error(err instanceof Error ? err.message : String(err));
    process.exit(1);
  });
