# Free Model Aggregator

An aggregator that answers both completion wires — OpenAI Chat Completions
(`POST /v1/chat/completions`) and Anthropic Messages (`POST /v1/messages`),
in JSON and in server-sent events — with real completions from the fourteen
free models of the brief, dialled through the grader's gateway on Chat
Completions. It never answers from its own text.

Entry for make.fun bounty `free-model-aggregator-02` ($500).

## Model choice

The model that serves a prompt is the highest-rated free model meeting
that prompt's requirements, from the brief's own ratings:

| Prompt needs | Model dialled | Rating |
|---|---|---|
| JSON mode | `openrouter/nvidia/nemotron-3.5-super-vl:free` | 1303 |
| anything else (incl. tools) | `openrouter/thinkingmachines/inkling-2-small:free` | 1475 |

When a call is refused (429/5xx), the server fails over to the next-best
model meeting the requirements and reports every refusal in the
`x-makefun-upstream-failures` header (at most 8 rows
`{"status","code","model"}`).

## Run

```sh
node main.ts --list   # print the catalog document and exit 0
node main.ts          # serve, register with the env, exit 0 when /work says done
```

Node 22, standard library only. The server reads the brief from
`MAKEFUN_PROMPT_FILE`, the env from `MAKEFUN_ENV_URL`, the gateway from
`OPENAI_BASE_URL` / `OPENAI_API_KEY`, the seed from `MAKEFUN_TRIAL_SEED`
and the trial's now from `MAKEFUN_TRIAL_NOW`. It listens on `0.0.0.0` at
a free port and registers that port and its catalog with the env.

Determinism: no wall clock and no unseeded randomness in any request —
response ids and the created instant come from the trial seed and
`MAKEFUN_TRIAL_NOW`, and the env is asked one request at a time.

## Layout

- `main.ts` — entrypoint: `--list`, brief loading, server start, register, `/work` polling
- `lib/catalog.ts` — the fourteen free models, the catalog document, model selection
- `lib/gateway.ts` — dialling the gateway (JSON + SSE) and the failures trail
- `lib/wire.ts` — the two wires: request parsing, answer translation, streaming
- `lib/server.ts` — the HTTP server and the failover loop

## License

MIT. See `LICENSE`.
