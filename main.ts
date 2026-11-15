// free-model-aggregator-01: one server, two client wires, a directory of free providers, all three
// upstream wires spoken.
//
// WHAT A TRIAL GIVES YOU
//   A trial gives this server a brief and an env, and runs it as `node /bot/main.ts` from your
//   repository's root, with Node 22's standard library alone (nothing is installed; TypeScript runs
//   directly, and a relative import needs its .ts extension). The brief is at the path in
//   $MAKEFUN_PROMPT_FILE. It carries three JSON documents:
//     - the provider directory: ten simulated free-tier providers, each with its id, wire (openai,
//       anthropic or gemini), auth scheme, key environment variable, the key the service takes, and the
//       base path you join to $MAKEFUN_ENV_URL;
//     - the quality table: every model's public rating and what it can do (tools, vision, JSON mode,
//       context length), and whether it is free;
//     - the workload manifest: the twenty requests the client will send, in order, with each one's api,
//       whether it streams, its weight and the requirements a model must meet to serve it.
//   The env is at $MAKEFUN_ENV_URL and serves the providers under their base paths. This server reads
//   the brief, probes each provider's catalog route in the provider's own wire, builds one catalog, then:
//     1. listens on 0.0.0.0 at a free port (never 8080: the env itself is there on one layout);
//     2. POSTs $MAKEFUN_ENV_URL/register {"protocol_version": 1, "port": <port>, "catalog": <catalog>};
//     3. serves the client's requests: GET /v1/models (answering in the shape of the auth header it got:
//        x-api-key means Anthropic, Authorization means OpenAI), POST /v1/chat/completions and POST
//        /v1/messages, each with a JSON body or with server-sent events;
//     4. polls GET $MAKEFUN_ENV_URL/work every 100 ms and exits 0 when the env says {"done": true}. A
//        process still running when the wall ends scores zero, so never wait for anything else.
//   `node main.ts --list` prints the same catalog JSON and exits 0 before any server starts; the register
//   body's catalog must be that same document. Build it once, as this file does, and use the one object.
//
// HOW IT ROUTES
//   - quality: every free model that meets the request's requirements (the quality table's capabilities
//     are the truth) is a candidate, ranked by the table's rating_milli, highest first. The client sends
//     the workload in manifest order — the heavier requests first — so serving each request from the best
//     candidate still standing spends the scarcest top models where they earn the most.
//   - failover: a candidate that answers 402, 429 or a 5xx is retried once (a refusal on a provider's
//     failure schedule spends one call; the next call often serves), then the request moves to the next
//     candidate. A provider that refuses three calls in a row is treated as spent or out and drops
//     behind the standing candidates for the rest of the trial; one success clears the mark.
//   - wires: the request is rewritten into the routed provider's own wire (Chat Completions, Messages
//     or generateContent), preserving the system prompt, the user text, the tool names, max_tokens, the
//     stops, the images and JSON mode exactly as the client sent them; the provider's answer, JSON or
//     SSE, is read into one shape and rendered back in the client's wire with its usage intact.
//   - catalog: the probe reports every provider that answered, with the quality table's free flag and
//     context length per model; GET /v1/models lists exactly the catalog's free models.
//
//   Everything is a function of the brief and the providers' answers: no wall clock, no randomness, so a
//   replay of the same trial makes the same calls in the same order.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicMessage,
  anthropicModelList,
  anthropicStreamText,
  anthropicToGemini,
  anthropicToOpenAI,
  openAICompletion,
  openAIError,
  openAIModelList,
  openAIStreamText,
  openAIRequestToAnthropic,
  openAIToGemini,
  parseAnthropicAnswer,
  parseGeminiAnswer,
  parseOpenAIAnswer,
  type NormalizedAnswer,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one provider call may take before the request moves to the next candidate. */
const CALL_TIMEOUT_MS = 15000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;
/** Attempts per candidate before the request moves on: the first try and one retry. */
const ATTEMPTS_PER_CANDIDATE = 2;
/** Consecutive refusals after which a provider is treated as spent or out for the rest of the trial. */
const SPENT_AFTER_REFUSALS = 3;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;
/** Consecutive refused calls per provider id; a success resets the count to zero. */
const refusals = new Map<string, number>();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

/** The free models the catalog registers, in catalog order: what GET /v1/models lists, on either wire. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const provider of catalog.providers) {
    const probe = probes.find((p) => p.provider.id === provider.id);
    for (const model of provider.models) {
      if (!model.free) continue;
      const listed = probe?.models.find((m) => m.id === model.id);
      out.push({ id: model.id, provider: provider.id, display_name: listed?.display_name ?? model.id });
    }
  }
  return out;
}

interface Route {
  probe: Probe;
  model: ProbedModel;
  rating: number;
}

/**
 * Every free candidate that can serve one request, best first: the quality table's rating orders them,
 * the brief's provider order breaks a tie, and a provider marked spent (three refusals in a row) drops
 * behind the standing candidates. One wire limit applies: an Anthropic-wire provider cannot echo JSON
 * mode, so an OpenAI client request that asks for a JSON response format never routes to one.
 */
function candidatesFor(requires: Requirement, clientApi: 'openai' | 'anthropic', body: Json): Route[] {
  const wantsJsonMode =
    clientApi === 'openai' && isRecord(body.response_format) && (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema');
  const routes: Route[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    if (wantsJsonMode && probe.provider.wire === 'anthropic') continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      routes.push({ probe, model, rating: row?.rating_milli ?? 0 });
    }
  }
  const providerOrder = new Map(brief.directory.providers.map((p, i) => [p.id, i]));
  routes.sort((a, b) => {
    const spentA = (refusals.get(a.probe.provider.id) ?? 0) >= SPENT_AFTER_REFUSALS ? 1 : 0;
    const spentB = (refusals.get(b.probe.provider.id) ?? 0) >= SPENT_AFTER_REFUSALS ? 1 : 0;
    if (spentA !== spentB) return spentA - spentB;
    if (a.rating !== b.rating) return b.rating - a.rating;
    const orderA = providerOrder.get(a.probe.provider.id) ?? 0;
    const orderB = providerOrder.get(b.probe.provider.id) ?? 0;
    if (orderA !== orderB) return orderA - orderB;
    return a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0;
  });
  return routes;
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

interface UpstreamCall {
  url: string;
  headers: Record<string, string>;
  payload: Json;
  stream: boolean;
}

/** One routed provider call, in the provider's own wire, for a client request in either wire. */
function upstreamCall(route: Route, clientApi: 'openai' | 'anthropic', body: Json): UpstreamCall {
  const provider = route.probe.provider;
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = { 'content-type': 'application/json', ...authHeaders(provider.auth, provider.key) };
  const stream = body.stream === true;
  if (provider.wire === 'openai') {
    const payload = clientApi === 'openai' ? { ...body, model: route.model.id } : anthropicToOpenAI(body, route.model.id);
    if (payload.stream === true) payload.stream_options = { include_usage: true };
    return { url: `${base}/v1/chat/completions`, headers, payload, stream: payload.stream === true };
  }
  if (provider.wire === 'anthropic') {
    headers['anthropic-version'] = '2023-06-01';
    const payload = clientApi === 'anthropic' ? { ...body, model: route.model.id } : openAIRequestToAnthropic(body, route.model.id);
    return { url: `${base}/v1/messages`, headers, payload, stream: payload.stream === true };
  }
  const payload = clientApi === 'openai' ? openAIToGemini(body) : anthropicToGemini(body);
  const action = stream ? 'streamGenerateContent?alt=sse' : 'generateContent';
  return { url: `${base}/v1beta/models/${encodeURIComponent(route.model.id)}:${action}`, headers, payload, stream };
}

/** A provider's answer text read into the one normalized shape; null when it is not a usable answer. */
function parseAnswer(wire: 'openai' | 'anthropic' | 'gemini', text: string, stream: boolean): NormalizedAnswer | null {
  if (wire === 'openai') return parseOpenAIAnswer(text, stream);
  if (wire === 'anthropic') return parseAnthropicAnswer(text, stream);
  return parseGeminiAnswer(text, stream);
}

/**
 * Serve one client request from the best candidate still standing. A refusal (402, 429 or 5xx) retries
 * the same candidate once, then moves to the next; any success answers and ends the request, so exactly
 * one provider call ever answers it. Returns null when every candidate refused.
 */
async function serve(clientApi: 'openai' | 'anthropic', body: Json): Promise<{ route: Route; answer: NormalizedAnswer } | null> {
  const candidates = candidatesFor(requirementsFor(body), clientApi, body);
  for (const route of candidates) {
    const call = upstreamCall(route, clientApi, body);
    // The retry exists for a one-off refusal on a provider's failure schedule. A provider that has
    // already refused a call in this trial is likelier spent or out than unlucky, so its candidates
    // get the one attempt only: a second refusal would spend a call the tie-break counts for nothing.
    const attempts = (refusals.get(route.probe.provider.id) ?? 0) === 0 ? ATTEMPTS_PER_CANDIDATE : 1;
    for (let attempt = 0; attempt < attempts; attempt++) {
      let upstream: Response;
      try {
        upstream = await fetch(call.url, {
          method: 'POST',
          headers: call.headers,
          body: JSON.stringify(call.payload),
          signal: AbortSignal.timeout(CALL_TIMEOUT_MS),
        });
      } catch {
        break; // Unreachable or timed out: the next candidate, not a retry of a dead route.
      }
      if (upstream.ok) {
        const answer = parseAnswer(route.probe.provider.wire, await upstream.text(), call.stream);
        if (answer !== null) {
          refusals.set(route.probe.provider.id, 0);
          return { route, answer };
        }
        break; // A 2xx that is not a usable answer: do not spend another call on this candidate.
      }
      const status = upstream.status;
      await upstream.text().catch(() => '');
      if (status === 402 || status === 429 || status >= 500) {
        refusals.set(route.probe.provider.id, (refusals.get(route.probe.provider.id) ?? 0) + 1);
        continue; // One retry, then the loop's next attempt or the next candidate takes over.
      }
      break; // A 400/401 names a fault in the call itself: the next candidate, never a retry.
    }
  }
  return null;
}

/** Render a served answer in the client's wire, as JSON or as an event stream. */
function render(res: ServerResponse, clientApi: 'openai' | 'anthropic', stream: boolean, answer: NormalizedAnswer, model: string): void {
  if (clientApi === 'openai') {
    if (stream) {
      res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
      res.end(openAIStreamText(answer, model));
    } else {
      sendJson(res, 200, openAICompletion(answer, model));
    }
    return;
  }
  if (stream) {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
    res.end(anthropicStreamText(answer, model));
  } else {
    sendJson(res, 200, anthropicMessage(answer, model));
  }
}

/** POST /v1/chat/completions: an OpenAI-shaped client. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const served = await serve('openai', body);
  if (served === null) return sendError(res, 'openai', 503, 'no_free_model', 'no free model on the directory meets this request');
  render(res, 'openai', body.stream === true, served.answer, served.route.model.id);
}

/** POST /v1/messages: an Anthropic-shaped client. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const served = await serve('anthropic', body);
  if (served === null) return sendError(res, 'anthropic', 503, 'overloaded_error', 'no free model on the directory meets this request');
  const requested = typeof body.model === 'string' && body.model !== '' ? body.model : served.route.model.id;
  render(res, 'anthropic', body.stream === true, served.answer, requested);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
