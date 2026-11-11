// Street-level photo geolocator for make.fun's photo-geolocator-01.
//
// For every photo it asks claude-opus-5-5 (effort low) to read the clues in the picture (driving side,
// road markings, signs and script, bollards, poles, plates, vegetation, terrain, architecture, sun) and
// commit to a latitude and longitude. The trial's $1.00 budget is the hard limit: the gateway reserves
// each call's worst case (its body at one token per 3 bytes at the dearest input rate, plus its whole
// output cap), so the agent keeps its own conservative ledger, sends the biggest photos first while the
// most budget is left, and steps down to claude-sonnet-5-5 and then claude-haiku-5-5 when Opus's worst
// case no longer fits. Any photo still unplaced gets a fallback point, so no photo is left unanswered.
//
// Deterministic for the replay: calls go out in rounds, and each round's photos, models and request
// bodies depend only on the photos and on the answers of earlier rounds (never on which answer of a
// round came back first). Answers are posted once, in photo order, at the end.
//
// Node 22, standard library only.

interface Photo {
  id: string;
  url: string;
  bytes: number;
  answered: boolean;
}

interface Guess {
  id: string;
  lat: number;
  lon: number;
}

interface ModelSpec {
  name: string;
  maxTokens: number;
  effort: string;
  /** USD per token: input, the dearest input class (cache write 1h, what the gateway reserves at), output. */
  input: number;
  dearestInput: number;
  output: number;
}

const M = 1e-6;
const OPUS: ModelSpec = { name: 'claude-opus-5-5', maxTokens: 3000, effort: 'low', input: 4 * M, dearestInput: 8 * M, output: 20 * M };
const SONNET: ModelSpec = { name: 'claude-sonnet-5-5', maxTokens: 2000, effort: 'low', input: 2 * M, dearestInput: 4 * M, output: 10 * M };
const HAIKU: ModelSpec = { name: 'claude-haiku-5-5', maxTokens: 1500, effort: 'low', input: 0.5 * M, dearestInput: 1 * M, output: 2.5 * M };
const LADDER = [OPUS, SONNET, HAIKU];

const BUDGET_USD = 1.0;
/** Headroom kept below the budget for rounding in the gateway's own meter. */
const SAFETY_USD = 0.03;
/** Spend recorded per call is the usage priced at list rates times this margin. */
const SPEND_MARGIN = 1.1;
const MAX_PER_ROUND = 3;
const MAX_REQUESTS = 190;
/** Where a photo goes when no model placed it: central Europe, where much of the dataset sits. */
const FALLBACK = { lat: 47.0, lon: 8.0 };

const SYSTEM = `You are an expert at geolocating street-level photos (the kind found on Mapillary and in GeoGuessr).
You are shown one photo taken somewhere on Earth from a road, street or path, with no metadata. Work out where it was taken.

Look for evidence, strongest first:
- Text: language, script, alphabet details, place names, phone number formats, domain names, shop names, route numbers.
- Road furniture: which side traffic drives on, centre and edge line colours and dash patterns, bollard shapes and colours, kerb paint, guardrails, sign shapes, colours and fonts, chevrons, pedestrian crossing style.
- Utility poles (wood, concrete, metal; crossarms, insulators), street lamps, fire hydrants, mailboxes.
- Vehicles and licence plates (colour, shape, EU strip), common car models, motorbikes, tuk-tuks.
- Architecture, roofing, walls, fences, building materials.
- Vegetation, crops, soil colour, terrain, climate, snow, the sun's direction and shadows (sun to the north means the southern hemisphere).
- Camera and coverage context of Mapillary uploads.

Then commit to the single most likely place and give coordinates of that specific area (a town or road), not a country's centre, when the evidence is specific. Your score is the distance error, so when you are torn between nearby regions, pick a point between them; when torn between far-apart regions, pick the one you think more likely.

Keep the reasoning short: at most 8 lines of clues and conclusions. End with exactly one final line in this form and nothing after it:
ANSWER {"lat": <decimal degrees, north positive>, "lon": <decimal degrees, east positive>}`;

const USER_TEXT = 'Where was this photo taken? Follow the instructions and end with the ANSWER line.';

/** Retried (with a fixed backoff) only when the gateway does not say x-should-retry: false; its refusals are final. */
const RETRY_STATUSES = [408, 429, 500, 502, 503, 504, 529];
const RETRIES = 3;

function need(name: string): string {
  const v = process.env[name];
  if (v === undefined || v === '') throw new Error(`${name} is not set`);
  return v;
}

const envUrl = need('MAKEFUN_ENV_URL').replace(/\/+$/, '');
const anthropicUrl = need('ANTHROPIC_BASE_URL').replace(/\/+$/, '');
const apiKey = need('ANTHROPIC_API_KEY');

const sleep = (ms: number): Promise<void> => new Promise((r) => setTimeout(r, ms));
const log = (s: string): void => {
  process.stderr.write(`geo: ${s}\n`);
};

function requestBody(model: ModelSpec, jpegB64: string): string {
  return JSON.stringify({
    model: model.name,
    max_tokens: model.maxTokens,
    output_config: { effort: model.effort },
    system: SYSTEM,
    messages: [
      {
        role: 'user',
        content: [
          { type: 'image', source: { type: 'base64', media_type: 'image/jpeg', data: jpegB64 } },
          { type: 'text', text: USER_TEXT },
        ],
      },
    ],
  });
}

/** The gateway's reservation for a call: its body at a token per 3 bytes at the dearest input rate, plus the whole output cap. */
function worstCaseUsd(model: ModelSpec, body: string): number {
  return Math.ceil(Buffer.byteLength(body) / 3) * model.dearestInput + model.maxTokens * model.output;
}

interface Usage {
  input_tokens?: number;
  output_tokens?: number;
  cache_creation_input_tokens?: number;
  cache_read_input_tokens?: number;
}

function spentUsd(model: ModelSpec, u: Usage | undefined, body: string): number {
  if (u === undefined) return worstCaseUsd(model, body);
  const cost =
    (u.input_tokens ?? 0) * model.input +
    (u.cache_creation_input_tokens ?? 0) * model.dearestInput +
    (u.cache_read_input_tokens ?? 0) * model.input +
    (u.output_tokens ?? 0) * model.output;
  return cost * SPEND_MARGIN;
}

interface CallResult {
  text: string | null;
  spent: number;
  /** True when the gateway refused the call (final, nothing charged). */
  refused: boolean;
}

async function call(model: ModelSpec, body: string): Promise<CallResult> {
  for (let attempt = 0; ; attempt += 1) {
    let res: Response;
    try {
      res = await fetch(`${anthropicUrl}/v1/messages`, {
        method: 'POST',
        headers: { 'content-type': 'application/json', 'x-api-key': apiKey, 'anthropic-version': '2023-06-01' },
        body,
      });
    } catch (err) {
      if (attempt < RETRIES) {
        await sleep(2000 * (attempt + 1));
        continue;
      }
      log(`${model.name} network error: ${String(err)}`);
      return { text: null, spent: worstCaseUsd(model, body), refused: false };
    }
    if (res.ok) {
      const msg = (await res.json()) as { content?: { type: string; text?: string }[]; usage?: Usage; stop_reason?: string };
      const text = (msg.content ?? [])
        .filter((b) => b.type === 'text')
        .map((b) => b.text ?? '')
        .join('\n');
      if (msg.stop_reason !== 'end_turn') log(`${model.name} stop_reason ${msg.stop_reason}`);
      return { text, spent: spentUsd(model, msg.usage, body), refused: false };
    }
    const errText = await res.text();
    if (RETRY_STATUSES.includes(res.status) && res.headers.get('x-should-retry') !== 'false' && attempt < RETRIES) {
      await sleep(2000 * (attempt + 1));
      continue;
    }
    log(`${model.name} HTTP ${res.status}: ${errText.slice(0, 300)}`);
    // A refusal by the gateway costs nothing; any other error may have been charged, so count its worst case.
    const refused = res.headers.get('x-should-retry') === 'false' || res.status === 400 || res.status === 401;
    return { text: null, spent: refused ? 0 : worstCaseUsd(model, body), refused };
  }
}

const isLat = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v) && v >= -90 && v <= 90;
const isLon = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v) && v >= -180 && v <= 180;

/** The coordinates an answer gives: the last JSON object holding lat and lon, else the last in-range number pair. */
function coordinates(text: string): { lat: number; lon: number } | null {
  const objects = [...text.matchAll(/\{[^{}]*\}/g)].reverse();
  for (const m of objects) {
    try {
      const o = JSON.parse(m[0]) as Record<string, unknown>;
      const lat = Number(o.lat ?? o.latitude);
      const lon = Number(o.lon ?? o.lng ?? o.longitude);
      if (isLat(lat) && isLon(lon)) return { lat, lon };
    } catch {
      // not JSON
    }
  }
  const nums = [...text.matchAll(/-?\d+(?:\.\d+)?/g)].map((m) => Number(m[0]));
  for (let i = nums.length - 2; i >= 0; i -= 1) {
    if (isLat(nums[i]) && isLon(nums[i + 1]) && (String(nums[i]).includes('.') || String(nums[i + 1]).includes('.'))) {
      return { lat: nums[i]!, lon: nums[i + 1]! };
    }
  }
  return null;
}

const round6 = (x: number): number => Math.round(x * 1e6) / 1e6;

async function main(): Promise<number> {
  const listed = (await (await fetch(`${envUrl}/photos`)).json()) as { photos: Photo[] };
  const photos = listed.photos.filter((p) => !p.answered);
  // One env request at a time, in a fixed order.
  const jpegs = new Map<string, string>();
  for (const p of photos) {
    const buf = Buffer.from(await (await fetch(`${envUrl}${p.url}`)).arrayBuffer());
    jpegs.set(p.id, buf.toString('base64'));
  }
  // Biggest photos first: their worst case is dearest, and the most budget is left at the start.
  const order = [...photos].sort((a, b) => b.bytes - a.bytes || (a.id < b.id ? -1 : 1));

  const placed = new Map<string, { lat: number; lon: number }>();
  /** Per photo, the rung of the ladder to try next. */
  const rung = new Map<string, number>(photos.map((p) => [p.id, 0]));
  let spent = 0;
  let requests = 0;
  const limit = BUDGET_USD - SAFETY_USD;

  for (let roundNo = 0; roundNo < 40; roundNo += 1) {
    const pending = order.filter((p) => !placed.has(p.id) && rung.get(p.id)! < LADDER.length);
    if (pending.length === 0) break;
    const batch: { photo: Photo; model: ModelSpec; body: string; worst: number }[] = [];
    let reserved = 0;
    for (const p of pending) {
      if (batch.length >= MAX_PER_ROUND || requests + batch.length >= MAX_REQUESTS) break;
      // The best model this photo could have on its own, with only what is already spent...
      let chosen: { model: ModelSpec; body: string; worst: number; r: number } | null = null;
      for (let r = rung.get(p.id)!; r < LADDER.length; r += 1) {
        const model = LADDER[r]!;
        const body = requestBody(model, jpegs.get(p.id)!);
        const worst = worstCaseUsd(model, body);
        if (spent + worst <= limit) {
          chosen = { model, body, worst, r };
          break;
        }
      }
      if (chosen === null) {
        // Nothing fits even alone: this photo gets the fallback point.
        rung.set(p.id, LADDER.length);
        continue;
      }
      // ...and it joins this round only if that model still fits beside the round's other calls, so a
      // photo is never downgraded just because it shared a round. Otherwise it waits for the next round.
      if (spent + reserved + chosen.worst > limit) break;
      rung.set(p.id, chosen.r);
      batch.push({ photo: p, model: chosen.model, body: chosen.body, worst: chosen.worst });
      reserved += chosen.worst;
    }
    if (batch.length === 0) continue;
    requests += batch.length;
    const results = await Promise.all(batch.map((b) => call(b.model, b.body)));
    batch.forEach((b, i) => {
      const r = results[i]!;
      spent += r.spent;
      const c = r.text === null ? null : coordinates(r.text);
      if (c !== null) {
        placed.set(b.photo.id, c);
        log(`${b.photo.id} ${b.model.name}: ${c.lat}, ${c.lon} (spent ${spent.toFixed(4)})`);
      } else {
        log(`${b.photo.id} ${b.model.name}: no coordinates${r.text ? `: ${r.text.slice(-200)}` : ''}`);
        rung.set(b.photo.id, rung.get(b.photo.id)! + 1);
      }
    });
  }

  const answers: Guess[] = photos.map((p) => {
    const c = placed.get(p.id) ?? FALLBACK;
    return { id: p.id, lat: round6(c.lat), lon: round6(c.lon) };
  });
  const res = await fetch(`${envUrl}/answers`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ answers }) });
  log(`${placed.size} of ${photos.length} placed by a model, ${requests} requests, about $${spent.toFixed(3)}: HTTP ${res.status} ${await res.text()}`);
  return res.ok ? 0 : 1;
}

/** After a crash: give every photo still unanswered the fallback point, so none counts the maximum error. */
async function rescue(): Promise<void> {
  const listed = (await (await fetch(`${envUrl}/photos`)).json()) as { photos: Photo[] };
  const answers = listed.photos.filter((p) => !p.answered).map((p) => ({ id: p.id, lat: FALLBACK.lat, lon: FALLBACK.lon }));
  if (answers.length === 0) return;
  const res = await fetch(`${envUrl}/answers`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ answers }) });
  log(`rescue: ${answers.length} fallback answers: HTTP ${res.status}`);
}

try {
  process.exitCode = await main();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exitCode = 1;
  try {
    await rescue();
  } catch (err2) {
    log(`rescue failed: ${String(err2)}`);
  }
}
