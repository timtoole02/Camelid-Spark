// Structural analyzer for OpenAI-style chat-completion SSE streams.
// Usage: node analyze_sse.mjs <file.sse>
// Emits a structural summary (NOT token values) for parity comparison between
// the llama-server oracle and Camelid: chunk count, ordering of finish_reason vs
// usage, whether content chunks omit `usage`, terminal empty-choices shape, and
// [DONE] placement.
import { readFileSync } from "node:fs";

const file = process.argv[2];
const raw = readFileSync(file, "utf8");
// SSE frames are separated by blank lines; each data frame is `data: <payload>`.
const frames = raw
  .split(/\r?\n/)
  .filter((l) => l.startsWith("data:"))
  .map((l) => l.slice(5).trim());

let doneSeen = false;
let doneIndex = -1;
const chunks = [];
frames.forEach((f, i) => {
  if (f === "[DONE]") {
    doneSeen = true;
    doneIndex = i;
    return;
  }
  chunks.push({ i, obj: JSON.parse(f) });
});

const hasUsage = (c) => Object.prototype.hasOwnProperty.call(c.obj, "usage");
const usageChunks = chunks.filter(hasUsage);
const finishChunks = chunks.filter(
  (c) => Array.isArray(c.obj.choices) && c.obj.choices.some((ch) => ch.finish_reason != null)
);
const contentChunks = chunks.filter(
  (c) => Array.isArray(c.obj.choices) && c.obj.choices.length > 0 && !hasUsage(c)
);
const emptyChoiceChunks = chunks.filter(
  (c) => Array.isArray(c.obj.choices) && c.obj.choices.length === 0
);

const lastFinish = finishChunks.length ? finishChunks[finishChunks.length - 1].i : -1;
const firstUsage = usageChunks.length ? usageChunks[0].i : -1;

const summary = {
  file,
  total_data_frames: frames.length,
  chunk_count_excluding_done: chunks.length,
  done_seen: doneSeen,
  done_is_last_frame: doneSeen && doneIndex === frames.length - 1,
  content_chunks_all_omit_usage: contentChunks.every((c) => !hasUsage(c)),
  usage_chunk_count: usageChunks.length,
  terminal_usage_chunk: usageChunks.length
    ? {
        choices_is_empty_array:
          Array.isArray(usageChunks[usageChunks.length - 1].obj.choices) &&
          usageChunks[usageChunks.length - 1].obj.choices.length === 0,
        has_id: "id" in usageChunks[usageChunks.length - 1].obj,
        has_created: "created" in usageChunks[usageChunks.length - 1].obj,
        has_model: "model" in usageChunks[usageChunks.length - 1].obj,
        object: usageChunks[usageChunks.length - 1].obj.object,
        usage_keys: Object.keys(usageChunks[usageChunks.length - 1].obj.usage).sort(),
        usage_values: usageChunks[usageChunks.length - 1].obj.usage,
      }
    : null,
  empty_choice_chunk_count: emptyChoiceChunks.length,
  usage_after_finish_reason: firstUsage > lastFinish && lastFinish >= 0,
  usage_before_done: usageChunks.length ? firstUsage < doneIndex : null,
};

console.log(JSON.stringify(summary, null, 2));
