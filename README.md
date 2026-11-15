# free-model-aggregator-01

A free-model aggregator for the make.fun `free-model-aggregator-01` bounty: one Node 22
server (`main.ts`, standard library only) that serves the OpenAI Chat Completions API and
the Anthropic Messages API — JSON and SSE — over the ten simulated free-tier providers a
trial hands it.

- **Quality routing:** every free model that meets a request's requirements (tools, vision,
  JSON mode, minimum context) is ranked by the quality table's `rating_milli`; the best
  standing model serves the request.
- **Failover:** a provider refusal (402, 429 or 5xx) is retried once, then the request moves
  to the next-best candidate; a provider that refuses repeatedly is treated as spent and
  drops behind the standing candidates.
- **All three upstream wires:** requests are rewritten into each provider's own wire
  (OpenAI Chat Completions, Anthropic Messages, Gemini generateContent), preserving the
  system prompt, tool names, `max_tokens`, stop sequences, images and JSON mode, and the
  answer is rendered back in the client's wire with its usage intact.
- **Catalog:** `node main.ts --list` prints the registered catalog — every provider that
  answered its catalog probe, with each model's free flag from the quality table.

Deterministic: no wall clock, no randomness, no model calls at runtime.

MIT licensed — see `LICENSE`.
