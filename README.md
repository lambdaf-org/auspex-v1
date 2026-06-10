# auspex

> A chat-archive analyst that cites every claim, computes confidence from falsification probes, and refuses to hand out IQ or MBTI scores.

![Rust](https://img.shields.io/badge/Rust-stable-orange)

Point auspex at your message exports and it builds a per-person profile and an interaction graph, all on your machine. Every observation links back to the exact message it came from. Every confidence number is computed by formula from counted evidence. Self-claims like "i'm an INTJ" get reconciled against behavior in a separate stream. Local Ollama for inference, local fastembed for retrieval, embedded D3 graph served on 127.0.0.1.

![auspex UI on the demo corpus: radar feed on the left, force-directed network in the center, evidence-cited sidebar on the right](assets/example.png)

## Quickstart

```bash
git clone https://github.com/lambdaf-org/auspex-v1
cd auspex-v1

ollama pull gpt-oss:20b                 # or qwen2.5:7b, llama3.1:8b, aya-expanse:8b
cp -r lexicons.example lexicons         # seed the gitignored translation files
cp config.example.json config.json      # then set self_name + handles + aliases
# drop chat exports into data/   format: YYYY-MM-DD HH:MM | sender | message

cargo build --release
OLLAMA_MODEL=gpt-oss:20b ./target/release/auspex data/*.txt
# server starts at http://localhost:8765/ when the pipeline finishes
```

Re-opening the UI without re-running the pipeline:

```bash
./target/release/auspex --serve     # or -s; serves the existing graph.html
```

Adding more exports later: drop them in `data/` and re-run the same command. The incremental pipeline does only the new work, and any person whose corpus fingerprint matches the previous run is loaded straight from cache.

## Features

- **Cited profiles.** Identity, anxiety, social style, growth, and vulnerability paragraphs per person. Every claim carries `support_ids` pointing at verbatim messages, and quotes are substring-verified before saving so hallucinated quotes get dropped.
- **Confidence computed from probes.** Phase 3 asks the model what behavior would contradict a theme. Phase 4 probes the embedding index for it one message at a time. Confidence is `(s+1)/(s+3·f+2)` from the counts, where `f` is the number of confirmed falsifications. The model never writes the number.
- **Self-claims reconciled separately.** A message like "i'm an INTJ" or "i have IQ 145" is routed out of trait extraction into its own stream and tagged `consistent`, `inconsistent`, `unverifiable`, or `not-literal` against behavioral evidence.
- **No IQ, no MBTI.** The synthesis prompt refuses to issue either, by design. Cognitive markers (abstract rate, conditional rate, lexical and integrative complexity, domain breadth, self-monitoring) are z-scored across your own corpus. Big Five ships as top-5 quoted messages per dimension with no score.
- **Real interaction graph.** Edges come from reply latency, who-addresses-whom, tone toward a specific person, topic overlap, and mentions. Thickness encodes intensity, color encodes warmth. Click an edge for the directional A to B and B to A breakdown with baseline and recent windows.
- **Ranked insight feed.** Each run diffs against the previous snapshot and ranks changes by urgency (`relationship_cooling`, `tone_shift_toward`, `alliance_forming`, `cognitive_shift`, `high_stakes_claim`, and more). Urgency is delta magnitude times baseline confidence times novelty.
- **Fully local.** One Rust binary, an embedded `std::net` HTTP server bound to 127.0.0.1, and a D3 UI delivered inline. No cloud, no telemetry, no remote API, no update server.

## How it works

Each person runs through a seven-phase pipeline:

```
parse → phase 0  classify every message (function, register, self-statement detection)
        phase 1  extract observations with cited, verified quotes
        phase 2  cluster observations into themes
        phase 3  deepen themes + emit falsification specs
        phase 4  probe each spec → validated theme + computed confidence
        phase 5  cognitive markers, Big Five citations, self-claim reconciliation, synthesis
        phase 6  predictions (intent clusters × validated themes)
                          │
        cross-person z-score · pair interactions · insight diff vs snapshot
                          │
              graph.html → embedded HTTP server :8765 → you
```

Measurements split into two windows: the full corpus (baseline) and the most recent quartile by timestamp. Themes carry a `recent_share` and a status of `active`, `stable`, or `fading`, so what is changing is a first-class output. The confidence formula is a smoothed proportion: falsifying evidence weighs 3x because a clean counter-example is more diagnostic than one more supporting example, and the Laplace `+1 / +2` terms return 0.5 under zero evidence. So 23 supporting messages with 2 of 9 falsification probes confirmed gives a confidence of 0.77.

## Configuration

`config.json` (gitignored, see `config.example.json`) sets your `self_name`, `self_handles`, and per-person `aliases`. Environment variables:

| Var | Default | Purpose |
|---|---|---|
| `OLLAMA_MODEL` | `llama3.2` | Used for every LLM call. `gpt-oss:20b` for synthesis depth; `qwen2.5:7b` / `aya-expanse:8b` / `llama3.1:8b` for speed and multilingual corpora. |
| `AUSPEX_PORT` | `8765` | HTTP server port. |
| `AUSPEX_NO_SERVE` | unset | Set to anything to skip the embedded server. |

Lexicons live as plain text under `lexicons/` (gitignored), one entry per line, `#` for comments. The loader merges every `.txt` under each category and is language-agnostic; `lexicons.example/` ships as an English-only template.

## Honest limits

- **Construct validity is unverified.** auspex measures textual signatures of communicative behavior. Chat-derived Big Five correlates roughly 0.3 to 0.5 with self-report inventories even under ideal conditions. Read outputs as one disciplined view of a message stream.
- **Pair tone-toward-person uses an adjacency heuristic.** When a message is classified `addressee: specific`, tone is attributed to whoever was replied to in chronological order. Interleaved group chat can misattribute. Reply-threading is on the roadmap.
- **First-run insights are sparse.** Most insight types are diff-based and need a prior snapshot. The first run yields only `high_stakes_claim` insights; value climbs from run two onward.
- **Chat context does not scale infinitely.** The chat panel dumps every profile into context, so around 50 people approaches the limit of a 128k-context model. Retrieval-augmented chat is on the roadmap.
- **LLM-dependent.** Phase 0 quality drives everything downstream. Test a weak or low-resource-language model on a sample before a multi-hour run.

## Stack

- Rust, one binary, ~5,255 LoC across 14 modules, no async runtime
- fastembed 5, local AllMiniLML6V2 embeddings, no GPU required
- regex, serde, serde_json, ureq
- `std::net` embedded HTTP server (no tokio, hyper, or axum)
- D3.js and marked.js delivered inline, no build step
- Ollama for local inference

## Contributing

Lambdaforge is open source and contributions are welcome. Start with the [contributor guide](https://github.com/lambdaf-org/contributing), and see the org-wide [CONTRIBUTING](https://github.com/lambdaf-org/.github/blob/main/CONTRIBUTING.md) and [Code of Conduct](https://github.com/lambdaf-org/.github/blob/main/CODE_OF_CONDUCT.md).

## License

This repository does not yet include a `LICENSE` file, so default copyright applies for now. A license is coming soon. If you want to use or build on this before then, please open an issue.
