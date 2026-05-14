//! The LLM pipeline: classification → observation → theme → falsification →
//! cognitive markers → Big Five → self-claim reconciliation → synthesis → predictions.
//! Plus profile_person which orchestrates everything for one sender, and
//! cross-person z-score calibration.

use crate::index::MessageIndex;
use crate::lexicons::*;
use crate::llm::*;
use crate::math::*;
use crate::metrics::*;
use crate::persist::*;
use crate::types::*;
use fastembed::TextEmbedding;
use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::time::Instant;

// ─── phase 0: message function classification ───
// classifies each message by what it IS (joke, quote, self-statement, code, ...)
// so downstream phases don't treat "i have 145 iq" the same as a behavioral signal.

/// Heuristic shortcut. Returns Some(classification) for messages that don't need an LLM call —
/// i.e. URLs, code blocks, and ≤2-char garble. We deliberately only auto-tag what's visually
/// unambiguous, since short messages can carry real signal ("ja", "no", "fr", "bruh" are not noise).
pub(crate) fn heuristic_tag(text: &str) -> Option<MsgClassification> {
    let t = text.trim();
    // sensible defaults for any heuristic-shortcut message so downstream code that
    // checks enum values doesn't have to special-case empty strings.
    let base = || MsgClassification {
        function: "literal".into(),
        speech_act: "share".into(),
        modality: "factual".into(),
        topic: "media".into(),
        addressee: "group".into(),
        tone: "neutral".into(),
        ..Default::default()
    };
    if t.is_empty() {
        return Some(MsgClassification {
            is_low_info: true,
            topic: "none".into(),
            ..base()
        });
    }
    let lower = t.to_lowercase();
    if (lower.starts_with("http://") || lower.starts_with("https://")) && !t.contains(' ') {
        return Some(MsgClassification {
            is_code_or_url: true,
            ..base()
        });
    }
    let url_share = [
        "tenor.com/", "media.discordapp", "cdn.discordapp", "youtu.be/", "youtube.com/watch",
    ];
    if url_share.iter().any(|p| lower.contains(p)) && t.split_whitespace().count() <= 4 {
        return Some(MsgClassification {
            is_code_or_url: true,
            ..base()
        });
    }
    if t.starts_with("```") || (t.starts_with('`') && t.ends_with('`') && t.len() >= 3) {
        return Some(MsgClassification {
            is_code_or_url: true,
            topic: "code".into(),
            ..base()
        });
    }
    None
}

pub(crate) fn phase0_classify(
    name: &str,
    messages: &[&RawMessage],
    corpus: &[RawMessage],
    id_to_pos: &HashMap<String, usize>,
    model: &str,
) -> HashMap<String, MsgClassification> {
    let mut out: HashMap<String, MsgClassification> = HashMap::new();

    // load cache + backfill legacy fields. Re-classify any entry that's missing
    // the rich schema (no `function` filled in) so old thin cache regenerates.
    if let Some(cached) = load_json::<Vec<MsgClassification>>(name, "classifications") {
        for mut c in cached {
            c.derive_legacy_fields();
            out.insert(c.msg_id.clone(), c);
        }
        eprintln!("    phase 0: loaded {} cached classifications", out.len());
    } else if let Some(partial) =
        load_json::<Vec<MsgClassification>>(name, "classifications_partial")
    {
        for mut c in partial {
            c.derive_legacy_fields();
            out.insert(c.msg_id.clone(), c);
        }
        eprintln!(
            "    phase 0: resumed from {} partial classifications",
            out.len()
        );
    }

    let mut to_classify: Vec<&&RawMessage> = Vec::new();
    for m in messages {
        // already have a rich classification for this message → skip
        if let Some(c) = out.get(&m.id) {
            if !c.function.is_empty() {
                continue;
            }
            // legacy thin cache entry → re-classify with rich prompt
            out.remove(&m.id);
        }
        if let Some(mut c) = heuristic_tag(&m.text) {
            c.msg_id = m.id.clone();
            c.derive_legacy_fields();
            out.insert(m.id.clone(), c);
        } else {
            to_classify.push(m);
        }
    }

    let batch_size = 10;
    let total_batches = (to_classify.len() + batch_size - 1) / batch_size;
    let start = Instant::now();
    let mut done_batches = 0usize;

    for batch in to_classify.chunks(batch_size) {
        done_batches += 1;
        eprint!(
            "\r    phase 0: classify {}/{}    ",
            done_batches, total_batches
        );
        std::io::stderr().flush().ok();

        // build the block with 2-3 prior messages of conversational context per target.
        // This gives the classifier enough surrounding chat to disambiguate references —
        // e.g. "devices mache ig" reads as a profession claim out of context, but with
        // the prior "what schoolwork are you doing" line it's obviously a project response.
        let mut block = String::new();
        for (i, m) in batch.iter().enumerate() {
            block.push_str(&format!("MSG {}:\n", i + 1));
            block.push_str("[CONTEXT — do not classify, just for disambiguation]\n");
            let pos = id_to_pos.get(&m.id).copied().unwrap_or(0);
            let start = pos.saturating_sub(3);
            let mut ctx_lines = 0;
            for ctx_pos in start..pos {
                if let Some(ctx) = corpus.get(ctx_pos) {
                    let ctx_txt: String = ctx.text.chars().take(160).collect();
                    block.push_str(&format!("  {}: {}\n", ctx.sender, ctx_txt));
                    ctx_lines += 1;
                }
            }
            if ctx_lines == 0 {
                block.push_str("  (no prior context — first message in stream)\n");
            }
            let txt: String = m.text.chars().take(500).collect();
            block.push_str(&format!("[TARGET to classify]\n  {}\n\n", txt));
        }

        let prompt = format!(
            "You are classifying Discord messages from ONE person. \
             Messages may be in any language or mixed across languages. \
             Classify by FUNCTION regardless of language — do NOT tag low-info just because it's not English.\n\n\
             For EACH message, fill ALL these fields:\n\n\
             FUNCTION (pick one): literal | ironic | joke | dramatic | rhetorical | quoted\n\
             SPEECH_ACT (pick one): assert | ask | command | concede | challenge | agree | disagree | hedge | exclaim\n\
             MODALITY (pick one): factual | opinion | hypothetical | normative | speculative\n\
             TOPIC: 1-2 word label (e.g. work, relationships, tech, mood, request, food, money, philosophy)\n\
             ADDRESSEE (pick one): self | specific | group | none\n\
             TONE: ONE word emotional tone (frustrated, calm, enthusiastic, anxious, defensive, warm, curious, bored, hostile, resigned, deadpan, amused, ...)\n\n\
             BOOLEANS (true/false):\n\
             self_statement = claim about themselves (identity, mood, traits, scores, history)\n\
             quoting = quoting/citing external content\n\
             code_or_url = code snippet, command, URL\n\
             low_info = genuinely empty of interpretable content (NOT just short or in another language)\n\
             conditional = uses if/when/unless structure or counterfactual reasoning\n\
             meta_cognitive = ANY self-reflection: noticing their own thinking/feeling/reaction, commenting on \
             how they're communicating, second-guessing themselves, catching themselves doing something, \
             explaining why they said what they said. Don't reserve this for deep philosophy — \
             \"i think i'm overthinking this\" counts, \"hmm wait actually no\" counts, \"i realize that...\" counts.\n\
             multi_perspective = considers more than one angle: contrasts views, considers a counterargument, \
             notes how someone else might see it, weighs two options, says \"on one hand X but on the other Y\". \
             Doesn't need to be balanced — even mentioning the alternative once counts.\n\n\
             Under-using these two booleans is a known failure mode. If a message contains ANY trace of \
             self-noticing or perspective-weighing, mark it true.\n\n\
             AFFECT:\n\
             valence: integer -2 (strong neg) to +2 (strong pos), 0 = neutral\n\
             intensity: integer 0 (flat) to 3 (extreme)\n\n\
             IF self_statement is true, ALSO fill:\n\
             self_claim: the verbatim claim phrase from the message\n\
             claim_dimension: intelligence | mood | profession | identity | history | preference | trait | other\n\
             claim_register: serious | ironic | hyperbole\n\
             claim_certainty: hedged | confident | absolute\n\
             else: leave those four as null.\n\n\
             implies: ONE short phrase of what's strongly implied but NOT explicitly said, or null.\n\n\
             Under-tagging is wrong. If a message is both opinion AND a joke, mark function=joke; if a self-statement is ironic, mark claim_register=ironic. Don't punt to 'low_info=true' just because language is non-English.\n\n\
             USE THE CONTEXT to resolve ambiguity. If \"devices mache ig\" follows \"what are you building for schoolwork?\" then it's a PROJECT description, NOT a profession claim. claim_dimension should reflect what the speaker actually meant given the surrounding conversation, not the most generic reading of the text in isolation. If context suggests the claim is contextual/situational (talking about a current task, a temporary state) rather than identity-defining, set claim_certainty to `hedged` or set is_self_statement=false.\n\n\
             Messages:\n{}\n\n\
             Return JSON: {{\"items\":[{{\"i\":1,\"function\":\"...\",\"speech_act\":\"...\",\"modality\":\"...\",\"topic\":\"...\",\"addressee\":\"...\",\"tone\":\"...\",\"self_statement\":false,\"quoting\":false,\"code_or_url\":false,\"low_info\":false,\"conditional\":false,\"meta_cognitive\":false,\"multi_perspective\":false,\"valence\":0,\"intensity\":0,\"self_claim\":null,\"claim_dimension\":null,\"claim_register\":null,\"claim_certainty\":null,\"implies\":null}}]}}\n\
             ONLY JSON.",
            block
        );

        let parsed = llm_json(model, &prompt);
        let items = parsed.as_ref().and_then(|j| {
            j.get("items")
                .and_then(|v| v.as_array())
                .or_else(|| j.as_array())
                .cloned()
        });

        if let Some(arr) = items {
            for item in arr {
                let i = item.get("i").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                if i == 0 || i > batch.len() {
                    continue;
                }
                let m = &batch[i - 1];
                let str_or = |k: &str, default: &str| -> String {
                    item.get(k)
                        .and_then(|v| v.as_str())
                        .filter(|s| !s.is_empty() && *s != "null")
                        .unwrap_or(default)
                        .to_string()
                };
                let bool_or = |k: &str| -> bool {
                    item.get(k).and_then(|v| v.as_bool()).unwrap_or(false)
                };
                let int_or = |k: &str, default: i32, lo: i32, hi: i32| -> i32 {
                    let v = item
                        .get(k)
                        .and_then(|v| v.as_i64())
                        .map(|n| n as i32)
                        .unwrap_or(default);
                    v.max(lo).min(hi)
                };
                let opt_str = |k: &str| -> Option<String> {
                    item.get(k)
                        .and_then(|v| v.as_str())
                        .filter(|s| !s.is_empty() && *s != "null" && *s != "None")
                        .map(|s| s.to_string())
                };

                let mut c = MsgClassification {
                    msg_id: m.id.clone(),
                    function: str_or("function", "literal"),
                    speech_act: str_or("speech_act", "assert"),
                    modality: str_or("modality", "opinion"),
                    topic: str_or("topic", "general"),
                    addressee: str_or("addressee", "group"),
                    tone: str_or("tone", "neutral"),
                    is_self_statement: bool_or("self_statement"),
                    is_quoting: bool_or("quoting"),
                    is_code_or_url: bool_or("code_or_url"),
                    is_low_info: bool_or("low_info"),
                    is_conditional: bool_or("conditional"),
                    is_meta_cognitive: bool_or("meta_cognitive"),
                    is_multi_perspective: bool_or("multi_perspective"),
                    valence: int_or("valence", 0, -2, 2),
                    intensity: int_or("intensity", 0, 0, 3),
                    self_claim: opt_str("self_claim"),
                    claim_dimension: str_or("claim_dimension", ""),
                    claim_register: str_or("claim_register", ""),
                    claim_certainty: str_or("claim_certainty", ""),
                    implies: opt_str("implies"),
                    ..Default::default()
                };
                c.derive_legacy_fields();
                out.insert(m.id.clone(), c);
            }
        }

        // checkpoint to the MAIN file every 5 batches — atomic, kill-safe.
        if done_batches % 5 == 0 {
            let v: Vec<MsgClassification> = out.values().cloned().collect();
            let _ = save_json(name, "classifications", &v);
        }
    }

    // anything still missing → conservative default
    for m in messages {
        out.entry(m.id.clone()).or_insert_with(|| {
            let mut c = MsgClassification {
                msg_id: m.id.clone(),
                function: "literal".into(),
                modality: "opinion".into(),
                topic: "general".into(),
                addressee: "group".into(),
                tone: "neutral".into(),
                ..Default::default()
            };
            c.derive_legacy_fields();
            c
        });
    }

    let elapsed = start.elapsed().as_secs();
    eprintln!(
        "\r    phase 0: done. {} classified in {}m{}s              ",
        out.len(),
        elapsed / 60,
        elapsed % 60
    );
    let v: Vec<MsgClassification> = out.values().cloned().collect();
    let _ = save_json(name, "classifications", &v);
    out
}

// excluded from trait extraction — these channels aren't authentic-self signal
pub(crate) fn is_excluded_for_traits(c: &MsgClassification) -> bool {
    c.is_low_info
        || c.is_code_or_url
        || c.is_quoting
        || c.function == "joke"
        || c.function == "quoted"
}

// excluded from lexical/style metrics — these are noise in vocab/length stats
pub(crate) fn is_excluded_for_style(c: &MsgClassification) -> bool {
    c.is_low_info || c.is_code_or_url || c.is_quoting || c.function == "quoted"
}

// ─── phase 1: micro-observation scan ───

pub(crate) fn phase1_scan(
    name: &str,
    messages: &[&RawMessage],
    classifications: &HashMap<String, MsgClassification>,
    model: &str,
) -> Vec<Observation> {
    // load any existing observations and figure out which messages have already been scanned.
    // we use the union of `support_ids` across observations as the "already-covered" set.
    // Fall back to the _partial sidecar so a prior interrupted run's work isn't lost.
    let mut all_obs: Vec<Observation> = load_json::<Vec<Observation>>(name, "observations")
        .or_else(|| {
            let partial = load_json::<Vec<Observation>>(name, "observations_partial");
            if partial.is_some() {
                eprintln!(
                    "    phase 1: recovering from observations_partial (no main file)"
                );
            }
            partial
        })
        .unwrap_or_default();
    let covered: HashSet<String> = all_obs
        .iter()
        .flat_map(|o| o.support_ids.iter().cloned())
        .collect();

    // only authentic-self channels go in: drop jokes, code, urls, quotes, low-info, roleplay.
    // also drop anything already covered by a prior observation — that's the incremental hook.
    let substantive: Vec<&&RawMessage> = messages
        .iter()
        .filter(|m| m.text.chars().count() >= 30)
        .filter(|m| {
            classifications
                .get(&m.id)
                .map_or(true, |c| !is_excluded_for_traits(c))
        })
        .filter(|m| !covered.contains(&m.id))
        .collect();

    if substantive.is_empty() {
        eprintln!(
            "    phase 1: nothing new to scan ({} cached observations cover this corpus)",
            all_obs.len()
        );
        return all_obs;
    }

    if !covered.is_empty() {
        eprintln!(
            "    phase 1: {} new substantive msgs (incremental — {} previously covered)",
            substantive.len(),
            covered.len()
        );
    }

    let batch_size = 6;
    let total_batches = (substantive.len() + batch_size - 1) / batch_size;
    let cached_count = all_obs.len();
    let start = Instant::now();

    for (batch_idx, batch) in substantive.chunks(batch_size).enumerate() {
        eprint!(
            "\r    phase 1: batch {}/{} ({} observations)    ",
            batch_idx + 1,
            total_batches,
            all_obs.len()
        );
        std::io::stderr().flush().ok();

        // build the message block with inline Phase 0 metadata as a hint.
        // tone + implies + register give the model a richer starting point per message.
        let mut msg_block = String::new();
        for (i, m) in batch.iter().enumerate() {
            let txt: String = m.text.chars().take(600).collect();
            let meta = classifications.get(&m.id).map(|c| {
                let mut bits: Vec<String> = Vec::new();
                if !c.tone.is_empty() && c.tone != "neutral" {
                    bits.push(format!("tone:{}", c.tone));
                }
                if !c.function.is_empty() && c.function != "literal" {
                    bits.push(format!("register:{}", c.function));
                }
                if !c.modality.is_empty() && c.modality != "opinion" {
                    bits.push(format!("mod:{}", c.modality));
                }
                if c.is_meta_cognitive {
                    bits.push("meta-cognitive".into());
                }
                if c.is_multi_perspective {
                    bits.push("multi-perspective".into());
                }
                if let Some(imp) = &c.implies {
                    if !imp.is_empty() {
                        bits.push(format!("implies:{}", imp.chars().take(80).collect::<String>()));
                    }
                }
                if bits.is_empty() {
                    String::new()
                } else {
                    format!("  [{}]", bits.join(" · "))
                }
            }).unwrap_or_default();
            msg_block.push_str(&format!("{}. {}{}\n", i + 1, txt, meta));
        }

        let prompt = format!(
            "These messages are by ONE person. Find 3-5 SPECIFIC psychological patterns (not generic).\n\
             For each pattern, you MUST cite the message numbers that drove it AND quote a verbatim phrase from one of them.\n\n\
             Do NOT take self-claims at face value (e.g. \"i'm an INTJ\" is a claim, not evidence of INTJ-ness — \
             treat it as data about self-presentation).\n\
             Do NOT restate what was said. Infer who they ARE.\n\n\
             Messages:\n{}\n\n\
             Return JSON:\n\
             {{\"observations\":[\n\
             {{\"trait\":\"specific 1-line pattern\",\
             \"evidence\":\"verbatim phrase from a cited message\",\
             \"dimension\":\"personality|cognition|emotional|social|defense|values|presentation\",\
             \"polarity\":\"exhibits\" or \"opposite-implied\",\
             \"support\":[<message numbers>]}}\
             ]}}\n\
             ONLY JSON.",
            msg_block
        );

        if let Some(json) = llm_json(model, &prompt) {
            let arr_opt = json
                .get("observations")
                .and_then(|v| v.as_array())
                .cloned()
                .or_else(|| json.as_array().cloned());

            if let Some(arr) = arr_opt {
                for o in arr {
                    let trait_name = match o.get("trait").and_then(|v| v.as_str()) {
                        Some(s) if !s.is_empty() => s.to_string(),
                        _ => continue,
                    };
                    let evidence = o
                        .get("evidence")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .trim()
                        .to_string();
                    let dimension = o
                        .get("dimension")
                        .and_then(|v| v.as_str())
                        .unwrap_or("personality")
                        .to_string();
                    let polarity = o
                        .get("polarity")
                        .and_then(|v| v.as_str())
                        .unwrap_or("exhibits")
                        .to_string();
                    let support_nums: Vec<usize> = o
                        .get("support")
                        .and_then(|v| v.as_array())
                        .map(|a| {
                            a.iter()
                                .filter_map(|x| x.as_u64().map(|n| n as usize))
                                .filter(|&n| n >= 1 && n <= batch.len())
                                .collect()
                        })
                        .unwrap_or_default();

                    let mut support_ids: Vec<String> = support_nums
                        .iter()
                        .map(|&n| batch[n - 1].id.clone())
                        .collect();

                    // verify evidence quote actually appears in one of the cited (or batch) messages
                    let needle = evidence.to_lowercase();
                    let needle_short = needle.chars().take(40).collect::<String>();
                    let mut verified = false;
                    if !evidence.is_empty() {
                        let cited: Vec<&&RawMessage> = if !support_nums.is_empty() {
                            support_nums.iter().map(|&n| batch[n - 1]).collect()
                        } else {
                            batch.iter().copied().collect()
                        };
                        for m in cited {
                            if m.text.to_lowercase().contains(&needle_short)
                                || m.text.to_lowercase().contains(&needle)
                            {
                                verified = true;
                                if support_ids.is_empty() {
                                    support_ids.push(m.id.clone());
                                }
                                break;
                            }
                        }
                    }
                    // if LLM gave no support nums and no evidence, fall back to whole batch
                    if support_ids.is_empty() {
                        if evidence.is_empty() {
                            // no anchor — drop
                            continue;
                        }
                        // last resort: scan full batch for evidence substring
                        for m in batch.iter() {
                            if m.text.to_lowercase().contains(&needle_short) {
                                support_ids.push(m.id.clone());
                                verified = true;
                                break;
                            }
                        }
                    }
                    if !verified && !evidence.is_empty() {
                        // evidence claimed but not present anywhere → likely hallucinated; drop
                        continue;
                    }
                    if support_ids.is_empty() {
                        continue;
                    }

                    all_obs.push(Observation {
                        trait_name,
                        evidence,
                        dimension,
                        polarity,
                        support_ids,
                    });
                }
            }
        }

        // checkpoint to the MAIN file every 5 batches. Atomic via tmp+rename in save_json,
        // so a hard kill at any point leaves the file consistent and recoverable.
        if (batch_idx + 1) % 5 == 0 {
            let _ = save_json(name, "observations", &all_obs);
        }
    }

    let elapsed = start.elapsed().as_secs();
    let new_obs = all_obs.len() - cached_count;
    eprintln!(
        "\r    phase 1: done. +{} new ({} total) in {}m{}s              ",
        new_obs,
        all_obs.len(),
        elapsed / 60,
        elapsed % 60
    );
    let _ = save_json(name, "observations", &all_obs);

    // intelligent update: if we added new observations, downstream caches are stale.
    // invalidate them so the next phases recompute against the new data.
    if new_obs > 0 && cached_count > 0 {
        eprintln!(
            "    phase 1: invalidating downstream caches ({} new obs)",
            new_obs
        );
        invalidate_phase_caches(
            name,
            &[
                "themes_raw",
                "themes_deep",
                "validated",
                "cognitive",
                "cognitive_recent",
                "big_five",
                "big_five_recent",
                "self_claims",
                "interpretation",
                "predictions",
            ],
        );
    }

    all_obs
}

pub(crate) fn invalidate_phase_caches(name: &str, suffixes: &[&str]) {
    for s in suffixes {
        let _ = std::fs::remove_file(format!("{}/{}_{}.json", PROFILE_DIR, name, s));
    }
}

// ─── phase 2: cluster observations into themes ───

pub(crate) fn phase2_cluster(
    name: &str,
    observations: &[Observation],
    emb_model: &mut TextEmbedding,
) -> Vec<(String, Vec<usize>)> {
    if let Some(cached) = load_json::<Vec<(String, Vec<usize>)>>(name, "themes_raw") {
        eprintln!("    phase 2: loaded {} cached themes", cached.len());
        return cached;
    }

    if observations.is_empty() {
        return Vec::new();
    }

    eprint!(
        "    phase 2: embedding {} observations... ",
        observations.len()
    );
    std::io::stderr().flush().ok();

    let texts: Vec<String> = observations.iter().map(|o| o.trait_name.clone()).collect();
    let embeddings = match emb_model.embed(texts, None) {
        Ok(e) => e,
        Err(_) => {
            eprintln!("embedding failed");
            return Vec::new();
        }
    };

    let k = ((observations.len() as f64).sqrt() / 2.5)
        .max(4.0)
        .min(15.0) as usize;
    let assignments = kmeans(&embeddings, k, 20);

    let mut themes: Vec<(String, Vec<usize>)> = Vec::new();
    for j in 0..k {
        let indices: Vec<usize> = assignments
            .iter()
            .enumerate()
            .filter(|(_, &a)| a == j)
            .map(|(i, _)| i)
            .collect();
        if indices.is_empty() {
            continue;
        }

        // find the observation closest to centroid as theme name
        let members: Vec<&Vec<f32>> = indices.iter().map(|&i| &embeddings[i]).collect();
        let centroid = vec_mean(&members);
        let best_idx = *indices
            .iter()
            .max_by(|&&a, &&b| {
                cosine_sim(&embeddings[a], &centroid)
                    .partial_cmp(&cosine_sim(&embeddings[b], &centroid))
                    .unwrap()
            })
            .unwrap();
        let theme_name = observations[best_idx].trait_name.clone();
        themes.push((theme_name, indices));
    }

    themes.sort_by(|a, b| b.1.len().cmp(&a.1.len()));
    eprintln!("{} themes", themes.len());
    let _ = save_json(name, "themes_raw", &themes);
    themes
}

// ─── phase 3: deepen each theme ───

pub(crate) fn phase3_deepen(
    name: &str,
    themes: &[(String, Vec<usize>)],
    observations: &[Observation],
    id_to_msg: &HashMap<String, &RawMessage>,
    model: &str,
) -> Vec<DeepTheme> {
    if let Some(cached) = load_json::<Vec<DeepTheme>>(name, "themes_deep") {
        eprintln!("    phase 3: loaded {} cached analyses", cached.len());
        return cached;
    }

    let mut results: Vec<DeepTheme> = Vec::new();
    let top = themes.iter().take(10);

    for (i, (theme_name, indices)) in top.enumerate() {
        eprint!(
            "\r    phase 3: deepening theme {}/{}    ",
            i + 1,
            themes.len().min(10)
        );
        std::io::stderr().flush().ok();

        // stratified pull of support_ids: round-robin across member observations
        let mut seen_ids: HashSet<String> = HashSet::new();
        let mut support_ids: Vec<String> = Vec::new();
        for round in 0..8 {
            for &obs_i in indices {
                if let Some(sid) = observations[obs_i].support_ids.get(round) {
                    if seen_ids.insert(sid.clone()) {
                        support_ids.push(sid.clone());
                    }
                }
            }
            if support_ids.len() >= 24 {
                break;
            }
        }
        support_ids.truncate(24);

        // build a message block from the supporting messages — verbatim, full text
        let mut msg_block = String::new();
        let mut included: Vec<(usize, &RawMessage)> = Vec::new();
        for sid in &support_ids {
            if let Some(m) = id_to_msg.get(sid) {
                let n = included.len() + 1;
                let txt: String = m.text.chars().take(400).collect();
                msg_block.push_str(&format!("{}. {}\n", n, txt));
                included.push((n, *m));
            }
        }
        if included.is_empty() {
            continue;
        }

        // also show a couple member observation labels for context
        let mut obs_block = String::new();
        for &obs_i in indices.iter().take(5) {
            obs_block.push_str(&format!(
                "  - [{}] {}\n",
                observations[obs_i].dimension, observations[obs_i].trait_name
            ));
        }

        let prompt = format!(
            "Hypothesis about ONE person: \"{}\".\n\
             Member observations:\n{}\n\
             Supporting messages (verbatim, numbered):\n{}\n\n\
             1. analysis: 2-3 sentences on what this pattern reveals psychologically. Be specific. \
             Avoid clichés. Don't restate the hypothesis.\n\
             2. quotes: 3-5 verbatim phrases pulled from the numbered messages above (do NOT invent). \
             For each, give the message number it came from.\n\
             3. falsifications: 2-3 specific BEHAVIORS that, if observed, would weaken or contradict \
             this hypothesis. For each behavior, give 2-3 search queries that would surface examples.\n\n\
             Return JSON:\n\
             {{\
             \"analysis\":\"...\",\
             \"quotes\":[{{\"i\":<msg_number>,\"quote\":\"...\"}}],\
             \"falsifications\":[{{\"behavior\":\"...\",\"queries\":[\"...\",\"...\"]}}]\
             }}\n\
             ONLY JSON.",
            theme_name, obs_block, msg_block
        );

        let mut analysis = String::new();
        let mut quotes: Vec<EvidenceQuote> = Vec::new();
        let mut falsifications: Vec<FalsificationSpec> = Vec::new();

        if let Some(json) = llm_json(model, &prompt) {
            analysis = json["analysis"].as_str().unwrap_or("").to_string();

            if let Some(arr) = json.get("quotes").and_then(|v| v.as_array()) {
                for q in arr {
                    let msg_num = q.get("i").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                    let qtext = q
                        .get("quote")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .trim()
                        .to_string();
                    if qtext.is_empty() {
                        continue;
                    }
                    // find the message that contains this quote
                    let needle = qtext.to_lowercase();
                    let needle_short: String = needle.chars().take(40).collect();
                    let candidate = included
                        .iter()
                        .find(|(n, _)| *n == msg_num)
                        .map(|(_, m)| *m);
                    let actual = match candidate {
                        Some(m)
                            if m.text.to_lowercase().contains(&needle)
                                || m.text.to_lowercase().contains(&needle_short) =>
                        {
                            Some(m)
                        }
                        _ => included
                            .iter()
                            .find(|(_, m)| {
                                m.text.to_lowercase().contains(&needle_short)
                            })
                            .map(|(_, m)| *m),
                    };
                    if let Some(m) = actual {
                        quotes.push(EvidenceQuote {
                            quote: qtext,
                            msg_id: m.id.clone(),
                        });
                    }
                }
            }

            if let Some(arr) = json.get("falsifications").and_then(|v| v.as_array()) {
                for f in arr {
                    let behavior = f
                        .get("behavior")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .trim()
                        .to_string();
                    if behavior.is_empty() {
                        continue;
                    }
                    let queries: Vec<String> = f
                        .get("queries")
                        .and_then(|v| v.as_array())
                        .map(|a| {
                            a.iter()
                                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                                .filter(|s| !s.is_empty())
                                .collect()
                        })
                        .unwrap_or_default();
                    if queries.is_empty() {
                        continue;
                    }
                    falsifications.push(FalsificationSpec { behavior, queries });
                }
            }
        }

        if analysis.is_empty() && quotes.is_empty() {
            continue;
        }

        results.push(DeepTheme {
            name: theme_name.clone(),
            obs_indices: indices.clone(),
            support_ids: support_ids.clone(),
            analysis,
            quotes,
            falsifications,
        });
    }

    eprintln!(
        "\r    phase 3: done. {} themes analyzed            ",
        results.len()
    );
    let _ = save_json(name, "themes_deep", &results);
    results
}

// ─── phase 4: adversarial validation ───

pub(crate) fn phase4_validate(
    name: &str,
    deep_themes: &[DeepTheme],
    index: &MessageIndex,
    classifications: &HashMap<String, MsgClassification>,
    recent_ids: &HashSet<String>,
    emb_model: &mut TextEmbedding,
    model: &str,
    person: &str,
) -> Vec<ValidatedTheme> {
    if let Some(cached) = load_json::<Vec<ValidatedTheme>>(name, "validated") {
        eprintln!("    phase 4: loaded {} cached validations", cached.len());
        return cached;
    }

    let mut validated: Vec<ValidatedTheme> = Vec::new();

    for (i, theme) in deep_themes.iter().enumerate() {
        eprint!(
            "\r    phase 4: falsifying theme {}/{}    ",
            i + 1,
            deep_themes.len()
        );
        std::io::stderr().flush().ok();

        let mut falsify_checked = 0usize;
        let mut falsify_confirmed = 0usize;
        let mut summary_pieces: Vec<String> = Vec::new();

        // for each falsification behavior, retrieve candidates and check each independently
        for (fi, spec) in theme.falsifications.iter().take(4).enumerate() {
            let mut candidates: Vec<(String, String)> = Vec::new(); // (id, text)
            let mut seen: HashSet<String> = HashSet::new();

            for q in spec.queries.iter().take(4) {
                let emb = match emb_model.embed(vec![q.clone()], None) {
                    Ok(v) => v.into_iter().next(),
                    Err(_) => None,
                };
                if let Some(e) = emb {
                    let results = index.search_for(&e, Some(person), 5);
                    for (_, entry) in results {
                        // skip messages that the theme itself was built on
                        if theme.support_ids.iter().any(|s| s == &entry.id) {
                            continue;
                        }
                        // skip jokes, code, urls, quotes — those aren't authentic evidence either way
                        let allowed = classifications
                            .get(&entry.id)
                            .map_or(true, |c| !is_excluded_for_traits(c));
                        if !allowed {
                            continue;
                        }
                        if !seen.insert(entry.id.clone()) {
                            continue;
                        }
                        candidates.push((entry.id.clone(), entry.text.clone()));
                    }
                }
            }

            // cap candidate checks per behavior
            candidates.truncate(6);

            for (cid, ctext) in &candidates {
                let txt: String = ctext.chars().take(400).collect();
                let prompt = format!(
                    "Behavior pattern claimed of a person: \"{}\".\n\
                     Their hypothesized trait: \"{}\".\n\
                     Candidate message: \"{}\"\n\n\
                     Does this single message clearly EXHIBIT the falsifying behavior \
                     (i.e. weaken the trait)?\n\
                     Be strict: only \"yes\" if it's a clear example. Otherwise \"no\".\n\
                     Return JSON: {{\"falsifies\": \"yes\"|\"no\"|\"weak\", \"reason\": \"1 short clause\"}}\n\
                     ONLY JSON.",
                    spec.behavior, theme.name, txt
                );
                falsify_checked += 1;
                if let Some(j) = llm_json(model, &prompt) {
                    let verdict = j
                        .get("falsifies")
                        .and_then(|v| v.as_str())
                        .unwrap_or("no")
                        .to_lowercase();
                    if verdict == "yes" {
                        falsify_confirmed += 1;
                        let reason = j
                            .get("reason")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        let short: String = ctext.chars().take(80).collect();
                        if summary_pieces.len() < 3 {
                            summary_pieces.push(format!("\"{}\" — {}", short, reason));
                        }
                        let _ = cid; // retained for future use
                    }
                }
            }

            // pacing log per behavior
            eprint!(
                "\r    phase 4: theme {}/{} (behavior {}/{}, {} checks, {} confirmed)  ",
                i + 1,
                deep_themes.len(),
                fi + 1,
                theme.falsifications.len().min(4),
                falsify_checked,
                falsify_confirmed
            );
            std::io::stderr().flush().ok();
        }

        // computed, not LLM-typed
        let n_support = theme.support_ids.len();
        let confidence = computed_confidence(n_support, falsify_confirmed);

        let summary = if falsify_confirmed == 0 {
            format!(
                "{} falsification probes issued; none confirmed.",
                falsify_checked
            )
        } else {
            format!(
                "{}/{} falsification probes confirmed. Examples: {}",
                falsify_confirmed,
                falsify_checked,
                summary_pieces.join(" · ")
            )
        };

        // temporal status: what fraction of supporting messages are in the recent quartile?
        let (recent_share, temporal_status) = if theme.support_ids.is_empty() {
            (0.0, "unknown".to_string())
        } else {
            let recent_hits = theme
                .support_ids
                .iter()
                .filter(|id| recent_ids.contains(*id))
                .count() as f64;
            let share = (recent_hits / theme.support_ids.len() as f64 * 100.0).round() / 100.0;
            // baseline expected share is .25 (the recent quartile is 25% of all messages).
            // > .40 = recent-skew (active), < .10 = old-skew (fading), else stable.
            let status = if share >= 0.40 {
                "active"
            } else if share <= 0.10 {
                "fading"
            } else {
                "stable"
            };
            (share, status.to_string())
        };

        validated.push(ValidatedTheme {
            name: theme.name.clone(),
            count: theme.obs_indices.len(),
            msg_count: n_support,
            analysis: theme.analysis.clone(),
            confidence,
            support: theme.quotes.clone(),
            falsifications_checked: falsify_checked,
            falsifications_confirmed: falsify_confirmed,
            falsification_summary: summary,
            recent_share,
            temporal_status,
        });
    }

    eprintln!(
        "\r    phase 4: done. {} themes validated            ",
        validated.len()
    );
    let _ = save_json(name, "validated", &validated);
    validated
}

pub(crate) fn computed_confidence(n_support: usize, n_falsify_confirmed: usize) -> f64 {
    let s = n_support as f64;
    // falsifications weigh 3x because they're more diagnostic than positive evidence
    let f = (n_falsify_confirmed as f64) * 3.0;
    ((s + 1.0) / (s + f + 2.0) * 100.0).round() / 100.0
}

// ─── phase 5: synthesis ───

// ─── phase 5a: cognitive markers (replaces IQ point estimate) ───
//
// Rationale: IQ-from-chat-lexicon is not psychometrically defensible. Instead we
// surface a *profile* of cognitive-style signals and z-score them across the
// corpus so the user can see relative position rather than a fake point score.

pub(crate) fn compute_cognitive_markers(
    name: &str,
    messages: &[&RawMessage],
    classifications: &HashMap<String, MsgClassification>,
    emb_model: &mut TextEmbedding,
    _model: &str,
) -> CognitiveMarkers {
    if let Some(cached) = load_json::<CognitiveMarkers>(name, "cognitive") {
        eprintln!("    phase 5a: loaded cached cognitive markers");
        return cached;
    }

    let substantive: Vec<&&RawMessage> = messages
        .iter()
        .filter(|m| m.text.chars().count() >= 30)
        .filter(|m| {
            classifications
                .get(&m.id)
                .map_or(true, |c| !is_excluded_for_style(c))
        })
        .collect();

    if substantive.is_empty() {
        return CognitiveMarkers::default();
    }

    let abstract_words: HashSet<&'static str> = [
        "concept", "idea", "principle", "theory", "system", "model", "structure", "framework",
        "approach", "method", "process", "pattern", "function", "mechanism", "abstraction",
        "implication", "consequence", "context", "instance", "category", "dimension",
        "perspective", "assumption", "premise", "argument", "logic", "reason", "cause", "effect",
        "tendency", "phenomenon", "constraint", "tradeoff", "trade-off", "interface", "boundary",
        "scope", "domain", "axis", "vector", "hierarchy", "ontology", "topology",
    ]
    .into_iter()
    .collect();

    let conditional_markers: HashSet<&'static str> = [
        "if", "unless", "whenever", "while", "until", "because", "though", "although", "however",
        "therefore", "thus", "hence", "suppose", "imagine", "assume", "would", "could", "might",
        "should", "wouldn't", "couldn't", "shouldn't", "otherwise", "unless", "given", "provided",
    ]
    .into_iter()
    .collect();

    let subordinators: HashSet<&'static str> = [
        "that", "which", "who", "whom", "whose", "because", "although", "since", "while",
        "before", "after", "until", "though", "whereas", "where",
    ]
    .into_iter()
    .collect();

    let hedging = hedging_words();
    let certainty = certainty_words();
    let self_ref = self_ref_words();

    let mut total_words = 0usize;
    let mut abstract_count = 0usize;
    let mut conditional_count = 0usize;
    let mut hedge_in_self = 0usize;
    let mut cert_in_self = 0usize;
    let mut clause_depths: Vec<f64> = Vec::new();

    for m in &substantive {
        let lower = m.text.to_lowercase();
        let words: Vec<&str> = lower.split_whitespace().collect();
        total_words += words.len();
        let mut clean: Vec<String> = Vec::with_capacity(words.len());
        for w in &words {
            let c: String = w
                .chars()
                .filter(|c| c.is_alphanumeric() || *c == '\'')
                .collect();
            if c.is_empty() {
                continue;
            }
            if abstract_words.contains(c.as_str()) {
                abstract_count += 1;
            }
            if conditional_markers.contains(c.as_str()) {
                conditional_count += 1;
            }
            clean.push(c);
        }
        // first-person utterance? then count hedging vs certainty within it
        if clean.iter().any(|w| self_ref.contains(w.as_str())) {
            for c in &clean {
                if hedging.contains(c.as_str()) {
                    hedge_in_self += 1;
                }
                if certainty.contains(c.as_str()) {
                    cert_in_self += 1;
                }
            }
        }
        // clause-depth proxy: punctuation + subordinators
        let depth = m.text.chars().filter(|c| *c == ',' || *c == ';' || *c == ':').count() as f64
            + clean
                .iter()
                .filter(|w| subordinators.contains(w.as_str()))
                .count() as f64;
        clause_depths.push(depth);
    }

    let n = substantive.len() as f64;
    let tw = total_words.max(1) as f64;
    let abstract_rate = abstract_count as f64 / tw * 100.0;
    let conditional_rate = conditional_count as f64 / tw * 100.0;
    let lexical_complexity = clause_depths.iter().sum::<f64>() / n;
    let self_total = (hedge_in_self + cert_in_self).max(1) as f64;
    let self_monitoring = hedge_in_self as f64 / self_total;

    // domain breadth: kmeans on a sample of message embeddings, count non-tiny clusters
    let sample_texts: Vec<String> = substantive
        .iter()
        .take(120)
        .map(|m| m.text.chars().take(400).collect())
        .collect();
    let domain_breadth = if sample_texts.len() >= 6 {
        match emb_model.embed(sample_texts, None) {
            Ok(embs) => {
                let k = ((embs.len() as f64).sqrt() / 2.0).max(3.0).min(8.0) as usize;
                let assign = kmeans(&embs, k, 12);
                let mut counts: HashMap<usize, usize> = HashMap::new();
                for a in &assign {
                    *counts.entry(*a).or_default() += 1;
                }
                let min_size = (embs.len() / (k * 2)).max(2);
                counts.values().filter(|c| **c >= min_size).count()
            }
            Err(_) => 0,
        }
    } else {
        0
    };

    // integrative complexity — derived from Phase 0 classifications, not a separate LLM pass.
    // mapping: differentiation = multi_perspective; integration = meta_cognitive + conditional;
    // baseline 1 for any substantive message; we report a continuous score on the 1-7 Suedfeld axis.
    let mut ic_sum = 0.0_f64;
    let mut ic_n = 0usize;
    let mut multi_n = 0usize;
    let mut meta_n = 0usize;
    let mut cond_class_n = 0usize;
    for m in &substantive {
        if let Some(c) = classifications.get(&m.id) {
            ic_n += 1;
            // baseline + (1 point for multi-perspective) + (1 point each for meta + cond)
            // + (2 points if all three present, integration over multiple dimensions)
            let mut s = 1.0_f64;
            if c.is_multi_perspective {
                s += 2.0;
                multi_n += 1;
            }
            if c.is_meta_cognitive {
                s += 1.5;
                meta_n += 1;
            }
            if c.is_conditional {
                s += 1.0;
                cond_class_n += 1;
            }
            if c.is_multi_perspective && c.is_meta_cognitive && c.is_conditional {
                s += 1.5;
            }
            ic_sum += s.min(7.0);
        }
    }
    let integrative_complexity = if ic_n > 0 {
        ic_sum / ic_n as f64
    } else {
        0.0
    };
    let _ = (multi_n, meta_n, cond_class_n); // counted for potential future surfacing

    let r2 = |v: f64| (v * 100.0).round() / 100.0;
    let markers = CognitiveMarkers {
        abstract_rate: r2(abstract_rate),
        conditional_rate: r2(conditional_rate),
        integrative_complexity: r2(integrative_complexity),
        lexical_complexity: r2(lexical_complexity),
        domain_breadth,
        self_monitoring: r2(self_monitoring),
        sample_size: substantive.len(),
        ..Default::default()
    };

    let _ = save_json(name, "cognitive", &markers);
    markers
}

/// Embedding-free cognitive markers over a message subset (used for the recent quartile).
/// Domain breadth requires embeddings and isn't meaningful on a small slice, so it's omitted.
pub(crate) fn compute_cognitive_markers_named(
    name: &str,
    cache_suffix: &str,
    messages: &[&RawMessage],
    classifications: &HashMap<String, MsgClassification>,
) -> CognitiveMarkers {
    if let Some(cached) = load_json::<CognitiveMarkers>(name, cache_suffix) {
        eprintln!("    phase 5a: loaded cached {} markers", cache_suffix);
        return cached;
    }

    let substantive: Vec<&&RawMessage> = messages
        .iter()
        .filter(|m| m.text.chars().count() >= 30)
        .filter(|m| {
            classifications
                .get(&m.id)
                .map_or(true, |c| !is_excluded_for_style(c))
        })
        .collect();

    if substantive.is_empty() {
        return CognitiveMarkers::default();
    }

    let abstract_words: HashSet<&'static str> = [
        "concept", "idea", "principle", "theory", "system", "model", "structure", "framework",
        "approach", "method", "process", "pattern", "function", "mechanism", "abstraction",
        "implication", "consequence", "context", "instance", "category", "dimension",
        "perspective", "assumption", "premise", "argument", "logic", "reason", "cause", "effect",
        "tendency", "phenomenon", "constraint", "tradeoff", "trade-off", "interface", "boundary",
        "scope", "domain", "axis", "vector", "hierarchy", "ontology", "topology",
    ]
    .into_iter()
    .collect();
    let conditional_markers: HashSet<&'static str> = [
        "if", "unless", "whenever", "while", "until", "because", "though", "although", "however",
        "therefore", "thus", "hence", "suppose", "imagine", "assume", "would", "could", "might",
        "should", "wouldn't", "couldn't", "shouldn't", "otherwise", "unless", "given", "provided",
    ]
    .into_iter()
    .collect();
    let subordinators: HashSet<&'static str> = [
        "that", "which", "who", "whom", "whose", "because", "although", "since", "while",
        "before", "after", "until", "though", "whereas", "where",
    ]
    .into_iter()
    .collect();

    let hedging = hedging_words();
    let certainty = certainty_words();
    let self_ref = self_ref_words();

    let mut total_words = 0usize;
    let mut abstract_count = 0usize;
    let mut conditional_count = 0usize;
    let mut hedge_in_self = 0usize;
    let mut cert_in_self = 0usize;
    let mut clause_depths: Vec<f64> = Vec::new();

    for m in &substantive {
        let lower = m.text.to_lowercase();
        let words: Vec<&str> = lower.split_whitespace().collect();
        total_words += words.len();
        let mut clean: Vec<String> = Vec::with_capacity(words.len());
        for w in &words {
            let c: String = w
                .chars()
                .filter(|c| c.is_alphanumeric() || *c == '\'')
                .collect();
            if c.is_empty() {
                continue;
            }
            if abstract_words.contains(c.as_str()) {
                abstract_count += 1;
            }
            if conditional_markers.contains(c.as_str()) {
                conditional_count += 1;
            }
            clean.push(c);
        }
        if clean.iter().any(|w| self_ref.contains(w.as_str())) {
            for c in &clean {
                if hedging.contains(c.as_str()) {
                    hedge_in_self += 1;
                }
                if certainty.contains(c.as_str()) {
                    cert_in_self += 1;
                }
            }
        }
        let depth = m.text.chars().filter(|c| *c == ',' || *c == ';' || *c == ':').count() as f64
            + clean
                .iter()
                .filter(|w| subordinators.contains(w.as_str()))
                .count() as f64;
        clause_depths.push(depth);
    }

    let n = substantive.len() as f64;
    let tw = total_words.max(1) as f64;
    let abstract_rate = abstract_count as f64 / tw * 100.0;
    let conditional_rate = conditional_count as f64 / tw * 100.0;
    let lexical_complexity = clause_depths.iter().sum::<f64>() / n;
    let self_total = (hedge_in_self + cert_in_self).max(1) as f64;
    let self_monitoring = hedge_in_self as f64 / self_total;

    // integrative complexity from Phase 0 classifications
    let mut ic_sum = 0.0_f64;
    let mut ic_n = 0usize;
    for m in &substantive {
        if let Some(c) = classifications.get(&m.id) {
            ic_n += 1;
            let mut s = 1.0_f64;
            if c.is_multi_perspective { s += 2.0; }
            if c.is_meta_cognitive { s += 1.5; }
            if c.is_conditional { s += 1.0; }
            if c.is_multi_perspective && c.is_meta_cognitive && c.is_conditional { s += 1.5; }
            ic_sum += s.min(7.0);
        }
    }
    let integrative_complexity = if ic_n > 0 { ic_sum / ic_n as f64 } else { 0.0 };

    let r2 = |v: f64| (v * 100.0).round() / 100.0;
    let markers = CognitiveMarkers {
        abstract_rate: r2(abstract_rate),
        conditional_rate: r2(conditional_rate),
        integrative_complexity: r2(integrative_complexity),
        lexical_complexity: r2(lexical_complexity),
        domain_breadth: 0, // not meaningful on the recent slice
        self_monitoring: r2(self_monitoring),
        sample_size: substantive.len(),
        ..Default::default()
    };

    let _ = save_json(name, cache_suffix, &markers);
    markers
}

// ─── phase 5b: Big Five signals (no scores — citations only, derived from Phase 0 tags) ───
//
// Each dimension is now grounded in Phase 0 classifications, not generic seed embeddings.
// We rank each person's messages by a per-dimension score and surface the top 5 quotes.

pub(crate) fn extract_big_five_signals(
    name: &str,
    person: &str,
    messages: &[&RawMessage],
    classifications: &HashMap<String, MsgClassification>,
) -> BigFiveSignals {
    extract_big_five_signals_named(name, "big_five", person, messages, classifications)
}

pub(crate) fn extract_big_five_signals_named(
    name: &str,
    cache_suffix: &str,
    person: &str,
    messages: &[&RawMessage],
    classifications: &HashMap<String, MsgClassification>,
) -> BigFiveSignals {
    if let Some(cached) = load_json::<BigFiveSignals>(name, cache_suffix) {
        eprintln!("    phase 5b: loaded cached {} signals", cache_suffix);
        return cached;
    }

    eprint!("    phase 5b: big-five ({}) ... ", cache_suffix);
    std::io::stderr().flush().ok();

    // domain-stop topics that bias toward openness vs everything else
    let abstract_topics: HashSet<&'static str> = [
        "philosophy", "theory", "tech", "math", "science", "art", "music",
        "psychology", "linguistics", "concept", "abstract", "idea",
    ]
    .into_iter()
    .collect();
    let work_topics: HashSet<&'static str> = [
        "work", "project", "deadline", "school", "study", "tasks", "planning",
    ]
    .into_iter()
    .collect();
    let social_topics: HashSet<&'static str> = [
        "friends", "relationships", "party", "meet", "social", "people",
    ]
    .into_iter()
    .collect();
    let distress_tones: HashSet<&'static str> = [
        "anxious", "frustrated", "defensive", "resigned", "hostile", "sad",
        "depressed", "lonely", "overwhelmed", "panicked", "worried", "angry",
    ]
    .into_iter()
    .collect();
    let warm_tones: HashSet<&'static str> = [
        "warm", "kind", "grateful", "supportive", "empathetic", "tender", "earnest",
    ]
    .into_iter()
    .collect();
    let high_energy: HashSet<&'static str> = [
        "enthusiastic", "excited", "playful", "energetic", "amused",
    ]
    .into_iter()
    .collect();

    // score each message per dimension; bigger = better example
    let score_message = |c: &MsgClassification| -> [f64; 5] {
        let topic_l = c.topic.to_lowercase();
        let tone_l = c.tone.to_lowercase();
        let mut o = 0.0;
        let mut co = 0.0;
        let mut e = 0.0;
        let mut a = 0.0;
        let mut nn = 0.0;

        // Openness: meta + multi-perspective + conditional + abstract topic
        if c.is_meta_cognitive { o += 1.5; }
        if c.is_multi_perspective { o += 2.0; }
        if c.is_conditional { o += 0.5; }
        if abstract_topics.contains(topic_l.as_str()) { o += 1.0; }
        if c.modality == "hypothetical" || c.modality == "speculative" { o += 1.0; }

        // Conscientiousness: planning / work topics + assertive speech act + normative modality
        if work_topics.contains(topic_l.as_str()) { co += 1.0; }
        if c.modality == "normative" { co += 1.5; }
        if c.speech_act == "assert" || c.speech_act == "command" { co += 0.5; }
        if c.claim_certainty == "confident" || c.claim_certainty == "absolute" { co += 0.5; }

        // Extraversion: specific addressee + high intensity + social topic + high-energy tone
        if c.addressee == "specific" { e += 1.0; }
        if c.intensity >= 2 { e += 0.5 * (c.intensity as f64); }
        if social_topics.contains(topic_l.as_str()) { e += 1.0; }
        if high_energy.contains(tone_l.as_str()) { e += 1.0; }
        if c.function == "joke" { e += 0.3; }

        // Agreeableness: agree/concede speech_act + warm tone + valence positive + addressee specific
        if c.speech_act == "agree" || c.speech_act == "concede" { a += 1.5; }
        if warm_tones.contains(tone_l.as_str()) { a += 1.0; }
        if c.valence >= 1 && c.addressee == "specific" { a += 0.7; }

        // Neuroticism: negative valence × intensity + distress tone + meta on self
        if c.valence <= -1 {
            nn += (c.intensity as f64) * 0.8 - (c.valence as f64) * 0.5; // both magnitudes
        }
        if distress_tones.contains(tone_l.as_str()) { nn += 1.5; }
        if c.is_meta_cognitive && c.valence <= 0 { nn += 0.8; }
        if c.claim_dimension == "mood" && c.valence <= 0 { nn += 1.0; }

        [o, co, e, a, nn]
    };

    let mut scored: Vec<(usize, [f64; 5], &&RawMessage)> = Vec::new();
    for m in messages {
        if m.sender != person {
            continue;
        }
        if m.text.chars().count() < 12 {
            continue;
        }
        let c = match classifications.get(&m.id) {
            Some(c) if !is_excluded_for_traits(c) => c,
            _ => continue,
        };
        let s = score_message(c);
        scored.push((scored.len(), s, m));
    }

    let dim_names = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"];
    let mut out = BigFiveSignals::default();
    for (dim_i, dim) in dim_names.iter().enumerate() {
        let mut ranked: Vec<&(usize, [f64; 5], &&RawMessage)> = scored
            .iter()
            .filter(|(_, s, _)| s[dim_i] > 0.0)
            .collect();
        ranked.sort_by(|a, b| b.1[dim_i].partial_cmp(&a.1[dim_i]).unwrap());
        let bucket: Vec<EvidenceQuote> = ranked
            .into_iter()
            .take(5)
            .map(|(_, _, m)| EvidenceQuote {
                quote: m.text.chars().take(220).collect(),
                msg_id: m.id.clone(),
            })
            .collect();
        match *dim {
            "openness" => out.openness = bucket,
            "conscientiousness" => out.conscientiousness = bucket,
            "extraversion" => out.extraversion = bucket,
            "agreeableness" => out.agreeableness = bucket,
            "neuroticism" => out.neuroticism = bucket,
            _ => {}
        }
    }

    eprintln!("done");
    let _ = save_json(name, cache_suffix, &out);
    out
}

// ─── phase 5c: self-claim reconciliation ───
//
// User's complaint: anyone can say "I have a 145 IQ" and it folds into the profile.
// Fix: pull self-claims out as a SEPARATE stream, retrieve behavioral evidence
// from the same person, and reconcile claim-vs-behavior explicitly.

pub(crate) fn reconcile_self_claims(
    name: &str,
    person: &str,
    messages: &[&RawMessage],
    classifications: &HashMap<String, MsgClassification>,
    index: &MessageIndex,
    emb_model: &mut TextEmbedding,
    model: &str,
) -> Vec<SelfClaim> {
    if let Some(cached) = load_json::<Vec<SelfClaim>>(name, "self_claims") {
        eprintln!("    phase 5c: loaded {} cached self-claims", cached.len());
        return cached;
    }

    // collect claims with their Phase 0 metadata, dedupe by lowercased claim text.
    // ironic / hyperbole claims auto-route to a non-LLM verdict — they're not literal.
    #[derive(Clone)]
    struct RawClaim {
        claim: String,
        msg_id: String,
        dimension: String,
        register: String,
        certainty: String,
    }
    let mut seen: HashSet<String> = HashSet::new();
    let mut raw_claims: Vec<RawClaim> = Vec::new();
    for m in messages {
        if let Some(c) = classifications.get(&m.id) {
            if let Some(claim) = &c.self_claim {
                let key = claim.to_lowercase();
                if key.chars().count() < 6 {
                    continue;
                }
                if !seen.insert(key) {
                    continue;
                }
                raw_claims.push(RawClaim {
                    claim: claim.clone(),
                    msg_id: m.id.clone(),
                    dimension: if c.claim_dimension.is_empty() {
                        "unknown".into()
                    } else {
                        c.claim_dimension.clone()
                    },
                    register: c.claim_register.clone(),
                    certainty: c.claim_certainty.clone(),
                });
            }
        }
    }

    // prioritize trait/identity/intelligence/profession claims; mood/state claims last
    raw_claims.sort_by_key(|c| {
        let prio = match c.dimension.as_str() {
            "intelligence" | "identity" | "trait" | "profession" | "history" => 0,
            "preference" => 1,
            "mood" => 2,
            _ => 1,
        };
        (prio, std::cmp::Reverse(c.claim.chars().count()))
    });
    raw_claims.truncate(30);

    if raw_claims.is_empty() {
        let _ = save_json(name, "self_claims", &Vec::<SelfClaim>::new());
        return Vec::new();
    }

    eprintln!("    phase 5c: reconciling {} self-claims", raw_claims.len());

    let mut out: Vec<SelfClaim> = Vec::new();
    for (i, rc) in raw_claims.iter().enumerate() {
        let claim = &rc.claim;
        let claim_msg_id = &rc.msg_id;
        eprint!("\r    phase 5c: claim {}/{}    ", i + 1, raw_claims.len());
        std::io::stderr().flush().ok();

        // ironic / hyperbole → don't waste an LLM call. Claim isn't literal.
        if rc.register == "ironic" || rc.register == "hyperbole" {
            out.push(SelfClaim {
                claim: claim.clone(),
                msg_id: claim_msg_id.clone(),
                dimension: rc.dimension.clone(),
                verdict: "not-literal".into(),
                rationale: format!(
                    "Phase 0 classified register as '{}' — not a literal claim about self.",
                    rc.register
                ),
                behavioral_evidence: Vec::new(),
            });
            continue;
        }

        // retrieve behavioral evidence from the same person, excluding the claim itself
        // and other self-statements (we want behavior, not more claims)
        let mut evidence: Vec<EvidenceQuote> = Vec::new();
        let emb = match emb_model.embed(vec![claim.clone()], None) {
            Ok(v) => v.into_iter().next(),
            Err(_) => None,
        };
        if let Some(e) = emb {
            let results = index.search_for(&e, Some(person), 10);
            for (_, entry) in results {
                if &entry.id == claim_msg_id {
                    continue;
                }
                // exclude other self-statements & jokes/quotes/code
                let c = classifications.get(&entry.id);
                let is_other_claim = c
                    .map(|c| c.tags.iter().any(|t| t == "self-statement"))
                    .unwrap_or(false);
                if is_other_claim {
                    continue;
                }
                if c.map_or(false, is_excluded_for_traits) {
                    continue;
                }
                evidence.push(EvidenceQuote {
                    quote: entry.text.chars().take(200).collect(),
                    msg_id: entry.id.clone(),
                });
                if evidence.len() >= 5 {
                    break;
                }
            }
        }

        let ev_block: String = evidence
            .iter()
            .enumerate()
            .map(|(j, q)| format!("{}. \"{}\"", j + 1, q.quote))
            .collect::<Vec<_>>()
            .join("\n");
        let ev_section = if evidence.is_empty() {
            "(none retrieved)".to_string()
        } else {
            ev_block
        };

        let certainty_hint = if rc.certainty.is_empty() {
            String::new()
        } else {
            format!(
                "Phase 0 marked the claim as `{}`. Higher-certainty claims face a stricter bar — \
                 unsupported confident claims should lean toward `inconsistent` rather than `unverifiable` \
                 when behavioral evidence runs the other way.\n\n",
                rc.certainty
            )
        };

        let prompt = format!(
            "A person made this self-claim about themselves (dimension: {}): \"{}\"\n\n\
             {}\
             Other things they said (behavioral evidence, NOT other self-claims):\n{}\n\n\
             Reconcile:\n\
             - verdict: \"consistent\" | \"inconsistent\" | \"unverifiable\"\n\
             - rationale: 1-2 sentences pointing to the evidence (or lack of it). \
             Be skeptical: unsupported = unverifiable, not consistent.\n\n\
             Return JSON: {{\"verdict\":\"...\",\"rationale\":\"...\"}}\n\
             ONLY JSON.",
            rc.dimension, claim, certainty_hint, ev_section
        );

        let mut verdict = "unverifiable".to_string();
        let mut rationale = String::new();
        if let Some(j) = llm_json(model, &prompt) {
            verdict = j
                .get("verdict")
                .and_then(|v| v.as_str())
                .unwrap_or("unverifiable")
                .to_string();
            rationale = j
                .get("rationale")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
        }
        out.push(SelfClaim {
            claim: claim.clone(),
            msg_id: claim_msg_id.clone(),
            dimension: rc.dimension.clone(),
            verdict,
            rationale,
            behavioral_evidence: evidence,
        });
    }
    eprintln!("\r    phase 5c: done. {} reconciled            ", out.len());

    let _ = save_json(name, "self_claims", &out);
    out
}

// ─── phase 5: synthesis ───
//
// Synthesis now writes the qualitative paragraphs from validated themes + cognitive
// markers + Big Five citations + self-claim reconciliation. NO IQ score, NO MBTI.

pub(crate) fn phase5_synthesize(
    name: &str,
    behavioral: &Behavioral,
    themes: &[ValidatedTheme],
    cognitive: &CognitiveMarkers,
    cognitive_recent: &CognitiveMarkers,
    big_five: &BigFiveSignals,
    big_five_recent: &BigFiveSignals,
    self_claims: &[SelfClaim],
    interaction: Option<&Interaction>,
    is_self: bool,
    model: &str,
) -> Option<Interpretation> {
    if let Some(cached) = load_json::<Interpretation>(name, "interpretation") {
        eprintln!("    phase 5: loaded cached interpretation");
        return Some(cached);
    }

    eprint!("    phase 5: synthesizing... ");
    std::io::stderr().flush().ok();

    let b = behavioral;
    let mut prompt = format!(
        "{} | {} msgs | {} substantive\n\
         Style: self-ref:{:.1}% certainty:{:.2} vocab-div:{:.2} clout:{:.2} auth:{:.2}\n\
         Cognitive markers (raw, NOT IQ): abstract-rate:{:.2} conditional-rate:{:.2} \
         integrative-complexity(1-7):{:.1} lexical-complexity:{:.2} domain-breadth:{} \
         self-monitoring:{:.2}\n",
        name,
        b.msg_count,
        b.substantive_count,
        b.self_ref_rate,
        b.certainty_ratio,
        b.vocab_diversity,
        b.clout,
        b.authenticity,
        cognitive.abstract_rate,
        cognitive.conditional_rate,
        cognitive.integrative_complexity,
        cognitive.lexical_complexity,
        cognitive.domain_breadth,
        cognitive.self_monitoring
    );

    if let Some(int) = interaction {
        prompt.push_str(&format!(
            "Engagement: {:.0}ch replies, {}↔{} init, mirror:{:.2}\n",
            int.self_avg_reply_len,
            int.self_initiations,
            int.other_initiations,
            int.mirroring
        ));
    }

    // recent-vs-baseline trends for cognitive markers
    let delta = |old: f64, new: f64| -> &'static str {
        let d = new - old;
        let scale = old.abs().max(0.1);
        let pct = d / scale;
        if pct > 0.20 { "↑↑" }
        else if pct > 0.07 { "↑" }
        else if pct < -0.20 { "↓↓" }
        else if pct < -0.07 { "↓" }
        else { "→" }
    };
    if cognitive_recent.sample_size > 0 {
        prompt.push_str(&format!(
            "\nTREND vs recent quartile ({} recent / {} all):\n\
             abstract {} ({:.2}→{:.2}) · conditional {} ({:.2}→{:.2}) · \
             integrative {} ({:.2}→{:.2}) · self-monitoring {} ({:.2}→{:.2})\n\
             Big Five signal counts recent: O:{} C:{} E:{} A:{} N:{}\n",
            cognitive_recent.sample_size,
            cognitive.sample_size,
            delta(cognitive.abstract_rate, cognitive_recent.abstract_rate),
            cognitive.abstract_rate, cognitive_recent.abstract_rate,
            delta(cognitive.conditional_rate, cognitive_recent.conditional_rate),
            cognitive.conditional_rate, cognitive_recent.conditional_rate,
            delta(cognitive.integrative_complexity, cognitive_recent.integrative_complexity),
            cognitive.integrative_complexity, cognitive_recent.integrative_complexity,
            delta(cognitive.self_monitoring, cognitive_recent.self_monitoring),
            cognitive.self_monitoring, cognitive_recent.self_monitoring,
            big_five_recent.openness.len(),
            big_five_recent.conscientiousness.len(),
            big_five_recent.extraversion.len(),
            big_five_recent.agreeableness.len(),
            big_five_recent.neuroticism.len(),
        ));
    }

    prompt.push_str("\nVALIDATED THEMES (only with confidence ≥ .4) — note temporal_status:\n");
    for t in themes.iter().filter(|t| t.confidence >= 0.4) {
        let tag = match t.temporal_status.as_str() {
            "active" => " [ACTIVE — recent-skew]",
            "fading" => " [FADING — old-skew]",
            _ => "",
        };
        prompt.push_str(&format!(
            "• {}{} (obs:{}, msgs:{}, conf:{:.0}%, recent-share:{:.2}, falsify {}/{}): {}\n",
            t.name,
            tag,
            t.count,
            t.msg_count,
            t.confidence * 100.0,
            t.recent_share,
            t.falsifications_confirmed,
            t.falsifications_checked,
            t.analysis
        ));
    }

    let bf_count = big_five.openness.len()
        + big_five.conscientiousness.len()
        + big_five.extraversion.len()
        + big_five.agreeableness.len()
        + big_five.neuroticism.len();
    prompt.push_str(&format!(
        "\nBIG FIVE SIGNAL COUNTS (NOT scores — citations only): \
         O:{} C:{} E:{} A:{} N:{} (total {})\n",
        big_five.openness.len(),
        big_five.conscientiousness.len(),
        big_five.extraversion.len(),
        big_five.agreeableness.len(),
        big_five.neuroticism.len(),
        bf_count
    ));

    if !self_claims.is_empty() {
        prompt.push_str("\nSELF-CLAIMS vs OBSERVED BEHAVIOR:\n");
        for c in self_claims.iter().take(8) {
            prompt.push_str(&format!(
                "  - claims \"{}\" [{}] → {}: {}\n",
                c.claim.chars().take(80).collect::<String>(),
                c.dimension,
                c.verdict,
                c.rationale.chars().take(120).collect::<String>()
            ));
        }
    }

    let role = if is_self {
        "This is the USER themselves. Note blind spots they probably can't see in themselves."
    } else {
        "This is one node in the user's social network. Be useful for their decision-making."
    };

    prompt.push_str(&format!(
        "\n{}\n\n\
         RULES:\n\
         - DO NOT issue an IQ number — there's no defensible way to derive one from chat.\n\
         - DO NOT issue an MBTI type — it's not reliable from this data.\n\
         - DO cite themes by name as evidence.\n\
         - DO distinguish CURRENT state from BASELINE state. If a theme is ACTIVE (recent-skew), \
         describe it in present tense; if FADING, describe it as something they've moved past.\n\
         - DO surface trends visible in the cognitive markers (↑/↓ arrows above) — say what's \
         shifting, not just what's stable.\n\
         - DO note where self-claims diverge from behavior, if any.\n\
         - DO acknowledge what you DON'T know (e.g. in-person behavior, off-platform context).\n\n\
         Return JSON: {{\"identity\": \"2-3 sentences\", \"anxiety\": \"2-3 sentences\", \
         \"social_style\": \"2-3 sentences\", \"growth\": \"2-3 sentences\", \
         \"vulnerability\": \"2-3 sentences\", \"summary\": \"one paragraph: describe cognitive \
         style qualitatively (NOT an IQ), key personality tendencies, what's currently shifting \
         (cite ACTIVE / FADING themes and any cognitive marker trends), what makes this person \
         distinctive in this corpus, and what we genuinely don't know\"}}\n\
         ONLY JSON.",
        role
    ));

    let result: Option<Interpretation> =
        llm_json(model, &prompt).and_then(|json| serde_json::from_value(json).ok());

    if let Some(ref interp) = result {
        let _ = save_json(name, "interpretation", interp);
        eprintln!("done");
    } else {
        eprintln!("failed");
    }
    result
}

// ─── phase 6: intent prediction ───

pub(crate) fn phase6_predict(
    name: &str,
    messages: &[&RawMessage],
    themes: &[ValidatedTheme],
    interpretation: Option<&Interpretation>,
    classifications: &HashMap<String, MsgClassification>,
    emb_model: &mut TextEmbedding,
    model: &str,
) -> Vec<Prediction> {
    if let Some(cached) = load_json::<Vec<Prediction>>(name, "predictions") {
        eprintln!("    phase 6: loaded {} cached predictions", cached.len());
        return cached;
    }

    eprint!("    phase 6: extracting intents... ");
    std::io::stderr().flush().ok();

    // intents only from authentic-self channels — don't predict from jokes/quotes/code
    let texts: Vec<&str> = messages
        .iter()
        .filter(|m| {
            classifications
                .get(&m.id)
                .map_or(true, |c| !is_excluded_for_traits(c))
        })
        .map(|m| m.text.as_str())
        .collect();

    let signals = extract_intents(&texts);
    if signals.is_empty() {
        eprintln!("no intent signals found");
        return Vec::new();
    }
    eprint!("{} signals... ", signals.len());

    let clusters = cluster_intents(&signals, emb_model);
    if clusters.is_empty() {
        eprintln!("no clusters");
        return Vec::new();
    }
    eprint!("{} intent clusters... LLM... ", clusters.len());
    std::io::stderr().flush().ok();

    let mut prompt = format!("Person: {}\n", name);
    if let Some(interp) = interpretation {
        prompt.push_str(&format!("Profile: {}\n\n", interp.summary));
    }
    prompt.push_str("ACTIVE INTENT CLUSTERS (ranked by recency-weighted frequency):\n");
    for (i, cluster) in clusters.iter().take(5).enumerate() {
        prompt.push_str(&format!(
            "  {}. [{}x, recency:{:.2}] \"{}\"\n",
            i + 1,
            cluster.count,
            cluster.recency_score,
            cluster.intent
        ));
        for sig in cluster.signals.iter().take(2) {
            prompt.push_str(&format!("     evidence: \"{}\"\n", sig));
        }
    }

    prompt.push_str("\nVALIDATED THEMES (only ≥ .4 confidence):\n");
    for t in themes.iter().filter(|t| t.confidence >= 0.4).take(5) {
        prompt.push_str(&format!(
            "  • {} ({}x, conf:{:.0}%): {}\n",
            t.name,
            t.count,
            t.confidence * 100.0,
            t.analysis
        ));
    }

    prompt.push_str(
        "\nUsing the psychological themes and intent clusters together, predict 3-5 actions \
         this person is LIKELY TO TAKE that they HAVE NOT EXPLICITLY STATED. \
         Do NOT restate what they said. Derive non-obvious implications.\n\
         For each prediction, list which theme name(s) and which intent cluster (by number) \
         imply it — that becomes its evidence array.\n\
         Confidence should be CONSERVATIVE. Above .7 only when multiple independent signals converge.\n\n\
         Return JSON: {\"predictions\": [\
         {\"action\": \"non-obvious predicted action\", \"confidence\": 0.0-1.0, \
         \"timeframe\": \"days/weeks/months\", \"evidence\": [\"theme name or intent #\"]}]}\n\
         ONLY JSON.");

    let predictions: Vec<Prediction> = llm_json(model, &prompt)
        .and_then(|json| {
            let arr = json
                .get("predictions")
                .and_then(|v| v.as_array())
                .or_else(|| json.as_array());
            arr.map(|a| {
                a.iter()
                    .filter_map(|p| {
                        Some(Prediction {
                            action: p.get("action").and_then(|v| v.as_str())?.to_string(),
                            confidence: p.get("confidence").and_then(|v| v.as_f64()).unwrap_or(0.5),
                            timeframe: p
                                .get("timeframe")
                                .and_then(|v| v.as_str())
                                .unwrap_or("unknown")
                                .to_string(),
                            evidence: p
                                .get("evidence")
                                .and_then(|v| v.as_array())
                                .map(|a| {
                                    a.iter()
                                        .filter_map(|v| v.as_str().map(|s| s.to_string()))
                                        .collect()
                                })
                                .unwrap_or_default(),
                        })
                    })
                    .collect()
            })
        })
        .unwrap_or_default();

    eprintln!("{} predictions", predictions.len());
    let _ = save_json(name, "predictions", &predictions);
    predictions
}

/// Returns the IDs of messages in the most recent quartile (by parse order, which is
/// chronological for typical chat exports). Used to compute "recent vs all" temporal deltas.
pub(crate) fn recent_quartile_ids<'a>(
    messages: &'a [&'a RawMessage],
) -> (HashSet<String>, Vec<&'a RawMessage>) {
    let n = messages.len();
    if n == 0 {
        return (HashSet::new(), Vec::new());
    }
    // sort by timestamp string (lexicographic = chronological for ISO-ish formats);
    // fall back to existing order for messages without timestamps.
    let mut sorted: Vec<&RawMessage> = messages.iter().copied().collect();
    sorted.sort_by(|a, b| match (a.timestamp.as_ref(), b.timestamp.as_ref()) {
        (Some(ta), Some(tb)) => ta.cmp(tb),
        (Some(_), None) => std::cmp::Ordering::Greater,
        (None, Some(_)) => std::cmp::Ordering::Less,
        (None, None) => std::cmp::Ordering::Equal,
    });
    let cutoff = (n * 3) / 4;
    let recent: Vec<&RawMessage> = sorted[cutoff..].to_vec();
    let ids: HashSet<String> = recent.iter().map(|m| m.id.clone()).collect();
    (ids, recent)
}

pub(crate) fn profile_person(
    name: &str,
    messages: &[&RawMessage],
    is_self: bool,
    interaction: Option<Interaction>,
    index: &MessageIndex,
    corpus: &[RawMessage],
    id_to_pos: &HashMap<String, usize>,
    emb_model: &mut TextEmbedding,
    model: &str,
) -> Profile {
    // Fast path: if this person's corpus fingerprint matches the previous run's snapshot
    // AND a cached Profile is on disk, skip the entire pipeline (no LLM, no scans, no
    // re-eval). Stays valid until new messages arrive for this specific person.
    let fingerprint = crate::insight::corpus_fingerprint(messages);
    if let Some(prev_snap) = crate::insight::load_snapshot(name) {
        if prev_snap.fingerprint == fingerprint {
            if let Some(mut cached) = load_json::<Profile>(name, "profile") {
                eprintln!(
                    "  ┌ {} ({} msgs) — UNCHANGED since last run, skipping pipeline",
                    name, messages.len()
                );
                // refresh the interaction block in case self-side stats moved
                cached.interaction = interaction;
                eprintln!("  └ {} loaded from cache", name);
                return cached;
            }
        }
    }

    eprintln!("  ┌ {} ({} msgs)", name, messages.len());

    // phase 0: classify each message by function (joke / quote / self-statement / code / ...)
    // sees corpus context per target so disambiguation has the surrounding conversation
    let classifications = phase0_classify(name, messages, corpus, id_to_pos, model);

    // behavioral metrics, with code/url/quote/low-info excluded so vocab stats reflect prose
    let excluded_for_style: HashSet<String> = classifications
        .iter()
        .filter(|(_, c)| is_excluded_for_style(c))
        .map(|(k, _)| k.clone())
        .collect();
    let behavioral = compute_behavioral(messages, &excluded_for_style);

    let id_to_msg: HashMap<String, &RawMessage> = messages
        .iter()
        .map(|m| (m.id.clone(), *m))
        .collect();

    // temporal split: recent quartile (chronologically last 25%) for trend awareness
    let (recent_ids, recent_msgs_owned) = recent_quartile_ids(messages);
    let recent_msgs: Vec<&RawMessage> = recent_msgs_owned;
    let recent_refs: Vec<&RawMessage> = recent_msgs.clone();
    eprintln!(
        "    temporal split: {} recent / {} total ({:.0}%)",
        recent_msgs.len(),
        messages.len(),
        100.0 * recent_msgs.len() as f64 / messages.len().max(1) as f64
    );

    let observations = phase1_scan(name, messages, &classifications, model);
    let theme_clusters = phase2_cluster(name, &observations, emb_model);
    let deep_themes = phase3_deepen(name, &theme_clusters, &observations, &id_to_msg, model);
    let validated = phase4_validate(
        name,
        &deep_themes,
        index,
        &classifications,
        &recent_ids,
        emb_model,
        model,
        name,
    );

    // cognitive markers + Big Five citations + self-claim reconciliation (replaces IQ/MBTI).
    // Each computed twice: once over all, once over recent quartile.
    let cognitive = compute_cognitive_markers(name, messages, &classifications, emb_model, model);
    let recent_ref_slice: Vec<&RawMessage> = recent_refs.iter().copied().collect();
    let recent_msg_slice: Vec<&RawMessage> =
        recent_ref_slice.iter().map(|m| *m).collect();
    let cognitive_recent =
        compute_cognitive_markers_named(name, "cognitive_recent", &recent_msg_slice, &classifications);
    let big_five = extract_big_five_signals(name, name, messages, &classifications);
    let big_five_recent = extract_big_five_signals_named(
        name,
        "big_five_recent",
        name,
        &recent_msg_slice,
        &classifications,
    );
    let self_claims =
        reconcile_self_claims(name, name, messages, &classifications, index, emb_model, model);

    let interpretation = phase5_synthesize(
        name,
        &behavioral,
        &validated,
        &cognitive,
        &cognitive_recent,
        &big_five,
        &big_five_recent,
        &self_claims,
        interaction.as_ref(),
        is_self,
        model,
    );
    let predictions = phase6_predict(
        name,
        messages,
        &validated,
        interpretation.as_ref(),
        &classifications,
        emb_model,
        model,
    );

    eprintln!(
        "  └ {} complete: {} class → {} obs → {} themes → {} validated → {} claims → {} predictions",
        name,
        classifications.len(),
        observations.len(),
        theme_clusters.len(),
        validated.len(),
        self_claims.len(),
        predictions.len()
    );

    let profile = Profile {
        name: name.to_string(),
        is_self,
        total_messages: messages.len(),
        behavioral,
        cognitive,
        cognitive_recent,
        big_five,
        big_five_recent,
        themes: validated,
        self_claims,
        interaction,
        interpretation,
        predictions,
    };
    // Persist the assembled Profile so the next run can short-circuit on fingerprint match.
    let _ = save_json(&profile.name, "profile", &profile);
    profile
}


// ─── cross-person calibration ───

pub(crate) fn apply_cognitive_zscores(profiles: &mut [Profile]) {
    if profiles.len() < 2 {
        return;
    }
    let n = profiles.len() as f64;

    fn stats<F: Fn(&CognitiveMarkers) -> f64>(profiles: &[Profile], f: F, n: f64) -> (f64, f64) {
        let mu = profiles.iter().map(|p| f(&p.cognitive)).sum::<f64>() / n;
        let var =
            profiles.iter().map(|p| (f(&p.cognitive) - mu).powi(2)).sum::<f64>() / n;
        (mu, var.sqrt().max(1e-6))
    }

    let (mu_abs, s_abs) = stats(profiles, |c| c.abstract_rate, n);
    let (mu_cond, s_cond) = stats(profiles, |c| c.conditional_rate, n);
    let (mu_int, s_int) = stats(profiles, |c| c.integrative_complexity, n);
    let (mu_lex, s_lex) = stats(profiles, |c| c.lexical_complexity, n);
    let (mu_br, s_br) = stats(profiles, |c| c.domain_breadth as f64, n);
    let (mu_sm, s_sm) = stats(profiles, |c| c.self_monitoring, n);

    let r2 = |v: f64| (v * 100.0).round() / 100.0;
    for p in profiles.iter_mut() {
        let c = &mut p.cognitive;
        c.z_abstract = r2((c.abstract_rate - mu_abs) / s_abs);
        c.z_conditional = r2((c.conditional_rate - mu_cond) / s_cond);
        c.z_integrative = r2((c.integrative_complexity - mu_int) / s_int);
        c.z_lexical = r2((c.lexical_complexity - mu_lex) / s_lex);
        c.z_breadth = r2((c.domain_breadth as f64 - mu_br) / s_br);
        c.z_self_monitoring = r2((c.self_monitoring - mu_sm) / s_sm);
    }
}
