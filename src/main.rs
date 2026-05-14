//! auspex — chat-archive psychological profiler with provenance, falsification,
//! temporal awareness, pair-network dynamics, and an incremental insight engine.
//! See module-level docs for each phase.
//!
//! Run: `./auspex data/*.txt`     — full pipeline + serve
//! Run: `./auspex --serve`        — skip pipeline, just open UI
//! Env: `OLLAMA_MODEL`, `AUSPEX_PORT`, `AUSPEX_NO_SERVE`.

use std::collections::{HashMap, HashSet};
use std::time::Instant;

use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};

mod config;
mod html;
mod http;
mod index;
mod insight;
mod lexicons;
mod llm;
mod math;
mod metrics;
mod parse;
mod persist;
mod pipeline;
mod types;

use config::load_config;
use html::generate_html;
use http::{http_serve, HTTP_PORT_DEFAULT};
use index::{IndexEntry, MessageIndex};
use insight::{
    append_to_feed, corpus_fingerprint, generate_insights, generate_pair_insights,
    load_prev_pair_interactions, load_snapshot, make_snapshot, save_latest_insights,
    save_pair_interactions, save_snapshot,
};
use metrics::{compute_interactions, compute_pair_interactions};
use parse::{compute_msg_id, parse_files};
use persist::load_json;
use pipeline::{apply_cognitive_zscores, profile_person};
use types::{Insight, MsgClassification, Profile, RawMessage};

pub(crate) const INSIGHTS_DIR: &str = "insights";





// ─── main ───

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // --serve (alias -s): skip the pipeline entirely and just serve the existing graph.html.
    // Use this when the pipeline has already produced output and you just want to open the UI.
    let flags: Vec<String> = std::env::args()
        .skip(1)
        .filter(|a| a.starts_with('-'))
        .collect();
    if flags.iter().any(|a| a == "--serve" || a == "-s") {
        if !std::path::Path::new("graph.html").exists() {
            eprintln!("graph.html not found in cwd — run the pipeline first to generate it");
            return Ok(());
        }
        let port: u16 = std::env::var("AUSPEX_PORT")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(HTTP_PORT_DEFAULT);
        let root = std::env::current_dir()
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_else(|_| ".".into());
        eprintln!("--serve mode: skipping pipeline, serving existing graph.html");
        http_serve(port, root);
        return Ok(());
    }

    let config = load_config();
    let model = std::env::var("OLLAMA_MODEL").unwrap_or_else(|_| "llama3.2".into());

    let args: Vec<String> = std::env::args()
        .skip(1)
        .filter(|a| !a.starts_with('-'))
        .collect();
    let raw_messages = if !args.is_empty() {
        eprintln!("parsing {} files...", args.len());
        parse_files(&args, &config)
    } else {
        eprintln!("no files, using demo. usage: auspex file1.txt ...");
        demo_data(&config.self_name)
    };
    eprintln!("parsed {} messages", raw_messages.len());

    let mut by_sender: HashMap<String, Vec<&RawMessage>> = HashMap::new();
    for msg in &raw_messages {
        by_sender.entry(msg.sender.clone()).or_default().push(msg);
    }

    eprintln!("loading embedding model...");
    let mut emb_model = TextEmbedding::try_new(
        InitOptions::new(EmbeddingModel::AllMiniLML6V2).with_show_download_progress(true),
    )?;

    let mut index = MessageIndex::load_or_new();
    let existing = index.existing_ids();
    let new_msgs: Vec<&RawMessage> = raw_messages
        .iter()
        .filter(|m| !existing.contains(&m.id))
        .collect();
    if !new_msgs.is_empty() {
        eprintln!("indexing {} new messages...", new_msgs.len());
        let texts: Vec<String> = new_msgs.iter().map(|m| m.text.clone()).collect();
        let new_embs = emb_model.embed(texts, None)?;
        let new_entries: Vec<IndexEntry> = new_msgs
            .iter()
            .map(|m| IndexEntry {
                id: m.id.clone(),
                sender: m.sender.clone(),
                text: m.text.clone(),
                timestamp: m.timestamp.clone(),
            })
            .collect();
        index.append(new_entries, new_embs);
        index.save()?;
    }
    eprintln!("index: {} entries", index.entries.len());

    let interactions = compute_interactions(&raw_messages, &config.self_name);
    let pipeline_start = Instant::now();
    let mut profiles: Vec<Profile> = Vec::new();

    // position map for corpus-wide conversational-context lookup in Phase 0
    let id_to_pos: HashMap<String, usize> = raw_messages
        .iter()
        .enumerate()
        .map(|(i, m)| (m.id.clone(), i))
        .collect();

    for (name, messages) in &by_sender {
        if messages.len() < 10 {
            continue;
        }
        let is_self = *name == config.self_name;
        let interaction = if is_self {
            None
        } else {
            interactions.get(name).cloned()
        };
        let profile = profile_person(
            name,
            messages,
            is_self,
            interaction,
            &index,
            &raw_messages,
            &id_to_pos,
            &mut emb_model,
            &model,
        );
        profiles.push(profile);
    }

    // ── cross-person calibration: z-score cognitive markers across the corpus ──
    // without this, everyone looks "highly analytical" because Discord users *are*
    apply_cognitive_zscores(&mut profiles);

    // ── insight engine: diff each profile against its prior snapshot ──
    eprintln!("\n── insights ──");
    let mut all_insights: Vec<Insight> = Vec::new();
    let id_to_msg: HashMap<String, &RawMessage> = raw_messages
        .iter()
        .map(|m| (m.id.clone(), m))
        .collect();
    for p in &profiles {
        let person_msgs: Vec<&RawMessage> = raw_messages
            .iter()
            .filter(|m| m.sender == p.name)
            .collect();
        let fp = corpus_fingerprint(&person_msgs);
        let prev = load_snapshot(&p.name);
        let unchanged = prev.as_ref().map_or(false, |s| s.fingerprint == fp);

        let insights = if unchanged {
            eprintln!("  {} — corpus unchanged, no new insights", p.name);
            Vec::new()
        } else {
            let ins = generate_insights(prev.as_ref(), p, &id_to_msg);
            if !ins.is_empty() {
                eprintln!(
                    "  {} — {} insights (top urgency {:.2})",
                    p.name,
                    ins.len(),
                    ins.first().map(|i| i.urgency).unwrap_or(0.0)
                );
            } else {
                eprintln!("  {} — no surfaced changes", p.name);
            }
            ins
        };

        all_insights.extend(insights);
        let snapshot = make_snapshot(p, fp);
        let _ = save_snapshot(&snapshot);
    }

    // ── pair interactions: real network signal ──
    eprintln!("\n── pair interactions ──");
    // collect classifications across all profiled people from cache
    let mut all_classifications: HashMap<String, MsgClassification> = HashMap::new();
    for p in &profiles {
        if let Some(cached) =
            load_json::<Vec<MsgClassification>>(&p.name, "classifications")
        {
            for c in cached {
                all_classifications.insert(c.msg_id.clone(), c);
            }
        }
    }
    // global chronological recent quartile (same time window across all pairs)
    let mut sorted_msgs: Vec<&RawMessage> = raw_messages.iter().collect();
    sorted_msgs.sort_by(|a, b| match (a.timestamp.as_ref(), b.timestamp.as_ref()) {
        (Some(ta), Some(tb)) => ta.cmp(tb),
        (Some(_), None) => std::cmp::Ordering::Greater,
        (None, Some(_)) => std::cmp::Ordering::Less,
        _ => std::cmp::Ordering::Equal,
    });
    let cutoff = (sorted_msgs.len() * 3) / 4;
    let global_recent_ids: HashSet<String> = sorted_msgs[cutoff..]
        .iter()
        .map(|m| m.id.clone())
        .collect();

    let person_names: Vec<String> = profiles.iter().map(|p| p.name.clone()).collect();
    let pair_interactions = compute_pair_interactions(
        &raw_messages,
        &all_classifications,
        &global_recent_ids,
        &person_names,
    );
    eprintln!(
        "  computed {} pairs (with ≥3 interactions)",
        pair_interactions.len()
    );

    // pair insights: diff against last run's pair state
    let prev_pairs = load_prev_pair_interactions();
    let pair_insights = generate_pair_insights(&prev_pairs, &pair_interactions);
    eprintln!("  emitted {} pair insights", pair_insights.len());
    let _ = save_pair_interactions(&pair_interactions);

    all_insights.extend(pair_insights);
    all_insights.sort_by(|a, b| b.urgency.partial_cmp(&a.urgency).unwrap());
    let top_insights: Vec<Insight> = all_insights.iter().take(40).cloned().collect();
    let _ = save_latest_insights(&top_insights);
    let _ = append_to_feed(&all_insights);
    eprintln!(
        "  ─ {} total insights this run ({} pair, {} person)",
        all_insights.len(),
        all_insights.iter().filter(|i| i.insight_type.contains("relationship") || i.insight_type.contains("tone_shift") || i.insight_type == "asymmetric_investment" || i.insight_type == "alliance_forming" || i.insight_type == "new_pair").count(),
        all_insights.iter().filter(|i| !i.insight_type.contains("relationship") && i.insight_type != "tone_shift_toward" && i.insight_type != "asymmetric_investment" && i.insight_type != "alliance_forming" && i.insight_type != "new_pair").count(),
    );

    let elapsed = pipeline_start.elapsed().as_secs();
    eprintln!("\npipeline complete: {}m{}s", elapsed / 60, elapsed % 60);

    std::fs::write(
        "graph.html",
        generate_html(
            &profiles,
            &pair_interactions,
            &config.self_name,
            &model,
            &top_insights,
        ),
    )?;
    eprintln!("wrote graph.html");

    eprintln!("\n── profiles ──");
    for p in &profiles {
        let tag = if p.is_self { " [self]" } else { "" };
        eprintln!(
            "  {}{} | {} msgs | {} themes | {} predictions | LLM: {}",
            p.name,
            tag,
            p.total_messages,
            p.themes.len(),
            p.predictions.len(),
            if p.interpretation.is_some() {
                "yes"
            } else {
                "no"
            }
        );
    }
    // ── start the embedded HTTP server (graph + Ollama proxy on one port) ──
    if std::env::var("AUSPEX_NO_SERVE").is_err() {
        let port: u16 = std::env::var("AUSPEX_PORT")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(HTTP_PORT_DEFAULT);
        let root = std::env::current_dir()
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_else(|_| ".".into());
        http_serve(port, root); // blocks until Ctrl-C
    } else {
        eprintln!("\nAUSPEX_NO_SERVE=1 — open graph.html manually");
    }
    Ok(())
}

fn demo_data(sov: &str) -> Vec<RawMessage> {
    let m = |s: &str, t: &str| RawMessage {
        id: compute_msg_id(s, None, t),
        sender: s.into(),
        text: t.into(),
        timestamp: None,
    };
    vec![
        m(sov, "just finished setting up the new CI pipeline, feels good"),
        m(sov, "been reading about distributed systems nonstop lately"),
        m(sov, "rust is genuinely making me a better programmer I think"),
        m(sov, "went climbing with alice saturday, she's getting really strong"),
        m(sov, "the fundamental problem with microservices is latency not complexity"),
        m(sov, "alice's approach to the auth refactor was really clean, learned from it"),
        m(sov, "because the underlying bottleneck is memory bandwidth not compute"),
        m(sov, "starting to feel burned out on the side project honestly"),
        m(sov, "charlie knows so much about payments infrastructure, kind of humbling"),
        m(sov, "did an online iq test for fun, scored pretty well actually"),
        m("alice", "just deployed the new api, finally got CI green after three days"),
        m("alice", "rust's borrow checker is brutal but I'm learning to love it"),
        m("alice", "the climbing gym just set new V6 routes, tried two"),
        m("alice", "hit my first V7 yesterday, fingers are completely destroyed"),
        m("alice", "because the underlying issue is the allocator not freeing properly when the reference count drops to zero"),
        m("alice", "debugging this memory leak is killing me, been at it since monday"),
        m("alice", "thinking about doing a mountaineering course this summer maybe"),
        m("alice", "the team standup is getting way too long, need to restructure it"),
        m("alice", "made homemade ramen last night, turned out actually amazing"),
        m("alice", "reading a novel right now, genuinely cannot put it down"),
        m("bob", "just finished a logo redesign for a client, pretty happy with how it turned out"),
        m("bob", "illustrator keeps crashing, might switch to figma full time honestly"),
        m("bob", "been learning rust on the side, still really confused by lifetimes"),
        m("bob", "went bouldering saturday, that new gym downtown is absolutely sick"),
        m("bob", "i think maybe i should probably update my portfolio site sometime"),
        m("bob", "freelance life is stressful but I honestly can't go back to an office"),
        m("bob", "cooking thai curry from scratch tonight, found a great recipe online"),
        m("bob", "met someone cool at the concert last night, she plays guitar in a band"),
        m("bob", "rent is getting insane, might need to find a new place"),
        m("bob", "saw an amazing band last night, absolutely incredible show"),
    ]
}
