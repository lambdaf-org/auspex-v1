//! Snapshot-based insight engine: diffs current profiles + pair interactions
//! against the prior run's snapshot to surface ranked, evidence-cited insights.
//! Persistent feed log + per-run latest.json.

use crate::types::*;
use crate::INSIGHTS_DIR;
use std::collections::HashMap;

pub(crate) fn save_pair_interactions(pairs: &[PairInteraction]) -> std::io::Result<()> {
    std::fs::create_dir_all(INSIGHTS_DIR)?;
    let path = format!("{}/pair_interactions.json", INSIGHTS_DIR);
    let tmp = format!("{}.tmp", path);
    std::fs::write(&tmp, serde_json::to_string_pretty(pairs)?)?;
    std::fs::rename(&tmp, &path)
}

pub(crate) fn load_prev_pair_interactions() -> Vec<PairInteraction> {
    std::fs::read_to_string(format!("{}/pair_interactions.json", INSIGHTS_DIR))
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

/// Diff prior vs current pair interactions → emit ranked pair insights.
pub(crate) fn generate_pair_insights(
    prev: &[PairInteraction],
    curr: &[PairInteraction],
) -> Vec<Insight> {
    let now = now_iso();
    let prev_map: HashMap<(String, String), &PairInteraction> = prev
        .iter()
        .map(|p| ((p.a.clone(), p.b.clone()), p))
        .collect();

    let mut out: Vec<Insight> = Vec::new();
    for p in curr {
        // ── relationship cooling/warming: compare recent reply counts vs baseline (within same run) ──
        let recent_total = p.recent.a_to_b.reply_count
            + p.recent.b_to_a.reply_count
            + p.recent.a_to_b.mention_count
            + p.recent.b_to_a.mention_count;
        let baseline_total = p.baseline.a_to_b.reply_count
            + p.baseline.b_to_a.reply_count
            + p.baseline.a_to_b.mention_count
            + p.baseline.b_to_a.mention_count;
        // baseline is 75% of corpus, recent is 25% — normalize to per-window rate
        let baseline_per_quartile = baseline_total as f64 / 3.0;
        if baseline_total >= 8 {
            let ratio = recent_total as f64 / baseline_per_quartile.max(0.5);
            if ratio <= 0.4 {
                out.push(Insight {
                    insight_type: "relationship_cooling".into(),
                    person: p.a.clone(),
                    summary: format!(
                        "{} ↔ {}: interaction dropped to {:.0}% of baseline (recent {} vs baseline-rate {:.1})",
                        p.a,
                        p.b,
                        ratio * 100.0,
                        recent_total,
                        baseline_per_quartile
                    ),
                    details: format!(
                        "Baseline window had {} adjacent exchanges + mentions; recent quartile has {}. \
                         Warmth: baseline {:.2}, recent {:.2}.",
                        baseline_total, recent_total, p.baseline.warmth, p.recent.warmth
                    ),
                    urgency: (0.55 + (0.4 - ratio).max(0.0) * 0.8).min(1.0),
                    evidence: Vec::new(),
                    first_seen: now.clone(),
                    related_theme: None,
                    related_claim: Some(p.b.clone()),
                });
            } else if ratio >= 2.0 {
                out.push(Insight {
                    insight_type: "relationship_warming".into(),
                    person: p.a.clone(),
                    summary: format!(
                        "{} ↔ {}: interaction surged to {:.0}% of baseline rate ({} recent)",
                        p.a,
                        p.b,
                        ratio * 100.0,
                        recent_total
                    ),
                    details: format!(
                        "Recent {} exchanges + mentions vs baseline rate {:.1}. Warmth recent {:.2}.",
                        recent_total, baseline_per_quartile, p.recent.warmth
                    ),
                    urgency: (0.45 + (ratio - 2.0).min(2.0) * 0.1).min(0.85),
                    evidence: Vec::new(),
                    first_seen: now.clone(),
                    related_theme: None,
                    related_claim: Some(p.b.clone()),
                });
            }
        }

        // ── tone shift toward the other ──
        let bd = crate::metrics::dominant_tone(&p.baseline.a_to_b.tones);
        let rd = crate::metrics::dominant_tone(&p.recent.a_to_b.tones);
        if let (Some(b1), Some(r1)) = (bd.as_ref(), rd.as_ref()) {
            if b1 != r1
                && p.baseline.a_to_b.tones.values().sum::<usize>() >= 4
                && p.recent.a_to_b.tones.values().sum::<usize>() >= 3
            {
                let bw = crate::metrics::warmth_score(&p.baseline.a_to_b.tones);
                let rw = crate::metrics::warmth_score(&p.recent.a_to_b.tones);
                let toward_negative = rw < bw;
                let urgency = if toward_negative { 0.7 } else { 0.45 };
                out.push(Insight {
                    insight_type: "tone_shift_toward".into(),
                    person: p.a.clone(),
                    summary: format!(
                        "{} → {}: tone shifted {} → {} (warmth {:.2} → {:.2})",
                        p.a, p.b, b1, r1, bw, rw
                    ),
                    details: format!(
                        "Direction-of-addressee tone changed in the recent quartile. \
                         Negative-direction shifts often precede cooling.",
                    ),
                    urgency,
                    evidence: Vec::new(),
                    first_seen: now.clone(),
                    related_theme: None,
                    related_claim: Some(p.b.clone()),
                });
            }
        }

        // ── asymmetric investment ──
        let a_recent = p.recent.a_to_b.reply_count + p.recent.a_to_b.initiation_count;
        let b_recent = p.recent.b_to_a.reply_count + p.recent.b_to_a.initiation_count;
        let max_side = a_recent.max(b_recent);
        let min_side = a_recent.min(b_recent);
        if max_side >= 6 && min_side as f64 / max_side as f64 <= 0.30 {
            let invester = if a_recent > b_recent {
                p.a.clone()
            } else {
                p.b.clone()
            };
            let target = if a_recent > b_recent {
                p.b.clone()
            } else {
                p.a.clone()
            };
            out.push(Insight {
                insight_type: "asymmetric_investment".into(),
                person: invester.clone(),
                summary: format!(
                    "{} invests heavily in {} ({} acts vs {} return)",
                    invester, target, max_side, min_side
                ),
                details: format!(
                    "Recent quartile: one-sided initiation/reply pattern. Could be courtship, \
                     hierarchy, or one party tuning out."
                ),
                urgency: 0.55,
                evidence: Vec::new(),
                first_seen: now.clone(),
                related_theme: None,
                related_claim: Some(target),
            });
        }

        // ── alliance forming: topic overlap rising + interaction warming together ──
        let topic_delta = p.recent.topic_overlap - p.baseline.topic_overlap;
        let warmth_delta = p.recent.warmth - p.baseline.warmth;
        if topic_delta >= 0.10 && warmth_delta >= 0.10 && p.recent.adjacent_pairs >= 4 {
            out.push(Insight {
                insight_type: "alliance_forming".into(),
                person: p.a.clone(),
                summary: format!(
                    "{} ↔ {}: alliance signal — topic overlap +{:.0}%, warmth +{:.2}",
                    p.a,
                    p.b,
                    topic_delta * 100.0,
                    warmth_delta
                ),
                details: format!(
                    "Shared topics: {}. Recent interaction strength {:.2}.",
                    p.recent.shared_topics.iter().take(4).cloned().collect::<Vec<_>>().join(", "),
                    p.edge_strength
                ),
                urgency: 0.5 + topic_delta.min(0.3),
                evidence: Vec::new(),
                first_seen: now.clone(),
                related_theme: None,
                related_claim: Some(p.b.clone()),
            });
        }

        // ── new pair: edge that didn't exist last run (>= 5 recent interactions) ──
        if !prev.is_empty() {
            let key = (p.a.clone(), p.b.clone());
            if !prev_map.contains_key(&key) && p.recent.adjacent_pairs >= 5 {
                out.push(Insight {
                    insight_type: "new_pair".into(),
                    person: p.a.clone(),
                    summary: format!(
                        "{} ↔ {}: new pair — {} recent exchanges",
                        p.a, p.b, p.recent.adjacent_pairs
                    ),
                    details: format!(
                        "This pair had no significant prior interaction. Edge strength now {:.2}.",
                        p.edge_strength
                    ),
                    urgency: 0.5,
                    evidence: Vec::new(),
                    first_seen: now.clone(),
                    related_theme: None,
                    related_claim: Some(p.b.clone()),
                });
            }
        }
    }

    out.sort_by(|a, b| b.urgency.partial_cmp(&a.urgency).unwrap());
    out
}





// ─── insight engine ───
//
// The point of this tool isn't the static profile — it's noticing *what changed*
// when new messages come in. Insights are computed by diffing the current Profile
// against a saved snapshot from the previous run.


pub(crate) fn now_iso() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let t = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    // simple ISO-ish: seconds since epoch is sortable + parseable enough for now
    format!("{}", t)
}

pub(crate) fn corpus_fingerprint(messages: &[&RawMessage]) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut ids: Vec<&str> = messages.iter().map(|m| m.id.as_str()).collect();
    ids.sort_unstable();
    let mut h = DefaultHasher::new();
    for i in &ids {
        i.hash(&mut h);
    }
    format!("{:016x}-{}", h.finish(), ids.len())
}

pub(crate) fn snapshot_path(name: &str) -> String {
    format!("{}/snapshots/{}.json", INSIGHTS_DIR, name)
}

pub(crate) fn load_snapshot(name: &str) -> Option<ProfileSnapshot> {
    std::fs::read_to_string(snapshot_path(name))
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
}

pub(crate) fn save_snapshot(snapshot: &ProfileSnapshot) -> std::io::Result<()> {
    std::fs::create_dir_all(format!("{}/snapshots", INSIGHTS_DIR))?;
    let path = snapshot_path(&snapshot.name);
    let tmp = format!("{}.tmp", path);
    std::fs::write(&tmp, serde_json::to_string_pretty(snapshot)?)?;
    std::fs::rename(&tmp, &path)
}

pub(crate) fn make_snapshot(profile: &Profile, fingerprint: String) -> ProfileSnapshot {
    ProfileSnapshot {
        name: profile.name.clone(),
        fingerprint,
        total_messages: profile.total_messages,
        snapshot_at: now_iso(),
        themes: profile
            .themes
            .iter()
            .map(|t| ThemeSnapshot {
                name: t.name.clone(),
                confidence: t.confidence,
                msg_count: t.msg_count,
                recent_share: t.recent_share,
                temporal_status: t.temporal_status.clone(),
            })
            .collect(),
        cognitive: profile.cognitive.clone(),
        cognitive_recent: profile.cognitive_recent.clone(),
        self_claims: profile
            .self_claims
            .iter()
            .map(|sc| SelfClaimSnapshot {
                claim: sc.claim.clone(),
                msg_id: sc.msg_id.clone(),
                dimension: sc.dimension.clone(),
                verdict: sc.verdict.clone(),
            })
            .collect(),
    }
}

/// True if this dimension/claim is high-stakes enough to surface even without comparison.
pub(crate) fn is_high_stakes_claim(dim: &str, claim: &str) -> bool {
    let dim_l = dim.to_lowercase();
    if matches!(
        dim_l.as_str(),
        "identity" | "intelligence" | "profession" | "history"
    ) {
        return true;
    }
    let lower = claim.to_lowercase();
    // mood claims that flag distress
    let red_flags = [
        "kill myself", "killing myself", "want to die", "end it", "suicidal",
        "im done", "i'm done", "giving up", "leaving", "quitting",
        "depressed", "depression", "anxious all the time", "panic",
        "burnt out", "burned out", "breaking down",
    ];
    red_flags.iter().any(|p| lower.contains(p))
}

pub(crate) fn generate_insights(
    prev: Option<&ProfileSnapshot>,
    curr: &Profile,
    msg_lookup: &HashMap<String, &RawMessage>,
) -> Vec<Insight> {
    let mut out: Vec<Insight> = Vec::new();
    let now = now_iso();

    // ── theme deltas ──
    let prev_themes: HashMap<String, &ThemeSnapshot> = prev
        .map(|p| p.themes.iter().map(|t| (t.name.clone(), t)).collect())
        .unwrap_or_default();

    for t in &curr.themes {
        if t.confidence < 0.4 {
            continue;
        }
        let support_evidence: Vec<EvidenceQuote> =
            t.support.iter().take(3).cloned().collect();

        match prev_themes.get(&t.name) {
            None if prev.is_some() => {
                // NEW theme (only meaningful if we had a prior snapshot)
                let urgency = 0.5
                    * t.confidence
                    * (1.0_f64).min(t.msg_count as f64 / 10.0)
                    * if t.temporal_status == "active" { 1.4 } else { 1.0 };
                out.push(Insight {
                    insight_type: "new_theme".into(),
                    person: curr.name.clone(),
                    summary: format!(
                        "new pattern emerged: {} ({}% confidence, {} msgs)",
                        t.name,
                        (t.confidence * 100.0).round(),
                        t.msg_count
                    ),
                    details: t.analysis.clone(),
                    urgency: urgency.min(1.0),
                    evidence: support_evidence.clone(),
                    first_seen: now.clone(),
                    related_theme: Some(t.name.clone()),
                    related_claim: None,
                });
            }
            Some(prev_t) => {
                // confidence jump
                let dconf = t.confidence - prev_t.confidence;
                if dconf.abs() >= 0.15 {
                    let direction = if dconf > 0.0 { "↑" } else { "↓" };
                    let urgency = (0.4 * dconf.abs() * 3.0).min(1.0);
                    out.push(Insight {
                        insight_type: "confidence_jump".into(),
                        person: curr.name.clone(),
                        summary: format!(
                            "theme '{}' confidence {} ({:.0}% → {:.0}%)",
                            t.name,
                            direction,
                            prev_t.confidence * 100.0,
                            t.confidence * 100.0
                        ),
                        details: format!(
                            "{} support changed: {} → {} msgs.",
                            t.analysis, prev_t.msg_count, t.msg_count
                        ),
                        urgency,
                        evidence: support_evidence.clone(),
                        first_seen: now.clone(),
                        related_theme: Some(t.name.clone()),
                        related_claim: None,
                    });
                }
                // temporal status change
                if t.temporal_status != prev_t.temporal_status
                    && !t.temporal_status.is_empty()
                    && !prev_t.temporal_status.is_empty()
                {
                    let severity = match (prev_t.temporal_status.as_str(), t.temporal_status.as_str()) {
                        ("fading", "active") | ("stable", "active") => 0.75,
                        ("active", "fading") => 0.5,
                        _ => 0.35,
                    };
                    let urgency = (severity * t.confidence).min(1.0);
                    out.push(Insight {
                        insight_type: "theme_status_change".into(),
                        person: curr.name.clone(),
                        summary: format!(
                            "'{}' {} → {} (recent-share {:.0}%)",
                            t.name,
                            prev_t.temporal_status,
                            t.temporal_status,
                            t.recent_share * 100.0
                        ),
                        details: t.analysis.clone(),
                        urgency,
                        evidence: support_evidence.clone(),
                        first_seen: now.clone(),
                        related_theme: Some(t.name.clone()),
                        related_claim: None,
                    });
                }
            }
            _ => {}
        }
    }

    // ── cognitive marker shifts (recent vs baseline of CURRENT profile) ──
    let cog = &curr.cognitive;
    let cog_r = &curr.cognitive_recent;
    if cog_r.sample_size > 0 {
        let shifts: [(&str, f64, f64); 4] = [
            ("integrative complexity", cog.integrative_complexity, cog_r.integrative_complexity),
            ("conditional rate", cog.conditional_rate, cog_r.conditional_rate),
            ("self-monitoring", cog.self_monitoring, cog_r.self_monitoring),
            ("abstract rate", cog.abstract_rate, cog_r.abstract_rate),
        ];
        for (name_m, old, new) in shifts {
            if old.abs() < 0.05 && new.abs() < 0.05 {
                continue;
            }
            let scale = old.abs().max(0.1);
            let pct = (new - old) / scale;
            if pct.abs() < 0.20 {
                continue;
            }
            let direction = if pct > 0.0 { "↑" } else { "↓" };
            let urgency = (0.35 * pct.abs()).min(1.0);
            out.push(Insight {
                insight_type: "cognitive_shift".into(),
                person: curr.name.clone(),
                summary: format!(
                    "{} {} {:.0}% in recent quartile ({:.2} → {:.2})",
                    name_m,
                    direction,
                    pct.abs() * 100.0,
                    old,
                    new
                ),
                details: format!(
                    "Recent quartile shows {} in {}. {} substantive messages in window.",
                    if pct > 0.0 { "elevated" } else { "reduced" },
                    name_m,
                    cog_r.sample_size
                ),
                urgency,
                evidence: Vec::new(),
                first_seen: now.clone(),
                related_theme: None,
                related_claim: None,
            });
        }
    }

    // ── self-claim deltas ──
    let prev_claims: HashMap<String, &SelfClaimSnapshot> = prev
        .map(|p| {
            p.self_claims
                .iter()
                .map(|c| (c.claim.to_lowercase(), c))
                .collect()
        })
        .unwrap_or_default();

    for c in &curr.self_claims {
        let key = c.claim.to_lowercase();
        let evidence = msg_lookup
            .get(&c.msg_id)
            .map(|m| {
                vec![EvidenceQuote {
                    quote: m.text.chars().take(220).collect(),
                    msg_id: c.msg_id.clone(),
                }]
            })
            .unwrap_or_default();

        match prev_claims.get(&key) {
            None if prev.is_some() => {
                let high_stakes = is_high_stakes_claim(&c.dimension, &c.claim);
                let severity: f64 = if high_stakes { 0.85 } else { 0.45 };
                let weight: f64 = match c.verdict.as_str() {
                    "inconsistent" => 1.0,
                    "consistent" => 0.5,
                    _ => 0.7,
                };
                let urgency = (severity * weight).min(1.0);
                out.push(Insight {
                    insight_type: if high_stakes {
                        "high_stakes_claim"
                    } else {
                        "new_self_claim"
                    }
                    .into(),
                    person: curr.name.clone(),
                    summary: format!(
                        "claim about {}: \"{}\" [{}]",
                        c.dimension,
                        c.claim.chars().take(80).collect::<String>(),
                        c.verdict
                    ),
                    details: c.rationale.clone(),
                    urgency,
                    evidence,
                    first_seen: now.clone(),
                    related_theme: None,
                    related_claim: Some(c.claim.clone()),
                });
            }
            Some(prev_c) => {
                if prev_c.verdict != c.verdict
                    && !prev_c.verdict.is_empty()
                    && !c.verdict.is_empty()
                {
                    let urgency: f64 = 0.7;
                    out.push(Insight {
                        insight_type: "claim_verdict_change".into(),
                        person: curr.name.clone(),
                        summary: format!(
                            "claim '{}' verdict {} → {}",
                            c.claim.chars().take(60).collect::<String>(),
                            prev_c.verdict,
                            c.verdict
                        ),
                        details: c.rationale.clone(),
                        urgency,
                        evidence,
                        first_seen: now.clone(),
                        related_theme: None,
                        related_claim: Some(c.claim.clone()),
                    });
                }
            }
            _ => {
                // first-run, no prior snapshot: still flag high-stakes claims (they don't decay)
                if is_high_stakes_claim(&c.dimension, &c.claim) {
                    out.push(Insight {
                        insight_type: "high_stakes_claim".into(),
                        person: curr.name.clone(),
                        summary: format!(
                            "high-stakes claim ({}): \"{}\" [{}]",
                            c.dimension,
                            c.claim.chars().take(80).collect::<String>(),
                            c.verdict
                        ),
                        details: c.rationale.clone(),
                        urgency: 0.75,
                        evidence,
                        first_seen: now.clone(),
                        related_theme: None,
                        related_claim: Some(c.claim.clone()),
                    });
                }
            }
        }
    }

    out.sort_by(|a, b| b.urgency.partial_cmp(&a.urgency).unwrap());
    out
}

pub(crate) fn append_to_feed(insights: &[Insight]) -> std::io::Result<()> {
    std::fs::create_dir_all(INSIGHTS_DIR)?;
    let path = format!("{}/feed.json", INSIGHTS_DIR);
    let mut feed: InsightFeed = std::fs::read_to_string(&path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default();
    feed.insights.extend_from_slice(insights);
    feed.last_run_at = now_iso();
    // cap accumulated feed at the most recent 500 to keep file size bounded
    let n = feed.insights.len();
    if n > 500 {
        feed.insights = feed.insights.split_off(n - 500);
    }
    let tmp = format!("{}.tmp", path);
    std::fs::write(&tmp, serde_json::to_string_pretty(&feed)?)?;
    std::fs::rename(&tmp, &path)
}

pub(crate) fn save_latest_insights(insights: &[Insight]) -> std::io::Result<()> {
    std::fs::create_dir_all(INSIGHTS_DIR)?;
    let path = format!("{}/latest.json", INSIGHTS_DIR);
    let tmp = format!("{}.tmp", path);
    std::fs::write(&tmp, serde_json::to_string_pretty(insights)?)?;
    std::fs::rename(&tmp, &path)
}




