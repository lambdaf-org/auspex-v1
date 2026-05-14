//! All metric / classification computation that runs BEFORE the LLM pipeline:
//! intent extraction, behavioral lexical stats, self↔other interaction,
//! all-pairs interaction modeling. Pure computation — no LLM, no I/O.

use crate::lexicons::*;
use crate::math::*;
use crate::types::*;
use fastembed::TextEmbedding;
use std::collections::{HashMap, HashSet};

// ─── intent extraction ───


pub(crate) fn extract_intents(messages: &[&str]) -> Vec<IntentSignal> {
    let patterns = intent_patterns();
    let mut signals = Vec::new();
    for msg in messages.iter() {
        let lower = msg.to_lowercase();
        for (category, phrases) in patterns {
            for phrase in phrases {
                if lower.contains(phrase.as_str()) {
                    signals.push(IntentSignal {
                        category: category.clone(),
                        text: msg.to_string(),
                        matched_pattern: phrase.clone(),
                    });
                    break; // one category match per message per category
                }
            }
        }
    }
    signals
}

pub(crate) fn cluster_intents(signals: &[IntentSignal], emb_model: &mut TextEmbedding) -> Vec<IntentCluster> {
    if signals.is_empty() {
        return Vec::new();
    }

    let texts: Vec<String> = signals.iter().map(|s| s.text.clone()).collect();
    let embeddings = match emb_model.embed(texts, None) {
        Ok(e) => e,
        Err(_) => return Vec::new(),
    };

    let k = ((signals.len() as f64).sqrt() / 2.0).max(2.0).min(10.0) as usize;
    let assignments = kmeans(&embeddings, k, 15);

    let mut clusters: Vec<IntentCluster> = Vec::new();
    let total = signals.len() as f64;

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

        // recency: signals later in the array are more recent
        let recency = indices.iter().map(|&i| i as f64 / total).sum::<f64>() / indices.len() as f64;

        // pick the most representative signal as the intent label
        let members: Vec<&Vec<f32>> = indices.iter().map(|&i| &embeddings[i]).collect();
        let centroid = vec_mean(&members);
        let best = *indices
            .iter()
            .max_by(|&&a, &&b| {
                cosine_sim(&embeddings[a], &centroid)
                    .partial_cmp(&cosine_sim(&embeddings[b], &centroid))
                    .unwrap()
            })
            .unwrap();

        let example_signals: Vec<String> = indices
            .iter()
            .rev()
            .take(3)
            .map(|&i| signals[i].text.chars().take(150).collect())
            .collect();

        clusters.push(IntentCluster {
            intent: signals[best].text.chars().take(120).collect(),
            count: indices.len(),
            recency_score: (recency * 100.0).round() / 100.0,
            signals: example_signals,
        });
    }

    // sort by recency-weighted count
    clusters.sort_by(|a, b| {
        let sa = a.count as f64 * (0.3 + a.recency_score * 0.7);
        let sb = b.count as f64 * (0.3 + b.recency_score * 0.7);
        sb.partial_cmp(&sa).unwrap()
    });
    clusters
}


// ─── behavioral metrics ───

pub(crate) fn compute_behavioral(messages: &[&RawMessage], excluded: &HashSet<String>) -> Behavioral {
    if messages.is_empty() {
        return Behavioral::default();
    }
    // exclude messages classified as low-info / code / pure-url so style stats reflect prose
    let messages: Vec<&str> = messages
        .iter()
        .filter(|m| !excluded.contains(&m.id))
        .map(|m| m.text.as_str())
        .collect();
    let messages = messages.as_slice();
    if messages.is_empty() {
        return Behavioral::default();
    }
    let self_ref = self_ref_words();
    let other_ref = other_ref_words();
    let certainty = certainty_words();
    let hedging = hedging_words();
    let positive = positive_words();
    let negative = negative_words();
    let analytical = analytical_words();

    let mut total_words = 0usize;
    let mut self_count = 0usize;
    let mut other_count = 0usize;
    let mut question_count = 0usize;
    let mut exclamation_count = 0usize;
    let mut link_count = 0usize;
    let mut msg_lengths: Vec<f64> = Vec::new();

    for msg in messages {
        let lower = msg.to_lowercase();
        let wc = lower.split_whitespace().count();
        total_words += wc;
        msg_lengths.push(wc as f64);
        if msg.contains('?') {
            question_count += 1;
        }
        if msg.contains('!') {
            exclamation_count += 1;
        }
        if lower.contains("http") {
            link_count += 1;
        }
        for w in lower.split_whitespace() {
            let clean: String = w
                .chars()
                .filter(|c| c.is_alphanumeric() || *c == '\'')
                .collect();
            if self_ref.contains(clean.as_str()) {
                self_count += 1;
            }
            if other_ref.contains(clean.as_str()) {
                other_count += 1;
            }
        }
    }
    let n = messages.len() as f64;
    let tw = total_words.max(1) as f64;
    let avg_len = tw / n;
    let variance = msg_lengths
        .iter()
        .map(|l| (l - avg_len).powi(2))
        .sum::<f64>()
        / n.max(1.0);

    let substantive: Vec<&str> = messages
        .iter()
        .filter(|m| {
            if m.len() >= 40 {
                return true;
            }
            let lower = m.to_lowercase();
            // short but high-signal: contains intent, emotion, or certainty markers
            intent_patterns()
                .iter()
                .any(|(_, phrases)| phrases.iter().any(|p| lower.contains(p)))
                || positive.iter().any(|w| lower.contains(w))
                || negative.iter().any(|w| lower.contains(w))
                || certainty.iter().any(|w| lower.contains(w))
        })
        .copied()
        .collect();
    let sub_n = substantive.len();
    let mut sub_words = 0usize;
    let mut sub_chars = 0usize;
    let mut sub_unique: HashSet<String> = HashSet::new();
    let mut sub_certainty = 0usize;
    let mut sub_hedging = 0usize;
    let mut sub_positive = 0usize;
    let mut sub_negative = 0usize;
    let mut sub_analytical = 0usize;
    let mut scored_messages: Vec<(f64, &str)> = Vec::new();

    for msg in &substantive {
        let lower = msg.to_lowercase();
        let words: Vec<&str> = lower.split_whitespace().collect();
        let wc = words.len();
        sub_words += wc;
        let mut msg_unique = HashSet::new();
        for w in &words {
            let clean: String = w
                .chars()
                .filter(|c| c.is_alphanumeric() || *c == '\'')
                .collect();
            if clean.is_empty() {
                continue;
            }
            sub_chars += clean.len();
            sub_unique.insert(clean.clone());
            msg_unique.insert(clean.clone());
            if certainty.contains(clean.as_str()) {
                sub_certainty += 1;
            }
            if hedging.contains(clean.as_str()) {
                sub_hedging += 1;
            }
            if positive.contains(clean.as_str()) {
                sub_positive += 1;
            }
            if negative.contains(clean.as_str()) {
                sub_negative += 1;
            }
            if analytical.contains(clean.as_str()) {
                sub_analytical += 1;
            }
        }
        let analytical_bonus = words.iter().any(|w| {
            analytical.contains(
                w.chars()
                    .filter(|c| c.is_alphanumeric())
                    .collect::<String>()
                    .as_str(),
            )
        });
        scored_messages.push((
            msg_unique.len() as f64 * (wc as f64).ln() * if analytical_bonus { 1.5 } else { 1.0 },
            msg,
        ));
    }
    scored_messages.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    let top_messages: Vec<String> = scored_messages
        .iter()
        .take(8)
        .map(|(_, m)| m.to_string())
        .collect();

    let stw = sub_words.max(1) as f64;
    let cert_total = (sub_certainty + sub_hedging).max(1) as f64;
    let emo_total = (sub_positive + sub_negative).max(1) as f64;
    let self_rate = self_count as f64 / tw * 100.0;
    let other_rate = other_count as f64 / tw * 100.0;
    let cert_ratio = sub_certainty as f64 / cert_total;
    let r2 = |v: f64| (v * 100.0).round() / 100.0;

    Behavioral {
        msg_count: messages.len(),
        total_words,
        substantive_count: sub_n,
        avg_msg_len: r2(avg_len),
        msg_len_variance: r2(variance),
        question_rate: r2(question_count as f64 / n),
        exclamation_rate: r2(exclamation_count as f64 / n),
        link_rate: r2(link_count as f64 / n),
        self_ref_rate: r2(self_rate),
        other_ref_rate: r2(other_rate),
        vocab_diversity: r2(sub_unique.len() as f64 / stw),
        certainty_ratio: r2(cert_ratio),
        analytical_rate: r2(sub_analytical as f64 / stw * 100.0),
        emotional_rate: r2((sub_positive + sub_negative) as f64 / stw * 100.0),
        emotional_valence: r2(sub_positive as f64 / emo_total),
        avg_word_len: r2(sub_chars as f64 / stw),
        clout: r2((other_rate / 100.0 * 0.4
            + cert_ratio * 0.4
            + (1.0 - self_rate / 100.0 / 15.0).max(0.0) * 0.2)
            .min(1.0)),
        authenticity: r2(((self_rate / 100.0 / 10.0).min(1.0) * 0.4
            + (1.0 - cert_ratio) * 0.3
            + (sub_positive + sub_negative) as f64 / stw * 10.0 * 0.3)
            .min(1.0)),
        top_messages,
    }
}

// ─── interaction metrics ───

pub(crate) fn compute_interactions(messages: &[RawMessage], self_name: &str) -> HashMap<String, Interaction> {
    let mut metrics: HashMap<String, Interaction> = HashMap::new();
    let mut sov_lens: HashMap<String, Vec<f64>> = HashMap::new();
    let mut other_lens: HashMap<String, Vec<f64>> = HashMap::new();
    for (i, msg) in messages.iter().enumerate() {
        if msg.sender == self_name && i > 0 && messages[i - 1].sender != self_name {
            let other = &messages[i - 1].sender;
            let m = metrics.entry(other.clone()).or_default();
            m.self_avg_reply_len += msg.text.len() as f64;
            m.other_initiations += 1;
            if msg.text.contains('?') {
                m.self_questions += 1;
            }
            sov_lens
                .entry(other.clone())
                .or_default()
                .push(msg.text.len() as f64);
        }
        if msg.sender != self_name && i > 0 && messages[i - 1].sender == self_name {
            other_lens
                .entry(msg.sender.clone())
                .or_default()
                .push(msg.text.len() as f64);
            metrics
                .entry(msg.sender.clone())
                .or_default()
                .self_initiations += 1;
        }
    }
    for (name, m) in metrics.iter_mut() {
        let replies = m.other_initiations.max(1) as f64;
        m.self_avg_reply_len = (m.self_avg_reply_len / replies).round();
        if let (Some(sl), Some(ol)) = (sov_lens.get(name), other_lens.get(name)) {
            let n = sl.len().min(ol.len());
            if n > 2 {
                let sa = sl[..n].iter().sum::<f64>() / n as f64;
                let oa = ol[..n].iter().sum::<f64>() / n as f64;
                let cov = sl[..n]
                    .iter()
                    .zip(&ol[..n])
                    .map(|(a, b)| (a - sa) * (b - oa))
                    .sum::<f64>()
                    / n as f64;
                let ss = (sl[..n].iter().map(|x| (x - sa).powi(2)).sum::<f64>() / n as f64).sqrt();
                let os = (ol[..n].iter().map(|x| (x - oa).powi(2)).sum::<f64>() / n as f64).sqrt();
                if ss > 0.0 && os > 0.0 {
                    m.mirroring = (cov / (ss * os) * 100.0).round() / 100.0;
                }
            }
        }
    }
    metrics
}

// ─── pair interaction metrics ───
//
// The point: real network intelligence isn't just per-person profiles. It's the
// pairwise signal — who replies to whom, how fast, with what tone, on what topics,
// who mentions whom in conversations with others, who's investing in whom. Everything
// below operates on (A, B) pairs and gets a baseline + recent split for trend detection.

/// Canonical (alphabetic) pair key.
pub(crate) fn canonical_pair(x: &str, y: &str) -> (String, String) {
    if x <= y {
        (x.to_string(), y.to_string())
    } else {
        (y.to_string(), x.to_string())
    }
}

/// Whole-word substring match — case-insensitive, ASCII-boundary aware.
pub(crate) fn word_contains(haystack_lower: &str, needle_lower: &str) -> bool {
    if needle_lower.is_empty() {
        return false;
    }
    let hb = haystack_lower.as_bytes();
    let nb = needle_lower.as_bytes();
    if nb.len() > hb.len() {
        return false;
    }
    let mut i = 0;
    while i + nb.len() <= hb.len() {
        if &hb[i..i + nb.len()] == nb {
            let left_ok = i == 0 || !hb[i - 1].is_ascii_alphanumeric();
            let right_ok =
                i + nb.len() == hb.len() || !hb[i + nb.len()].is_ascii_alphanumeric();
            if left_ok && right_ok {
                return true;
            }
        }
        i += 1;
    }
    false
}

pub(crate) fn extract_mentions(text: &str, names: &[String], self_sender: &str) -> Vec<String> {
    let lower = text.to_lowercase();
    let mut out: Vec<String> = Vec::new();
    for name in names {
        if name == self_sender {
            continue;
        }
        let nl = name.to_lowercase();
        if nl.chars().count() < 3 {
            continue;
        }
        if word_contains(&lower, &nl) && !out.contains(name) {
            out.push(name.clone());
        }
    }
    out
}

/// Parse a "YYYY-MM-DD HH:MM" timestamp into seconds-since-epoch-ish (lexicographic ordering OK).
pub(crate) fn parse_ts_seconds(ts: &str) -> Option<i64> {
    // expecting "YYYY-MM-DD HH:MM"
    let mut parts = ts.splitn(2, ' ');
    let date = parts.next()?;
    let time = parts.next().unwrap_or("00:00");
    let mut d = date.split('-');
    let y: i64 = d.next()?.parse().ok()?;
    let m: i64 = d.next()?.parse().ok()?;
    let day: i64 = d.next()?.parse().ok()?;
    let mut t = time.split(':');
    let h: i64 = t.next()?.parse().ok()?;
    let min: i64 = t.next().and_then(|s| s.parse().ok()).unwrap_or(0);
    // very rough conversion — adequate for relative latency
    let days = (y - 1970) * 365 + (m - 1) * 30 + (day - 1);
    Some(days * 86400 + h * 3600 + min * 60)
}

pub(crate) fn median(mut v: Vec<f64>) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

pub(crate) const PAIR_WARM_TONES: &[&str] = &[
    "warm", "supportive", "playful", "amused", "kind", "earnest", "enthusiastic",
    "curious", "tender", "affectionate", "grateful",
];
pub(crate) const PAIR_COLD_TONES: &[&str] = &[
    "hostile", "frustrated", "defensive", "resigned", "annoyed", "cold", "dismissive",
    "contemptuous", "bitter",
];

pub(crate) fn warmth_score(tones: &HashMap<String, usize>) -> f64 {
    let total = tones.values().sum::<usize>().max(1) as f64;
    let warm = PAIR_WARM_TONES
        .iter()
        .filter_map(|k| tones.get(*k))
        .copied()
        .sum::<usize>() as f64;
    let cold = PAIR_COLD_TONES
        .iter()
        .filter_map(|k| tones.get(*k))
        .copied()
        .sum::<usize>() as f64;
    (warm - cold) / total
}

pub(crate) fn dominant_tone(tones: &HashMap<String, usize>) -> Option<String> {
    tones
        .iter()
        .filter(|(_, &c)| c > 0)
        .max_by_key(|(_, c)| *c)
        .map(|(k, _)| k.clone())
}

pub(crate) fn merge_directed(
    target: &mut DirectedPairMetrics,
    other: &DirectedPairMetrics,
) -> DirectedPairMetrics {
    let mut combined = target.clone();
    combined.reply_count += other.reply_count;
    combined.initiation_count += other.initiation_count;
    combined.mention_count += other.mention_count;
    for (k, v) in &other.tones {
        *combined.tones.entry(k.clone()).or_default() += v;
    }
    *target = combined.clone();
    combined
}

pub(crate) fn compute_pair_interactions(
    messages: &[RawMessage],
    classifications: &HashMap<String, MsgClassification>,
    recent_ids: &HashSet<String>,
    person_names: &[String],
) -> Vec<PairInteraction> {
    use std::collections::BTreeMap;
    // accumulators: pair_key → (baseline PairBlock, recent PairBlock)
    let mut blocks: BTreeMap<(String, String), [PairBlock; 2]> = BTreeMap::new();
    // per-person topic distribution per window: window 0=baseline, 1=recent
    let mut topics_by_person: HashMap<String, [HashMap<String, usize>; 2]> = HashMap::new();
    // latency samples per (direction)
    let mut latencies: HashMap<(String, String, usize), Vec<f64>> = HashMap::new();
    // -> "A replied to B with latency L in window W"

    let person_set: HashSet<&str> = person_names.iter().map(|s| s.as_str()).collect();

    for (i, m) in messages.iter().enumerate() {
        if !person_set.contains(m.sender.as_str()) {
            continue;
        }
        let win: usize = if recent_ids.contains(&m.id) { 1 } else { 0 };
        // topic distribution for this person
        if let Some(c) = classifications.get(&m.id) {
            if !c.topic.is_empty() && c.topic != "media" && c.topic != "code" && c.topic != "none" {
                *topics_by_person
                    .entry(m.sender.clone())
                    .or_default()[win]
                    .entry(c.topic.clone())
                    .or_default() += 1;
            }
        }

        // mentions: who does m talk about?
        let mentioned = extract_mentions(&m.text, person_names, &m.sender);
        for other in &mentioned {
            let pair = canonical_pair(&m.sender, other);
            let block = &mut blocks
                .entry(pair.clone())
                .or_insert_with(|| [PairBlock::default(), PairBlock::default()])[win];
            // m.sender mentioned `other`
            if pair.0 == m.sender {
                block.a_to_b.mention_count += 1;
            } else {
                block.b_to_a.mention_count += 1;
            }
        }

        // adjacent-message reply detection
        if i > 0 {
            let prev = &messages[i - 1];
            if prev.sender != m.sender && person_set.contains(prev.sender.as_str()) {
                let pair = canonical_pair(&m.sender, &prev.sender);
                let block = &mut blocks
                    .entry(pair.clone())
                    .or_insert_with(|| [PairBlock::default(), PairBlock::default()])[win];
                block.adjacent_pairs += 1;

                // m replied to prev
                let (replier_to_initiator_a_to_b_dir, _init_dir_a_to_b) = if pair.0 == m.sender {
                    (true, false) // m.sender = a, prev = b → a replied to b; prev (b) initiated
                } else {
                    (false, true)
                };
                if replier_to_initiator_a_to_b_dir {
                    block.a_to_b.reply_count += 1;
                    block.b_to_a.initiation_count += 1;
                } else {
                    block.b_to_a.reply_count += 1;
                    block.a_to_b.initiation_count += 1;
                }

                // tone capture: when classifier says m is addressed to a specific person,
                // assume that person is the one m just replied to.
                if let Some(c) = classifications.get(&m.id) {
                    if c.addressee == "specific" && !c.tone.is_empty() {
                        if replier_to_initiator_a_to_b_dir {
                            *block.a_to_b.tones.entry(c.tone.clone()).or_default() += 1;
                        } else {
                            *block.b_to_a.tones.entry(c.tone.clone()).or_default() += 1;
                        }
                    }
                }

                // latency
                if let (Some(t_prev), Some(t_curr)) = (
                    prev.timestamp.as_deref().and_then(parse_ts_seconds),
                    m.timestamp.as_deref().and_then(parse_ts_seconds),
                ) {
                    let dt = (t_curr - t_prev).max(0) as f64;
                    if dt < 7.0 * 86400.0 {
                        // ignore gaps > 1 week as "different conversation"
                        let from = m.sender.clone();
                        let to = prev.sender.clone();
                        latencies
                            .entry((from, to, win))
                            .or_default()
                            .push(dt);
                    }
                }
            }
        }
    }

    // finalize: compute topic_overlap, latencies, warmth, edge_strength
    let mut out: Vec<PairInteraction> = Vec::new();
    for (pair, mut wins) in blocks {
        for win in 0..2 {
            // topic overlap between pair.0 and pair.1 (using window-specific topic dist)
            let empty = HashMap::new();
            let a_topics = topics_by_person
                .get(&pair.0)
                .map(|w| &w[win])
                .unwrap_or(&empty);
            let b_topics = topics_by_person
                .get(&pair.1)
                .map(|w| &w[win])
                .unwrap_or(&empty);
            let a_set: HashSet<&String> = a_topics.keys().collect();
            let b_set: HashSet<&String> = b_topics.keys().collect();
            let inter = a_set.intersection(&b_set).count() as f64;
            let union = a_set.union(&b_set).count().max(1) as f64;
            wins[win].topic_overlap = (inter / union * 100.0).round() / 100.0;
            wins[win].shared_topics =
                a_set.intersection(&b_set).map(|s| (*s).clone()).collect();
            // warmth: pooled tones from both directions
            let mut pooled: HashMap<String, usize> = wins[win].a_to_b.tones.clone();
            for (k, v) in &wins[win].b_to_a.tones {
                *pooled.entry(k.clone()).or_default() += v;
            }
            wins[win].warmth = (warmth_score(&pooled) * 100.0).round() / 100.0;
            // latencies
            if let Some(samples) = latencies.remove(&(pair.0.clone(), pair.1.clone(), win)) {
                wins[win].a_to_b.reply_latency_p50_secs = median(samples).round();
            }
            if let Some(samples) = latencies.remove(&(pair.1.clone(), pair.0.clone(), win)) {
                wins[win].b_to_a.reply_latency_p50_secs = median(samples).round();
            }
        }
        // edge_strength: log-scaled interaction × (topic+0.1) × warmth-boost, computed on RECENT
        let r = &wins[1];
        let intensity = ((r.a_to_b.reply_count + r.b_to_a.reply_count + r.a_to_b.mention_count + r.b_to_a.mention_count) as f64 + 1.0).ln();
        let topic = r.topic_overlap + 0.10;
        let warmth_boost = 1.0 + 0.6 * r.warmth.max(-0.5);
        let edge_strength = (intensity * topic * warmth_boost * 100.0).round() / 100.0;

        out.push(PairInteraction {
            a: pair.0,
            b: pair.1,
            baseline: std::mem::take(&mut wins[0]),
            recent: std::mem::take(&mut wins[1]),
            edge_strength,
        });
    }

    // drop pairs with no meaningful interaction at all
    out.retain(|p| {
        let total = p.baseline.adjacent_pairs
            + p.recent.adjacent_pairs
            + p.baseline.a_to_b.mention_count
            + p.baseline.b_to_a.mention_count
            + p.recent.a_to_b.mention_count
            + p.recent.b_to_a.mention_count;
        total >= 3
    });

    // normalize edge_strength to 0..1 (max over all surviving pairs)
    let max_es = out.iter().map(|p| p.edge_strength).fold(0.0_f64, f64::max);
    if max_es > 0.0 {
        for p in out.iter_mut() {
            p.edge_strength = (p.edge_strength / max_es * 100.0).round() / 100.0;
        }
    }

    out.sort_by(|a, b| b.edge_strength.partial_cmp(&a.edge_strength).unwrap());
    out
}
