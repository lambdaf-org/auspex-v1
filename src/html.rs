//! Renders `graph.html` from the embedded template in `assets/index.html`.
//!
//! The template is plain HTML/CSS/JS — no Rust-format escaping. Data is injected
//! as a single `window.__DATA__` JSON blob by replacing the `/*AUSPEX_DATA*/null`
//! sentinel. Edit the template directly; recompile only when the binary changes.

use crate::types::*;
use std::collections::HashSet;

const TEMPLATE: &str = include_str!("../assets/index.html");
const SENTINEL: &str = "/*AUSPEX_DATA*/null";

pub(crate) fn generate_html(
    profiles: &[Profile],
    pairs: &[PairInteraction],
    self_name: &str,
    model: &str,
    insights: &[Insight],
) -> String {
    let nodes = build_nodes(profiles);
    let edges = build_edges(profiles, pairs);
    let insights_json = build_insights(insights);

    let data = serde_json::json!({
        "nodes": nodes,
        "edges": edges,
        "insights": insights_json,
        "sov": self_name,
        "model": model,
    });
    let data_str = serde_json::to_string(&data).unwrap_or_else(|_| "null".into());

    TEMPLATE.replacen(SENTINEL, &data_str, 1)
}

// ─── node JSON ───────────────────────────────────────────────────────────────

fn build_nodes(profiles: &[Profile]) -> Vec<serde_json::Value> {
    profiles.iter().map(|p| {
        let i = p.interpretation.clone().unwrap_or_default();
        let b = &p.behavioral;
        let c = &p.cognitive;
        let cr = &p.cognitive_recent;
        let bf = &p.big_five;
        let bfr = &p.big_five_recent;
        let themes: Vec<serde_json::Value> = p.themes.iter().map(|t| {
            let support: Vec<serde_json::Value> = t.support.iter().map(|q| serde_json::json!({
                "quote": q.quote, "msgId": q.msg_id,
            })).collect();
            serde_json::json!({
                "name": t.name, "count": t.count, "msgCount": t.msg_count,
                "confidence": t.confidence, "analysis": t.analysis,
                "support": support,
                "falsificationsChecked": t.falsifications_checked,
                "falsificationsConfirmed": t.falsifications_confirmed,
                "falsificationSummary": t.falsification_summary,
                "recentShare": t.recent_share,
                "temporalStatus": t.temporal_status,
            })
        }).collect();
        let bf_json = |v: &Vec<EvidenceQuote>| -> Vec<serde_json::Value> {
            v.iter().map(|q| serde_json::json!({"quote": q.quote, "msgId": q.msg_id})).collect()
        };
        let claims: Vec<serde_json::Value> = p.self_claims.iter().map(|sc| {
            let ev: Vec<serde_json::Value> = sc.behavioral_evidence.iter()
                .map(|q| serde_json::json!({"quote": q.quote, "msgId": q.msg_id}))
                .collect();
            serde_json::json!({
                "claim": sc.claim, "msgId": sc.msg_id, "dimension": sc.dimension,
                "verdict": sc.verdict, "rationale": sc.rationale,
                "behavioralEvidence": ev,
            })
        }).collect();
        let mut n = serde_json::json!({
            "id":p.name,"messages":p.total_messages,"isSelf":p.is_self,
            "themeCount":p.themes.len(),"themes":themes,
            "identity":i.identity,"anxiety":i.anxiety,"social_style":i.social_style,
            "growth":i.growth,"vulnerability":i.vulnerability,"summary":i.summary,
            "beh":{"selfRef":b.self_ref_rate,"otherRef":b.other_ref_rate,"certainty":b.certainty_ratio,
                "vocabDiv":b.vocab_diversity,"analytical":b.analytical_rate,"emotional":b.emotional_rate,
                "valence":b.emotional_valence,"clout":b.clout,"authenticity":b.authenticity,
                "questionRate":b.question_rate,"avgMsgLen":b.avg_msg_len,"substantive":b.substantive_count,
                "avgWordLen":b.avg_word_len},
            "cognitive": {
                "abstractRate": c.abstract_rate, "conditionalRate": c.conditional_rate,
                "integrativeComplexity": c.integrative_complexity,
                "lexicalComplexity": c.lexical_complexity,
                "domainBreadth": c.domain_breadth, "selfMonitoring": c.self_monitoring,
                "zAbstract": c.z_abstract, "zConditional": c.z_conditional,
                "zIntegrative": c.z_integrative, "zLexical": c.z_lexical,
                "zBreadth": c.z_breadth, "zSelfMonitoring": c.z_self_monitoring,
                "sampleSize": c.sample_size,
            },
            "cognitiveRecent": {
                "abstractRate": cr.abstract_rate, "conditionalRate": cr.conditional_rate,
                "integrativeComplexity": cr.integrative_complexity,
                "lexicalComplexity": cr.lexical_complexity,
                "selfMonitoring": cr.self_monitoring,
                "sampleSize": cr.sample_size,
            },
            "bigFive": {
                "openness": bf_json(&bf.openness),
                "conscientiousness": bf_json(&bf.conscientiousness),
                "extraversion": bf_json(&bf.extraversion),
                "agreeableness": bf_json(&bf.agreeableness),
                "neuroticism": bf_json(&bf.neuroticism),
            },
            "bigFiveRecent": {
                "openness": bf_json(&bfr.openness),
                "conscientiousness": bf_json(&bfr.conscientiousness),
                "extraversion": bf_json(&bfr.extraversion),
                "agreeableness": bf_json(&bfr.agreeableness),
                "neuroticism": bf_json(&bfr.neuroticism),
            },
            "selfClaims": claims,
        });
        if let Some(r) = &p.interaction {
            n["reaction"] = serde_json::json!({"avgReply":r.self_avg_reply_len,
                "sovInit":r.self_initiations,"otherInit":r.other_initiations,
                "questions":r.self_questions,"mirroring":r.mirroring});
        }
        let preds: Vec<serde_json::Value> = p.predictions.iter().map(|pr| serde_json::json!({
            "action": pr.action, "confidence": pr.confidence,
            "timeframe": pr.timeframe, "evidence": pr.evidence,
        })).collect();
        n["predictions"] = serde_json::json!(preds);
        n
    }).collect()
}

// ─── edge JSON (derived from pair interactions) ──────────────────────────────

fn build_edges(profiles: &[Profile], pairs: &[PairInteraction]) -> Vec<serde_json::Value> {
    let profiled: HashSet<&str> = profiles.iter().map(|p| p.name.as_str()).collect();
    pairs
        .iter()
        .filter(|p| profiled.contains(p.a.as_str()) && profiled.contains(p.b.as_str()))
        .filter(|p| p.edge_strength > 0.05)
        .map(|p| {
            let dir = |d: &DirectedPairMetrics| {
                let tones: Vec<serde_json::Value> = d
                    .tones
                    .iter()
                    .map(|(k, v)| serde_json::json!({"tone": k, "n": v}))
                    .collect();
                serde_json::json!({
                    "replyCount": d.reply_count,
                    "initiationCount": d.initiation_count,
                    "mentionCount": d.mention_count,
                    "replyLatencyP50Secs": d.reply_latency_p50_secs,
                    "tones": tones,
                })
            };
            let block = |b: &PairBlock| {
                serde_json::json!({
                    "aToB": dir(&b.a_to_b),
                    "bToA": dir(&b.b_to_a),
                    "topicOverlap": b.topic_overlap,
                    "sharedTopics": b.shared_topics,
                    "adjacentPairs": b.adjacent_pairs,
                    "warmth": b.warmth,
                })
            };
            serde_json::json!({
                "source": p.a,
                "target": p.b,
                "score": p.edge_strength,
                "warmth": p.recent.warmth,
                "baseline": block(&p.baseline),
                "recent": block(&p.recent),
            })
        })
        .collect()
}

// ─── insights JSON ───────────────────────────────────────────────────────────

fn build_insights(insights: &[Insight]) -> Vec<serde_json::Value> {
    insights
        .iter()
        .map(|i| {
            let ev: Vec<serde_json::Value> = i
                .evidence
                .iter()
                .map(|q| serde_json::json!({"quote": q.quote, "msgId": q.msg_id}))
                .collect();
            serde_json::json!({
                "type": i.insight_type,
                "person": i.person,
                "summary": i.summary,
                "details": i.details,
                "urgency": i.urgency,
                "evidence": ev,
                "firstSeen": i.first_seen,
                "relatedTheme": i.related_theme,
                "relatedClaim": i.related_claim,
            })
        })
        .collect()
}
