//! All serializable data types used across the pipeline.
//! Pure data — no I/O, no business logic. Trivial helpers (default-field
//! derivations, back-compat field projection) live here when they're tied
//! to a single struct.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ─── intent extraction ───

#[derive(Serialize, Deserialize, Clone)]
pub(crate) struct IntentSignal {
    pub(crate) category: String,
    pub(crate) text: String,
    pub(crate) matched_pattern: String,
}

#[derive(Serialize, Deserialize, Clone)]
pub(crate) struct IntentCluster {
    pub(crate) intent: String,
    pub(crate) count: usize,
    pub(crate) recency_score: f64,
    pub(crate) signals: Vec<String>,
}

#[derive(Serialize, Deserialize, Clone)]
pub(crate) struct Prediction {
    pub(crate) action: String,
    pub(crate) confidence: f64,
    pub(crate) timeframe: String,
    pub(crate) evidence: Vec<String>,
}

// ─── raw message (chat input) ───

#[derive(Clone, Serialize, Deserialize)]
pub(crate) struct RawMessage {
    pub(crate) id: String,
    pub(crate) sender: String,
    pub(crate) text: String,
    pub(crate) timestamp: Option<String>,
}

// ─── behavioral metrics ───

#[derive(Serialize, Deserialize, Clone, Default)]
pub(crate) struct Behavioral {
    pub(crate) msg_count: usize,
    pub(crate) total_words: usize,
    pub(crate) substantive_count: usize,
    pub(crate) avg_msg_len: f64,
    pub(crate) msg_len_variance: f64,
    pub(crate) question_rate: f64,
    pub(crate) exclamation_rate: f64,
    pub(crate) link_rate: f64,
    pub(crate) self_ref_rate: f64,
    pub(crate) other_ref_rate: f64,
    pub(crate) vocab_diversity: f64,
    pub(crate) certainty_ratio: f64,
    pub(crate) analytical_rate: f64,
    pub(crate) emotional_rate: f64,
    pub(crate) emotional_valence: f64,
    pub(crate) avg_word_len: f64,
    pub(crate) clout: f64,
    pub(crate) authenticity: f64,
    pub(crate) top_messages: Vec<String>,
}

// ─── self↔other interaction (legacy) ───

#[derive(Serialize, Deserialize, Clone, Default)]
pub(crate) struct Interaction {
    pub(crate) self_avg_reply_len: f64,
    pub(crate) self_initiations: usize,
    pub(crate) other_initiations: usize,
    pub(crate) self_questions: usize,
    pub(crate) mirroring: f64,
}

// ─── pair interaction (all-pairs) ───

#[derive(Serialize, Deserialize, Clone, Default)]
pub(crate) struct DirectedPairMetrics {
    pub(crate) reply_count: usize,
    pub(crate) reply_latency_p50_secs: f64,
    pub(crate) initiation_count: usize,
    pub(crate) mention_count: usize,
    #[serde(default)]
    pub(crate) tones: HashMap<String, usize>,
}

#[derive(Serialize, Deserialize, Clone, Default)]
pub(crate) struct PairBlock {
    pub(crate) a_to_b: DirectedPairMetrics,
    pub(crate) b_to_a: DirectedPairMetrics,
    pub(crate) topic_overlap: f64,
    #[serde(default)]
    pub(crate) shared_topics: Vec<String>,
    pub(crate) adjacent_pairs: usize,
    pub(crate) warmth: f64,
}

#[derive(Serialize, Deserialize, Clone, Default)]
pub(crate) struct PairInteraction {
    pub(crate) a: String,
    pub(crate) b: String,
    pub(crate) baseline: PairBlock,
    pub(crate) recent: PairBlock,
    pub(crate) edge_strength: f64,
}

// ─── evidence quote (verbatim msg snippet + msg_id) ───

#[derive(Serialize, Deserialize, Clone, Default, Debug)]
pub(crate) struct EvidenceQuote {
    pub(crate) quote: String,
    pub(crate) msg_id: String,
}

// ─── Phase 0 message classification (19-field rich) ───

#[derive(Serialize, Deserialize, Clone, Default)]
pub(crate) struct MsgClassification {
    pub(crate) msg_id: String,

    #[serde(default)]
    pub(crate) tags: Vec<String>,
    #[serde(default)]
    pub(crate) sentiment: String,
    #[serde(default)]
    pub(crate) self_claim: Option<String>,

    #[serde(default)]
    pub(crate) function: String,
    #[serde(default)]
    pub(crate) speech_act: String,
    #[serde(default)]
    pub(crate) modality: String,
    #[serde(default)]
    pub(crate) topic: String,
    #[serde(default)]
    pub(crate) addressee: String,

    #[serde(default)]
    pub(crate) is_self_statement: bool,
    #[serde(default)]
    pub(crate) is_quoting: bool,
    #[serde(default)]
    pub(crate) is_code_or_url: bool,
    #[serde(default)]
    pub(crate) is_low_info: bool,
    #[serde(default)]
    pub(crate) is_conditional: bool,
    #[serde(default)]
    pub(crate) is_meta_cognitive: bool,
    #[serde(default)]
    pub(crate) is_multi_perspective: bool,

    #[serde(default)]
    pub(crate) valence: i32,
    #[serde(default)]
    pub(crate) intensity: i32,
    #[serde(default)]
    pub(crate) tone: String,

    #[serde(default)]
    pub(crate) claim_dimension: String,
    #[serde(default)]
    pub(crate) claim_register: String,
    #[serde(default)]
    pub(crate) claim_certainty: String,

    #[serde(default)]
    pub(crate) implies: Option<String>,
}

impl MsgClassification {
    /// Project rich fields into the legacy `tags` array so older downstream code keeps working.
    pub(crate) fn derive_legacy_fields(&mut self) {
        if self.tags.is_empty() {
            let mut t: Vec<String> = Vec::new();
            if self.is_low_info {
                t.push("low-info".into());
            }
            if self.is_code_or_url {
                t.push("code-or-url".into());
            }
            if self.is_quoting || self.function == "quoted" {
                t.push("quote".into());
            }
            if self.function == "joke" {
                t.push("joke".into());
            }
            if self.function == "ironic" {
                t.push("ironic".into());
            }
            if self.is_self_statement {
                t.push("self-statement".into());
            }
            if t.is_empty() {
                t.push(if self.modality == "factual" {
                    "factual".into()
                } else {
                    "opinion".into()
                });
            }
            self.tags = t;
        }
        if self.sentiment.is_empty() {
            self.sentiment = match self.valence {
                v if v <= -1 => "neg".into(),
                v if v >= 1 => "pos".into(),
                _ => "neu".into(),
            };
        }
    }
}

// ─── Phase 1-4: observation / theme / falsification ───

#[derive(Serialize, Deserialize, Clone)]
pub(crate) struct Observation {
    pub(crate) trait_name: String,
    pub(crate) evidence: String,
    pub(crate) dimension: String,
    #[serde(default = "default_polarity")]
    pub(crate) polarity: String,
    #[serde(default)]
    pub(crate) support_ids: Vec<String>,
}

pub(crate) fn default_polarity() -> String {
    "exhibits".to_string()
}

#[derive(Serialize, Deserialize, Clone)]
pub(crate) struct FalsificationSpec {
    pub(crate) behavior: String,
    pub(crate) queries: Vec<String>,
}

#[derive(Serialize, Deserialize, Clone)]
pub(crate) struct DeepTheme {
    pub(crate) name: String,
    pub(crate) obs_indices: Vec<usize>,
    pub(crate) support_ids: Vec<String>,
    pub(crate) analysis: String,
    pub(crate) quotes: Vec<EvidenceQuote>,
    pub(crate) falsifications: Vec<FalsificationSpec>,
}

#[derive(Serialize, Deserialize, Clone)]
pub(crate) struct ValidatedTheme {
    pub(crate) name: String,
    pub(crate) count: usize,
    #[serde(default)]
    pub(crate) msg_count: usize,
    pub(crate) analysis: String,
    pub(crate) confidence: f64,
    #[serde(default)]
    pub(crate) support: Vec<EvidenceQuote>,
    #[serde(default)]
    pub(crate) falsifications_checked: usize,
    #[serde(default)]
    pub(crate) falsifications_confirmed: usize,
    #[serde(default)]
    pub(crate) falsification_summary: String,
    /// fraction of support messages that fall in the recent quartile.
    /// > 0.4 = "active" (trend-on), < 0.1 = "fading", in-between = stable.
    #[serde(default)]
    pub(crate) recent_share: f64,
    /// status string derived from recent_share: "active" | "stable" | "fading" | "unknown"
    #[serde(default)]
    pub(crate) temporal_status: String,
}

#[derive(Serialize, Deserialize, Clone, Default)]
pub(crate) struct Interpretation {
    pub(crate) identity: String,
    pub(crate) anxiety: String,
    pub(crate) social_style: String,
    pub(crate) growth: String,
    pub(crate) vulnerability: String,
    pub(crate) summary: String,
}

#[derive(Serialize, Deserialize, Clone, Default)]
pub(crate) struct CognitiveMarkers {
    pub(crate) abstract_rate: f64,
    pub(crate) conditional_rate: f64,
    pub(crate) integrative_complexity: f64,
    pub(crate) lexical_complexity: f64,
    pub(crate) domain_breadth: usize,
    pub(crate) self_monitoring: f64,
    pub(crate) z_abstract: f64,
    pub(crate) z_conditional: f64,
    pub(crate) z_integrative: f64,
    pub(crate) z_lexical: f64,
    pub(crate) z_breadth: f64,
    pub(crate) z_self_monitoring: f64,
    pub(crate) sample_size: usize,
}

#[derive(Serialize, Deserialize, Clone, Default)]
pub(crate) struct BigFiveSignals {
    pub(crate) openness: Vec<EvidenceQuote>,
    pub(crate) conscientiousness: Vec<EvidenceQuote>,
    pub(crate) extraversion: Vec<EvidenceQuote>,
    pub(crate) agreeableness: Vec<EvidenceQuote>,
    pub(crate) neuroticism: Vec<EvidenceQuote>,
}

#[derive(Serialize, Deserialize, Clone)]
pub(crate) struct SelfClaim {
    pub(crate) claim: String,
    pub(crate) msg_id: String,
    pub(crate) dimension: String,
    pub(crate) verdict: String,
    pub(crate) rationale: String,
    #[serde(default)]
    pub(crate) behavioral_evidence: Vec<EvidenceQuote>,
}

// ─── insight engine snapshot & feed types ───

#[derive(Serialize, Deserialize, Clone, Default)]
pub(crate) struct ProfileSnapshot {
    pub(crate) name: String,
    pub(crate) fingerprint: String,
    pub(crate) total_messages: usize,
    pub(crate) snapshot_at: String,
    pub(crate) themes: Vec<ThemeSnapshot>,
    pub(crate) cognitive: CognitiveMarkers,
    pub(crate) cognitive_recent: CognitiveMarkers,
    pub(crate) self_claims: Vec<SelfClaimSnapshot>,
}

#[derive(Serialize, Deserialize, Clone, Default)]
pub(crate) struct ThemeSnapshot {
    pub(crate) name: String,
    pub(crate) confidence: f64,
    pub(crate) msg_count: usize,
    pub(crate) recent_share: f64,
    pub(crate) temporal_status: String,
}

#[derive(Serialize, Deserialize, Clone, Default)]
pub(crate) struct SelfClaimSnapshot {
    pub(crate) claim: String,
    pub(crate) msg_id: String,
    pub(crate) dimension: String,
    pub(crate) verdict: String,
}

#[derive(Serialize, Deserialize, Clone)]
pub(crate) struct Insight {
    pub(crate) insight_type: String,
    pub(crate) person: String,
    pub(crate) summary: String,
    pub(crate) details: String,
    pub(crate) urgency: f64,
    pub(crate) evidence: Vec<EvidenceQuote>,
    pub(crate) first_seen: String,
    #[serde(default)]
    pub(crate) related_theme: Option<String>,
    #[serde(default)]
    pub(crate) related_claim: Option<String>,
}

#[derive(Serialize, Deserialize, Clone, Default)]
pub(crate) struct InsightFeed {
    pub(crate) insights: Vec<Insight>,
    pub(crate) last_run_at: String,
}

// ─── top-level profile (aggregate over all phases) ───

#[derive(Serialize, Deserialize, Clone)]
pub(crate) struct Profile {
    pub(crate) name: String,
    pub(crate) is_self: bool,
    pub(crate) total_messages: usize,
    pub(crate) behavioral: Behavioral,
    #[serde(default)]
    pub(crate) cognitive: CognitiveMarkers,
    /// cognitive markers computed over the recent quartile only — for trend awareness
    #[serde(default)]
    pub(crate) cognitive_recent: CognitiveMarkers,
    #[serde(default)]
    pub(crate) big_five: BigFiveSignals,
    /// Big Five signals computed over the recent quartile only
    #[serde(default)]
    pub(crate) big_five_recent: BigFiveSignals,
    pub(crate) themes: Vec<ValidatedTheme>,
    #[serde(default)]
    pub(crate) self_claims: Vec<SelfClaim>,
    pub(crate) interaction: Option<Interaction>,
    pub(crate) interpretation: Option<Interpretation>,
    pub(crate) predictions: Vec<Prediction>,
}
