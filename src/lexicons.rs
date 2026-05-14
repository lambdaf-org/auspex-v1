//! Runtime lexicon loader.
//!
//! Word lists and intent phrases live on disk under `./lexicons/`, NOT in source.
//! Layout:
//!     lexicons/<category>/<lang>.txt              # word lists, one entry per line
//!     lexicons/intents/<intent>/<lang>.txt         # intent phrases, one per line
//!
//! Each file is plain text. Blank lines and lines starting with `#` are ignored.
//! All matched entries from all language files for a given category are merged
//! into a single case-folded set / list. The directory is git-ignored — these
//! are local translations, treated like an `.env`.
//!
//! If `lexicons/` is missing or empty the loader returns empty sets; downstream
//! metrics gracefully zero out rather than crash.

use std::collections::HashSet;
use std::sync::OnceLock;

const LEXICON_DIR: &str = "lexicons";

/// Parse a lexicon file: one entry per line, lowercased, blank/comment lines ignored.
fn parse_file(path: &std::path::Path) -> Vec<String> {
    let mut out = Vec::new();
    let Ok(content) = std::fs::read_to_string(path) else {
        return out;
    };
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        out.push(trimmed.to_lowercase());
    }
    out
}

/// Load all `<category>/*.txt` files under `lexicons/`, merging into a single set.
fn load_word_set(category: &str) -> HashSet<String> {
    let mut out = HashSet::new();
    let dir = std::path::Path::new(LEXICON_DIR).join(category);
    let Ok(entries) = std::fs::read_dir(&dir) else {
        eprintln!(
            "    lexicon: {:?} missing — {} returns empty set",
            dir, category
        );
        return out;
    };
    let mut langs: Vec<String> = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("txt") {
            continue;
        }
        let lang = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("?")
            .to_string();
        let words = parse_file(&path);
        if !words.is_empty() {
            langs.push(lang);
            for w in words {
                out.insert(w);
            }
        }
    }
    if !langs.is_empty() {
        eprintln!(
            "    lexicon: {} loaded ({} entries from langs: {})",
            category,
            out.len(),
            langs.join(", ")
        );
    }
    out
}

macro_rules! cached_set {
    ($fn_name:ident, $category:literal) => {
        pub(crate) fn $fn_name() -> &'static HashSet<String> {
            static CELL: OnceLock<HashSet<String>> = OnceLock::new();
            CELL.get_or_init(|| load_word_set($category))
        }
    };
}

cached_set!(self_ref_words, "self_ref");
cached_set!(other_ref_words, "other_ref");
cached_set!(certainty_words, "certainty");
cached_set!(hedging_words, "hedging");
cached_set!(positive_words, "positive");
cached_set!(negative_words, "negative");
cached_set!(analytical_words, "analytical");

/// Intent patterns: `Vec<(intent_name, phrases)>`. Each intent gets its own
/// subdirectory with per-language files.
fn load_intent_patterns() -> Vec<(String, Vec<String>)> {
    let mut out: Vec<(String, Vec<String>)> = Vec::new();
    let dir = std::path::Path::new(LEXICON_DIR).join("intents");
    let Ok(entries) = std::fs::read_dir(&dir) else {
        eprintln!("    lexicon: {:?} missing — intent_patterns is empty", dir);
        return out;
    };
    let mut intents: Vec<std::path::PathBuf> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .collect();
    intents.sort();
    for intent_dir in intents {
        let intent_name = match intent_dir.file_name().and_then(|s| s.to_str()) {
            Some(n) => n.to_string(),
            None => continue,
        };
        let mut phrases: Vec<String> = Vec::new();
        if let Ok(lang_entries) = std::fs::read_dir(&intent_dir) {
            for lang_entry in lang_entries.flatten() {
                let path = lang_entry.path();
                if path.extension().and_then(|s| s.to_str()) != Some("txt") {
                    continue;
                }
                phrases.extend(parse_file(&path));
            }
        }
        if !phrases.is_empty() {
            out.push((intent_name, phrases));
        }
    }
    if !out.is_empty() {
        eprintln!(
            "    lexicon: intent_patterns loaded ({} intents, {} phrases total)",
            out.len(),
            out.iter().map(|(_, p)| p.len()).sum::<usize>()
        );
    }
    out
}

pub(crate) fn intent_patterns() -> &'static Vec<(String, Vec<String>)> {
    static CELL: OnceLock<Vec<(String, Vec<String>)>> = OnceLock::new();
    CELL.get_or_init(load_intent_patterns)
}
