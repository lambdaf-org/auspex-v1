//! Chat-log parsing: extracts `RawMessage`s from one-line-per-message text files,
//! resolves sender aliases, and computes a stable `msg_id` per message.

use crate::config::Config;
use crate::types::RawMessage;
use std::collections::HashMap;

pub(crate) fn compute_msg_id(sender: &str, timestamp: Option<&str>, text: &str) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut h = DefaultHasher::new();
    sender.hash(&mut h);
    timestamp.unwrap_or("").hash(&mut h);
    text.hash(&mut h);
    format!("{:016x}", h.finish())
}

pub(crate) fn parse_files(paths: &[String], config: &Config) -> Vec<RawMessage> {
    let patterns = [regex::Regex::new(
        r"^(?P<date>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2})\s*\|\s*(?P<name>[^|]+?)\s*\|\s*(?P<msg>.+)$",
    )
    .unwrap()];
    let alias_map = build_alias_map(config);
    let mut messages = Vec::new();
    for path in paths {
        let content = match std::fs::read_to_string(path) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("skip {}: {}", path, e);
                continue;
            }
        };
        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            for pat in &patterns {
                if let Some(caps) = pat.captures(line) {
                    let name = caps.name("name").unwrap().as_str().trim();
                    let msg = caps.name("msg").unwrap().as_str().trim();
                    let date = caps.name("date").map(|d| d.as_str().to_string());
                    if msg.is_empty() {
                        break;
                    }
                    let resolved = resolve_name(name, &alias_map);
                    let id = compute_msg_id(&resolved, date.as_deref(), msg);
                    messages.push(RawMessage {
                        id,
                        sender: resolved,
                        text: msg.to_string(),
                        timestamp: date,
                    });
                    break;
                }
            }
        }
    }
    messages
}

pub(crate) fn build_alias_map(config: &Config) -> HashMap<String, String> {
    let mut map = HashMap::new();
    for h in &config.self_handles {
        map.insert(h.to_lowercase(), config.self_name.clone());
    }
    for (canonical, handles) in &config.aliases {
        for h in handles {
            map.insert(h.to_lowercase(), canonical.clone());
        }
    }
    map
}

pub(crate) fn resolve_name(raw: &str, alias_map: &HashMap<String, String>) -> String {
    alias_map
        .get(&raw.to_lowercase())
        .cloned()
        .unwrap_or_else(|| raw.to_string())
}
