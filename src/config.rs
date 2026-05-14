//! Identity / alias config loaded from `config.json` in the cwd.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Serialize, Deserialize)]
pub(crate) struct Config {
    pub(crate) self_name: String,
    pub(crate) self_handles: Vec<String>,
    pub(crate) aliases: HashMap<String, Vec<String>>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            self_name: "you".into(),
            self_handles: vec!["you".into(), "me".into(), "Me".into(), "You".into()],
            aliases: HashMap::new(),
        }
    }
}

pub(crate) fn load_config() -> Config {
    std::fs::read_to_string("config.json")
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_else(|| {
            let cfg = Config::default();
            let _ = std::fs::write("config.json", serde_json::to_string_pretty(&cfg).unwrap());
            eprintln!("wrote default config.json");
            cfg
        })
}
