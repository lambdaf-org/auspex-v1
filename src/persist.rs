//! Per-profile JSON cache: save/load under `profiles/<name>_<suffix>.json`.
//! Atomic via tmp-then-rename.

use serde::Serialize;

pub(crate) const PROFILE_DIR: &str = "profiles";

pub(crate) fn save_json<T: Serialize>(name: &str, suffix: &str, data: &T) -> std::io::Result<()> {
    std::fs::create_dir_all(PROFILE_DIR)?;
    let tmp = format!("{}/{}_{}.json.tmp", PROFILE_DIR, name, suffix);
    let dst = format!("{}/{}_{}.json", PROFILE_DIR, name, suffix);
    std::fs::write(&tmp, serde_json::to_string_pretty(data)?)?;
    std::fs::rename(&tmp, &dst)
}
pub(crate) fn load_json<T: serde::de::DeserializeOwned>(name: &str, suffix: &str) -> Option<T> {
    std::fs::read_to_string(format!("{}/{}_{}.json", PROFILE_DIR, name, suffix))
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
}
