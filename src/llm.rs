//! Local Ollama wrappers: JSON-mode + plain text.

// ─── llm helper ───

pub(crate) fn llm_json(model: &str, prompt: &str) -> Option<serde_json::Value> {
    let body = serde_json::json!({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": false, "format": "json",
        "options": {"num_ctx": 32768, "temperature": 0.3}
    });
    let resp = ureq::post("http://localhost:11434/api/chat")
        .timeout(std::time::Duration::from_secs(600))
        .set("Content-Type", "application/json")
        .send_string(&body.to_string())
        .ok()?;
    let text = resp.into_string().ok()?;
    let json: serde_json::Value = serde_json::from_str(&text).ok()?;
    let content = json["message"]["content"].as_str()?;

    let mut clean = content.trim().to_string();
    // strip markdown fences
    if let Some(start) = clean.find("```") {
        if let Some(end) = clean[start + 3..].find("```") {
            clean = clean[start + 3..start + 3 + end].trim().to_string();
            if clean.starts_with("json") {
                clean = clean[4..].trim().to_string();
            }
        }
    }
    // strip thinking tags
    if let Some(pos) = clean.find("</think>") {
        clean = clean[pos + 8..].trim().to_string();
    }
    // find first JSON start
    if let Some(pos) = clean.find('{') {
        if pos > 0 {
            clean = clean[pos..].to_string();
        }
    } else if let Some(pos) = clean.find('[') {
        if pos > 0 {
            clean = clean[pos..].to_string();
        }
    }

    if let Ok(v) = serde_json::from_str::<serde_json::Value>(&clean) {
        return Some(v);
    }

    eprintln!("    JSON parse failed: {}", &clean[..clean.len().min(150)]);
    None
}

pub(crate) fn llm_text(model: &str, system: &str, user: &str) -> Option<String> {
    let body = serde_json::json!({
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        "stream": false,
        "options": {"num_ctx": 4096}
    });
    let resp = ureq::post("http://localhost:11434/api/chat")
        .timeout(std::time::Duration::from_secs(600))
        .set("Content-Type", "application/json")
        .send_string(&body.to_string())
        .ok()?;
    let text = resp.into_string().ok()?;
    let json: serde_json::Value = serde_json::from_str(&text).ok()?;
    json["message"]["content"].as_str().map(|s| s.to_string())
}

