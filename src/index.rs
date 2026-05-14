//! On-disk embedding index + metadata.
//!
//! [index/embeddings.bin] = packed f32 vectors, [index/meta.json] = sender+text+id+timestamp.
//! Backfills msg_id for old metadata files lacking the field.

use crate::math::cosine_sim;
use crate::parse::compute_msg_id;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::io::Write;

// ─── message index ───

pub(crate) const INDEX_DIR: &str = "index";

#[derive(Serialize, Deserialize)]
pub(crate) struct IndexMeta {
    pub(crate) entries: Vec<IndexEntry>,
}
#[derive(Serialize, Deserialize, Clone)]
pub(crate) struct IndexEntry {
    #[serde(default)]
    pub(crate) id: String,
    pub(crate) sender: String,
    pub(crate) text: String,
    pub(crate) timestamp: Option<String>,
}

pub(crate) struct MessageIndex {
    pub(crate) entries: Vec<IndexEntry>,
    pub(crate) embeddings: Vec<Vec<f32>>,
}

impl MessageIndex {
    pub(crate) fn load_or_new() -> Self {
        let mut meta: Vec<IndexEntry> =
            std::fs::read_to_string(format!("{}/meta.json", INDEX_DIR))
                .ok()
                .and_then(|s| serde_json::from_str::<IndexMeta>(&s).ok())
                .map(|m| m.entries)
                .unwrap_or_default();
        let embeddings =
            load_embeddings_bin(&format!("{}/embeddings.bin", INDEX_DIR)).unwrap_or_default();
        if meta.len() != embeddings.len() {
            return Self {
                entries: Vec::new(),
                embeddings: Vec::new(),
            };
        }
        // backfill ids for entries from older index versions
        for e in meta.iter_mut() {
            if e.id.is_empty() {
                e.id = compute_msg_id(&e.sender, e.timestamp.as_deref(), &e.text);
            }
        }
        Self {
            entries: meta,
            embeddings,
        }
    }
    pub(crate) fn existing_ids(&self) -> HashSet<String> {
        self.entries.iter().map(|e| e.id.clone()).collect()
    }
    pub(crate) fn append(&mut self, entries: Vec<IndexEntry>, embeddings: Vec<Vec<f32>>) {
        self.entries.extend(entries);
        self.embeddings.extend(embeddings);
    }
    pub(crate) fn save(&self) -> std::io::Result<()> {
        std::fs::create_dir_all(INDEX_DIR)?;
        std::fs::write(
            format!("{}/meta.json", INDEX_DIR),
            serde_json::to_string(&IndexMeta {
                entries: self.entries.clone(),
            })?,
        )?;
        save_embeddings_bin(&self.embeddings, &format!("{}/embeddings.bin", INDEX_DIR))
    }
    pub(crate) fn search_for(
        &self,
        query_emb: &[f32],
        person: Option<&str>,
        top_k: usize,
    ) -> Vec<(f32, &IndexEntry)> {
        let mut scored: Vec<(f32, &IndexEntry)> = self
            .entries
            .iter()
            .enumerate()
            .filter(|(_, e)| person.map_or(true, |p| e.sender == p))
            .filter(|(_, e)| e.text.len() >= 20)
            .map(|(i, e)| (cosine_sim(query_emb, &self.embeddings[i]), e))
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        scored.truncate(top_k);
        scored
    }
}

pub(crate) fn save_embeddings_bin(embeddings: &[Vec<f32>], path: &str) -> std::io::Result<()> {
    let mut f = std::fs::File::create(path)?;
    let n = embeddings.len() as u32;
    let dim = if n > 0 { embeddings[0].len() as u32 } else { 0 };
    f.write_all(&n.to_le_bytes())?;
    f.write_all(&dim.to_le_bytes())?;
    for emb in embeddings {
        for &val in emb {
            f.write_all(&val.to_le_bytes())?;
        }
    }
    Ok(())
}
pub(crate) fn load_embeddings_bin(path: &str) -> std::io::Result<Vec<Vec<f32>>> {
    let data = std::fs::read(path)?;
    if data.len() < 8 {
        return Ok(Vec::new());
    }
    let n = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
    let dim = u32::from_le_bytes([data[4], data[5], data[6], data[7]]) as usize;
    let mut embeddings = Vec::with_capacity(n);
    let mut off = 8;
    for _ in 0..n {
        let mut emb = Vec::with_capacity(dim);
        for _ in 0..dim {
            if off + 4 > data.len() {
                break;
            }
            emb.push(f32::from_le_bytes([
                data[off],
                data[off + 1],
                data[off + 2],
                data[off + 3],
            ]));
            off += 4;
        }
        embeddings.push(emb);
    }
    Ok(embeddings)
}
