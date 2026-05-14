//! Vector arithmetic + k-means clustering. No external dependencies.

// ─── vector math + k-means ───

pub(crate) fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}
pub(crate) fn vec_mean(vecs: &[&Vec<f32>]) -> Vec<f32> {
    let dim = vecs[0].len();
    let mut out = vec![0.0f32; dim];
    for v in vecs {
        for (i, val) in v.iter().enumerate() {
            out[i] += val;
        }
    }
    let n = vecs.len() as f32;
    out.iter_mut().for_each(|x| *x /= n);
    out
}
pub(crate) fn kmeans(embeddings: &[Vec<f32>], k: usize, iters: usize) -> Vec<usize> {
    let n = embeddings.len();
    if n <= k {
        return (0..n).collect();
    }
    let mut centroids: Vec<Vec<f32>> = (0..k).map(|i| embeddings[i * n / k].clone()).collect();
    let mut assign = vec![0usize; n];
    for _ in 0..iters {
        for (i, emb) in embeddings.iter().enumerate() {
            assign[i] = centroids
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| {
                    cosine_sim(emb, a).partial_cmp(&cosine_sim(emb, b)).unwrap()
                })
                .unwrap()
                .0;
        }
        for j in 0..k {
            let members: Vec<&Vec<f32>> = embeddings
                .iter()
                .enumerate()
                .filter(|(i, _)| assign[*i] == j)
                .map(|(_, e)| e)
                .collect();
            if !members.is_empty() {
                centroids[j] = vec_mean(&members);
            }
        }
    }
    assign
}
