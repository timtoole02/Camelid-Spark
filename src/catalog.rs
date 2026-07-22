//! `camelid pull` — fetch a supported model so the user never has to hunt for a
//! compatible GGUF by hand.
//!
//! Camelid only serves specific Q8_0 (and one Q4_0 QAT) rows, and most GGUFs on
//! the web are other quantizations that fail closed. This command downloads one
//! of the known-good rows from the same curated catalog the web UI uses, into
//! `./models`, and prints the exact `camelid serve` command to run next.

use std::path::{Path, PathBuf};

use crate::api::{curated_catalog, CatalogItem};

/// Entry point for the `Pull` subcommand. With no `query`, prints the catalog;
/// otherwise resolves `query` to exactly one row and downloads it into
/// `models_dir`.
pub fn run_pull(query: Option<&str>, models_dir: &Path) -> anyhow::Result<()> {
    let entries = curated_catalog();

    let Some(query) = query else {
        print_catalog(&entries);
        eprintln!("\nDownload one with:  camelid pull <id>   (e.g. camelid pull llama32_3b)");
        return Ok(());
    };

    let item = resolve(&entries, query)?;
    let dest = download(&item, models_dir)?;

    eprintln!("\n✓ {} is ready at {}", item.name, dest.display());
    eprintln!(
        "\nStart chatting:\n  camelid serve --model {}",
        dest.display()
    );
    Ok(())
}

/// Match `query` against the catalog by id or name, ignoring case and the
/// `-`/`_`/`.`/space separators people mix up. Errors helpfully on no match or
/// an ambiguous match rather than guessing.
fn resolve(entries: &[CatalogItem], query: &str) -> anyhow::Result<CatalogItem> {
    let needle = normalize(query);
    let matches: Vec<&CatalogItem> = entries
        .iter()
        .filter(|item| {
            normalize(item.catalog_id).contains(&needle) || normalize(item.name).contains(&needle)
        })
        .collect();

    match matches.as_slice() {
        [] => {
            print_catalog(entries);
            anyhow::bail!(
                "no supported model matches \"{query}\" — pick an id from the list above"
            );
        }
        [only] => Ok((*only).clone()),
        many => {
            let ids: Vec<&str> = many.iter().map(|item| item.catalog_id).collect();
            anyhow::bail!(
                "\"{query}\" matches several models ({}); be more specific",
                ids.join(", ")
            );
        }
    }
}

/// Ask the Hugging Face Hub for the current byte size of `item`'s GGUF.
///
/// The catalog ships a `size_bytes` constant, but uploaders occasionally
/// re-publish a row (a re-quant, a metadata fix) and the byte count shifts.
/// Gating "is this download complete?" on a baked-in constant then breaks: a
/// fully-downloaded file stops matching, so `pull` tries to resume a file that
/// is already whole. Querying the Hub's file tree keeps the check honest.
///
/// Returns `None` when offline or the response can't be parsed; callers then
/// fall back to the catalog constant.
fn remote_size(item: &CatalogItem) -> Option<u64> {
    let url = format!(
        "https://huggingface.co/api/models/{}/tree/main?recursive=1",
        item.repo_id
    );
    let output = std::process::Command::new("curl")
        .args(["-fsSL", &url])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let tree: serde_json::Value = serde_json::from_slice(&output.stdout).ok()?;
    for entry in tree.as_array()? {
        if entry.get("path").and_then(|p| p.as_str()) == Some(item.filename) {
            // LFS/xet-backed files report the real content size under `lfs.size`;
            // the top-level `size` for those is just the pointer's byte count.
            return entry
                .get("lfs")
                .and_then(|lfs| lfs.get("size"))
                .and_then(|s| s.as_u64())
                .or_else(|| entry.get("size").and_then(|s| s.as_u64()));
        }
    }
    None
}

/// Download `item` into `models_dir` via `curl` (resumable, streaming progress
/// to the terminal). Skips a complete copy, resumes a partial one, and re-fetches
/// a stale/oversized one — judged against the Hub's *current* file size, not a
/// baked-in constant, so a re-published row can't trick `pull` into resuming a
/// file that is already whole.
fn download(item: &CatalogItem, models_dir: &Path) -> anyhow::Result<PathBuf> {
    std::fs::create_dir_all(models_dir)?;
    let dest = models_dir.join(item.filename);

    // FLINT: split models (the Hub only ships some large rows as gguf-split
    // shards). Fetch every part, then merge into the single `filename`.
    if !item.parts.is_empty() {
        return download_multipart(item, models_dir, &dest);
    }

    // Authoritative size from the Hub; fall back to the catalog constant offline.
    let expected = remote_size(item);
    let target = expected.unwrap_or(item.size_bytes);

    if let Ok(meta) = std::fs::metadata(&dest) {
        let have = meta.len();
        if have == target {
            eprintln!(
                "{} already downloaded at {} ({:.1} GB, size-verified)",
                item.name,
                dest.display(),
                target as f64 / 1e9
            );
            return Ok(dest);
        }
        if expected.is_some() && have > target {
            // Larger than the Hub's current file: a stale or corrupt copy. A
            // byte-range resume can't repair that, so start clean.
            eprintln!(
                "Local {} is {have} bytes but the Hub file is {target} — re-downloading fresh",
                item.filename
            );
            let _ = std::fs::remove_file(&dest);
        }
        // Otherwise the file is shorter than expected: a partial download that
        // `curl -C -` resumes below.
    }

    let url = format!(
        "https://huggingface.co/{}/resolve/main/{}",
        item.repo_id, item.filename
    );
    eprintln!(
        "Downloading {} ({:.1} GB) from {}",
        item.name,
        target as f64 / 1e9,
        item.repo_id
    );

    // -L follow redirects, -C - resume a partial file, --fail surface HTTP
    // errors as a non-zero exit. curl writes its own progress bar to stderr.
    let status = std::process::Command::new("curl")
        .args(["-L", "-C", "-", "--fail", "-o"])
        .arg(&dest)
        .arg(&url)
        .status()
        .map_err(|err| anyhow::anyhow!("could not run curl (is it installed?): {err}"))?;

    let have_after = std::fs::metadata(&dest).map(|m| m.len()).unwrap_or(0);

    if !status.success() {
        // curl returns non-zero (HTTP 416) when asked to resume a file that is
        // already complete. If every byte is here, that's success, not failure;
        // otherwise the download genuinely broke.
        if have_after == 0 || have_after != target {
            anyhow::bail!("download failed (curl exited with {status}); re-run to resume");
        }
    }

    // Final integrity gate: never hand back a short or oversized file when the
    // Hub told us the size to expect.
    if expected.is_some() && have_after != target {
        anyhow::bail!(
            "download incomplete: {} is {have_after} bytes, expected {target} — re-run to resume",
            item.filename
        );
    }

    Ok(dest)
}

/// Multi-part variant of [`download`]: fetch each gguf-split shard (resumable,
/// exact-size gated against the catalog's per-part byte counts), merge them
/// into the single `filename` with `crate::gguf::merge`, then delete the parts.
/// A pre-existing merged file that parses as GGUF is treated as complete.
fn download_multipart(
    item: &CatalogItem,
    models_dir: &Path,
    dest: &Path,
) -> anyhow::Result<PathBuf> {
    if let Ok(meta) = std::fs::metadata(dest) {
        if meta.len() > 0 && crate::gguf::read_metadata(dest).is_ok() {
            eprintln!(
                "{} already downloaded and parseable at {} ({:.1} GB)",
                item.name,
                dest.display(),
                meta.len() as f64 / 1e9
            );
            // A crash between a prior run's merge-rename and its part cleanup
            // orphans tens of GB of .shardNN files no listing ever shows;
            // sweep them now that the merged file is proven present.
            for i in 1..=item.parts.len() {
                let leftover = models_dir.join(format!("{}.shard{i:02}", item.filename));
                if std::fs::remove_file(&leftover).is_ok() {
                    eprintln!("  removed leftover part {}", leftover.display());
                }
            }
            return Ok(dest.to_path_buf());
        }
    }

    let total: u64 = item.parts.iter().map(|p| p.size_bytes).sum();
    eprintln!(
        "Downloading {} in {} parts ({:.1} GB total) from {}",
        item.name,
        item.parts.len(),
        total as f64 / 1e9,
        item.repo_id
    );

    let mut shard_paths = Vec::with_capacity(item.parts.len());
    for (i, part) in item.parts.iter().enumerate() {
        // Deliberately NOT a `.gguf` name: the models scan lists `*.gguf`, and
        // a half-downloaded shard must never show up as a (broken) model. The
        // merger takes the ordered path list directly, so names don't matter.
        let shard_dest = models_dir.join(format!("{}.shard{:02}", item.filename, i + 1));
        if std::fs::metadata(&shard_dest).map(|m| m.len()).ok() == Some(part.size_bytes) {
            eprintln!(
                "  part {}/{}: already complete ({})",
                i + 1,
                item.parts.len(),
                shard_dest.display()
            );
            shard_paths.push(shard_dest);
            continue;
        }
        let url = format!(
            "https://huggingface.co/{}/resolve/main/{}",
            item.repo_id, part.remote_path
        );
        eprintln!(
            "  part {}/{}: {} ({:.1} GB)",
            i + 1,
            item.parts.len(),
            part.remote_path,
            part.size_bytes as f64 / 1e9
        );
        let status = std::process::Command::new("curl")
            .args(["-L", "-C", "-", "--fail", "-o"])
            .arg(&shard_dest)
            .arg(&url)
            .status()
            .map_err(|err| anyhow::anyhow!("could not run curl (is it installed?): {err}"))?;
        let have = std::fs::metadata(&shard_dest).map(|m| m.len()).unwrap_or(0);
        // Exact-size gate: passes a resume-of-complete (curl 416), fails any
        // truncated or size-shifted part regardless of exit code.
        if have != part.size_bytes {
            if have > part.size_bytes {
                // A range resume can never SHRINK a file — an oversized shard
                // would wedge every retry forever. Start that part clean.
                let _ = std::fs::remove_file(&shard_dest);
                anyhow::bail!(
                    "part {} was {have} bytes but the catalog pins {} — removed {}; re-run to re-download it",
                    i + 1,
                    part.size_bytes,
                    shard_dest.display()
                );
            }
            if !status.success() {
                anyhow::bail!(
                    "part {} download failed (curl exited with {status}); re-run to resume",
                    i + 1
                );
            }
            anyhow::bail!(
                "part {} is {have} bytes, expected {} — re-run to resume",
                i + 1,
                part.size_bytes
            );
        }
        shard_paths.push(shard_dest);
    }

    eprintln!(
        "Merging {} parts into {} …",
        item.parts.len(),
        dest.display()
    );
    let tmp = dest.with_extension("gguf.merge-tmp");
    match crate::gguf::merge::merge_shards(&shard_paths, &tmp) {
        Ok(report) => {
            std::fs::rename(&tmp, dest)?;
            for p in &shard_paths {
                let _ = std::fs::remove_file(p);
            }
            eprintln!(
                "Merged {} tensors ({} split keys dropped); parts deleted.",
                report.tensors, report.kv_dropped
            );
            Ok(dest.to_path_buf())
        }
        Err(e) => {
            let _ = std::fs::remove_file(&tmp);
            anyhow::bail!("shard merge failed: {e} — parts kept; re-run to retry");
        }
    }
}

fn print_catalog(entries: &[CatalogItem]) {
    eprintln!("Supported models (download into ./models):\n");
    // Annotate each row with a capacity verdict for THIS host (fit axis only — not
    // a support claim). Probed once, reused across rows.
    let hw = crate::capability::HardwareProfile::cached();
    eprintln!(
        "  {:<28} {:<8} {:>8}  {:<15} NAME",
        "ID", "QUANT", "SIZE", "FIT (this host)"
    );
    for item in entries {
        let verdict = crate::fit::assess(hw, &crate::fit::advisory_footprint(item.size_bytes));
        eprintln!(
            "  {:<28} {:<8} {:>6.1} GB  {:<15} {}",
            item.catalog_id,
            item.quant,
            item.size_bytes as f64 / 1e9,
            verdict.cli_label(),
            item.name,
        );
    }
}

fn normalize(value: &str) -> String {
    value
        .chars()
        .filter(|c| !matches!(c, '-' | '_' | '.' | ' '))
        .flat_map(char::to_lowercase)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_by_id_fragment_ignoring_separators() {
        let entries = curated_catalog();
        let item = resolve(&entries, "llama32-3b").unwrap();
        assert_eq!(item.catalog_id, "llama32_3b_instruct_q8_0");
    }

    #[test]
    fn resolves_by_name_fragment() {
        let entries = curated_catalog();
        let item = resolve(&entries, "tinyllama").unwrap();
        assert!(item.catalog_id.contains("tinyllama"));
    }

    #[test]
    fn unknown_query_is_an_error() {
        let entries = curated_catalog();
        assert!(resolve(&entries, "gpt-9-turbo").is_err());
    }
}
