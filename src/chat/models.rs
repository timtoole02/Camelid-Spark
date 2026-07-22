//! The supported-model list the picker shows: the `/api/capabilities` ledger
//! (supported rows only) joined to the `pull` catalog for downloadability and
//! on-disk availability.
//!
//! The list is derived entirely at runtime — capabilities rows from the engine,
//! catalog rows from `curated_catalog()` — so it grows automatically as the
//! ledger promotes rows, with no edits here.

use std::path::{Path, PathBuf};

use camelid::api::{curated_catalog, CatalogItem};

use super::client::CompatRow;

/// Whether the row's GGUF can be fetched, and whether it is already present.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Availability {
    /// The catalog GGUF is on disk at full size — selectable immediately.
    Ready,
    /// Supported and in the pull catalog, but not yet downloaded.
    NotDownloaded,
    /// Supported but the catalog has no alias for it (can't auto-fetch).
    NoPullAlias,
}

/// One supported row, ready to render and act on. Support posture for the prompt
/// line is owned by the session (read from the ledger by id), so it is not
/// duplicated here.
pub struct PickerRow {
    pub id: String,
    pub quant: String,
    /// The catalog entry, when this supported row is also pullable.
    pub catalog: Option<CatalogItem>,
    pub availability: Availability,
}

impl PickerRow {
    /// Absolute path the GGUF would live at, when this row is in the catalog.
    pub fn local_path(&self, models_dir: &Path) -> Option<PathBuf> {
        self.catalog
            .as_ref()
            .map(|item| models_dir.join(item.filename))
    }
}

/// Build the picker's supported rows from the ledger, joined to the catalog and
/// annotated with on-disk availability under `models_dir`.
pub fn supported_rows(ledger: &[CompatRow], models_dir: &Path) -> Vec<PickerRow> {
    let catalog = curated_catalog();
    ledger
        .iter()
        .filter(|row| row.is_supported())
        .map(|row| {
            let item = catalog
                .iter()
                .find(|item| item.catalog_id == row.id)
                .cloned();
            let availability = match &item {
                Some(item) if is_downloaded(item, models_dir) => Availability::Ready,
                Some(_) => Availability::NotDownloaded,
                None => Availability::NoPullAlias,
            };
            PickerRow {
                id: row.id.clone(),
                quant: row.quantization.clone(),
                catalog: item,
                availability,
            }
        })
        .collect()
}

/// True when the catalog GGUF exists at its full expected size.
fn is_downloaded(item: &CatalogItem, models_dir: &Path) -> bool {
    std::fs::metadata(models_dir.join(item.filename))
        .map(|meta| meta.len() == item.size_bytes)
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(id: &str, status: &str) -> CompatRow {
        CompatRow {
            id: id.into(),
            family: "fam".into(),
            quantization: "Q8_0".into(),
            status: status.into(),
            tool_capable: false,
        }
    }

    #[test]
    fn picker_is_ledger_derived_supported_only_and_joins_catalog() {
        // A ledger with a catalogued supported row, a supported row with no
        // catalog alias, and a planned row that must be excluded.
        let ledger = vec![
            row("tinyllama_1_1b_chat_q8_0", "supported_exact_row_smoke"),
            row(
                "mistral_instruct_exact_7b_v0_3_q8_0",
                "supported_exact_row_smoke_lane",
            ),
            row("qwen2_5_7b_instruct_q8_0", "planned"),
        ];
        let rows = supported_rows(&ledger, Path::new("/nonexistent-models-dir"));

        // Planned row dropped; both supported rows present.
        assert_eq!(rows.len(), 2);
        assert!(rows.iter().all(|r| r.id != "qwen2_5_7b_instruct_q8_0"));

        // The catalogued row joins (catalog Some) but is not on disk here.
        let tiny = rows
            .iter()
            .find(|r| r.id == "tinyllama_1_1b_chat_q8_0")
            .unwrap();
        assert!(tiny.catalog.is_some());
        assert_eq!(tiny.availability, Availability::NotDownloaded);

        // The supported row absent from the catalog is flagged NoPullAlias.
        let mistral = rows
            .iter()
            .find(|r| r.id == "mistral_instruct_exact_7b_v0_3_q8_0")
            .unwrap();
        assert!(mistral.catalog.is_none());
        assert_eq!(mistral.availability, Availability::NoPullAlias);
    }
}
