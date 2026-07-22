mod reader;

pub use reader::{GgufFile, GgufMetadataValue, GgufTensorDescriptor, GgufTensorType};

use std::path::Path;

use crate::Result;

pub fn read_metadata(path: impl AsRef<Path>) -> Result<GgufFile> {
    reader::read_metadata(path.as_ref())
}

/// Parse a GGUF header prefix, validating tensor bounds against `declared_len`
/// instead of the on-disk length. See [`reader::read_metadata_with_len`].
pub fn read_metadata_with_len(path: impl AsRef<Path>, declared_len: u64) -> Result<GgufFile> {
    reader::read_metadata_with_len(path.as_ref(), declared_len)
}
