//! Cross-platform positioned file I/O.
//!
//! The Q8_0 file-backed weight path reads exact byte ranges from a GGUF file at
//! explicit offsets, often from several rayon worker threads sharing one
//! `Arc<File>`. On Unix this is `pread(2)` via `FileExt::read_exact_at`, which
//! is positioned and never touches the file cursor, so concurrent positioned
//! reads on one handle are safe. Windows has no `pread`; the equivalent is
//! `ReadFile` with an explicit offset, exposed by `std::os::windows::fs::FileExt::seek_read`.
//! Each `seek_read` call carries its own offset, so concurrent reads return the
//! correct bytes regardless of the shared cursor — the cursor side effect is
//! irrelevant here because every caller passes an absolute offset and never
//! relies on sequential cursor position. The token-parity gate exercises this
//! path under rayon and would surface any divergence.

use std::fs::File;
use std::io::Result;

/// Read exactly `buf.len()` bytes from `file` starting at absolute byte
/// `offset`, without relying on the file cursor. Equivalent to `pread` on Unix.
#[cfg(unix)]
pub fn read_exact_at(file: &File, buf: &mut [u8], offset: u64) -> Result<()> {
    use std::os::unix::fs::FileExt;
    file.read_exact_at(buf, offset)
}

/// Read exactly `buf.len()` bytes from `file` starting at absolute byte
/// `offset`. Loops over `seek_read` (positioned `ReadFile`) until the buffer is
/// filled, matching `read_exact_at`'s "fill or error" contract.
#[cfg(windows)]
pub fn read_exact_at(file: &File, buf: &mut [u8], offset: u64) -> Result<()> {
    use std::io::{Error, ErrorKind};
    use std::os::windows::fs::FileExt;
    let mut filled = 0usize;
    while filled < buf.len() {
        match file.seek_read(&mut buf[filled..], offset + filled as u64) {
            Ok(0) => {
                return Err(Error::new(
                    ErrorKind::UnexpectedEof,
                    "failed to fill whole buffer",
                ));
            }
            Ok(n) => filled += n,
            Err(ref err) if err.kind() == ErrorKind::Interrupted => {}
            Err(err) => return Err(err),
        }
    }
    Ok(())
}
