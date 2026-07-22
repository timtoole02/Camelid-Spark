//! A Windows kill-on-close Job Object for process-tree teardown.
//!
//! Assign a freshly spawned child to the job; terminating the job (or dropping
//! its last handle) kills the child AND every descendant it spawned. Used by
//! `run_windows_command` so a timeout cannot leave orphaned PowerShell — and,
//! in a later phase, subagent — processes running. `std::process::Child::kill`
//! alone only reaps the direct child, not its tree.

use std::os::windows::io::RawHandle;

use windows_sys::Win32::Foundation::{CloseHandle, HANDLE};
use windows_sys::Win32::System::JobObjects::{
    AssignProcessToJobObject, CreateJobObjectW, JobObjectExtendedLimitInformation,
    SetInformationJobObject, TerminateJobObject, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
};

/// An owned job-object handle. Created with `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`,
/// so closing it (Drop) terminates any process still in the job.
pub struct JobObject {
    handle: HANDLE,
}

// SAFETY: a Windows job-object handle is a process-wide kernel handle, not tied
// to the thread that created it; create/assign/terminate/close are all valid from
// any thread. This lets a JobObject be held in the process-global subagent
// registry (a `Mutex`-guarded static). It is never used concurrently without that
// lock.
unsafe impl Send for JobObject {}

impl JobObject {
    /// Create a new unnamed kill-on-close job object.
    pub fn new() -> std::io::Result<Self> {
        // SAFETY: null security attributes + null name create a fresh unnamed job
        // and return a null handle on failure.
        let handle = unsafe { CreateJobObjectW(std::ptr::null(), std::ptr::null()) };
        if handle.is_null() {
            return Err(std::io::Error::last_os_error());
        }
        let job = JobObject { handle };

        // SAFETY: the struct is plain-old-data (integers/handles); all-zero is a
        // valid initial state, and we set the one field we rely on below.
        let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = unsafe { std::mem::zeroed() };
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        // SAFETY: `info` is a correctly sized, zero-initialized struct of the type
        // the ExtendedLimitInformation class expects; we pass its true byte length.
        let ok = unsafe {
            SetInformationJobObject(
                job.handle,
                JobObjectExtendedLimitInformation,
                std::ptr::addr_of!(info) as *const core::ffi::c_void,
                std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
            )
        };
        if ok == 0 {
            return Err(std::io::Error::last_os_error());
        }
        Ok(job)
    }

    /// Assign a spawned child process to the job. Descendants the child spawns
    /// after assignment are captured by the job too.
    pub fn assign(&self, process: RawHandle) -> std::io::Result<()> {
        // SAFETY: `process` is a live process handle owned by the caller's
        // `std::process::Child` for the duration of this call.
        let ok = unsafe { AssignProcessToJobObject(self.handle, process as HANDLE) };
        if ok == 0 {
            Err(std::io::Error::last_os_error())
        } else {
            Ok(())
        }
    }

    /// Immediately terminate every process in the job.
    pub fn terminate(&self) {
        // SAFETY: terminating a valid, owned job handle; the exit code is arbitrary.
        unsafe {
            TerminateJobObject(self.handle, 1);
        }
    }
}

impl Drop for JobObject {
    fn drop(&mut self) {
        // SAFETY: `handle` came from CreateJobObjectW and has not been closed.
        // Closing the last handle on a kill-on-close job terminates survivors.
        unsafe {
            CloseHandle(self.handle);
        }
    }
}
