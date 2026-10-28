//! In-process Kapitsa. Requests retain the native engine and are one-shot.
//! Pass UTF-8 JSON using the schema documented in sdk/README.md.
use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::ptr::{self, NonNull};
use std::sync::Arc;

#[repr(C)]
struct NativeEngine {
    _private: [u8; 0],
}
#[repr(C)]
struct NativeRequest {
    _private: [u8; 0],
}
extern "C" {
    fn kapitsa_abi_version() -> u32;
    fn kapitsa_engine_create(
        json: *const c_char,
        out: *mut *mut NativeEngine,
        error: *mut *mut c_char,
    ) -> c_int;
    fn kapitsa_engine_destroy(engine: *mut NativeEngine);
    fn kapitsa_request_create(
        engine: *mut NativeEngine,
        json: *const c_char,
        out: *mut *mut NativeRequest,
        error: *mut *mut c_char,
    ) -> c_int;
    fn kapitsa_generate(
        request: *mut NativeRequest,
        callback: Option<unsafe extern "C" fn(*const c_char, usize, *mut c_void) -> c_int>,
        user: *mut c_void,
        result: *mut *mut c_char,
        error: *mut *mut c_char,
    ) -> c_int;
    fn kapitsa_request_cancel(request: *mut NativeRequest);
    fn kapitsa_request_destroy(request: *mut NativeRequest);
    fn kapitsa_string_free(string: *mut c_char);
}
#[derive(Debug)]
pub struct Error {
    pub status: i32,
    pub message: String,
}
impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} (status {})", self.message, self.status)
    }
}
impl std::error::Error for Error {}
fn json(s: &str) -> Result<CString, Error> {
    CString::new(s).map_err(|_| Error {
        status: 1,
        message: "JSON contains NUL".into(),
    })
}
unsafe fn take(p: *mut c_char) -> String {
    if p.is_null() {
        return String::new();
    }
    let value = CStr::from_ptr(p).to_string_lossy().into_owned();
    kapitsa_string_free(p);
    value
}
struct EngineInner(NonNull<NativeEngine>);
// The ABI permits concurrent request creation; Arc prevents destruction races.
unsafe impl Send for EngineInner {}
unsafe impl Sync for EngineInner {}
impl Drop for EngineInner {
    fn drop(&mut self) {
        unsafe { kapitsa_engine_destroy(self.0.as_ptr()) }
    }
}
#[derive(Clone)]
pub struct Engine(Arc<EngineInner>);
impl Engine {
    pub fn new(options_json: &str) -> Result<Self, Error> {
        let options = json(options_json)?;
        unsafe {
            if kapitsa_abi_version() != 1 {
                return Err(Error {
                    status: 4,
                    message: "unsupported native ABI".into(),
                });
            }
            let (mut out, mut error) = (ptr::null_mut(), ptr::null_mut());
            let status = kapitsa_engine_create(options.as_ptr(), &mut out, &mut error);
            let message = take(error);
            if status != 0 {
                return Err(Error { status, message });
            }
            Ok(Self(Arc::new(EngineInner(
                NonNull::new(out).expect("native engine"),
            ))))
        }
    }
    pub fn request(&self, request_json: &str) -> Result<Request, Error> {
        let request = json(request_json)?;
        unsafe {
            let (mut out, mut error) = (ptr::null_mut(), ptr::null_mut());
            let status =
                kapitsa_request_create(self.0 .0.as_ptr(), request.as_ptr(), &mut out, &mut error);
            let message = take(error);
            if status != 0 {
                return Err(Error { status, message });
            }
            Ok(Request(Arc::new(RequestInner(
                NonNull::new(out).expect("native request"),
            ))))
        }
    }
}
struct RequestInner(NonNull<NativeRequest>);
// Only cancel is callable through shared handles; generate consumes Request.
unsafe impl Send for RequestInner {}
unsafe impl Sync for RequestInner {}
impl Drop for RequestInner {
    fn drop(&mut self) {
        unsafe { kapitsa_request_destroy(self.0.as_ptr()) }
    }
}
pub struct Request(Arc<RequestInner>);
#[derive(Clone)]
pub struct Cancellation(Arc<RequestInner>);
impl Cancellation {
    pub fn cancel(&self) {
        unsafe { kapitsa_request_cancel(self.0 .0.as_ptr()) }
    }
}
#[derive(Debug)]
pub struct Output {
    pub json: String,
    pub cancelled: bool,
}
impl Request {
    pub fn cancellation(&self) -> Cancellation {
        Cancellation(self.0.clone())
    }
    pub fn generate<F: FnMut(&str) -> bool>(self, mut on_delta: F) -> Result<Output, Error> {
        struct Context<'a, F> {
            callback: &'a mut F,
            panicked: bool,
        }
        unsafe extern "C" fn delta<F: FnMut(&str) -> bool>(
            p: *const c_char,
            n: usize,
            user: *mut c_void,
        ) -> c_int {
            let ctx = &mut *(user as *mut Context<'_, F>);
            let bytes = std::slice::from_raw_parts(p as *const u8, n);
            let Ok(text) = std::str::from_utf8(bytes) else {
                return 0;
            };
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| (ctx.callback)(text))) {
                Ok(keep) => i32::from(keep),
                Err(_) => {
                    ctx.panicked = true;
                    0
                }
            }
        }
        let mut ctx = Context {
            callback: &mut on_delta,
            panicked: false,
        };
        unsafe {
            let (mut result, mut error) = (ptr::null_mut(), ptr::null_mut());
            let status = kapitsa_generate(
                self.0 .0.as_ptr(),
                Some(delta::<F>),
                &mut ctx as *mut _ as *mut c_void,
                &mut result,
                &mut error,
            );
            let output = take(result);
            let message = take(error);
            if ctx.panicked {
                return Err(Error {
                    status: 4,
                    message: "stream callback panicked; request cancelled".into(),
                });
            }
            if status == 0 || status == 3 {
                Ok(Output {
                    json: output,
                    cancelled: status == 3,
                })
            } else {
                Err(Error { status, message })
            }
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn missing_model_is_error() {
        assert!(Engine::new(r#"{"model_dir":"/missing-sdk-model"}"#).is_err());
    }
    #[test]
    fn nul_is_error() {
        assert!(Engine::new("\0").is_err());
    }
    #[test]
    fn abi_matches() {
        assert_eq!(unsafe { kapitsa_abi_version() }, 1);
    }
}
