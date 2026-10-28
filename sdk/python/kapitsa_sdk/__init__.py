"""In-process Kapitsa bindings. No HTTP or child process is created."""
import ctypes
import json
import os
import threading
from pathlib import Path


class KapitsaError(RuntimeError):
    def __init__(self, status, message):
        self.status = status
        super().__init__(message or f"Kapitsa status {status}")


class _Library:
    callback_type = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p)

    def __init__(self, path=None):
        path = path or os.environ.get("KAPITSA_LIBRARY")
        if path is None:
            path = Path(__file__).resolve().parents[3] / "build/libkapitsa_sdk.dylib"
        self.native = ctypes.CDLL(str(path))
        pointer = ctypes.c_void_p
        output = ctypes.POINTER(pointer)
        signatures = {
            "kapitsa_engine_create": [ctypes.c_char_p, output, output],
            "kapitsa_request_create": [pointer, ctypes.c_char_p, output, output],
            "kapitsa_generate": [pointer, self.callback_type, pointer, output, output],
        }
        for name, args in signatures.items():
            function = getattr(self.native, name)
            function.argtypes = args
            function.restype = ctypes.c_int
        for name in ("kapitsa_engine_destroy", "kapitsa_request_destroy", "kapitsa_request_cancel", "kapitsa_string_free"):
            function = getattr(self.native, name)
            function.argtypes = [pointer]
            function.restype = None
        self.native.kapitsa_abi_version.argtypes = []
        self.native.kapitsa_abi_version.restype = ctypes.c_uint32
        if self.native.kapitsa_abi_version() != 1:
            raise KapitsaError(4, "unsupported native ABI")

    def take(self, pointer):
        if not pointer.value:
            return ""
        try:
            return ctypes.string_at(pointer).decode("utf-8")
        finally:
            self.native.kapitsa_string_free(pointer)

    def take_result(self, result, error):
        # Both belong to us even if decoding either string raises.
        try:
            output = self.take(result)
        finally:
            message = self.take(error)
        return output, message


def _encode(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")


class Engine:
    def __init__(self, model_dir, *, library=None, **options):
        self._library = _Library(library)
        self._lock = threading.RLock()
        self._handle = ctypes.c_void_p()
        error = ctypes.c_void_p()
        status = self._library.native.kapitsa_engine_create(
            _encode({**options, "model_dir": str(model_dir)}), ctypes.byref(self._handle), ctypes.byref(error))
        message = self._library.take(error)
        if status:
            raise KapitsaError(status, message)

    def request(self, messages, **options):
        with self._lock:
            if not self._handle.value:
                raise RuntimeError("engine is closed")
            handle, error = ctypes.c_void_p(), ctypes.c_void_p()
            status = self._library.native.kapitsa_request_create(
                self._handle, _encode({**options, "messages": messages}), ctypes.byref(handle), ctypes.byref(error))
            message = self._library.take(error)
            if status:
                raise KapitsaError(status, message)
            return Request(self._library, handle)

    def generate(self, messages, *, on_delta=None, **options):
        with self.request(messages, **options) as request:
            return request.generate(on_delta=on_delta)

    def close(self):
        with self._lock:
            if self._handle.value:
                self._library.native.kapitsa_engine_destroy(self._handle)
                self._handle = ctypes.c_void_p()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        if hasattr(self, "_lock"):
            self.close()


class Request:
    def __init__(self, library, handle):
        self._library, self._handle = library, handle
        self._lock = threading.RLock()
        self._running = False
        self._started = False

    def cancel(self):
        with self._lock:
            if self._handle.value:
                self._library.native.kapitsa_request_cancel(self._handle)

    def generate(self, *, on_delta=None):
        with self._lock:
            if not self._handle.value or self._started:
                raise RuntimeError("request is closed or already started")
            self._started = self._running = True
        callback_error = []

        try:
            @self._library.callback_type
            def callback(pointer, length, _user):
                try:
                    delta = json.loads(ctypes.string_at(pointer, length).decode("utf-8"))
                    return 0 if on_delta is not None and on_delta(delta) is False else 1
                except BaseException as exc:
                    callback_error.append(exc)
                    return 0

            result, error = ctypes.c_void_p(), ctypes.c_void_p()
            status = self._library.native.kapitsa_generate(
                self._handle, callback, None, ctypes.byref(result), ctypes.byref(error))
            output, message = self._library.take_result(result, error)
            if callback_error:
                raise callback_error[0]
            if status not in (0, 3):
                raise KapitsaError(status, message)
            return json.loads(output) if output else {"finish_reason": "cancelled"}
        finally:
            with self._lock:
                self._running = False

    def close(self):
        with self._lock:
            if self._running:
                raise RuntimeError("cancel and wait for generate to return before closing request")
            if self._handle.value:
                self._library.native.kapitsa_request_destroy(self._handle)
                self._handle = ctypes.c_void_p()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        if hasattr(self, "_lock") and not self._running:
            self.close()
