import ctypes
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

from kapitsa_sdk import Engine, KapitsaError, Request, _Library


class ScriptedLibrary(_Library):
    """CPU-only stand-in at the C boundary; no engine, model or GPU is created."""

    def __init__(self):
        self.buffers = {}
        self.freed = []
        self.cancelled = False
        self.destroyed = 0
        self.before_delta = lambda: None
        self.native = SimpleNamespace(
            kapitsa_generate=self.generate,
            kapitsa_request_cancel=self.cancel,
            kapitsa_request_destroy=self.destroy,
            kapitsa_string_free=self.free,
        )

    def allocate(self, output, value):
        buffer = ctypes.create_string_buffer(value)
        address = ctypes.addressof(buffer)
        self.buffers[address] = buffer
        output._obj.value = address

    def generate(self, handle, callback, user, result, error):
        self.before_delta()
        delta = ctypes.create_string_buffer(b'{"content":"hello","reasoning":"","tool_calls":[]}')
        keep = callback(ctypes.addressof(delta), len(delta.value), user)
        cancelled = self.cancelled or not keep
        reason = b'cancelled' if cancelled else b'stop'
        self.allocate(result, b'{"content":"hello","finish_reason":"' + reason + b'"}')
        return 3 if cancelled else 0

    def cancel(self, handle):
        self.cancelled = True

    def destroy(self, handle):
        self.destroyed += 1

    def free(self, pointer):
        self.freed.append(pointer.value)
        del self.buffers[pointer.value]


class ContractTests(unittest.TestCase):
    def test_missing_model(self):
        with self.assertRaises(KapitsaError) as caught:
            Engine("/missing-python-sdk-model")
        self.assertEqual(caught.exception.status, 1)

    def test_unknown_option(self):
        with self.assertRaises(KapitsaError):
            Engine("/missing-python-sdk-model", typo=True)

    def test_nan_rejected(self):
        with self.assertRaises(ValueError):
            Engine("/missing-python-sdk-model", mem_budget_gb=float("nan"))

    def request(self):
        library = ScriptedLibrary()
        request = Request(library, ctypes.c_void_p(1))
        self.addCleanup(request.close)
        return library, request

    def test_callback_exception_is_reraised_after_native_returns(self):
        library, request = self.request()
        failure = ValueError("application callback failed")

        def callback(_delta):
            raise failure

        with self.assertRaises(ValueError) as caught:
            request.generate(on_delta=callback)
        self.assertIs(caught.exception, failure)
        self.assertEqual(len(library.freed), 1)
        self.assertFalse(library.buffers)
        request.close()
        request.close()
        self.assertEqual(library.destroyed, 1)

    def test_request_is_one_shot_and_callback_can_cancel(self):
        library, request = self.request()
        result = request.generate(on_delta=lambda _delta: False)
        self.assertEqual(result["finish_reason"], "cancelled")
        with self.assertRaises(RuntimeError):
            request.generate()
        self.assertFalse(library.buffers)

    def test_running_request_refuses_close_but_accepts_cancel(self):
        library, request = self.request()
        entered, resume = threading.Event(), threading.Event()

        def wait_for_cancel():
            entered.set()
            if not resume.wait(5):
                raise TimeoutError("test did not release generation")

        library.before_delta = wait_for_cancel
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(request.generate)
            try:
                self.assertTrue(entered.wait(5))
                with self.assertRaises(RuntimeError):
                    request.close()
                self.assertEqual(library.destroyed, 0)
                request.cancel()
            finally:
                resume.set()
            self.assertEqual(future.result(timeout=5)["finish_reason"], "cancelled")
        request.close()
        self.assertEqual(library.destroyed, 1)

    def test_callback_setup_failure_allows_cleanup(self):
        library, request = self.request()
        library.callback_type = Mock(side_effect=MemoryError("callback allocation failed"))
        with self.assertRaises(MemoryError):
            request.generate()
        request.close()
        self.assertEqual(library.destroyed, 1)

    def test_both_native_strings_are_freed_if_result_is_invalid_utf8(self):
        library, request = self.request()

        def invalid_result(handle, callback, user, result, error):
            library.allocate(result, b'\xff')
            library.allocate(error, b'error detail')
            return 4

        library.native.kapitsa_generate = invalid_result
        with self.assertRaises(UnicodeDecodeError):
            request.generate()
        self.assertEqual(len(library.freed), 2)
        self.assertFalse(library.buffers)


if __name__ == "__main__":
    unittest.main()
