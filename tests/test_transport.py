"""Bounded localhost protocol check with a fake policy; never loads model weights."""
import json
from pathlib import Path
import socket
import subprocess
import sys
import time
import unittest


class TransportTests(unittest.TestCase):
    def test_smoke_client_against_fake_policy(self):
        root = Path(__file__).resolve().parents[1]
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        code = '''
import sys
import numpy as np
from contract import PROMPT, SCHEMA, validate_request
from openpi.serving.websocket_policy_server import WebsocketPolicyServer
class Fake:
    def infer(self, data):
        validate_request(data)
        return {"request_id": data["request_id"], "actions": np.zeros((10,7), dtype=np.float32), "infer_ms": 0.0}
WebsocketPolicyServer(Fake(), host="127.0.0.1", port=int(sys.argv[1]),
    metadata={"schema":SCHEMA, "prompt":PROMPT, "model_identity":"TEST_FAKE_NOT_A_MODEL"}).serve_forever()
'''
        process = subprocess.Popen([sys.executable, "-c", code, str(port)], cwd=root,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            deadline = time.monotonic() + 15
            while True:
                if process.poll() is not None:
                    self.fail("Fake policy server exited during startup")
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                        break
                except OSError:
                    if time.monotonic() >= deadline:
                        self.fail("Fake policy server startup timed out")
                    time.sleep(0.1)
            result = subprocess.run([sys.executable, "smoke_client.py", "--port", str(port),
                                     "--timeout", "5"], cwd=root, capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(result.stdout)
            self.assertEqual(report["action_shape"], [10, 7])
            self.assertFalse(report["robot_moved"])
            self.assertEqual(report["model_identity"], "TEST_FAKE_NOT_A_MODEL")
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()
