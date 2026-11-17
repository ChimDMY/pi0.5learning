import unittest
import numpy as np
from contract import PROMPT, SCHEMA, validate_request, validate_actions


class ContractTests(unittest.TestCase):
    def request(self):
        return {"schema": SCHEMA, "request_id": "test", "prompt": PROMPT, "noise_seed": 812,
            "observation/state": np.array([0, 1, -1, 0, 0, 0, -1], dtype=np.float32),
            "observation/image": np.zeros((240, 320, 3), dtype=np.uint8),
            "observation/wrist_image": np.ones((240, 320, 3), dtype=np.uint8)}

    def test_preserve_joint_units_gripper_and_camera_order(self):
        data = self.request()
        obs, noise = validate_request(data)
        np.testing.assert_array_equal(obs["observation/state"], data["observation/state"])
        self.assertEqual(int(obs["observation/image"].max()), 0)
        self.assertEqual(int(obs["observation/wrist_image"].min()), 1)
        np.testing.assert_array_equal(noise, validate_request(data)[1])

    def test_reject_wrong_task_missing_camera_invalid_state(self):
        bad_values = [("prompt", "stack"), ("observation/state", np.zeros(6)),
                      ("observation/state", np.full(7, np.nan)),
                      ("observation/wrist_image", np.zeros((240, 320, 3), dtype=float)),
                      ("noise_seed", -1)]
        for key, value in bad_values:
            with self.subTest(key=key):
                data = self.request(); data[key] = value
                with self.assertRaises(ValueError): validate_request(data)
        data = self.request(); del data["observation/wrist_image"]
        with self.assertRaises(KeyError): validate_request(data)

    def test_actions_are_not_silently_clipped_or_inverted(self):
        raw = np.zeros((10, 7), dtype=np.float32); raw[:, 6] = 0.02
        np.testing.assert_array_equal(validate_actions(raw), raw)
        with self.assertRaises(ValueError): validate_actions(np.zeros((10, 32)))
        with self.assertRaises(ValueError): validate_actions(np.full((10, 7), np.inf))


if __name__ == "__main__":
    unittest.main()
