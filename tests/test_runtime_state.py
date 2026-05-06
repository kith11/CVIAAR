import time
import unittest

from modules.runtime_state import ThreadSafeTTLStore


class ThreadSafeTTLStoreTests(unittest.TestCase):
    def test_expired_values_are_not_returned(self):
        store = ThreadSafeTTLStore(default_ttl=0.05)
        store.set("k", "v")

        time.sleep(0.08)

        self.assertIsNone(store.get("k"))

    def test_increment_is_atomic_for_simple_counter_usage(self):
        store = ThreadSafeTTLStore(default_ttl=1.0)

        for _ in range(5):
            current = store.increment("requests")

        self.assertEqual(current, 5)
        self.assertEqual(store.get("requests"), 5)

    def test_snapshot_only_returns_live_entries(self):
        store = ThreadSafeTTLStore()
        store.set("active", 1)
        store.set("expired", 2, ttl=0.01)

        time.sleep(0.03)

        self.assertEqual(store.snapshot(), {"active": 1})


if __name__ == "__main__":
    unittest.main()
