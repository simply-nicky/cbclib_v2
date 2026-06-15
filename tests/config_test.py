from cbclib_v2 import get_cpu_config, reset_cpu_config, set_cpu_pool_worker

class TestCPUConfig():
    def teardown_method(self) -> None:
        set_cpu_pool_worker(False)
        reset_cpu_config()

    def test_effective_num_threads_uses_config_in_main_thread(self) -> None:
        get_cpu_config().num_threads = 4

        assert get_cpu_config().effective_num_threads() == 4

    def test_effective_num_threads_uses_one_thread_in_pool_worker(self) -> None:
        get_cpu_config().num_threads = 4
        set_cpu_pool_worker(True)

        assert get_cpu_config().effective_num_threads() == 1

    def test_reset_cpu_config_preserves_pool_worker_context(self) -> None:
        get_cpu_config().num_threads = 4
        set_cpu_pool_worker(True)

        reset_cpu_config()
        get_cpu_config().num_threads = 4

        assert get_cpu_config().effective_num_threads() == 1
