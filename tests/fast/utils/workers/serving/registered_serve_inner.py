from tests.fast.utils.workers.serving.registered_serve import register_test_serve_specs

from miles.utils.workers.serving import serve_inner

if __name__ == "__main__":
    register_test_serve_specs()
    serve_inner.main()
