def pytest_addoption(parser):
    parser.addoption("--snapshot", action="store_true", help="Generate snapshots from current code")
    parser.addoption("--compare", action="store_true", help="Compare current code against saved snapshots")
