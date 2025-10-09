import argparse

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--test-name", type=str)
    args = arg_parser.parse_args()
    print(f"{args=}")

    TODO
