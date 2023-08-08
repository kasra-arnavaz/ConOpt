import unittest


def main(test_directory):
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover(test_directory)
    test_runner = unittest.TextTestRunner()
    test_runner.run(test_suite)


if __name__ == "__main__":
    main("tests/integration")
