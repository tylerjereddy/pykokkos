import sys
import argparse
import os
import shutil
import pytest

# purge pk_cpp folder so that the
# test suite actually translates and
# compiles the code under test
cwd = os.getcwd()
shutil.rmtree(os.path.join(cwd, "pk_cpp"),
              ignore_errors=True)


# try to support command line arguments to
# runtests.py that mirror direct usage of
# pytest
pytest_args = []

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--specifictests', type=str)
parser.add_argument('-d', '--durations', type=int)
parser.add_argument('-c', '--coverage', type=str)
args = parser.parse_args()

if args.specifictests:
    pytest_args.append(args.specifictests)
if args.durations:
    pytest_args.append(f"--durations={args.durations}")
if args.coverage:
    pytest_args.append(f"--cov-report xml")


# force pytest to actually import
# all the test modules directly
ret = pytest.main(pytest_args)
sys.exit(ret)
