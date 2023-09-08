from pHierCC import phierCC
from click.testing import CliRunner
import filecmp
import gzip
import os 
import shutil

def test_true():
    assert 1 == 1

def test_string_error():
    runner = CliRunner()
    result = runner.invoke(phierCC, ['-p', 'tests/profiles_string_test.txt', '-o', 'test_string_file', '-n', '12'])
    # decompress output 
    with gzip.open('test_string_file.HierCC.gz', 'rb') as f_in:
        with open('test_string_file.HierCC', 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    # compare output
    assert filecmp.cmp('tests/output_string_test.txt', 'test_string_file.HierCC')

def test_missing_error():
    runner = CliRunner()
    result = runner.invoke(phierCC, ['-p', 'tests/profiles_missing_test.txt', '-o', 'test_missing_file', '-n', '12'])
    # decompress output 
  #  with gzip.open('test_missing_file.HierCC.gz', 'rb') as f_in:
 #       with open('test_missing_file.HierCC', 'wb') as f_out:
   #         shutil.copyfileobj(f_in, f_out)
    # compare output
    ##assert filecmp.cmp('tests/output_missing_test.txt', 'test_missing_file.HierCC')
  

# Path: test_basic.py