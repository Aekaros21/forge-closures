"""python -m forge_fixed hash src/comparators/<dir>   prints the comparator source_sha256."""
import sys

from forge_fixed import source_sha256

if len(sys.argv) != 3 or sys.argv[1] != 'hash':
    raise SystemExit('usage: python -m forge_fixed hash src/comparators/<dir>')
print(source_sha256(sys.argv[2]))
