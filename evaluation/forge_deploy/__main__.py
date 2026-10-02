"""Comparator deployment commands on a FORGE root (the HX1 verification root, or this checkout).

  python -m forge_deploy check DIR...                       folder checks and source_sha256 (no build)
  python -m forge_deploy manifest DIR... [--extra FILE...]  files the comparator staging sends
  python -m forge_deploy build DIR... [--expect FILE]       build each library (FILE: {"<dir>": "<source_sha256>"})
  python -m forge_deploy describe [CLASS...]                built libraries on this root, verified
"""
import argparse
import json
import sys

from forge_deploy import build_comparator, check_tree, describe, staging_manifest


def parser():
    p = argparse.ArgumentParser(prog='python -m forge_deploy', description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    sub.add_parser('check').add_argument('dirs', nargs='+')
    m = sub.add_parser('manifest')
    m.add_argument('dirs', nargs='+')
    m.add_argument('--extra', nargs='*', default=[])
    b = sub.add_parser('build')
    b.add_argument('dirs', nargs='+')
    b.add_argument('--expect', help='JSON file {"<dir>": "<source_sha256>"}; every built dir must be listed')
    sub.add_parser('describe').add_argument('classes', nargs='*')
    return p


def main(argv=None):
    from forge.core import ROOT
    args = parser().parse_args(argv)
    if args.command == 'check':
        print(json.dumps({d: check_tree(ROOT, d) for d in args.dirs}, indent=2, sort_keys=True))
    elif args.command == 'manifest':
        print('\n'.join(staging_manifest(ROOT, args.dirs, args.extra)))
    elif args.command == 'build':
        expected = json.loads(open(args.expect).read()) if args.expect else {}
        missing = [d for d in args.dirs if args.expect and d not in expected]
        if missing:
            raise SystemExit(f'--expect lists no source_sha256 for {missing}')
        done = {}
        for d in args.dirs:
            record = build_comparator(ROOT, d, expected_source_sha256=expected.get(d))
            done[d] = {k: record[k] for k in ('ras_model', 'library', 'library_sha256', 'source_sha256')}
            print(f'built {d}: {record["library"]} {record["library_sha256"]}', file=sys.stderr, flush=True)
        print(json.dumps(done, indent=2, sort_keys=True))
    else:
        print(json.dumps(describe(ROOT, args.classes or None), indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
