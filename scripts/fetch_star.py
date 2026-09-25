"""Explicit public STAR download/export; no download occurs during experiments.

Requires base R (Rscript) for reading the source .rda file. No AER package or
R add-on is needed. Alternatively pass an already downloaded source with
--source-rda. The source checksum and ordered CSV contents are verified.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from replication.io import safe_relative, write_json
from replication.star import (SOURCE_URL, SOURCE_SHA256, DOCUMENTATION_URL,
                              PAPER_ROWS_SHA256, read_star_rows, rows_sha256)


def fetch_star(output='data/STAR.csv', source_rda=None, rscript='Rscript'):
    destination = safe_relative(output)
    if destination.exists():
        rows = read_star_rows(destination)
        if rows_sha256(rows) != PAPER_ROWS_SHA256:
            raise FileExistsError('Existing CSV differs from the paper source; use another --output path')
        print('Verified existing STAR CSV: ' + str(destination.relative_to(ROOT)))
        return destination
    executable = shutil.which(rscript)
    if executable is None:
        raise RuntimeError('Install base R or pass --rscript PATH; data preparation needs Rscript once')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='star_source_') as scratch:
        scratch = Path(scratch)
        if source_rda is None:
            source = scratch / 'STAR.rda'
            request = urllib.request.Request(SOURCE_URL, headers={'User-Agent': 'STAR-replication'})
            with urllib.request.urlopen(request, timeout=120) as response:
                source.write_bytes(response.read())
        else:
            source = Path(source_rda).resolve()
        checksum = hashlib.sha256(source.read_bytes()).hexdigest()
        if checksum != SOURCE_SHA256:
            raise ValueError('STAR.rda checksum differs from the paper source; source data were not accepted')
        exported = scratch / 'STAR.csv'
        subprocess.run([executable, str(ROOT / 'scripts' / 'export_star.R'), str(source), str(exported)], check=True)
        rows = read_star_rows(exported)
        semantic = rows_sha256(rows)
        if semantic != PAPER_ROWS_SHA256:
            raise ValueError('STAR export differs from the frozen ordered source values')
        # Replace atomically only after both source and export have been checked.
        temporary = destination.with_suffix('.csv.tmp')
        shutil.copyfile(exported, temporary)
        temporary.replace(destination)
    provenance = dict(source_url=SOURCE_URL, source_documentation=DOCUMENTATION_URL,
                      source_rda_sha256=checksum, source_rows_sha256=semantic,
                      csv_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
                      n_source_rows=len(rows), exported_columns=list(rows[0]),
                      row_order='Original AER::STAR row order',
                      export='Base R; kindergarten and first-grade mean reading/mathematics scores')
    write_json(destination.with_suffix('.provenance.json'), provenance)
    print(json.dumps(dict(output=str(destination.relative_to(ROOT)), **provenance), indent=2))
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='data/STAR.csv', help='Relative path within this repository')
    parser.add_argument('--source-rda', help='Use an existing public AER STAR.rda instead of downloading')
    parser.add_argument('--rscript', default='Rscript', help='Rscript executable or explicit path')
    args = parser.parse_args()
    fetch_star(args.output, args.source_rda, args.rscript)


if __name__ == '__main__':
    main()
