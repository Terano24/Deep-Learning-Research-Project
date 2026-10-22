"""
Download T-cell assay records from the IEDB Query API (https://query-api.iedb.org).

Keeps the same selection criteria as the paper: linear epitopes, T-cell assays,
MHC class I, human host, peptide length 9/10. Lung-cancer records are flagged
(DOID:1324 and descendants) rather than filtered, so the model can be trained on
the full set and evaluated on the lung-cancer subset.
"""
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent
IEDB_URL = 'https://query-api.iedb.org/tcell_search'
LUNG_CANCER_DOID = 'DOID:1324'
PAGE_SIZE = 10000

SELECT = ','.join([
    'tcell_id', 'linear_sequence', 'linear_sequence_length', 'mhc_restriction',
    'mhc_allele_resolution', 'qualitative_measure', 'disease_names',
    'disease_iri_search', 'assay_names', 'reference_id', 'pubmed_id',
])
FILTERS = {
    'mhc_class': 'eq.I',
    'host_organism_iri': 'eq.NCBITaxon:9606',
    'linear_sequence_length': 'in.(9,10)',
}


def _get(url, retries=4, timeout=120):
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                return json.load(resp)
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(min(2 ** attempt, 30))


def fetch_iedb_data(out_path=None):
    out_path = Path(out_path) if out_path else BASE_DIR / 'data' / 'raw' / 'iedb_tcell_mhc1.csv'
    rows, offset = [], 0
    while True:
        params = dict(FILTERS, select=SELECT, order='tcell_id', limit=PAGE_SIZE, offset=offset)
        page = _get(IEDB_URL + '?' + urllib.parse.urlencode(params, safe='.,():'))
        rows.extend(page)
        print('fetched', len(rows))
        if len(page) < PAGE_SIZE:
            break
        offset += PAGE_SIZE

    df = pd.DataFrame(rows)
    df['lung_cancer'] = df['disease_iri_search'].apply(
        lambda x: LUNG_CANCER_DOID in x if isinstance(x, list) else False)
    df['disease_names'] = df['disease_names'].apply(lambda x: '|'.join(x) if isinstance(x, list) else x)
    df = df.drop(columns=['disease_iri_search'])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print('saved', len(df), 'rows to', out_path)
    return df


BINDING_SELECT = ','.join([
    'linear_sequence', 'linear_sequence_length', 'mhc_restriction',
    'qualitative_measure', 'quantitative_measure', 'assay_names',
])
BINDING_FILTERS = {
    'mhc_class': 'eq.I',
    'host_organism_iri': 'eq.NCBITaxon:9606',
    'linear_sequence_length': 'in.(9,10)',
    'mhc_restriction': 'like.HLA-*',
    'qualitative_measure': 'not.is.null',
}


def fetch_iedb_binding_data(out_path=None, max_rows=400000):
    """MHC binding / ligand-presentation assays, used to pretrain the encoder (transfer learning).

    Each page is appended to the CSV as it arrives, so an interrupted run resumes where it stopped.
    """
    out_path = Path(out_path) if out_path else BASE_DIR / 'data' / 'raw' / 'iedb_mhc_binding.csv'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    offset = 0
    if out_path.exists():
        offset = sum(1 for _ in open(out_path, encoding='utf-8')) - 1
        offset = (offset // PAGE_SIZE) * PAGE_SIZE      # drop a possibly half-written page below
        keep = pd.read_csv(out_path).iloc[:offset]
        keep.to_csv(out_path, index=False)
        print('resuming at', offset, flush=True)
    while offset < max_rows:
        params = dict(BINDING_FILTERS, select=BINDING_SELECT, order='structure_id,mhc_restriction',
                      limit=PAGE_SIZE, offset=offset)
        page = _get('https://query-api.iedb.org/mhc_search?' + urllib.parse.urlencode(params, safe='.,():*'),
                    retries=8, timeout=300)
        if not page:
            break
        pd.DataFrame(page)[BINDING_SELECT.split(',')].to_csv(
            out_path, mode='a', header=not out_path.exists() or offset == 0, index=False)
        offset += len(page)
        print('fetched', offset, flush=True)
        if len(page) < PAGE_SIZE:
            break
    print('saved', offset, 'rows to', out_path)


def fetch_imgt_data():
    """HLA pseudo-sequences are kept in data/raw/ParatopeIMGTopsi.txt."""
    pass


if __name__ == '__main__':
    import sys
    fetch_iedb_binding_data() if 'binding' in sys.argv else fetch_iedb_data()
