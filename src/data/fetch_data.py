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


def _get(url, retries=4):
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=120) as resp:
                return json.load(resp)
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


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


def fetch_imgt_data():
    """HLA pseudo-sequences are kept in data/raw/ParatopeIMGTopsi.txt."""
    pass


if __name__ == '__main__':
    fetch_iedb_data()
