import pandas as pd
import numpy as np

import argparse

parser = argparse.ArgumentParser(description='Process a list of submission as CSV files to aggregate them for ensembling.')
parser.add_argument('files', nargs='+', help='List of CSV files to ensemble')
parser.add_argument('--output', default='ensemble_submission.csv', help='Output file name')

args = parser.parse_args()

submits = []
# Access the list of files using args.files
for file in args.files:
    print(f"Processing file: {file}")
    submits.append(pd.read_csv(file))

# Check that all files have the same number of rows
for i in range(1, len(submits)):
    assert len(submits[i]) == len(submits[i-1])

# Check that all files have the same IDs
for i in range(1, len(submits)):
    assert np.all(submits[i]['ID'] == submits[i-1]['ID'])

ensemble = pd.DataFrame()
ensemble_df = submits[0].copy()
cols = submits[0].columns.difference(['ID'])

print(f"Ensembling {len(submits)} files")
for col in cols:
    ensemble_df[col] = sum(df[col] for df in submits) / len(submits)

print(f"Saving to {args.output}")
ensemble_df.to_csv(args.output, index=False)

print("Done!")