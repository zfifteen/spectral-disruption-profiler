"""
Download and prepare public gRNA efficacy datasets.

This script downloads datasets from public sources and prepares them
for validation. All datasets are verified with SHA-256 checksums.
"""

import os
import sys
import hashlib
import argparse
from pathlib import Path
import urllib.request
import pandas as pd


DATASETS_DIR = Path(__file__).parent.parent / 'datasets'

# Dataset metadata
DATASETS = {
    'doench2016': {
        'name': 'Doench 2016',
        'description': 'Optimized sgRNA design to maximize activity and minimize off-target effects',
        'url': 'https://www.nature.com/articles/nbt.3437',  # Paper URL
        'data_url': None,  # Placeholder - needs actual data URL
        'n_samples': '~5000-15000',
        'reference': 'Doench et al., Nature Biotechnology 2016',
        'license': 'Check publication supplementary info',
        'notes': 'Industry standard benchmark for gRNA design. Data may need manual download from supplementary materials.'
    },
    'wang2019': {
        'name': 'Wang 2019',
        'description': 'CRISPR-Cas9 screen data',
        'url': 'https://doi.org/10.1016/j.cell.2019.02.041',  # Example
        'data_url': None,
        'n_samples': '>1000',
        'reference': 'Wang et al., Cell 2019 (example)',
        'license': 'Check publication',
        'notes': 'Additional dataset for cross-validation. URL is placeholder.'
    }
}


def compute_sha256(filepath):
    """Compute SHA-256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def download_file(url, destination, description="file"):
    """Download a file from URL to destination."""
    print(f"Downloading {description} from {url}...")
    print(f"Destination: {destination}")
    
    try:
        urllib.request.urlretrieve(url, destination)
        print(f"✓ Downloaded successfully")
        return True
    except Exception as e:
        print(f"✗ Download failed: {e}")
        return False


def prepare_doench2016():
    """
    Prepare Doench 2016 dataset.
    
    NOTE: This is a placeholder implementation. The actual data needs to be
    downloaded from the Nature Biotechnology supplementary materials:
    https://www.nature.com/articles/nbt.3437
    
    The supplementary data typically includes:
    - Guide sequences (20-mers)
    - Measured efficacy scores
    - Experimental conditions
    
    Manual steps required:
    1. Visit the publication page
    2. Download supplementary data files
    3. Extract relevant columns (sequence, efficacy)
    4. Save as CSV in datasets/doench2016/raw/
    """
    dataset_dir = DATASETS_DIR / 'doench2016'
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    raw_dir = dataset_dir / 'raw'
    raw_dir.mkdir(exist_ok=True)
    
    processed_file = dataset_dir / 'doench2016_processed.csv'
    
    print("\n" + "="*80)
    print("Doench 2016 Dataset Preparation")
    print("="*80)
    print("\n⚠️  This dataset requires manual download from:")
    print("    https://www.nature.com/articles/nbt.3437")
    print("\nSteps:")
    print("  1. Visit the publication page")
    print("  2. Download supplementary data files")
    print("  3. Place CSV with columns 'sequence' and 'efficacy' in:")
    print(f"     {raw_dir}/")
    print("  4. Re-run this script to process the data")
    print("\nExpected columns in raw CSV:")
    print("  - sequence: 20-mer gRNA sequence (string)")
    print("  - efficacy: Measured on-target efficacy (float, 0-1 or normalized)")
    print("  - (optional) other metadata columns")
    
    # Check if raw data exists
    raw_files = list(raw_dir.glob('*.csv'))
    if not raw_files:
        print(f"\n✗ No CSV files found in {raw_dir}/")
        print("  Dataset preparation incomplete.")
        return False
    
    print(f"\n✓ Found {len(raw_files)} CSV file(s) in raw directory")
    
    # Try to load and process
    for raw_file in raw_files:
        print(f"\nProcessing: {raw_file.name}")
        try:
            df = pd.read_csv(raw_file)
            print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
            print(f"  Columns: {list(df.columns)}")
            
            # Check for required columns
            if 'sequence' in df.columns and 'efficacy' in df.columns:
                print("  ✓ Found required columns: sequence, efficacy")
                
                # Basic validation
                print(f"  Sequence lengths: {df['sequence'].str.len().unique()}")
                print(f"  Efficacy range: [{df['efficacy'].min():.3f}, {df['efficacy'].max():.3f}]")
                
                # Save processed version
                processed_df = df[['sequence', 'efficacy']].copy()
                processed_df.to_csv(processed_file, index=False)
                
                # Compute checksum
                checksum = compute_sha256(processed_file)
                checksum_file = dataset_dir / 'SHA256'
                with open(checksum_file, 'w') as f:
                    f.write(f"{checksum}  {processed_file.name}\n")
                
                print(f"\n✓ Processed dataset saved to: {processed_file}")
                print(f"  SHA-256: {checksum}")
                print(f"  Checksum saved to: {checksum_file}")
                
                return True
            else:
                print("  ✗ Missing required columns (sequence, efficacy)")
                
        except Exception as e:
            print(f"  ✗ Error processing file: {e}")
    
    return False


def list_datasets():
    """List all available datasets."""
    print("\n" + "="*80)
    print("Available Datasets")
    print("="*80)
    
    for key, info in DATASETS.items():
        print(f"\n{key}:")
        print(f"  Name: {info['name']}")
        print(f"  Description: {info['description']}")
        print(f"  Sample size: {info['n_samples']}")
        print(f"  Reference: {info['reference']}")
        print(f"  URL: {info['url']}")
        
        # Check if downloaded
        dataset_dir = DATASETS_DIR / key
        processed_file = dataset_dir / f"{key}_processed.csv"
        
        if processed_file.exists():
            print(f"  Status: ✓ Downloaded and processed")
            print(f"  Location: {processed_file}")
            
            # Load and show stats
            try:
                df = pd.read_csv(processed_file)
                print(f"  Rows: {len(df)}")
                print(f"  Columns: {list(df.columns)}")
            except:
                pass
        else:
            print(f"  Status: ✗ Not downloaded")
            print(f"  Notes: {info['notes']}")


def verify_dataset(dataset_name):
    """Verify dataset integrity using SHA-256 checksum."""
    dataset_dir = DATASETS_DIR / dataset_name
    checksum_file = dataset_dir / 'SHA256'
    
    if not checksum_file.exists():
        print(f"✗ No checksum file found for {dataset_name}")
        return False
    
    print(f"\nVerifying {dataset_name}...")
    
    with open(checksum_file, 'r') as f:
        for line in f:
            expected_hash, filename = line.strip().split(maxsplit=1)
            filepath = dataset_dir / filename
            
            if not filepath.exists():
                print(f"✗ File not found: {filepath}")
                return False
            
            actual_hash = compute_sha256(filepath)
            
            if actual_hash == expected_hash:
                print(f"✓ {filename}: OK")
            else:
                print(f"✗ {filename}: FAILED")
                print(f"  Expected: {expected_hash}")
                print(f"  Actual:   {actual_hash}")
                return False
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Download and prepare public gRNA efficacy datasets'
    )
    parser.add_argument(
        'action',
        choices=['list', 'download', 'verify'],
        help='Action to perform'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        choices=list(DATASETS.keys()) + ['all'],
        help='Dataset to download or verify'
    )
    
    args = parser.parse_args()
    
    # Create datasets directory
    DATASETS_DIR.mkdir(exist_ok=True)
    
    if args.action == 'list':
        list_datasets()
    
    elif args.action == 'download':
        if not args.dataset:
            print("Error: --dataset required for download action")
            parser.print_help()
            return 1
        
        if args.dataset == 'all':
            datasets_to_download = list(DATASETS.keys())
        else:
            datasets_to_download = [args.dataset]
        
        for dataset in datasets_to_download:
            if dataset == 'doench2016':
                prepare_doench2016()
            # Add more dataset handlers here
            else:
                print(f"\n✗ Download handler not implemented for {dataset}")
    
    elif args.action == 'verify':
        if not args.dataset:
            print("Error: --dataset required for verify action")
            parser.print_help()
            return 1
        
        if args.dataset == 'all':
            datasets_to_verify = list(DATASETS.keys())
        else:
            datasets_to_verify = [args.dataset]
        
        all_ok = True
        for dataset in datasets_to_verify:
            if not verify_dataset(dataset):
                all_ok = False
        
        if all_ok:
            print("\n✓ All datasets verified successfully")
            return 0
        else:
            print("\n✗ Some datasets failed verification")
            return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
