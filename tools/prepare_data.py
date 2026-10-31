"""Prepare the exact two-file CSIC + local-synthetic corpus used in the study."""
from pathlib import Path
import shutil
import requests

ROOT = Path(__file__).resolve().parents[1]
BASE = 'https://raw.githubusercontent.com/yungshenglu/TF-IDS-Detection/master/data/'

def main() -> None:
    raw = ROOT / 'data' / 'raw'
    raw.mkdir(parents=True, exist_ok=True)
    for name in ('normalTrafficTest.txt', 'anomalousTrafficTest.txt'):
        target = raw / name
        if not target.exists():
            response = requests.get(BASE + name, timeout=120)
            response.raise_for_status()
            target.write_bytes(response.content)
        print(f'Ready: {name} ({target.stat().st_size} bytes)')
    for source in sorted((ROOT / 'data' / 'synthetic').glob('*.txt')):
        shutil.copy2(source, raw / source.name)
    print('Prepared two CSIC files and five fixed local-synthetic files.')
    print('Do not add normalTrafficTrain.txt when reproducing the bundled results.')

if __name__ == '__main__':
    main()
