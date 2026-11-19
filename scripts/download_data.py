"""Download the two public datasets into data/raw/.

  Fin-Fact  (MIT)        https://github.com/IIT-DM/Fin-Fact   - Rangapur et al., 2023, arXiv:2309.08793
  SMS Spam Collection    https://archive.ics.uci.edu/dataset/228/sms+spam+collection
            (CC BY 4.0)  Almeida & Gomez Hidalgo, 2011. Mirror used: justmarkham/pycon-2016-tutorial
"""
import urllib.request
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
FILES = {
    "finfact.json": "https://raw.githubusercontent.com/IIT-DM/Fin-Fact/FinFact/finfact.json",
    "sms_spam.tsv": "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv",
}

if __name__ == "__main__":
    RAW.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        dest = RAW / name
        if dest.exists():
            print(f"exists: {dest}")
            continue
        print(f"downloading {url}")
        urllib.request.urlretrieve(url, dest)
        print(f"  saved {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
