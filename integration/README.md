## TLS inspection lab

`tls-inspection-lab/` generates WordPress brute-force and scan traffic behind an nginx TLS terminator and records both the
encrypted side and the plain HTTP hop behind it, to compare Suricata and the model on the same attacks encrypted and decrypted.
It reuses the modules in this directory.

From the repository root, once:

```bash
python integration/fetch_hikari.py --csv-only
python integrate.py train
integration/setup.sh
pip install scapy
```

Then see `tls-inspection-lab/README.md` (needs Docker).
