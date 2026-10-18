# TLS inspection lab

WordPress brute-force and scan traffic behind an nginx TLS terminator. A recorder in nginx's network sees both the encrypted
side (port 443) and the plain HTTP hop to WordPress (port 80), so Suricata and the model can be run on the same attacks
encrypted and decrypted.

```
attackers ---HTTPS---> nginx (TLS 1.3) ---HTTP---> WordPress + MariaDB
                          |
                       recorder (tcpdump)
```

The encrypted side goes through Zeek + zeek-flowmeter (flows for the model) and Suricata. On the plain hop,
`rewrite_client_addresses.py` restores each client's address and port from the headers nginx passes on.
nginx closes every connection after one response, so each request is one flow.

## Run

Needs Docker, Zeek, Suricata (`integration/setup.sh` at the repository root), `tcpdump` and `pip install scapy`.
Create the certificates once with `certs/make_certs.sh`, and train the HIKARI model at the repository root first.

```bash
./lab.sh up run-a        # start the stack, install WordPress
./lab.sh run run-a       # benign browsing, browser and XML-RPC brute force, WPScan
./lab.sh stop run-a
python analyze.py run-a
./lab.sh down run-a
```

`hikari.sh` does the same with the HIKARI authors' own scripts (`AutoSignPost.py`, `autobrowse.py`, cloned when the image
is built), their host names and their mitmproxy setting; probing is WPScan. It uses `analyze_authorscripts.py` and
`analyze_background.py`. `BENIGN_ROUNDS`, `BRUTE_ATTEMPTS` and `XMLRPC_ATTEMPTS` set the volume.

## Results

Share of flows flagged with the authors' scripts (`runs/author-scripts/results/`):

| Traffic | Flows | Suricata, encrypted | Suricata, decrypted |
| --- | --- | --- | --- |
| Brute force (browser) | 2,709 | 0% | 18.9% |
| Brute force XML-RPC | 5,145 | 0% | 0.6% |
| Probing (WPScan) | 182 | 0% | 100% |
| Benign signups | 752 | 0% | 0% |

The version with our own attack code (`runs/full/`, 50,629 attack and 65,097 benign flows) gives the same picture: 80.8% of
attack flows caught decrypted, 12.3% of benign flows flagged. The generated traffic is similar to HIKARI-2021 but not the same
dataset: probing is WPScan, there is no cryptominer traffic, and the target is a single WordPress server.
