"""
Rewrites the capture of the nginx-to-WordPress hop so each connection shows the original client address and port, as a decrypting
proxy would. The addresses come from the X-Forwarded-For and X-Client-Port headers that nginx adds.

    python rewrite_client_addresses.py runs/<id>/plain/plain.pcap runs/<id>/plain/plain-clients.pcap
"""

import re
import sys
from bisect import bisect_right

from scapy.all import IP, TCP, PcapReader, PcapWriter

FORWARDED = re.compile(rb"X-Forwarded-For: ([0-9.]+)\r\nX-Client-Port: (\d+)")


def main(source: str, destination: str) -> None:
    # nginx reuses its temporary ports, so a (address, port) pair belongs to a new client after every SYN
    connections = {}
    with PcapReader(source) as packets:
        for packet in packets:
            if TCP in packet and packet[TCP].dport == 80:
                ip, tcp = packet[IP], packet[TCP]
                key = (ip.src, tcp.sport)
                if tcp.flags.S and not tcp.flags.A:
                    connections.setdefault(key, []).append([float(packet.time), None])
                elif tcp.payload and key in connections and connections[key][-1][1] is None:
                    found = FORWARDED.search(bytes(tcp.payload))
                    if found:
                        connections[key][-1][1] = (found.group(1).decode(), int(found.group(2)))
    starts = {key: [start for start, _ in instances] for key, instances in connections.items()}

    def client_of(key, time):
        if key not in connections:
            return None
        index = bisect_right(starts[key], time) - 1
        return connections[key][index][1] if index >= 0 else None

    rewritten = 0
    with PcapReader(source) as packets, PcapWriter(destination, sync=True) as output:
        for packet in packets:
            if TCP in packet:
                ip, tcp = packet[IP], packet[TCP]
                if tcp.dport == 80 and (client := client_of((ip.src, tcp.sport), float(packet.time))):
                    ip.src, tcp.sport = client
                    rewritten += 1
                elif tcp.sport == 80 and (client := client_of((ip.dst, tcp.dport), float(packet.time))):
                    ip.dst, tcp.dport = client
                    rewritten += 1
                del ip.chksum, tcp.chksum
            output.write(packet)
    print(f"{sum(len(i) for i in connections.values())} connections, {rewritten} packets rewritten")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
