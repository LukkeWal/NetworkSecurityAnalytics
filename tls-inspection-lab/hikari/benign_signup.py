"""
Signs up random users with Firefox and Chrome through the authors' mitmproxy setting (their appSignupWordpress.py logic).
"""

import sys
import time

sys.path.insert(0, "/lab/profiles/benign")
from selenium import webdriver  # noqa: E402
from AutoSignPost import AutoSignPost  # noqa: E402
from LibDGA import LibDGA  # noqa: E402

webdriver.DesiredCapabilities.FIREFOX["acceptInsecureCerts"] = True


def main(url: str, signups: int) -> None:
    names = LibDGA()
    tool = AutoSignPost(PROXY="localhost:8080")
    for number in range(signups):
        try:
            tool.signUp(url, names.getFirstname(), names.getRandomEmail(), "firefox" if number % 2 == 0 else "chrome")
        except Exception as error:
            print("signup failed:", type(error).__name__)
        time.sleep(3)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
