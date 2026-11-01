"""Ordinary use of the shop: read pages, log in with the right password, open the dashboard."""

import os
import random
import sys
import time

from selenium import webdriver
from selenium.webdriver.common.by import By

SCALE = float(os.environ.get("SLEEP_SCALE", "1"))


def pause(low: float, high: float) -> None:
    time.sleep(random.uniform(low, high) * SCALE)


def one_round(driver) -> None:
    for path in ["/", "/?p=4", "/?s=shop", "/wp-login.php?action=lostpassword", "/?feed=rss2"]:
        driver.get("https://tls" + path)
        pause(0.5, 2.5)
    driver.get("https://tls/wp-login.php")
    driver.find_element(By.ID, "user_login").send_keys("alice")
    driver.find_element(By.ID, "user_pass").send_keys("Alice-lab-Pw-4421")
    driver.find_element(By.ID, "wp-submit").click()
    pause(1, 3)
    for path in ["/wp-admin/", "/wp-admin/edit.php", "/wp-admin/post-new.php", "/wp-admin/profile.php"]:
        driver.get("https://tls" + path)
        pause(0.5, 2.5)
    driver.get("https://tls/wp-login.php?action=logout")
    pause(0.5, 1.5)


def main(grid: str, rounds: int) -> None:
    options = webdriver.ChromeOptions()
    options.add_argument("--ignore-certificate-errors")
    driver = webdriver.Remote(command_executor=grid, options=options)
    random.seed()
    failed = 0
    for _ in range(rounds):
        try:
            one_round(driver)
        except Exception:
            failed += 1
    print("benign browsing done:", rounds, "rounds,", failed, "failed")
    driver.quit()


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
