from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import time
import pickle
from selenium.webdriver.support.ui import Select
import os
from log_in import SeleniumDriverLinkedin, SignInLinkedin
import yacs.config
from config import get_default_config
from crawl_info import CrawlInfor

def load_config() -> yacs.config.CfgNode:
    config = get_default_config()
    return config


def crawl_data():
    selenium_driver = SeleniumDriverLinkedin()
    driver = selenium_driver.driver
    config = load_config()
    account = {
        "email": config.account.user_name,
        "password": config.account.password
    }
    sign_in = SignInLinkedin(account, driver)
    response = sign_in.run()
    if response == 'Sign in successful!':
        crawl_info = CrawlInfor(driver, config)
        crawl_info.load_data(200)
def main():
    crawl_data()

if __name__ == '__main__':
    main()
