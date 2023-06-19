from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import time
import pickle
from selenium.webdriver.support.ui import Select
import os
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.service import Service as FirefoxService
from webdriver_manager.firefox import GeckoDriverManager

class SeleniumDriverLinkedin():
    def __init__(self):
        options = webdriver.FirefoxOptions()
        options.headless = False
        self.driver = webdriver.Firefox(service=FirefoxService(GeckoDriverManager().install()), options=options)
        self.driver.set_window_size(1920,1080)

    def close_all(self):
        # close all open tabs
        if len(self.driver.window_handles) < 1:
            return
        for window_handle in self.driver.window_handles[:]:
            self.driver.switch_to.window(window_handle)
            self.driver.close()

    def quit(self):
        # self.save_cookies()
        self.close_all()


class SignInLinkedin():
    def __init__(self, data, driver):
        self.driver = driver
        self.data = data
        self.limit = 0
        self.number_page = 0

    def sign_in_linkedin(self):
        self.driver.get('https://www.linkedin.com/login?fromSignIn=true&trk=guest_homepage-basic_nav-header-signin')
        time.sleep(5)
        try:
            time.sleep(5)
            username = self.driver.find_element(By.ID, "username")
            username.clear()
            username.send_keys(self.data["email"])
            self.driver.implicitly_wait(5)
            password = self.driver.find_element(By.ID, "password")
            password.clear()
            password.send_keys(self.data["password"])
            agree = self.driver.find_element(By.XPATH, '//button[normalize-space()="Sign in"]')
            agree.click()
        except:
            return "Can't send keys password and email in Sign in"
        try:
            error_username = self.driver.find_element(By.ID, "error-for-username")
            if error_username:
                return "We don't recognize that email or username"
            else:
                error_password = self.driver.find_element(By.ID, "error-for-password")
                if error_password:
                    return "That's not the right password"
        except:
            return "Sign in successful!"

    def run(self):
        # self.get_url()
        response = self.sign_in_linkedin()
        return response
