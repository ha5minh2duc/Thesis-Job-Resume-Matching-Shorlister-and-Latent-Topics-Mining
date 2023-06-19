from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import time
from datetime import datetime
from query_database import QueryDatabase

class CrawlInfor():
    def __init__(self, driver,config):
        self.driver = driver
        self.limit = 0
        self.config = config

    def load_data(self, limit):
        page_number = 1
        number_crawl = 0
        number_start = 200
        time.sleep(5)
        while True:
            self.driver.get('https://www.linkedin.com/jobs/search/?currentJobId=3597475148&geoId=104195383&keywords=machine%20learning&start={}'.format(number_start))
            time.sleep(5)
            jobs = self.driver.find_elements(By.CLASS_NAME, 'ember-view.jobs-search-results__list-item.occludable-update.p0.relative.scaffold-layout__list-item')
            page_number += 1
            time.sleep(5)
            for job in jobs:
                job.click()
                time.sleep(3)
                name_job = self.driver.find_element(By.CLASS_NAME, 't-24.t-bold.jobs-unified-top-card__job-title')
                name_job = name_job.text
                context = self.driver.find_element(By.CLASS_NAME, 'jobs-box--fadein.jobs-box--full-width.jobs-box--with-cta-large.jobs-description.jobs-description--reformatted')
                context = context.text
                data = {
                    "name_jobs": name_job,
                    "context": context
                }
                database = QueryDatabase(data, config=self.config)
                database.insert_JD_to_database()
                number_crawl += 1
                if number_crawl == limit:
                    return "Crawl finish"
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")   
            time.sleep(2)
            number_start+= 25
            
