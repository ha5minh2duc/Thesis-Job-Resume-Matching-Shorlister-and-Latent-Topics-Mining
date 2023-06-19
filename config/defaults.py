from .config_node import ConfigNode
import os
path_root = os.getcwd()

config = ConfigNode()
# connect database
config.database = ConfigNode()
config.database.username = 'root'
config.database.password = 'Abcde@12345'
config.database.host = 'localhost'
config.database.nameDB = 'mydb'
config.database.port = '3306'

config.openai = ConfigNode()
config.openai.api_key = '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'

config.save_data = ConfigNode()
config.save_data.resume = 'static/input/resume'
config.save_data.job = 'static/input/job'

def get_default_config():
    return config.clone()