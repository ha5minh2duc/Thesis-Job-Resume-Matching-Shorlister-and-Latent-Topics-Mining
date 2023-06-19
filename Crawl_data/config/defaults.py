from .config_node import ConfigNode

config = ConfigNode()
# connect database
config.database = ConfigNode()
config.database.username = 'root'
config.database.password = 'Abcde@12345'
config.database.host = 'localhost'
config.database.nameDB = 'MyDB'
config.database.port = '3306'
# secret key

config.account = ConfigNode()
config.account.user_name = 'huuphuongtp@gmail.com'
config.account.password = '1234512345'

def get_default_config():
    return config.clone()
