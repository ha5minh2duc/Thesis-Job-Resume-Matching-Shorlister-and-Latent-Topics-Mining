import mysql.connector 
from datetime import datetime


class QueryDatabase():
    def __init__(self, data, config):
        self.data = data
        self.config = config
        self.connect_database()
        self.mycursor = self.cnx.cursor()

    def connect_database(self):
        self.cnx = mysql.connector.connect(
            user = self.config.database.username,
            password = self.config.database.password,
            host = self.config.database.host,
            database = self.config.database.nameDB,
            port = self.config.database.port
        )

    def create_time(self):
        now = datetime.now()
        formatted_date = now.strftime('%Y-%m-%d %H:%M:%S')
        return formatted_date

    def insert_JD_to_database(self):
        create_time = self.create_time()
        sql = "insert into {}.jd (name_jobs, context, create_time) values (%s,%s,%s)".format(self.config.database.nameDB)
        value = (self.data["name_jobs"], self.data["context"], create_time)
        self.mycursor.execute(sql, value)
        self.cnx.commit()
        