import mysql.connector 
from datetime import datetime
import pandas as pd
import yacs.config

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
    
    def get_resumes_by_keyword(self,list_keyword):
        if "," in list_keyword:
            list_keyword = list_keyword.split(",")
        else:
            list_keyword = [list_keyword]
        sql = "select Name, Context from {}.resumes where ".format(self.config.database.nameDB)
        for keyword in list_keyword:
            sql = sql + "context like " + "'%" + keyword + "%' and " 
        sql = sql[:-5]
        self.mycursor.execute(sql)
        results = self.mycursor.fetchall()
        datas = []
        for result in results:
            data = {
                "Name": result[0],
                "Context": result[1]
            }
            datas.append(data)
        df = pd.DataFrame.from_dict(datas, orient="columns")
        return df

    def insert_resume_database(self, path_file_csv):
        data = pd.read_csv(path_file_csv)
        for i in range(len(data["Name"])):
            create_time = self.create_time()
            sql = "insert into {}.resumes (Name, Context, Category, create_time) values (%s,%s,%s,%s)".format(self.config.database.nameDB)
            value = (str(data["Name"][i]), data["Context"][i], "Machine learning", create_time)
            self.mycursor.execute(sql, value)
            self.cnx.commit()

    def insert_resume_it_viec_database(self, path_file_csv):
        data = pd.read_csv(path_file_csv)
        for i in range(len(data["Category"])):
            create_time = self.create_time()
            sql = "insert into {}.resumes (Name, Context, Category, create_time) values (%s,%s,%s,%s)".format(self.config.database.nameDB)
            value = ("IT Viec " + str(i), data["Resume"][i], data["Category"][i], create_time)
            self.mycursor.execute(sql, value)
            self.cnx.commit()

    def insert_job_it_viec_database(self, path_file_csv):
        data = pd.read_csv(path_file_csv)
        for i in range(len(data["job_id"])):
            create_time = self.create_time()
            sql = "insert into {}.jd (Name, Context, create_time) values (%s,%s,%s)".format(self.config.database.nameDB)
            value = (data["job_name"][i], data["description"][i], create_time)
            self.mycursor.execute(sql, value)
            self.cnx.commit()

