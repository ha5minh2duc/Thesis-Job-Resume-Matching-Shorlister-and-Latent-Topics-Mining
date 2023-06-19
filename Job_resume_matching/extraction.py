import pandas as pd
import json
from Job_resume_matching.JobInfoExtraction import JobInfoExtraction
# from JobInfoExtraction import JobInfoExtraction


def transform_dataframe_to_json(dataframe):

    # transforms the dataframe into json
    result = dataframe.to_json(orient="records")
    parsed = json.loads(result)
    json_data = json.dumps(parsed, indent=4)

    return json_data

def extraction(job_descriptions, index):
    degrees_patterns_path = 'Resources/data/degrees.jsonl'
    majors_patterns_path = 'Resources/data/majors.jsonl'
    skills_patterns_path = 'Resources/data/skills.jsonl'
    # jobs = pd.read_csv(job_descriptions, index_col=0)
    jobs = job_descriptions
    jobs.set_index('Name', inplace = True)
    job_extraction = JobInfoExtraction(skills_patterns_path, majors_patterns_path, degrees_patterns_path, jobs)
    jobs = job_extraction.extract_entities(jobs)
    # jobs_json = transform_dataframe_to_json(jobs)
    # print(type(jobs_json))
    return jobs.iloc[[index]]

def main():
    extraction('/home/phuonghuu/Phuong_workspace/Naive-Resume-Matching/Demo/Job_Data.csv',1)

if __name__ == "__main__":
    main()