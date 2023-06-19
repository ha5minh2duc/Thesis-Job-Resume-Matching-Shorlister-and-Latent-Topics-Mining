from spacy.lang.en import English
import pandas as pd

DEGREES_IMPORTANCE = {'high school': 0, 'associate': 1, 'BS-LEVEL': 2, 'MS-LEVEL': 3, 'PHD-LEVEL': 4}
ENTITIES = ['BS-LEVEL', 'MS-LEVEL', 'PHD-LEVEL', 'DEV', 'AI', 'CODING', 'DATA SCIENCES', 'AUTOMATION', 'BIG DATA',
            'WEB-DEVELOPMENT', 'MOBILE-DEVELOPMENT']
class JobInfoExtraction:

    def __init__(self, skills_patterns_path, majors_patterns_path, degrees_patterns_path, jobs):
        # self.jobs = jobs[['Qualifications']]
        self.skills_patterns_path = skills_patterns_path
        self.majors_patterns_path = majors_patterns_path
        self.degrees_patterns_path = degrees_patterns_path
        self.degrees_importance = DEGREES_IMPORTANCE

    @staticmethod
    def match_majors_by_spacy(self, job):
        nlp = English()
        # Add the pattern to the matcher
        patterns_path = self.majors_patterns_path
        ruler = nlp.add_pipe("entity_ruler")
        ruler.from_disk(patterns_path)
        # Process some text
        doc1 = nlp(job)
        acceptable_majors = []
        for ent in doc1.ents:
            labels_parts = ent.label_.split('|')
            if labels_parts[0] == 'MAJOR':
                if labels_parts[2].replace('-', ' ') not in acceptable_majors:
                    acceptable_majors.append(labels_parts[2].replace('-', ' '))
                if labels_parts[2].replace('-', ' ') not in acceptable_majors:
                    acceptable_majors.append(labels_parts[2].replace('-', ' '))
        return acceptable_majors

    @staticmethod
    def match_degrees_by_spacy(self, job):
        nlp = English()
        # Add the pattern to the matcher
        patterns_path = self.degrees_patterns_path
        ruler = nlp.add_pipe("entity_ruler")
        ruler.from_disk(patterns_path)
        # Process some text
        doc1 = nlp(job)
        degree_levels = []
        for ent in doc1.ents:
            labels_parts = ent.label_.split('|')
            if labels_parts[0] == 'DEGREE':
                # print((ent.text, ent.label_))
                if labels_parts[1] not in degree_levels:
                    degree_levels.append(labels_parts[1])
        return degree_levels

    @staticmethod
    def match_skills_by_spacy(self, job):
        nlp = English()
        patterns_path = self.skills_patterns_path
        ruler = nlp.add_pipe("entity_ruler")
        ruler.from_disk(patterns_path)
        # Process some text
        doc1 = nlp(job)
        job_skills = []
        for ent in doc1.ents:
            labels_parts = ent.label_.split('|')
            if labels_parts[0] == 'SKILL':
                # print((ent.text, ent.label_))
                if labels_parts[1].replace('-', ' ') not in job_skills:
                    job_skills.append(labels_parts[1].replace('-', ' '))
        return job_skills

    @staticmethod
    def get_minimum_degree(self, degrees):
        """get the minimum degree that the candidate has"""
        d = {degree: self.degrees_importance[degree] for degree in degrees}
        return min(d, key=d.get)

    def extract_entities(self, jobs):
        # recognize and extract entities
        jobs['Minimum degree level'] = ""
        jobs['Acceptable majors'] = ""
        jobs['Skills'] = ""
        for i, row in jobs.iterrows():
            job = jobs['Context'][i].replace('. ', ' ')
            degrees = self.match_degrees_by_spacy(self, job)
            if len(degrees) != 0:
                jobs['Minimum degree level'][i] = self.get_minimum_degree(self, degrees)
            else:
                jobs['Minimum degree level'][i] = ""
            jobs['Acceptable majors'][i] = self.match_majors_by_spacy(self, job)
            jobs['Skills'][i] = self.match_skills_by_spacy(self, job)
        return jobs
    
def main():
    degrees_patterns_path = 'Resources/data/degrees.jsonl'
    majors_patterns_path = 'Resources/data/majors.jsonl'
    skills_patterns_path = 'Resources/data/skills.jsonl'
    # jobs = pd.read_csv(job_descriptions, index_col=0)
    jobs = pd.read_csv('/home/phuonghuu/Phuong_workspace/Naive-Resume-Matching/Job_Data.csv')
    jobs.set_index('Name', inplace = True)
    job_extraction = JobInfoExtraction(skills_patterns_path, majors_patterns_path, degrees_patterns_path, jobs)
    jobs = job_extraction.extract_entities(jobs)
    print(jobs)

main()
