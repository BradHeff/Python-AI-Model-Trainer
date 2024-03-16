import os
from openai import OpenAI


def createTrainingJob():
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    client.files.create(file=open("mydata.jsonl", "rb"), purpose="fine-tune")
    client.fine_tuning.jobs.create(
        training_file="file-UO1VpbyFSj8OoEOeUsOoQc4c",
        model="ft:gpt-3.5-turbo-1106:trinity-cloud:hcs-ai:93IYX0Rd",
    )
