from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableLambda
import pandas as pd

file_path = "car_kick.csv"  # MA Sample


def create_df(file_path):
    df = pd.read_csv(file_path)
    return df


def suggestions(model, df):

    num_rows = df.shape[0]
    num_columns = df.shape[1]
    first_five = df.head().to_string(index=False)
    df_string = df.to_string(index=False)

    messages = [
        ("system", "You are a data analyst who helps get data insight."),
        (
            "human",
            "Given the csv. There are {num_columns} number of columns and the {num_rows} number of rows. The file path is {file_path}. The data of csv - {df_string} \n.The target columns is the last column. Give me 3 best suggestions of classical machine learning models you think would be best for this data in a json format model:,reason:",
        ),
    ]
    prompt_template = ChatPromptTemplate.from_messages(messages)
    chain = (
        prompt_template
        | model
        | StrOutputParser()
    )
    result = chain.invoke(
        {
            "num_columns": num_columns,
            "num_rows": num_rows,
            "file_path": file_path,
            "first_five": first_five,
            "df_string": "df_string",
        }
    )
    print(result)


df = create_df(file_path)
model = OllamaLLM(model="mistral-nemo")

suggestions(model, df)
