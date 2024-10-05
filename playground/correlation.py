from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableLambda
import pandas as pd
import re, os, subprocess

file_path = 'car_kick.csv' # Take from user

def create_df(file_path):
    df = pd.read_csv(file_path)
    return df

def grab_code(result):
    code_pattern = r'```python(.*?)```'
    code_block = re.search(code_pattern, result, re.DOTALL)

    if code_block:
        code = code_block.group(1).strip()
        return code
    else:
        return "No code"

def create_python_file(code):
    os.makedirs("corr", exist_ok=True)
    filename = "corr/corr.py"
    with open(filename, 'w') as f:
        f.write(code)
    return filename

def run_code(filename):
    result = ""
    try:
        result = subprocess.run(
            ["python", filename], capture_output=True, text=True, check=True
        )
        print(f"{filename} executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error executing {filename}: {e}")


def correlation_heatmap(model,df):

    num_rows = df.shape[0]
    num_columns = df.shape[1]
    first_five = df.head().to_string(index = False)
    df_string = df.to_string(index=False)

    messages = [
        ("system", "You are a data analyst who helps get data insight."),
        (
            "human",
            "Given the csv. There are {num_columns} number of columns and the {num_rows} number of rows. The file path is {file_path}. The head of csv - {first_five} \n.The target columns is the last column. Write a python file for correlation heatmap visualization of numeric values. Save the graph as corr.png. Make sure to use 'corr/' in the file paths when saving the plot. Don't forget to import numpy",
        ),
    ]
    prompt_template = ChatPromptTemplate.from_messages(messages)
    chain = (
        prompt_template
        | model
        | StrOutputParser()
        | RunnableLambda(lambda x: grab_code(x))
        | RunnableLambda(lambda x: create_python_file(x))
        | RunnableLambda(lambda x: run_code(x))
    )
    result = chain.invoke(
        {
            "num_columns": num_columns,
            "num_rows": num_rows,
            "file_path": file_path,
            "first_five": first_five,
        }
    )

df = create_df(file_path)
model = OllamaLLM(model="mistral-nemo")

correlation_heatmap(model,df)
