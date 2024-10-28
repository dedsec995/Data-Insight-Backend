from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableLambda
import pandas as pd
import re, os, subprocess

file_path = "car_kick.csv"  # Take from user


def create_df(file_path):
    df = pd.read_csv(file_path)
    return df


def grab_code(result):
    code_pattern = r"```python(.*?)```"
    code_block = re.search(code_pattern, result, re.DOTALL)

    if code_block:
        code = code_block.group(1).strip()
        return code
    else:
        return "No code"


def create_python_file(code):
    os.makedirs("model", exist_ok=True)
    filename = "model/model.py"
    with open(filename, "w") as f:
        f.write(code)
    return filename


def run_code(filename):
    result = ""
    try:
        result = subprocess.run(
            ["python", filename], capture_output=True, text=True, check=True
        )
        if result.returncode != 0:
            print(f"Error: {result.stderr}")
            return None
        output = result.stdout.strip()
        print(f"{filename} executed successfully.")
        file_paths = [
            line.strip() for line in output.split("\n") if line.strip().endswith(".png")
        ]
        return file_paths

    except subprocess.CalledProcessError as e:
        print(f"Error executing {filename}: {e}")
        return None


def classical_ml_model(model, df):

    num_rows = df.shape[0]
    num_columns = df.shape[1]
    first_five = df.head().to_string(index=False)
    df_string = df.to_string(index=False)
    recommended_model = "logistic regression"

    messages = [
        ("system", "You are a data analyst who helps get data insight."),
        (
            "human",
            "Given the csv. There are {num_columns} number of columns and the {num_rows} number of rows. The file path is {file_path}. The head of csv - {first_five} \n.The target columns is the last column. Write a python script to create a {recommended_model} machine learning model using scikit-learnk. There may be textual or categorical data, encode it accordinaly. Train it on the file. Save all the graphs like confusion matrix as png files inside the 'model/' folder. Keep a list of the file paths created and print it. Make sure to use 'model/' in the file paths when saving the plots. Save the model in 'model/' folder Also print the saved model path",
        ),
    ]
    prompt_template = ChatPromptTemplate.from_messages(messages)
    # print(prompt_template.format_prompt(num_columns=num_columns,num_rows=num_rows,file_path=file_path,first_five=first_five,df_string=df_string,recommended_model=recommended_model))
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
            "recommended_model": recommended_model,
        }
    )
    print(result)


def main():
    df = create_df(file_path)
    model = OllamaLLM(model="deepseek-coder-v2")
    classical_ml_model(model, df)


if __name__ == "__main__":
    main()
