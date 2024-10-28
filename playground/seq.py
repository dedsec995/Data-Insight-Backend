from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableLambda
import pandas as pd
import re, os, subprocess

# Global Var
file_path = "car_kick.csv"


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


def create_python_file(code, main_folder, file_name):
    os.makedirs(main_folder, exist_ok=True)
    filename = os.path.join(main_folder, f"{file_name}.py")
    with open(filename, "w") as f:
        f.write(code)
    return filename


def run_code(filename):
    try:
        result = subprocess.run(
            ["python", filename], capture_output=True, text=True, check=True
        )
        print(f"{filename} executed successfully.")
        output = result.stdout.strip()
        print(output)
    except subprocess.CalledProcessError as e:
        print(f"Error executing {filename}: {e}")
        return None


def classical_ml_model(model, df):
    num_rows = df.shape[0]
    num_columns = df.shape[1]
    first_five = df.head().to_csv(index=False,sep=",")
    recommended_model = "decision tree"

    messages_1 = [
        ("system", "You are a data analyst who helps get data insight."),
        (
            "human",
            "Given the csv. There are {num_columns} number of columns and the {num_rows} number of rows. The file path is {file_path}. The head of csv - {first_five} \n. The target columns is the last column. Write a python script to create a {recommended_model} machine learning model using scikit-learn. There may be textual or categorical data, encode it accordinaly. Preprocess and split the data into 80 20 ratio. ",
        ),
    ]
    prompt_1 = ChatPromptTemplate.from_messages(messages_1)
    # print(prompt_1.format_prompt(num_columns=num_columns,num_rows=num_rows,file_path=file_path,first_five=first_five,df_string=df_string,recommended_model=recommended_model))
    chain_1 = (
        prompt_1
        | model
        | StrOutputParser()
        | RunnableLambda(lambda x: grab_code(x))
    )
    code = chain_1.invoke(
        {
            "num_columns": num_columns,
            "num_rows": num_rows,
            "file_path": file_path,
            "first_five": first_five,
            "recommended_model": recommended_model,
        }
    )
    # print(code)
    # print("---"*50)
    messages_2 = [
        ("system", "You are a data analyst who helps get data insight."),
        (
            "human",
            "Here is the {code} \n I have preprocessed the data. Now Train the model on the data. Save the model trained in 'model/' folder. Save confusion matrix as png files inside the 'model/' folder.",
        ),
    ]
    prompt_2 = ChatPromptTemplate.from_messages(messages_2)
    # print(prompt_2.format_prompt(code=code,first_five=first_five))
    chain_2 = (
        prompt_2
        | model
        | StrOutputParser()
    )

    train_code = chain_2.invoke(
        {
            "code" : code,
        }
    )
    messages_3 = [
        ("system", "You are a data analyst who helps get data insight."),
        (
            "human",
            "Here is the {train_code} \n I want result like accuracy, recoil using classification report. Json format {{'model_path':'model_path','conf_path':'conf_path','result'['recoil':'recoil',:'precision':'precision,'f1':'f1','support':'support']}} Just the json to be printed.",
        ),
    ]
    prompt_3 = ChatPromptTemplate.from_messages(messages_3)
    chain_3 = (
        prompt_3
        | model
        | StrOutputParser()
        | RunnableLambda(lambda x: grab_code(x))
        | RunnableLambda(lambda x: create_python_file(x, "model/", "model"))
        | RunnableLambda(lambda x: run_code(x))
    )
    final_code = chain_3.invoke(
        {
            "train_code": train_code,
        }
    )
    print(final_code)
    print("---" * 50)

def main():
    df = create_df(file_path)
    model = OllamaLLM(model="deepseek-coder-v2")
    classical_ml_model(model, df)

if __name__ == "__main__":
    main()
