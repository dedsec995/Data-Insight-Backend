from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableLambda
import pandas as pd
import re, os, subprocess

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

def create_python_file(code, main_folder, sub_folder):
    full_path = os.path.join(main_folder, sub_folder)
    os.makedirs(full_path, exist_ok=True)
    filename = os.path.join(full_path, f"{sub_folder}.py")
    with open(filename, "w") as f:
        f.write(code)
    return filename

def run_code(filename):
    try:
        result = subprocess.run(
            ["python", filename], capture_output=True, text=True, check=True
        )
        # print(f"{filename} executed successfully.")
        output = result.stdout.strip()
        file_paths = [
            line.strip() for line in output.split("\n") if line.strip().endswith(".png")
        ]
        return file_paths
    except subprocess.CalledProcessError as e:
        print(f"Error executing {filename}: {e}")
        return None

def extract_paths(data, main_folder, sub_folder):
    paths = []
    for item in data:
        if os.path.join(main_folder, sub_folder) in item:
            # Use os.path.relpath with the parent of main_folder as the start
            path = os.path.relpath(item, os.path.dirname(main_folder))
            # Add 'upload/' to the beginning of the path
            path = os.path.join("uploads", path)
            paths.append(path)
    return list(dict.fromkeys(paths))

def extract_corr_path(result):
    if not result:
        return None
    if isinstance(result, list):
        if len(result) == 1 and isinstance(result[0], str):
            result = result[0]
    match = re.search(r'(uploads/.+)', result)
    if match:
        return match.group(1).strip()
    return result

def generate_visualizations(model, df, file_path, main_folder, visualization_type):
    num_rows, num_columns = df.shape
    first_five = df.head().to_string(index=False)

    if visualization_type == "correlation":
        sub_folder = "corr"
        message = f"Given the csv. There are {num_columns} number of columns and the {num_rows} number of rows. The file path is {file_path}. The head of csv - {first_five} \n.The target columns is the last column. Write a python script for correlation heatmap visualization of numeric values. Only save the graph as corr.png. Make sure to use '{os.path.join(main_folder, 'corr')}/' in the file paths when saving the plot. Don't forget to import. Just print the file path created"
    elif visualization_type == "histogram":
        sub_folder = "hist"
        message = f"Given the csv. There are {num_columns} number of columns and the {num_rows} number of rows. The file path is {file_path}. The head of csv - {first_five} \n.The target columns is the last column. Write a python script to create different histogram graphs of all features you think are important for visualization. Save all the graphs as png files inside the '{os.path.join(main_folder, 'hist')}/' folder. Keep a list of the file paths created and print it. Make sure to use '{os.path.join(main_folder, 'hist')}/' in the file paths when saving the plots."

    messages = [
        ("system", "You are a data analyst who helps get data insight."),
        ("human", message),
    ]
    prompt_template = ChatPromptTemplate.from_messages(messages)
    chain = (
        prompt_template
        | model
        | StrOutputParser()
        | RunnableLambda(lambda x: grab_code(x))
        | RunnableLambda(lambda x: create_python_file(x, main_folder, sub_folder))
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
    if visualization_type == "histogram":
        result = extract_paths(result, main_folder, sub_folder)
    if visualization_type == "correlation":
        result = extract_corr_path(result)
    return result

def combined_visualizations(file_path, main_folder):
    df = create_df(file_path)
    model = OllamaLLM(model="deepseek-coder-v2")
    os.makedirs(main_folder, exist_ok=True)
    correlation_result = generate_visualizations(
        model, df, file_path, main_folder, "correlation"
    )
    histogram_result = generate_visualizations(
        model, df, file_path, main_folder, "histogram"
    )
    return correlation_result, histogram_result

# Usage
# file_path = "car_kick.csv"  # This will be provided by the user
# main_folder = "output_visualizations"  # This will be provided by the user
# corr_result, hist_result = combined_visualizations(file_path, main_folder)
# print("Correlation Heatmap Result:", corr_result)
# print("Histogram Results:", hist_result)
