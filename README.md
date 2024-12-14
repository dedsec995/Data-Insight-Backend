# Data Insight

Data Insight is a web application designed to help data analysts and scientists generate insights from CSV data files. It uses AI to visualize data, suggest the best machine learning models, and assist in the creation of these models. The app leverages powerful tools like the Deepseek-Coder-V2 model, hosted via Ollama, and integrates with LangChain to help automate data analysis tasks.

### Tech Stack
- Backend: Python (Flask)
- Frontend: React.js
- AI Model: Deepseek-Coder-V2
- Model Serving: Ollama
- Data Processing & Visualization: Pandas, Matplotlib, Seaborn
- AI Integration: LangChain (Python)
- Machine Learning: scikit-learn

### Installation

#### Prerequisites
Make sure you have the following installed:

- Python 3.10 or higher
- Node.js (>=14.x)
- npm (>=6.x)

#### Install Ollama on Linux
Ollama is used to serve the Deepseek-Coder-V2 model. Paste the command to install Ollama and download the model:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```
Once ollama is installed, you need to download the Large Language model:

```bash
ollama pull deepseek-coder-v2
```
This will download the required model for use in the application

If you are on any other platform or want more detailed installation instructions, you can visit [Ollama's official installation page.](https://ollama.com/)

