# Data Insight

Data Insight is a web application designed to help data analysts and scientists generate insights from CSV data files. It uses AI to visualize data, suggest the best machine learning models, and assist in the creation of these models. The app leverages powerful tools like the Deepseek-Coder-V2 model, hosted via Ollama, and integrates with LangChain to help automate data analysis tasks.

## Demo
![Data Insight Demo](assets/d.gif)

## Tech Stack
- **Backend**: Python (Flask)
- **Frontend**: React.js
- **AI Model**: Deepseek-Coder-V2
- **Model Serving**: Ollama
- **Data Processing & Visualization**: Pandas, Matplotlib, Seaborn
- **AI Integration**: LangChain (Python)
- **Machine Learning**: scikit-learn

## Prerequisites
Make sure you have the following installed:

- Python 3.10 or higher
- Node.js (>=14.x)
- npm (>=6.x)

## Installation

### 1. Clone Repositories

#### Backend
```bash
# Clone the backend repository
git clone https://github.com/dedsec995/Data-Insight-Backend.git
cd data-insight-backend
```

#### Frontend
```bash
# Clone the frontend repository
git clone https://github.com/dedsec995/Data-Insight-Frontend.git
cd data-insight-frontend
```

### 2. Set Up Backend

#### Install Python Dependencies
Navigate to the backend directory and install the required Python dependencies:

```bash
cd data-insight-backend
pip install -r requirements.txt
```

#### Install Ollama on Linux
Ollama is used to serve the Deepseek-Coder-V2 model. Run the following command to install Ollama:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Once Ollama is installed, download the Large Language model:

```bash
ollama pull deepseek-coder-v2
```
This will download the required model for use in the application

If you are on any other platform or want more detailed installation instructions, you can visit [Ollama's official installation page.](https://ollama.com/)

#### Run the Backend Server
To start the Flask application, execute:

```bash
python app.py
```

The backend server will start and listen on `http://127.0.0.1:5000` by default.

### 3. Set Up Frontend

Navigate to the `data-insight-frontend` directory:

```bash
cd ../data-insight-frontend
```

#### Install Node.js Dependencies
Install the required Node.js dependencies using npm:

```bash
npm install
```

#### Start the Frontend Server
To run the React application:

```bash
npm start
```

The frontend server will start and listen on `http://localhost:3000` by default.

### 4. Run the Application
Make sure both the backend and frontend servers are running. Open a browser and navigate to `http://localhost:3000` to start using the Data Insight application.

## Screenshots
### Data Visualization
![Data Visualization](assets/graph.png)

### Model
![Result](assets/result.png)


## Usage
- Upload a CSV file for analysis.
- Generate visualizations and insights with the help of AI.
- Get suggestions for the best machine learning models for your data.
- Build and evaluate machine learning models directly in the app.

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request with your improvements or bug fixes.


