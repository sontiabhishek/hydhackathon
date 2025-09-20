# Text Document Summarizer

A simple Streamlit app for summarizing text documents.

It uses Streamlit for the UI and libraries like NLTK and Scikit-learn for text summarization.

## Features

- Upload up to 3 documents in `.docx`, `.txt`, or `.pdf` format.
- Upload `.zip` archives containing supported document types.
- Generate a concise summary using the TextRank algorithm.
- Interactively adjust the summary length.
- Choose to download summaries as either PDF or .txt files.
- Track usage with a session-based counter and billing summary.
- Generate and save a comprehensive report for all uploaded documents.

## Developer Setup Guide

**Critical Note:** Following these steps precisely is essential to prevent `ModuleNotFoundError`. This setup creates an isolated environment and installs all dependencies, including `scikit-learn`, which provides the `sklearn` module.

### Prerequisites

- Git
- Python 3.7+
- pip

### Step-by-Step Environment Setup

1.  **Clone the Repository** (if you haven't already)
    ```bash
    git clone <your-repository-url>
    cd smart-doc-checker
    ```

2.  **Make the Setup Script Executable (for macOS/Linux only)**

    Before running the setup script, you must make it executable:
    ```bash
    chmod +x setup.sh
    ```

3.  **Run the Automated Setup Script**

    This script will create a virtual environment and install all required libraries from `requirements.txt`.
    ```bash
    # For macOS and Linux
    ./setup.sh
    # For Windows
    setup.bat
    ```

4.  **Activate the Virtual Environment**

    After the setup script completes, you must activate the environment for your current terminal session.
    ```bash
    # For macOS and Linux
    source .venv/bin/activate
    # For Windows
    .venv\Scripts\activate
    ```
    You will know the environment is active because your terminal prompt will change to show a `(.venv)` prefix.

5.  **Launch the Application**

    With the virtual environment active, use the `launch.py` script to start the app. This script first verifies that all dependencies are installed correctly before starting the Streamlit server.
    ```bash
    python launch.py
    ```
    After running this command, your terminal will display status messages. **These are not new commands to run.** Simply wait, and your web browser will automatically open with the running application.

### Troubleshooting

**`ModuleNotFoundError` (e.g., "No module named 'sklearn' or 'fpdf'")**

This error means a required Python library is not installed in your active environment.
1.  **Check if your virtual environment is active.** Your terminal prompt must show `(.venv)`. If it doesn't, run the activation command from Step 3 again.
2.  **Manually install all dependencies.** The most reliable way to fix missing library errors is to run the installation command again within your active virtual environment. This ensures all packages, including their correct versions, are installed.
    ```bash
    pip install -r requirements.txt
    ```
3.  **Restart the app.** After the installation completes, stop the app if it's running (`Ctrl+C` in the terminal) and restart it using `python launch.py`.

**`command not found` (e.g., "bash: git: command not found" or "bash: python3: command not found")**

This error means a required program is not installed on your system or is not available in your system's `PATH`.
- **`git: command not found`**: You need to install Git. You can download it from the official Git website.
- **`python3: command not found`**: You need to install Python. You can download it from the official Python website. Make sure to check the box to "Add Python to PATH" during installation on Windows.
- **`./setup.sh: command not found`**: This can happen if you are not in the project's root directory. Use the `ls` command to check for the `setup.sh` file.

## Directory Structure

```
Smart Doc Checker/
├── app.py
├── README.md
├── requirements.txt
│   Includes the libraries: streamlit, textblob, python-docx
├── sample_docs/
│   Includes sample documents

├── reports/
└── sample_docs/
```

## TODO

- Add more file type support