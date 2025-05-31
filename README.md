# PDF Chatbot with Gemini

A Streamlit application that allows you to chat with your PDF documents using Google's Gemini AI model.

## Features

- Upload and process multiple PDF documents
- Interactive chat interface
- Real-time progress tracking during PDF processing
- Powered by Google's Gemini 1.5 Pro model
- Maintains conversation history

## Setup

1. Clone the repository:
```bash
git clone <your-repo-url>
cd <repo-name>
```

2. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root and add your Google API key:
```
GOOGLE_API_KEY=your_api_key_here
```

5. Run the application:
```bash
streamlit run app.py
```

## Requirements

- Python 3.8+
- Google API key with access to Gemini models
- Required Python packages (see requirements.txt)

## Usage

1. Open the application in your browser (default: http://localhost:8501)
2. Upload one or more PDF documents using the sidebar
3. Click "Process PDFs" to analyze the documents
4. Start chatting with your documents!

## Note

Make sure to keep your Google API key secure and never commit it to version control.
