# RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that combines vector search with a language model to provide accurate and context-aware responses.

## Features

- **Vector-based Search**: Uses FAISS for efficient similarity search
- **RAG Architecture**: Combines retrieval with language model generation
- **Web Interface**: Simple Flask-based web server
- **Persistence**: Stores Q&A pairs and embeddings for future use
- **Dimensionality Reduction**: Utilizes PCA for efficient vector storage and search

## Prerequisites

- Python 3.7+
- pip (Python package manager)
- Google Gemini API key (set as environment variable `GEMINI_API_KEY`)

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd rag-chatbot
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up your environment variables:
   ```bash
   export GEMINI_API_KEY='your-gemini-api-key-here'
   ```
   (Add this to your shell's rc file for persistence)

## Usage

1. Start the Flask development server:
   ```bash
   python app.py
   ```

2. The server will start on `http://localhost:8000`

### API Endpoints

- `POST /add` - Add new Q&A pairs to the knowledge base
  ```json
  {
    "question": "Your question here",
    "answer": "The answer to store"
  }
  ```

- `POST /chat` - Chat with the RAG system
  ```json
  {
    "message": "Your message to the chatbot"
  }
  ```

## Project Structure

- `app.py` - Main application file containing the Flask server and RAG logic
- `requirements.txt` - Python dependencies
- `data/` - Directory containing the SQLite database and FAISS index
  - `qa_store.db` - SQLite database storing Q&A pairs
  - `faiss.index` - FAISS index for vector search
  - `meta.json` - Metadata about the vector space

## How It Works

1. **Data Ingestion**: Q&A pairs are stored in an SQLite database
2. **Embedding Generation**: Questions are converted to vector embeddings using Google's Gemini API
3. **Indexing**: Vectors are indexed using FAISS for efficient similarity search
4. **Querying**: User queries are embedded and matched against stored vectors
5. **Response Generation**: The most relevant context is used to generate a response

## Dependencies

- Flask - Web framework
- FAISS - Vector similarity search
- NumPy - Numerical computing
- scikit-learn - For PCA (Principal Component Analysis)
- python-dotenv - Environment variable management
- gunicorn - Production WSGI server
- requests - HTTP client

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- Google Gemini for the language model API
- Facebook Research for FAISS
- The open-source community for the various libraries used in this project
