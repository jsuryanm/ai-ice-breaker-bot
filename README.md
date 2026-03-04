# LinkedIn Icebreaker Bot

An **AI-powered LinkedIn profile analyzer and conversational assistant** built using **Retrieval-Augmented Generation (RAG)**.
The application extracts LinkedIn profile data, converts it into embeddings, and uses a Large Language Model (LLM) to generate insights and answer questions about a person's career.

The project demonstrates how to build a **RAG pipeline using LlamaIndex, Groq LLMs, and vector embeddings**, along with an interactive **Gradio web interface**.

---

# Features

* Extract LinkedIn profile data using **ProxyCurl API** or mock data (we will be using mock data in this project)
* Convert profile information into **semantic embeddings**
* Store embeddings in a **vector index**
* Generate **interesting facts** about the person
* Ask **follow-up questions about the profile**
* Interactive **Gradio UI**
* Structured output using **Pydantic**
* Configurable **LLM models**

---

# Architecture

The system follows a **Retrieval-Augmented Generation (RAG)** pipeline.

```
LinkedIn Profile
        │
        ▼
Data Extraction (ProxyCurl / Mock Data)
        │
        ▼
Document Chunking
        │
        ▼
Embedding Model
        │
        ▼
Vector Store Index
        │
        ▼
Retriever
        │
        ▼
LLM (Groq)
        │
        ▼
Generated Insights & Answers
```

---

# Tech Stack

**Frameworks**

* LlamaIndex

**LLM Inference**

* Groq

**Embedding Model**

* BAAI/bge-small-en-v1.5

**UI**

* Gradio

**Validation**

* Pydantic

**Other Tools**

* Python
* Requests
* dotenv

---

# Project Structure

```
icebreaker/
│
├── modules/
│   ├── app.py                # Gradio web interface
│   ├── config.py             # Configuration settings
│   ├── data_extraction.py    # LinkedIn profile extraction
│   ├── data_processing.py    # Chunking + vector index creation
│   ├── llm_interface.py      # LLM initialization
│   ├── query_engine.py       # RAG pipeline logic
│   └── output_schema.py      # Structured output schema
│
├── main.py                   # CLI version of the application
│
└── README.md
```

---

# Installation

## 1. Clone the repository

```
git clone https://github.com/yourusername/ai-ice-breaker-bot.git
cd ai-ice-breaker-bot
```

---

## 2. Create a virtual environment

```
pip install uv 
uv venv .venv --python 3.12.12
uv init
```

Activate it:

### Windows

```
.venv\Scripts\activate
```

### Mac / Linux

```
source .venv/bin/activate
```

---

## 3. Install dependencies

```
uv add -r requirements.txt
```

Example dependencies:

```
llama-index
llama-index-llms-groq
sentence-transformers
gradio
pydantic
python-dotenv
requests
```

---

# Environment Setup

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_api_key
PROXYCURL_API_KEY=your_proxycurl_api_key 
```

You can also run the application using **mock LinkedIn data** if you don't have a ProxyCurl key.
Note: In this project I had used the mock data.
---

# Running the Application

Run the Gradio interface:

```
python -m icebreaker.modules.app
```

The application will start locally:

```
http://127.0.0.1:7860
```

Gradio may also generate a **temporary public share link**.

---

# Usage

### Step 1: Process LinkedIn Profile

Enter a LinkedIn profile URL and click **Process Profile**.

The system will:

1. Extract LinkedIn profile data
2. Split data into chunks
3. Generate embeddings
4. Store them in a vector index
5. Generate **3 interesting facts**

---

### Step 2: Chat with the Profile

You can ask questions such as:

```
What is this person's current job title?
Where did they study?
What companies have they worked at?
What technologies do they specialize in?
```

The system retrieves relevant context from the profile and generates answers using RAG.

---

# Example Output

```
Here are 3 interesting facts about this person:

• Worked nearly three decades at IBM contributing to database systems.

• Delivered a keynote on AI at the Digital CPA conference.

• Holds an Honours BSc in Computer Science from York University.
```

---

# Key Concepts Demonstrated

### Retrieval-Augmented Generation (RAG)

Instead of relying solely on the LLM's training data, the system retrieves relevant profile information and uses it as context for generating answers.

### Semantic Search

Embeddings allow the system to search for meaning rather than exact keywords.

### Structured Output

Pydantic ensures the LLM returns structured responses that can be parsed safely.

---

# Future Improvements

* Persistent vector database (Chroma / Qdrant)
* Streaming responses
* Conversation memory
* FastAPI backend
* Docker deployment
* Profile caching
* Multi-profile comparison

---

# License

MIT License
