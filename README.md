# RAG Chat App with Streamlit

This is a Retrieval-Augmented Generation (RAG) chat application built with Streamlit and LangChain. It allows users to upload documents or provide URLs, and then ask questions about the content.

## Features

- Upload text files (.txt), PDFs (.pdf), or Word documents (.docx)
- Input URLs for web content retrieval
- Ask questions about the uploaded or retrieved content
- Utilizes OpenAI's language models for generating responses
- Integrates with LangSmith for monitoring and debugging

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/genius-dev65/ragbot
   cd ragbot
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up your environment variables:
   - Create a `.env` file in the project root
   - Add the following variables:
     ```
     OPENAI_API_KEY=your_openai_api_key
     LANGCHAIN_TRACING_V2=your_langchain_tracing_v2
     LANGCHAIN_ENDPOINT=your_langchain_endpoint
     LANGCHAIN_API_KEY=your_langchain_api_key
     ```

## Usage

Run the Streamlit app locally:
```
streamlit run rag_chat_app.py
```

## LangSmith Integration

This app uses LangSmith for monitoring and debugging LangChain applications. To use LangSmith:

1. Sign up for a LangSmith account at https://smith.langchain.com/
2. Set up your LangSmith API key in the `.env` file
3. Use the LangSmith dashboard to monitor your app's performance and debug issues

## Deployment

You can easily deploy this app on Streamlit Cloud:

1. Fork this repository to your GitHub account
2. Sign up for a Streamlit Cloud account at https://streamlit.io/cloud
3. Create a new app in Streamlit Cloud and connect it to your forked repository
4. Set up the environment variables in the Streamlit Cloud dashboard
5. Deploy and access your app through the provided URL

For more information on deploying Streamlit apps, visit: https://docs.streamlit.io/streamlit-cloud/get-started/deploy-an-app
