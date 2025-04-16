# main.py
import os
from agno.llms import GeminiLLM
from agno.embeddings import GeminiEmbeddings
from memory.pinecone_memory import PineconeMemory
import openai
import pinecone

# Set up your environment variables for API keys
GEMINI_API_KEY = "your-gemini-api-key"
PINECONE_API_KEY = "your-pinecone-api-key"
PINECONE_ENV = "your-pinecone-environment"  # e.g., "us-west1-gcp"

# Initialize Gemini LLM (Large Language Model)
llm = GeminiLLM(api_key=GEMINI_API_KEY)

# Initialize Gemini Embeddings (for text embeddings)
embeddings = GeminiEmbeddings(api_key=GEMINI_API_KEY)

# Initialize Pinecone Memory (for storing and recalling data)
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
memory = PineconeMemory(index_name="chat-memory", api_key=PINECONE_API_KEY)

def create_gemini_embeddings(text):
    """
    Function to create embeddings from text using Gemini model.
    """
    try:
        embedding = embeddings.embed(text)
        return embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

def store_in_pinecone(text, metadata):
    """
    Function to store the text and metadata in Pinecone memory.
    """
    try:
        # Generate embeddings for the text
        embedding = create_gemini_embeddings(text)
        if embedding is not None:
            # Store the data in Pinecone memory
            memory.store(text, embedding, metadata)
            print("Data stored successfully in Pinecone.")
        else:
            print("Failed to create embedding. Data not stored.")
    except Exception as e:
        print(f"Error storing data in Pinecone: {e}")

def retrieve_from_pinecone(query):
    """
    Function to retrieve similar text from Pinecone memory using a query.
    """
    try:
        # Generate embeddings for the query
        embedding = create_gemini_embeddings(query)
        if embedding is not None:
            # Retrieve from Pinecone memory
            results = memory.retrieve(query, embedding)
            print("Retrieval results:", results)
            return results
        else:
            print("Failed to create embedding for the query.")
            return []
    except Exception as e:
        print(f"Error retrieving data from Pinecone: {e}")
        return []

def evaluate_project(project_description):
    """
    Function to evaluate a GitHub project by processing its description.
    You can customize this function as needed for project evaluation.
    """
    # Example: Use Gemini model for evaluation
    try:
        evaluation_result = llm.ask(f"Evaluate the following project description:\n{project_description}")
        print("Evaluation Result:", evaluation_result)
        return evaluation_result
    except Exception as e:
        print(f"Error during project evaluation: {e}")
        return None

def main():
    """
    Main entry point for the backend logic.
    """
    print("Initializing GitHub Project Evaluator Chatbot...")
    
    # Sample project description for evaluation
    project_description = """
    This is an open-source project to help developers quickly evaluate the performance of their code repositories.
    The project uses GitHub API for fetching details, Agno for NLP tasks, and Pinecone for memory storage.
    """

    # Evaluate project
    evaluation = evaluate_project(project_description)
    if evaluation:
        print(f"Evaluation: {evaluation}")
    
    # Store the project description and evaluation in Pinecone
    metadata = {"project_name": "GitHub Evaluator", "status": "active"}
    store_in_pinecone(project_description, metadata)
    
    # Query Pinecone memory for similar projects (example query)
    query = "Evaluate GitHub repositories"
    similar_projects = retrieve_from_pinecone(query)

if __name__ == "__main__":
    main()
