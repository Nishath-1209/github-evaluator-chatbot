from agno.llms import GeminiLLM
from agno.embeddings import GeminiEmbeddings
from memory.pinecone_memory import PineconeMemory

llm = GeminiLLM(api_key="your-gemini-api-key")
embeddings = GeminiEmbeddings(api_key="your-gemini-api-key")
memory = PineconeMemory(index_name="chat-memory", api_key="your-pinecone-api-key")