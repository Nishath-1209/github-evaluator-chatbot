import pinecone

class PineconeMemory:
    def __init__(self, index_name, api_key):
        pinecone.init(api_key=api_key)
        self.index = pinecone.Index(index_name)

    def store(self, query, response):
        self.index.upsert([{"id": query[:20], "values": [hash(response) % 10000]}])

    def retrieve(self):
        return []