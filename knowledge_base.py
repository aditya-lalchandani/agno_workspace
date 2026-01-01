from agno.knowledge.reader.pdf_reader import PDFReader
from agno.knowledge.chunking.semantic import SemanticChunking
from agno.knowledge.chunking.recursive import RecursiveChunking
from agno.knowledge.embedder.openai import OpenAIEmbedder
from agno.vectordb.lancedb import LanceDb
from agno.knowledge.knowledge import Knowledge
from dotenv import load_dotenv

# load the api key
load_dotenv()

# define the embedder
embedder = OpenAIEmbedder()

# define the chunking strategy
chunking_strategy = RecursiveChunking(chunk_size=1000,
                                    overlap=500)

# create the pdf reader
reader = PDFReader(chunking_strategy=chunking_strategy)

# create the vector db
vector_db = LanceDb(uri="vector_db/lancedb",
                    embedder=embedder,
                    table_name="knowledge_table")


# create a knowledge base
knowledge_base = Knowledge(name="Knowledge_Base",
                        description="Contains the research paper 'Attention is all you need'",
                        vector_db=vector_db)

if __name__ == "__main__":
    knowledge_base.add_content(path="attention_is_all_you_need.pdf",
                            reader=reader)