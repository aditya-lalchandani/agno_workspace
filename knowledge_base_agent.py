from agno.agent import Agent
from agno.models.openai import OpenAIChat
from knowledge_base import knowledge_base
from dotenv import load_dotenv

# load the api keys
load_dotenv()

# create the model
llm = OpenAIChat(id="gpt-4.1-mini")

# create the agent
agent = Agent(
    model=llm,
    knowledge=knowledge_base,
    name="Knowledge_Agent",
    search_knowledge=True,
    instructions=["you are a helpful assistant",
                "whenever asked about transformer model, use the knowledge base to get the context",
                "try not to hallucinate while responding",
                "if you don't know the answer to a particular query just say I dont know"],
    stream=True,
    markdown=True
)

agent.print_response("Hi how are you. Can you tell me the capital of India")

agent.print_response("Can you tell me what hardware was used for training the transformer model?")