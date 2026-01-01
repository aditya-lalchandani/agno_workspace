from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.os import AgentOS
from agno.team import Team
from agents import arxiv_research_agent, web_search_agent, hackernews_research_agent,news_article_research_agent,wikipedia_research_agent,youtube_research_agent,gmail_agent
from agno.db.in_memory import InMemoryDb
from agno.tools.local_file_system import LocalFileSystemTools
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

dir_path = Path("./research_papers/")
dir_path.mkdir(exist_ok=True)

target_dir = Path("./medium_articles/")
target_dir.mkdir(exist_ok=True)

orchestrator_model = OpenAIChat(id="gpt-4.1")

db = InMemoryDb()

medium_article_team = Team(
    id="medium-article-creation-team",
    name="Medium Article Creation Team",
    role="Team Leader which manages research and content creation",
    db=db,
    members=[arxiv_research_agent,
            web_search_agent,
            hackernews_research_agent,
            news_article_research_agent,
            wikipedia_research_agent,
            youtube_research_agent,
            gmail_agent],
    model=orchestrator_model,
    instructions=["You are a team leader managing multiple sub agents in your team",
                "You have access to agents which can do research based on the topic on various sources such as arxiv, X(formerly twitter), youtube, reddit, wikipedia, hackernews, newspaper articles, web search using google search and duckduckgo search",
                "you also have the capability to read and write emails",
                "your task is to understand the topic given by the user and fetch relevant research information using your team members",
                "once you have enough research material, your primary task is to create medium(platform) styled articles",
                "for checking how articles are written on medium you can use the article research agent",
                "once you have created the medium article, show the user the final draft",
                "only when the user confirms the draft, save it to the filesystem in a markdown format as a .md file using the filename suggested by user",
                "if the user does not give a filename, then use a self created name based on the topic on which the article was created"],
    add_datetime_to_context=True,
    add_history_to_context=True,
    num_history_runs=10,
    tools=[LocalFileSystemTools(target_directory=target_dir,
                                default_extension="md")],
    stream=True,
    markdown=True
)


agent_os = AgentOS(
    id="medium-article",
    name="Medium Article Generator OS",
    description="An Agent that conducts research of latest tech topics across multiple platforms and generates medium articles based on its findings",
    teams=[medium_article_team]
)

app = agent_os.get_app()

if __name__ == "__main__":
    agent_os.serve(
        app="app:app",
        reload=True
    )