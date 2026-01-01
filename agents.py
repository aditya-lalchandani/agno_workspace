from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.arxiv import ArxivTools
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.serpapi import SerpApiTools
from agno.tools.hackernews import HackerNewsTools
from agno.tools.newspaper4k import Newspaper4kTools
from agno.tools.wikipedia import WikipediaTools
from agno.tools.youtube import YouTubeTools
from agno.tools.gmail import GmailTools
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

dir_path = Path("./research_papers/")
dir_path.mkdir(exist_ok=True)


model = OpenAIChat(id="gpt-4.1-mini")

arxiv_research_agent = Agent(
    id="archive-research-agent",
    name="Archive Research Agent",
    model=model,
    role="Arxiv Research Assistant",
    instructions=["You are a research assistant that gathers research papers from Arxiv",
                "Use the available tools to search for research papers, authors and topics as per user's request",
                "Summarize your findings clearly and concisely"],
    tools=[ArxivTools(download_dir=dir_path)],
    add_datetime_to_context=True
)


web_search_agent = Agent(
    id="web-search-agent",
    name="Web Search Agent",
    role="Web Research Assistant",
    model=model,
    instructions=["You are a research assistant that gathers information from the web",
                "Use the available tools to search for articles, summaries and other important material based on the topic the user requested about",
                "Summarize your findings along with the resource links"],
    add_datetime_to_context=True,
    tools=[DuckDuckGoTools(), SerpApiTools()]
)

hackernews_research_agent = Agent(
    id="hackernews-research-agent",
    name="HeckerNews Research Agent",
    model=model,
    role="HackerNews Research Assistant",
    instructions=["You are an expert research assistant that can access HackerNews",
                "Get relevant information for the recent topics, and get information about the articles for the topic user requested for",
                "Summarize your findings in proper format"],
    add_datetime_to_context=True,
    tools=[HackerNewsTools()]
)


news_article_research_agent = Agent(
    id="news-article-research-agent",
    name="News Article Research Agent",
    model=model,
    role="News Article Research Agent",
    instructions=["You are a research assistant that can read the contents of articles",
                "Whenever an url is provided you can read the content of the article and can also get its data",
                "Using the available tools search for articles and summarize them and gather relevant information"],
    add_datetime_to_context=True,
    tools=[Newspaper4kTools(include_summary=True)]
)

wikipedia_research_agent = Agent(
    id="wikipedia-research-agent",
    name="Wikipedia Research Agent",
    role="Wikipedia Research Assistant",
    model=model,
    instructions=["You are a research assistant that gathers information from Wikipedia based on the input topic",
                "You have the capability to search for articles and gather its content",
                "Summarize the findings and mention the appropriate resources and references in your output"],
    add_datetime_to_context=True,
    tools=[WikipediaTools()]
)

youtube_research_agent = Agent(
    id="youtube-research-agent",
    name="Youtube Research Agent",
    model=model,
    role="Youtube Research Assistant",
    instructions=["You are a research assistant that gathers information from Youtube",
                "You have the capability to read youtube video transcripts and summarize them",
                "You can also read metadata related to youtube videos",
                "you can also fetch timestamps of a particular video",
                "summarize the transcripts in clear and concised manner"],
    add_datetime_to_context=True,
    tools=[YouTubeTools()]
)

gmail_agent = Agent(
    id="gmail-agent",
    name="Gmail Agent",
    role="Manages mails through Gmail",
    instructions=["You are a helpful agent that can draft messages using gmail",
                "you have the capability to draft mails, read mails and send mails whenever requested by the user",
                "you have the capability to search for mails and read them",
                "make sure to always confirm the drafted before sending it to the mail id",
                "make sure to write the mails in proper format including a relevant subject line"],
    model=model,
    add_datetime_to_context=True,
    tools=[GmailTools(port=8000)]
)