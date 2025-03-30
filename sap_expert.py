from __future__ import annotations as _annotations

from dataclasses import dataclass
from dotenv import load_dotenv
import logfire
import asyncio
import httpx
import os
import json
import re
from bs4 import BeautifulSoup
from urllib.parse import quote_plus

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.openai import OpenAIModel
from openai import AsyncOpenAI
from supabase import Client
from typing import List, Dict, Any

load_dotenv()

llm = os.getenv('LLM_MODEL', 'gpt-4o-mini')
# Initialize OpenAI model
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_API_BASE_URL"))

model = OpenAIModel(
    llm,
    openai_client=client
)

logfire.configure(send_to_logfire='if-token-present')

@dataclass
class PydanticAIDeps:
    supabase: Client
    openai_client: AsyncOpenAI
    ollama_client: AsyncClient

system_prompt = """
You are an expert in SAP Technologies and you have access to all the documentation to,
including examples, an API reference, and other resources to help you answer SAP related questions.

You have the following capabilities:
1. Access to stored SAP documentation through RAG and direct page retrieval
2. Ability to search the web using SearXNG for the latest information
3. Ability to fetch and analyze content from web pages

Your only job is to assist with SAP-related questions and you don't answer other questions besides describing what you are able to do.

When answering a question:
1. First check the stored documentation using RAG
2. If needed, search for specific pages in the documentation
3. If the answer isn't found or might be outdated or you need more information, use web search for the latest information
4. Fetch specific web pages when you need more detailed information

If necessary, you can repeat the steps up to 10 times to find the best answer.

Don't ask the user before taking an action, just do it. Be honest when you can't find an answer.
"""

sap_expert = Agent(
    model,
    system_prompt=system_prompt,
    deps_type=PydanticAIDeps,
    retries=2
)

async def get_embedding(text: str, openai_client: AsyncOpenAI, ollama_client:AsyncClient) -> List[float]:
    """Get embedding vector from OpenAI."""
    try:
        # response = await ollama_client.embeddings.create(
        #     model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        #     input=text
        # )
        # return response.data[0].embedding
        response = await ollama_client.embed(
            model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
            input=text
        )
        return response.embeddings[0]
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return [0] * 1536  # Return zero vector on error

@sap_expert.tool
async def retrieve_relevant_documentation(ctx: RunContext[PydanticAIDeps], user_query: str) -> str:
    """
    Retrieve relevant documentation chunks based on the query with RAG.
    
    Args:
        ctx: The context including the Supabase client and OpenAI client
        user_query: The user's question or query
        
    Returns:
        A formatted string containing the top 5 most relevant documentation chunks
    """
    try:
        # Get the embedding for the query
        query_embedding = await get_embedding(user_query, ctx.deps.openai_client, ctx.deps.ollama_client)
        
        # Query Supabase for relevant documents
        result = ctx.deps.supabase.rpc(
            'match_site_pages',
            {
                'query_embedding': query_embedding,
                'match_count': 5,
                'filter': {'source': 'sap_docs'}
            }
        ).execute()
        
        if not result.data:
            return "No relevant documentation found."
            
        # Format the results
        formatted_chunks = []
        for doc in result.data:
            print("retrieved: ", doc)
            chunk_text = f"""
# {doc['title']}

{doc['content']}
"""
            formatted_chunks.append(chunk_text)
            
        # Join all chunks with a separator
        return "\n\n---\n\n".join(formatted_chunks)
        
    except Exception as e:
        print(f"Error retrieving documentation: {e}")
        return f"Error retrieving documentation: {str(e)}"

@sap_expert.tool
async def list_documentation_pages(ctx: RunContext[PydanticAIDeps]) -> List[str]:
    """
    Retrieve a list of all available SAP documentation pages.
    
    Returns:
        List[str]: List of unique URLs for all documentation pages
    """
    try:
        # Query Supabase for unique URLs where source is sap_docs
        result = ctx.deps.supabase.from_('site_pages') \
            .select('url') \
            .eq('metadata->>source', 'sap_docs') \
            .execute()
        
        if not result.data:
            return []
            
        # Extract unique URLs
        urls = sorted(set(doc['url'] for doc in result.data))
        return urls
        
    except Exception as e:
        print(f"Error retrieving documentation pages: {e}")
        return []

@sap_expert.tool
async def get_page_content(ctx: RunContext[PydanticAIDeps], url: str) -> str:
    """
    Retrieve the full content of a specific documentation page by combining all its chunks.
    
    Args:
        ctx: The context including the Supabase client
        url: The URL of the page to retrieve
        
    Returns:
        str: The complete page content with all chunks combined in order
    """
    try:
        # Query Supabase for all chunks of this URL, ordered by chunk_number
        result = ctx.deps.supabase.from_('site_pages') \
            .select('title, content, chunk_number') \
            .eq('url', url) \
            .eq('metadata->>source', 'sap_docs') \
            .order('chunk_number') \
            .execute()
        
        if not result.data:
            return f"No content found for URL: {url}"
            
        # Format the page with its title and all chunks
        page_title = result.data[0]['title'].split(' - ')[0]  # Get the main title
        formatted_content = [f"# {page_title}\n"]
        
        # Add each chunk's content
        for chunk in result.data:
            formatted_content.append(chunk['content'])
            
        # Join everything together
        return "\n\n".join(formatted_content)
        
    except Exception as e:
        print(f"Error retrieving page content: {e}")
        return f"Error retrieving page content: {str(e)}"

@sap_expert.tool
async def search_web(ctx: RunContext[PydanticAIDeps], query: str, num_results: int = 5) -> List[Dict[str, str]]:
    """
    Search the web using SearXNG for the latest information.
    
    Args:
        ctx: The context
        query: The search query
        num_results: Number of results to return (default: 5)
        
    Returns:
        A list of search results with title, snippet, and URL
    """
    try:
        print("searching: ", query)
        # Configure your SearXNG instance URL
        searxng_url = os.getenv("SEARXNG_URL", "http://localhost:8080")
        
        # Format the search URL with parameters
        search_url = f"{searxng_url}/search"
        params = {
            "q": query,
            "format": "json",
            # "engines": "google,bing,duckduckgo",
            "results": str(num_results)
        }
        
        # Make the search request
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(search_url, params=params)
            response.raise_for_status()
            results = response.json()
            
        # Extract and format the results
        formatted_results = []
        for result in results.get("results", [])[:num_results]:
            formatted_results.append({
                "title": result.get("title", "No title"),
                "snippet": result.get("content", "No snippet"),
                "url": result.get("url", "")
            })
            
        if not formatted_results:
            return [{"title": "No results", "snippet": "No search results found", "url": ""}]
            
        return formatted_results
        
    except Exception as e:
        print(f"Error searching web: {e}")
        return [{"title": "Error", "snippet": f"Error searching web: {str(e)}", "url": ""}]

@sap_expert.tool
async def fetch_webpage_content(ctx: RunContext[PydanticAIDeps], url: str) -> str:
    """
    Fetch and extract the main content from a webpage.
    
    Args:
        ctx: The context
        url: The URL of the webpage to fetch
        
    Returns:
        The extracted main content of the webpage as text
    """
    try:
        print("fetching: ", url)
        # Make the HTTP request
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            html_content = response.text
            
        # Parse the HTML and extract main content
        soup = BeautifulSoup(html_content, "html.parser")
        
        # Remove unwanted elements
        for element in soup(["script", "style", "nav", "footer", "header"]):
            element.decompose()
            
        # Extract the text
        text = soup.get_text(separator="\n", strip=True)
        
        # Clean up the text
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        text = "\n\n".join(lines)
        
        # Limit the text length if it's too long
        if len(text) > 10000:
            text = text[:10000] + "...\n\n[Content truncated due to length]"
            
        return text
        
    except Exception as e:
        print(f"Error fetching webpage: {e}")
        return f"Error fetching webpage content: {str(e)}"