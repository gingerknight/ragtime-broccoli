"""
response = client.models.generate_content(
    model="gemini-2.5-flash", contents="Why is Boot.dev such a great place to learn about RAG? Use one paragraph maximum."
)
print(response.text)
prompt_tokens = response.usage_metadata.prompt_token_count
candidate_tokens = response.usage_metadata.candidates_token_count

print(f"Prompt Tokens: {prompt_tokens}")
print(f"Response Tokens: {candidate_tokens}")
"""

import os

from dotenv import load_dotenv
from google import genai


class GeminiClient:
    def __init__(self):
        load_dotenv()
        self._api_key = os.environ.get("GEMINI_API_KEY")
        self.client = genai.Client(api_key=self._api_key)

    def spell_check(self, query):
        # provide the query to Gemini, and ask it to give you back a query with corrected spelling.
        prompt = f"""
        Fix any spelling errors in this movie search query.
        Only correct obvious typos. Don't change correctly spelled words.
        Query: "{query}"
        If no errors, return the original query.
        Corrected:
        """

        response = self.client.models.generate_content(
            model="gemini-2.5-flash", 
            contents=prompt
            )
        
        return response
    
    def rewrite_query(self, query):
        # rewrite the query to be more optimal for search
        prompt = f"""Rewrite the user-provided movie search query below to be more specific and searchable.

        Consider:
        - Common movie knowledge (famous actors, popular films)
        - Genre conventions (horror = scary, animation = cartoon)
        - Keep the rewritten query concise (under 10 words)
        - It should be a Google-style search query, specific enough to yield relevant results
        - Don't use boolean logic

        Examples:
        - "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
        - "movie about bear in london with marmalade" -> "Paddington London marmalade"
        - "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

        If you cannot improve the query, output the original unchanged.
        Output only the rewritten query text, nothing else.

        User query: "{query}"
        """

        response = self.client.models.generate_content(
            model="gemini-2.5-flash", 
            contents=prompt
            )
        
        return response

