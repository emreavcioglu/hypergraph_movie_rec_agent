import os
from groq import Groq

class MovieNarrator:
    def __init__(self, api_key=None):
        self.client = None
        if api_key:
            try:
                self.client = Groq(api_key=api_key)
                print("✅ Groq client initialized successfully.")
            except Exception as e:
                print(f"⚠️ Narration initialization error: {e}")

    def explain_journey(self, start_movie, path_data, intent):
        """
        Generates a narration of the agent's journey through the movie graph.
        """
        # 1. Check if Client exists
        if not self.client:
            return "Narration unavailable: Groq client not initialized."
            
        # 2. Safety check for empty path
        if not path_data:
            return "No path data available to explain."

        # 3. Format the path string safely
        # We use .get() to avoid crashes if 'genre' is missing in a step
        try:
            steps_text = " -> ".join([f"{step.get('movie', 'Unknown')} ({step.get('genre', 'Unknown')})" for step in path_data])
            final_movie = path_data[-1].get('movie', 'Unknown')
        except Exception as e:
            return f"Error formatting path data: {e}"

        # 4. Construct Prompt
        prompt = (
            f"You are a film critic with a witty, insightful personality. "
            f"A user started with the movie '{start_movie}' and asked for '{intent}'. "
            f"They landed at: {steps_text}. "
            f"Explain in 2 sentences why '{final_movie}' is the perfect destination. "
            f"Mention the thematic connection."
        )

        # 5. Call API
        try:
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a helpful movie expert."},
                    {"role": "user", "content": prompt}
                ],
                model="llama-3.3-70b-versatile",
                temperature=0.7,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Narration API error: {e}"