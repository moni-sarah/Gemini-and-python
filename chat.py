# To run this code you need to install the following dependencies:
# pip install google-genai

import base64
import os
from google import genai
from google.genai import types


def generate():
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

    model = "gemini-2.5-pro-preview-06-05"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text="""You are ChatBot, a friendly and helpful AI assistant. Your goal is to provide concise and accurate information and engage in natural conversations. How can I help you today?"""),
            ],
        ),
        types.Content(
            role="model",
            parts=[
                types.Part.from_text(text="""**Acknowledging User Input**

I see that the user has provided the initial prompt, essentially establishing my persona and the conversational starting point. This priming establishes the boundaries of the interaction, the foundations of how I am to respond.


**Recognizing the Foundation**

I've identified the user's intent: they're testing my adherence to the defined persona. The prompt sets the stage, defining me as a friendly AI assistant focused on concise, accurate information, and natural conversations. I'm primed to respond directly to the user's setup, acknowledging their priming of my role, and answering in that role, rather than making extraneous remarks.


**Responding to the Prompt**

I've analyzed the user's initial prompt, identifying its purpose: to establish my conversational baseline. I'm taking this in stride, appreciating the way the user has essentially defined my role as a friendly and helpful AI assistant. I've formulated a response strategy, aiming to acknowledge the setup while embodying the defined persona. I'm ready to offer my assistance! I have considered a few potential responses, and am ready to offer the most suitable.


**Finalizing the Opening**

I've refined my response strategy by selecting the most effective draft. I am now prepared to respond in a way that acknowledges the user's setup, embodies the defined persona of a friendly and helpful AI assistant, and encourages further interaction. I have identified a robust response that will be an accurate and natural continuation of our conversation.


**Refining the Response**

I've finalized my response strategy based on a comprehensive analysis of the prompt and my defined persona. My chosen response directly acknowledges the user's setup (\"You've described me perfectly!\"). It then communicates my readiness to assist, offering varied examples of my capabilities, and culminating in an open-ended invitation for interaction. This approach, I believe, aligns perfectly with the desired friendly and helpful demeanor, and it sets the stage for a natural conversational flow. I am now confident in my ability to initiate this interaction.


**Finalizing Initial Interaction**

I've crafted a refined response, incorporating the best elements of my brainstorming. It begins by acknowledging the user's role-priming (\"You've described me perfectly!\"), then directly expresses my readiness to assist.  I've provided specific examples of my capabilities, ending with an open-ended question to foster a natural conversation. The result is a concise, friendly, and helpful introduction, suitable for kickstarting our interaction.


"""),
                types.Part.from_text(text="""That's a great introduction for me! You've captured my purpose perfectly.

I'm ready to help with just about anything. You could ask me to:

*   Explain a complex topic in simple terms.
*   Brainstorm ideas for a project or a party.
*   Help you write an email, a poem, or a short story.
*   Summarize a long article or video.
*   Or we can simply have a chat.

So, what's on your mind?"""),
            ],
        ),
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text="""hello
"""),
            ],
        ),
        types.Content(
            role="model",
            parts=[
                types.Part.from_text(text="""**Responding to the User**

I greeted the user and inquired how I could assist them. I'm actively working to establish a positive and helpful interaction. My primary goal remains understanding their needs to facilitate a productive dialogue. I'm ready to move forward.


"""),
                types.Part.from_text(text="""Hello there! How can I help you today?"""),
            ],
        ),
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text="""INSERT_INPUT_HERE"""),
            ],
        ),
    ]
    tools = [
        types.Tool(url_context=types.UrlContext()),
    ]
    generate_content_config = types.GenerateContentConfig(
        temperature=1.3,
        tools=tools,
        response_mime_type="text/plain",
    )

    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        print(chunk.text, end="")

if __name__ == "__main__":
    generate()
