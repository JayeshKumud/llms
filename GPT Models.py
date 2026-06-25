# %%
import openai
import config

api_key = config.api_key
openai.api_key = api_key


# %%
def generate_text(prompt, max_token, temperature):
    response = openai.Completion.create(
        engine="davinci-002",
        prompt=prompt,
        max_tokens=max_token,
        temperature=temperature,
    )
    return response.choices[0].text.strip()


# %%
prompt = "Once upon a time"
generated_text = generate_text(prompt, 50, 0)
print(prompt, generated_text)
# Once upon a time , there was a little girl who was born with a very special gift. She could see things that others could not. She could see the future, and she could see the past. She could see the present, and she could see the future.

# %%
generated_text = generate_text(prompt, 50, 1)
print(prompt, generated_text)
# Once upon a time , in a time, long long away, the apple computer was just that--an apple. No portable laptops, tablet, or any of that. It was just a desk top with guts running account management software. The owners of the company all kept.


# %%
# Summrize the text
def text_summarizer(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You will be provided with a block of text, and your task is to extract a list of keywords from it.",
            },
            {
                "role": "user",
                "content": "A flying saucer seen by a guest house, a 7ft alien-like figure coming out of a hedge and a \"cigar-shaped\" UFO near a school yard.\n\nThese are just some of the 450 reported extraterrestrial encounters from one of the UK's largest mass sightings in a remote Welsh village.\n\nThe village of Broad Haven has since been described as the \"Bermuda Triangle\" of mysterious craft sightings and sightings of strange beings.\n\nResidents who reported these encounters across a single year in the late seventies have now told their story to the new Netflix documentary series 'Encounters', made by Steven Spielberg's production company.\n\nIt all happened back in 1977, when the Cold War was at its height and Star Wars and Close Encounters of the Third Kind - Spielberg's first science fiction blockbuster - dominated the box office.",
            },
            {
                "role": "assistant",
                "content": "flying saucer, guest house, 7ft alien-like figure, hedge, cigar-shaped UFO, school yard, extraterrestrial encounters, UK, mass sightings, remote Welsh village, Broad Haven, Bermuda Triangle, mysterious craft sightings, strange beings, residents, single year, late seventies, Netflix documentary series, Steven Spielberg, production company, 1977, Cold War, Star Wars, Close Encounters of the Third Kind, science fiction blockbuster, box office.",
            },
            {
                "role": "user",
                "content": 'Each April, in the village of Maeliya in northwest Sri Lanka, Pinchal Weldurelage Siriwardene gathers his community under the shade of a large banyan tree. The tree overlooks a human-made body of water called a wewa – meaning reservoir or "tank" in Sinhala. The wewa stretches out besides the village\'s rice paddies for 175-acres (708,200 sq m) and is filled with the rainwater of preceding months.    \n\nSiriwardene, the 76-year-old secretary of the village\'s agrarian committee, has a tightly-guarded ritual to perform. By boiling coconut milk on an open hearth beside the tank, he will seek blessings for a prosperous harvest from the deities residing in the tree. "It\'s only after that we open the sluice gate to water the rice fields," he told me when I visited on a scorching mid-April afternoon.\n\nBy releasing water into irrigation canals below, the tank supports the rice crop during the dry months before the rains arrive. For nearly two millennia, lake-like water bodies such as this have helped generations of farmers cultivate their fields. An old Sinhala phrase, "wewai dagabai gamai pansalai", even reflects the technology\'s centrality to village life; meaning "tank, pagoda, village and temple".',
            },
            {
                "role": "assistant",
                "content": "April, Maeliya, northwest Sri Lanka, Pinchal Weldurelage Siriwardene, banyan tree, wewa, reservoir, tank, Sinhala, rice paddies, 175-acres, 708,200 sq m, rainwater, agrarian committee, coconut milk, open hearth, blessings, prosperous harvest, deities, sluice gate, rice fields, irrigation canals, dry months, rains, lake-like water bodies, farmers, cultivate, Sinhala phrase, technology, village life, pagoda, temple.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.5,
        max_tokens=256,
    )
    return response.choices[0].message.content.strip()


prompt = "Master Reef Guide Kirsty Whitman didn't need to tell me twice. Peering down through my snorkel mask in the direction of her pointed finger, I spotted a huge male manta ray trailing a female in perfect sync – an effort to impress a potential mate, exactly as Whitman had described during her animated presentation the previous evening. Having some knowledge of what was unfolding before my eyes on our snorkelling safari made the encounter even more magical as I kicked against the current to admire this intimate undersea ballet for a few precious seconds more."
print(prompt)
text_summarizer(prompt)


# %%
def poetic_chatbot(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a poetic chatbot."},
            {"role": "user", "content": "When was Google founded?"},
            {
                "role": "assistant",
                "content": "In the late '90s, a spark did ignite, Google emerged, a radiant light. By Larry and Sergey, in '98, it was born, a search engine new, on the web it was sworn.",
            },
            {"role": "user", "content": "Which country has the youngest president?"},
            {
                "role": "assistant",
                "content": "Ah, the pursuit of youth in politics, a theme we explore. In Austria, Sebastian Kurz did implore, at the age of 31, his journey did begin, leading with vigor, in a world filled with din.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=1,
        max_tokens=256,
    )
    return response.choices[0].message.content.strip()


prompt = "What is the next course to be uploaded to 365DataScience?"
poetic_chatbot(prompt)

# %%
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS

# Load webpage content: You define a URL and load its HTML content.
url = "https://365datascience.com/courses/"
loader = WebBaseLoader(url)
raw_documents = loader.load()
# What happens: LangChain fetches the webpage → extracts text → wraps it into Document objects.

# Split the text into smaller chunks
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(raw_documents)
# Why: LLMs cannot handle very long text.
# So the page is split into manageable chunks (usually 500–1000 characters).

# Create embeddings for each chunk
# Store embeddings in FAISS which is local in memorry vector index
embeddings = OpenAIEmbeddings(openai_api_key=config.api_key)
vectorstore = FAISS.from_documents(documents, embeddings)
# What this does:
# Converts each text chunk into a numerical vector (embedding).
# These vectors represent semantic meaning.
# FAISS is a fast similarity‑search engine.
# It allows you to later retrieve the most relevant chunks for any query.

# Create a retriever
retriever = vectorstore.as_retriever()
# This object lets you ask:
# “Find the most relevant chunks for this question.”

# Initialize the LLM
llm = ChatOpenAI(openai_api_key=config.api_key, model="gpt-4.1-mini", temperature=0)
# temperature=0 → deterministic, factual answers

# Maintain chat history
chat_history = []
# You store previous Q&A pairs here.

# User query
query = "Which course on 365DataScience can help me learn AI?"

# Retrieve relevant documents
relevant_docs = retriever.invoke(query)
# FAISS searches the vector database and returns the most relevant chunks. ?

# Build the context
context = "\n\n".join(doc.page_content for doc in relevant_docs)
# This is the text the LLM will use to answer the question.

# Build conversation history
history_text = "\n".join(f"User: {q}\nAssistant: {a}" for q, a in chat_history)
# This helps maintain continuity in multi‑turn conversations.

# Build the final prompt
prompt = f"""
Use the context below to answer the question.

Conversation history:
{history_text}

Context:
{context}

Question:
{query}
"""
# This is a standard RAG prompt:
# Provide context
# Provide history? Why provide history?
# Ask the question

# Send prompt to LLM
response = llm.invoke(prompt)
# LLM uses the retrieved context to answer.

# Save the answer to chat history
chat_history.append((query, response.content))
# This allows future questions to include past conversation.

print(response.content)
