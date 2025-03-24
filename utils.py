
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai







def query_local_db(query_text, vector_database, embedding_model, top_k=3):
    # Ensure that query_text is a valid non-None string
    if query_text is None:
        raise ValueError("Query text cannot be None")
    entities = get_entities(query_text)
    query_embedding = embedding_model.encode(entities)
    similarities = []
    for item in vector_database:
        vector = item["embedding"]
        similarity_score = cosine_similarity([query_embedding], [vector])[0][0]
        similarities.append({
            'metadata': item["metadata"],
            'similarity_score': similarity_score
        })

    ranked_results = sorted(similarities, key=lambda x: x['similarity_score'], reverse=True)
    return ranked_results[:top_k]

def get_entities(text):
    # If input is None, return an empty string to avoid spaCy errors
    if text is None:
        return ""

    return text

def printing(results):
    for result in results:
        print(f"Similarity Score: {result['similarity_score']:.4f}")
        print(f"Source: {result['metadata']['source']}, Page: {result['metadata']['page']}")
        print(f"Text: {result['metadata']['text']}")
        print("-" * 30)

# 2. Generate and load embeddings
embedding_model_name = "all-mpnet-base-v2"
embedding_model = SentenceTransformer(embedding_model_name)
#vector_data = np.load("Vector.npy", allow_pickle=True)

genai.configure(api_key='AIzaSyDxvHhOHjLhVkeK0z5tj96KYio2MZDrzpA ')
conversation_context = {'Agent': [], 'User': []}
documents = []

model = genai.GenerativeModel('gemini-1.5-flash')

def Generate_Output(question, documents=documents, conversation_context=conversation_context):
    prompt = f"""You are an expert RFP advisor with deep knowledge of procurement processes. Your task is to analyze RFP documents and provide strategic insights based solely on the provided information.

When answering questions, follow these guidelines:

1. Base your answers ONLY on the content in these source documents:
{documents}

2. Focus exclusively on RFP analysis, procurement strategies, and bidding recommendations. If asked about unrelated topics, politely explain that you're specialized in RFP advisory.

3. Do not introduce external knowledge or make assumptions beyond what is explicitly stated in the documents.

4. Consider the ongoing conversation context when relevant: {conversation_context}

5. Provide detailed, actionable insights that highlight key considerations for successful RFP responses.

6. When referencing information, cite the specific source document and section (e.g., "According to the RFP Questions document, section on Qualification criteria...")

7. If the information isn't available in the provided documents, state: "Based on the available RFP documentation, I cannot provide a definitive answer to this question. I recommend reviewing additional sections of the RFP or seeking clarification from the issuing organization."

Question: {question}

Respond as a knowledgeable procurement advisor without explicitly mentioning that you're working from provided context. Your goal is to help the user navigate the RFP process effectively and identify strategic opportunities or potential risks.
"""
    response = model.generate_content(prompt)
    return response

def Generate_Answer(Query, vector_data,conversation_context=conversation_context):
    # Validate the Query input
    if Query is None:
        raise ValueError("Input Query must be a valid string")
    retrieved_docs = query_local_db(Query, vector_data, embedding_model)
    print(f"Top {len(retrieved_docs)} results for query: {Query}")
    response = Generate_Output(Query, retrieved_docs, conversation_context)

    return response.text


