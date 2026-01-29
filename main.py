from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import numpy as np

# 1. Read PDF
reader = PdfReader("school_data.pdf")
texts = []

for page in reader.pages:
    page_text = page.extract_text()
    if page_text:
        texts.append(page_text)

# 2. Create embeddings
embedder = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedder.encode(texts)

# 3. Load open-source LLM
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

qa_pipeline = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=256
)

print("School AI Assistant (type 'exit' to quit)")

while True:
    query = input("\nAsk a question: ")
    if query.lower() == "exit":
        break

    # 4. Embed query
    query_embedding = embedder.encode([query])

    # 5. Similarity search
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    best_index = np.argmax(similarities)

    # 6. Domain restriction
    if similarities[best_index] < 0.3:
        print("Answer: I don't know")
        continue

    context = texts[best_index]

    # 7. Prompt the model
    prompt = f"""
Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{query}
"""

    result = qa_pipeline(prompt)
    print("Answer:", result[0]["generated_text"])