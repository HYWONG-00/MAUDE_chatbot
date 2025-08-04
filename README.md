### RAG chatbot for coronary drug-eluting stent from MAUDE

- I'm using pre-trained sentence transformer for embedding, and using Gemini 1.5 pro as LLM for processing the query and handling whole interactive chat. To run the code:
  python -m streamlit run 1_pretrained.py

References:

1. Building RAG chatbot with langchain and gemini pro (Getting idea of how to build the interface with Gemini 1.5 Pro)

- https://www.youtube.com/watch?v=WeVb0cE3rrs

2. Storing everything in ChromaDB (For metadata filtering and query searching)

- https://medium.com/keeping-up-with-ai/how-i-built-a-rag-based-ai-chatbot-from-my-personal-data-88eec0d3483c

3. Paper that uses gemini 1.5 pro to build RAG chatbot
   https://www.politesi.polimi.it/handle/10589/234631

Evaluation Metrics:
https://arxiv.org/pdf/2408.03562
✅ A. Retrieval Accuracy (How well your retriever fetches relevant context)
✅ B. Generation Accuracy (How correct and grounded the LLM's responses are)
https://learn.microsoft.com/en-us/azure/databricks/generative-ai/tutorials/ai-cookbook/evaluate-assess-performance

https://arxiv.org/pdf/2401.06800 - Evaluation - GPT-4-based automatic evaluation pipeline
The idea is to get evaluation dataset with questions and expected answer, then use the GPT-4 to generate score(1-5) for each metrics.

How many of them are based on the ground truth




### To do: 1) Choose right metrics: What are the level of accuracy we can trust? need some metrics score

# what are the number of most reported cases in this year?
# different prompts/different way of asking should always give same answer (semantic embeddings and training)
# Others objective: which devices are most reliable throughout the year / showing decreasing trends


# Does the chatbot retrieve the correct record(s) from the MAUDE dataset?
# Does the final answer reflect the true content from those records?
# Does it semantically match the user’s question (even if paraphrased)? - Depends on the semantic embeddings model u used


# Q: What adverse events were reported in 2023 related to drug eluting stents?

### Google Genrative AI is not working


# https://anonymous.4open.science/r/ragas_updated-FFC6/README.md
# https://medium.com/keeping-up-with-ai/how-i-built-a-rag-based-ai-chatbot-from-my-personal-data-88eec0d3483c