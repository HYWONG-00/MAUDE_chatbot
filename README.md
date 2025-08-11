### RAG chatbot for coronary drug-eluting stent from MAUDE

This chatbot was found on https://maudechatbot-ajgpwyobfuysmegnrorykf.streamlit.app/

For this chatbot:
- I'm using allMiniLM-L6-v2 for embedding, and using fine-tuned GPT-2 as LLM for processing the query and handling whole interactive chat. 
To run the code:
  python -m streamlit run rag-final.py

What does it done?
- Retrieving right documents for you (based on MDR report key)
- Explaining trends on patient gender and stent brands over year
For example, you can ask:
1. What were the adverse events reported in MDR Report Key 20211227? 
2. What was the patient's sex in MDR Report Key 18141819? 
3. In what year was MDR Report Key 18029867 reported? 
5. How many MDR reports are from 2024?
Give me random 5 reports about thrombosis. List in format, (id, gender, year)

Pros for this chatbot:
- Can retrieve correct documents if report key is given and then based on it give the right answer for the field asked in query
Issues:
- It has some issue with handling other kind of the requests especially for the trends one, the retrieved documents was correct, but the answer from llm are not always correct


Things to consider:
Choose right metrics: What are the level of accuracy we can trust? I used BERT-Score in this case
different prompts/different way of asking should always give same answer (semantic embeddings and training)
Others objective: which devices are most reliable throughout the year / showing decreasing trends

Does the chatbot retrieve the correct record(s) from the MAUDE dataset? For retrieving, if the user provides report key
Does the final answer reflect the true content from those records?
Does it semantically match the user’s question (even if paraphrased)? - Depends on the semantic embeddings model u used

