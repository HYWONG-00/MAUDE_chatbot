### Topic: RAG chatbot that how adverse events of coronary DES
Main Objective: I hope this chatbot can help medical practitioners in the field to decide the alternative stent or treatment to be used by the patient. 
Objectives:
1)	Extract all the adverse events particularly for coronary drug-eluting stent with fine-tuned biomedical NLP models such as BioBERT/ClinicalBERT
2)	Identify the clusters or apply topic modelling for these founded adverse events and suggests the topics for them
3)	Identify the emerging trends of 
      - report count changes for each clusters (find most common symptoms)
      - patient demographic changes for each clusters (find most vulnerable patient group)
      - device and manufacturer for each clusters (find the device with most problem for particular issue)
5)	Create RAG chatbot/Power BI dashboard to explain the whole trends including other analysis to the medical practitioners.

Limitation:
1) In this project, I only identify as much adverse events as I could using existing NER or BERT from the medical devices.However, this NER model should be fine-tuned for finding more AEs.
2) Some AEs found did not really happen in the end. It could mentioned Device malfunction problem "to be investigated" but it still tag "device malfunction" for this report. Therefore, we should dive into SEMANTIC UNDERSTANDING of the report
