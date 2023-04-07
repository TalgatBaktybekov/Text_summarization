# Text_summarization
This code implements a text summarization problem. It takes in a text document, preprocesses the sentences by lemmatizing, 
removing stop words and punctuations. Then it creates a similarity matrix between all pairs of sentences based on their cosine similarity 
using the TF-IDF scores of the words in the sentences. Finally, it ranks the sentences based on their similarity scores and selects
the top N sentences to form a summary of the original text document. The code uses the NetworkX library to perform page rank algorithm 
for ranking the sentences.
