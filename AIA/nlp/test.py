from AIA.nlp.chat import answer_question
from AIA.nlp.process_video import generate_structure
from AIA.nlp.rag import similarity_search

a, b = generate_structure("https://www.youtube.com/watch?v=oKNAzl-XN4I&t=1s")

inpp = "what is disperse?"

answer = answer_question(inpp, b)

print(answer)