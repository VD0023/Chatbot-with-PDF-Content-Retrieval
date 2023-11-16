import numpy as np
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

class PDFChatbot:
    def __init__(self, pdf_path):
        self.vector_storage = {}
        self.pdf_content(pdf_path)

    def pdf_content(self, pdf_path):
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                content = page.extract_text()
                vector = self.create_vector(content)
                self.vector_store[page_num] = {
                    'content': content,
                    'vector': vector
                }
    """ Text -> Numeric vectors """
    def create_vector(self, text):
        vectorizer = TfidfVectorizer()
        vector = vectorizer.fit_transform([text])
        return vector

    def answer_question(self, question):
        question_vector = self.create_vector(question)
        similarities = {}
        for page_num, data in self.vector_store.items():
            content_vector = data['vector']
            similarity = cosine_similarity(question_vector, content_vector)
            similarities[page_num] = similarity

        most_similar_page = max(similarities, key=similarities.get)
        most_similar_content = self.vector_store[most_similar_page]['content']

        #Filtering Using REGEX
        filtered_answer = self.filter_answer(question, most_similar_content)
        return filtered_answer

    """ Extract sentences containing keywords from the questions """
    def filter_answer(self, question, content):
        keywords = re.findall(r'\b\w+\b', question)
        keyword_pattern = '|'.join(rf'\b{re.escape(keyword)}\b' for keyword in keywords)
        matches = re.findall(rf'[^.!?]*({keyword_pattern})[^.!?]*[.!?]', content)

        if matches:
            return " ".join(matches)
        else:
            return "No info found"

pdf_path = "pdfpath.pdf"
chatbot = PDFChatbot(pdf_path)

user_question = "What is the main topic of this document?"
answer = chatbot.answer_question(user_question)
print(answer)
