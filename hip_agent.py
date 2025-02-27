import openai
import rag
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class HIPAgent:
    def __init__(self):
        # Load book chunks from 'textbook.txt'
        self.book_chunks = rag.load_and_split_book('textbook.txt',n=4)
        self.chunk_embeddings = rag.load_or_compute_embeddings('./bio_book_embeddings.pkl', self.book_chunks)
        self.topk_relevance = 3

    
    def get_response(self, question, answer_choices):
        """
        Calls the OpenAI 3.5 API to generate a response to the question.
        The response is then matched to one of the answer choices and the index of the
        matching answer choice is returned. If the response does not match any answer choice,
        -1 is returned.

        Args:
            question: The question to be asked.
            answer_choices: A list of answer choices.

        Returns:
            The index of the answer choice that matches the response, or -1 if the response
            does not match any answer choice.
        """
        option_list = ['A. ', 'B. ', 'C. ', 'D. ']
        answer_str = ''
        for i, opt in enumerate(option_list):
            answer_str += opt + answer_choices[i] + '\n'

        # Generate embeddings for the question along with the options
        question_embedding = rag.get_embedding(question +'\n' + answer_str)

        # Compute cosine similarity between the question embedding and each chunk embedding
        similarities = cosine_similarity([question_embedding], self.chunk_embeddings)[0]

        # Find relevant chunks based on the top K highest similarity scores
        relevant_indices = sorted(np.argsort(similarities)[::-1][:self.topk_relevance])
        relevant_chunks = [self.book_chunks[i] for i in relevant_indices]
        relevant_chunks_string = "".join(relevant_chunks)

        # Create the prompt.
        rag_prompt = ("Textbook related paragraph: {relevant_chunks_string} \n"
                  "Question: '{question}'\n"
                  "{answer_str}"
                  "Based on the related paragraph and the question with options, extract the most related information."
                  "Output the extracted information without answering the question. "
                )
                
        # Call the OpenAI 3.5 API to obtain context.
        context = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": rag_prompt.format(relevant_chunks_string=relevant_chunks_string, question=question, answer_str=answer_str)},
            ],
            temperature=0.,
            top_p=0.45
        )

        context_info = context.choices[0].message.content

        prompt = ("Based on the paragraph in the textbook answer the following question:\n"
                  "Question: {question}\n"
                  "{answer_str} \n"
                  "Textbook paragraph:  {context_info} \n\n"
                  "Based on the textbook, generate you answer to the question. "
                  "Your final answer must be one of the A, B, C, D, without any additional characters or symbols. "
                )
        # Call the OpenAI 3.5 API to get the answer.
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt.format(context_info=context_info, question=question, answer_str=answer_str)},
            ],
            temperature=0.,
            top_p=0.45
        )
        response_text = response.choices[0].message.content
        response_selection = response_text.split('Answer:')[-1].strip().replace('.','').split(' ')[0]

        # Match the response to one of the answer choices.
        for i, answer_choice in enumerate(['A','B','C','D']):
            if response_selection == answer_choice:
                return i

        # If the response does not match any answer choice, return -1.
        return -1
