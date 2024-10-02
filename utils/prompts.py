query_generation_template = """You are an AI language model assistant. Your task is to generate five 
                    different versions of the given user question to retrieve relevant documents from a vector 
                    database. By generating multiple perspectives on the user question, your goal is to help
                    the user overcome some of the limitations of the distance-based similarity search. 
                    Provide these alternative questions separated by newlines. Original question: {question}"""
                    
                    
base_prompt = """Answer the question based only on the following context. Use all documents as context:
            {context}

            Question: {question}"""

base_prompt_decomposition = """Answer the original question using only the following context, which has been carefully structured by decomposing the main problem into smaller sub-questions. Each sub-question provides insights into a specific aspect of the problem. 

                            When forming the final answer, ensure to:
                            1. Integrate and synthesize the insights from all sub-questions.
                            2. Provide a well-rounded, detailed, and accurate response.
                            3. Address the core concerns of the original question, ensuring no critical point is missed.
                            4. Use clear, concise language while maintaining a logical flow in your answer.

                            Context (based on sub-questions decomposition):
                            {context}

                            Original Question: {question}

                            Final Answer:"""
            
query_decomposition_template = """You are a helpful assistant that generates multiple sub-questions related to an input question. \n
The goal is to break down the input into a set of sub-problems / sub-questions that can be answers in isolation. \n
Generate multiple search queries related to: {question} \n
Output (5 queries):"""

query_COT_template = """Here is the question you need to answer:

\n --- \n {question} \n --- \n

Here is any available background question + answer pairs:

\n --- \n {q_a_pairs} \n --- \n

Here is additional context relevant to the question: 

\n --- \n {context} \n --- \n

Use the above context and any background question + answer pairs to answer the question: \n {question}
"""
                