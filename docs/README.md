## # AI Tutor Chatbot

## Abstract

The **AI Tutor Chatbot** is an intelligent educational assistant designed to support students by providing relevant answers to academic questions through a combination of **information retrieval, Natural Language Processing (NLP), Retrieval-Augmented Generation (RAG), and Large Language Models (LLMs)**.

The system processes and cleans educational datasets before using text-based similarity techniques to identify relevant learning content. User queries are preprocessed and compared against the available knowledge base using **TF-IDF and cosine similarity**. When sufficiently relevant information is retrieved, it can be used as contextual information for generating an appropriate response. For queries where the retrieval system cannot identify sufficiently relevant information, an LLM-based fallback mechanism can provide a response.

The application is implemented as a **Flask-based web application** with database integration and user authentication. The project demonstrates the integration of traditional information-retrieval techniques with modern generative AI to create an interactive learning assistant.

---

**## 1. Introduction**

Students often need quick access to reliable learning resources while studying. Traditional search-based systems can require students to manually identify and navigate through large amounts of educational material.

Recent developments in Artificial Intelligence and Natural Language Processing have enabled conversational systems capable of understanding natural-language questions and generating responses.

This project explores the development of an **AI-powered tutoring system** that combines retrieval techniques with generative AI.

Rather than relying exclusively on an LLM, the system first attempts to retrieve relevant information from a prepared educational dataset. This provides a structured approach to answering questions while allowing an LLM fallback when relevant information cannot be retrieved.

---

## 2. Problem Statement

Students may encounter difficulties when searching for specific information within large collections of learning material.

Traditional keyword-based searches can fail when a student's question is phrased differently from the information stored in the dataset.

At the same time, purely generative AI systems may produce responses that are not directly grounded in the available educational resources.

Therefore, this project investigates a hybrid approach combining:

- Information retrieval
- NLP-based text processing
- Similarity matching
- Retrieval-Augmented Generation
- Large Language Models

The objective is to create a system capable of retrieving relevant educational information while maintaining the flexibility of generative AI.

---

## 3. Aim

The primary aim of the project is to develop an **AI-powered educational chatbot** capable of answering student queries by retrieving relevant learning content and using generative AI when appropriate.

---

## 4. Objectives

The main objectives are:

1. Develop a web-based AI tutoring application.
2. Prepare and clean an educational question-and-answer dataset.
3. Implement text preprocessing for user queries and learning content.
4. Implement TF-IDF-based information retrieval.
5. Calculate cosine similarity between queries and stored content.
6. Establish similarity thresholds for retrieval confidence.
7. Implement a Retrieval-Augmented Generation workflow.
8. Integrate an LLM fallback mechanism.
9. Develop user authentication and database functionality.
10. Test the application and its backend endpoints.
11. Provide technical documentation for the system.
12. Evaluate limitations and identify future improvements.

---

## 5. System Architecture

The overall architecture consists of a frontend interface, Flask backend, database, retrieval system and AI response generation.

```text
                         ┌──────────────────┐
                         │     Student      │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │   Web Interface  │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │  Flask Backend   │
                         └────────┬─────────┘
                                  │
                     ┌────────────┴────────────┐
                     │                         │
                     ▼                         ▼
              ┌─────────────┐          ┌──────────────┐
              │   MySQL DB  │          │ Query        │
              │ Authentication│        │ Preprocessing│
              └─────────────┘          └──────┬───────┘
                                               │
                                               ▼
                                      ┌────────────────┐
                                      │ TF-IDF         │
                                      │ Vectorisation  │
                                      └───────┬────────┘
                                              │
                                              ▼
                                      ┌────────────────┐
                                      │ Cosine         │
                                      │ Similarity     │
                                      └───────┬────────┘
                                              │
                                ┌─────────────┴────────────┐
                                │                          │
                                ▼                          ▼
                       Relevant Content              Low Similarity
                                │                          │
                                ▼                          ▼
                         ┌────────────┐             ┌────────────┐
                         │ RAG / LLM  │             │ LLM        │
                         │ Response   │             │ Fallback   │
                         └─────┬──────┘             └─────┬──────┘
                               │                          │
                               └────────────┬─────────────┘
                                            ▼
                                   ┌─────────────────┐
                                   │  Final Answer   │
                                   └─────────────────┘
```

---

## 6. Methodology

### 6.1 Dataset Preparation

The educational dataset is processed before being used by the retrieval system.

The preprocessing workflow includes:

- Removing duplicate records
- Handling missing information
- Normalising text
- Cleaning question-and-answer pairs
- Preparing structured learning content

The objective is to ensure that the retrieval system works with consistent and relevant information.

---

**### 6.2 Query Processing**

When a student submits a question, the query is first normalised.

The preprocessing includes:

- Converting text to lowercase
- Removing unnecessary whitespace
- Normalising the input
- Preparing the query for vectorisation

This reduces differences caused by formatting and capitalisation.

---

### 6.3 TF-IDF Retrieval

The system uses **Term Frequency-Inverse Document Frequency (TF-IDF)** to represent text.

TF-IDF assigns importance to terms based on their occurrence within a document relative to their occurrence across the dataset.

The resulting vectors allow the system to compare the user's question against available educational content.

---

**### 6.4 Cosine Similarity**

Cosine similarity is used to determine the similarity between the user's query and stored learning content.

Conceptually:

```text
User Query
     ↓
TF-IDF Vector
     ↓
Compare with Dataset Vectors
     ↓
Cosine Similarity Score
     ↓
Determine Relevance
```

A similarity threshold is used to distinguish between sufficiently relevant and low-confidence retrieval results.

---

### 6.5 Retrieval-Augmented Generation

When relevant information is retrieved, it can be provided as context to the generative AI component.

```text
Question
   ↓
Retriever
   ↓
Relevant Learning Content
   ↓
Context
   ↓
Language Model
   ↓
Generated Response
```

This approach allows the chatbot to combine information retrieval with generative AI.

---

**### 6.6 LLM Fallback**

If the retrieval system produces a low similarity score, the system can use an LLM fallback.

This provides additional coverage for questions that are not sufficiently represented within the prepared dataset.

```text
Similarity Score
       │
       ├── High ──→ Retrieved Context → Response
       │
       └── Low ───→ LLM Fallback → Response
```

Low-similarity queries can also be logged and analysed to identify potential improvements to the dataset.

---

## 7. Web Application

The application is developed using **Python and Flask**.

The web application provides the interface through which users can interact with the AI Tutor.

The project includes:

- Flask backend
- HTML templates
- Static resources
- User authentication
- Database integration
- Chatbot functionality
- Retrieval pipeline
- Testing components

---

## 8. Database

A relational database is used to support application functionality and user management.

Database functionality includes:

- User information
- Authentication
- Account-related data
- Application data management

The project uses **MySQL** as part of the backend architecture.

---

**## 9. Testing**

Testing is performed to validate the behaviour of the application and its backend components.

Testing covers areas including:

- API endpoints
- Backend functionality
- Authentication
- Retrieval behaviour
- Response handling

The project also includes endpoint testing files within the repository.

---

**## 10. Technologies Used**

| Category | Technologies |
|---|---|
| Programming | Python, HTML, CSS, JavaScript |
| Backend | Flask |
| Database | MySQL, SQLite |
| NLP | TF-IDF, Cosine Similarity |
| AI | RAG, LLM |
| Testing | Python testing tools |
| Version Control | Git, GitHub |

---

**## 11. Project Structure**

```text
AI-Tutor/
│
├── app/
├── datasets/
├── docs/
├── static/
├── templates/
├── tests/
├── Report/
├── test_endpoints.py
├── users.db
└── .gitignore
```

---

**## 12. Key Features**

- 🤖 AI-powered tutoring chatbot
- 🔎 Educational content retrieval
- 🧠 TF-IDF-based similarity matching
- 📚 Retrieval-Augmented Generation
- 💬 LLM fallback
- 👤 User authentication
- 🗄️ Database integration
- 🧹 Dataset preprocessing
- 🧪 Backend endpoint testing
- 🌐 Web-based interface

---

**## 13. Results and Evaluation**

The developed system demonstrates the feasibility of combining traditional information retrieval with generative AI within an educational chatbot.

The retrieval component provides a mechanism for identifying relevant information from the prepared dataset, while the LLM fallback improves the system's ability to respond when a suitable dataset match cannot be identified.

The project also demonstrates the integration of:

- NLP preprocessing
- Database systems
- Web development
- Information retrieval
- Generative AI

Further quantitative evaluation using larger datasets and established retrieval and response-quality metrics would be required for a comprehensive assessment.

---

**## 14. Limitations**

The current system has several limitations:

- Retrieval performance depends on dataset quality.
- TF-IDF primarily relies on lexical similarity.
- Questions using significantly different terminology may receive lower similarity scores.
- LLM responses can vary depending on the underlying model.
- The current dataset may not cover every possible student question.
- Larger-scale deployment would require additional performance and security considerations.

---

**## 15. Future Work**

Potential future improvements include:

- 🔹 Semantic embedding-based retrieval
- 🔹 Vector database integration
- 🔹 Improved RAG evaluation
- 🔹 Conversation memory
- 🔹 Personalised learning recommendations
- 🔹 Student progress tracking
- 🔹 Improved similarity thresholds
- 🔹 Larger educational datasets
- 🔹 More extensive automated testing
- 🔹 Cloud deployment
- 🔹 Scalable microservice architecture

---

## 16. Learning Outcomes

This project provided practical experience in several areas of software and AI development.

### Artificial Intelligence

Experience with:

- Retrieval-Augmented Generation
- LLM integration
- Information retrieval
- NLP techniques

### Software Development

Experience with:

- Python
- Flask
- Backend APIs
- Web application development
- Database integration

**Data Processing**

Experience with:

- Dataset cleaning
- Data preprocessing
- Text normalisation
- TF-IDF vectorisation

**Software Engineering**

Experience with:

- Git/GitHub
- Testing
- Documentation
- System architecture
- Collaborative development

**## 17. Conclusion**

The AI Tutor project demonstrates a practical implementation of an intelligent educational assistant using a combination of **NLP, information retrieval, RAG and generative AI**.

By combining a retrieval-based approach with an LLM fallback mechanism, the system provides a flexible architecture for answering student questions while making use of structured educational content.

The project also demonstrates the integration of **Python, Flask, databases, web technologies and AI techniques** into a complete software application.





