# Technical Justifications for the Three Assignments

## 1) Why use a ReAct-style agent around the RAG tool?
I used a ReAct-style agent because the system no longer performs only one fixed task. It now needs to decide whether to retrieve information from agronomy documents, perform arithmetic, save data, query stored notes, or answer directly. A plain RAG pipeline is suitable when the flow is always fixed, but an agent is more appropriate when the model must reason about which action to take. In this assignment, the agent chooses between available tools based on the user’s request, which is the core idea behind ReAct: reasoning, acting through tools, then responding.

## 2) Why wrap the RAG pipeline as a tool?
My original notebook already contained the most important knowledge pipeline: document loading, chunking, embedding, retrieval, and grounded answering. Wrapping that pipeline as a tool allowed me to preserve the exact retrieval logic I had already built and reuse it inside the agent architecture. This made the knowledge access layer modular, reusable, and easier to test independently from the rest of the system.

## 3) Prompting strategy
The system prompt was intentionally narrow and task-specific. It tells the model when to use the agronomy retrieval tool, when to use the calculator or database-related tools, and when to avoid answering from general model knowledge. It also instructs the model not to invent facts outside the indexed documents and to keep answers concise and practical. This prompt design reduces hallucinations.

## 4) Why use a supervisor-style design for Assignment 2?
For Assignment 2, the system had to handle different responsibilities: agronomy retrieval, note saving, and safe database querying. A supervisor-style design fits this requirement because it separates responsibilities instead of mixing everything into one large undifferentiated workflow. The main controller handles the conversation and decides which tool or specialist function to use, while each task-specific component remains focused on a single responsibility.

## 5) Why keep chat history in RAM?
The assignment specifically asked for simple chat history in RAM rather than persistent storage. I implemented this using a Python list that stores prior user and assistant messages during runtime. This satisfies the requirement while keeping the design simple and easy. It also allows the system to maintain conversational continuity without adding unnecessary database complexity.

## 6) Why use simple guardrails?
The assignment asked for simple guardrails, so I implemented lightweight but practical protections. These include blocking obvious prompt-injection style phrases, restricting unsafe user instructions, blocking destructive SQL operations, and allowing only safe read-only SQL queries on the intended tables. This makes the system safer without overcomplicating the design.

## 7) Why use a separate `save_note` tool if there is already an SQL executor?
I separated the note-saving function from the SQL executor because they serve different purposes. The `save_note` tool provides one safe and predictable write path into the database, while the SQL executor remains read-only and is used only for inspection and retrieval. This separation is technically safer and easier to justify than allowing the model to generate arbitrary database write statements.

## 8) Why use Whisper for voice input?
I used Whisper for speech-to-text because it provides a simple Python interface for transcription and is well suited for converting spoken questions into text that can be passed into the RAG system. This made it straightforward to extend the text-based assistant into a voice-enabled assistant without redesigning the rest of the pipeline.

## 9) Why use `pyttsx3` and `gTTS` for voice output?
I used `pyttsx3` as the first option because it can work offline and generate speech files directly on the machine. I included `gTTS` as a fallback because it is simple, widely used, and reliable for producing audio output when offline synthesis is unavailable. This dual approach improves robustness while keeping implementation simple.

## 10) Why use manual evaluation instead of RAGAS?
The assignment allowed either RAGAS or manual evaluation. I chose manual evaluation because it was simpler to complete and easier to explain. My evaluation setup uses 15 test questions, compares the answers against expected sources and concepts, and exports detailed results to a CSV file. I also included blank human-scoring columns so correctness, groundedness, and helpfulness can be rated manually if needed. This makes the evaluation transparent and easy to inspect.

## 11) Why these evaluation metrics?
I used a set of lightweight proxy metrics that are easy to compute and interpret:

- **retrieval_hit_at_k**: checks whether the expected source appeared among the retrieved results  
- **citation_present**: checks whether the answer includes citation-style evidence markers  
- **keyword_recall_proxy**: checks whether the answer covers important expected concepts  
- **top1_similarity**: measures how strong the best retrieved chunk match was  

These are not meant to replace full human evaluation or advanced frameworks, but they provide a practical and defensible way to assess whether the retriever found the right information and whether the generated answer reflects the important content.

## 12) Why these technical choices are appropriate overall
Across the three assignments, I prioritized modularity, clarity, and stability. I kept the RAG engine reusable, exposed it through tools, added only the minimum required agent structure, used simple but effective guardrails, and designed the evaluation to be understandable and reproducible. This made the project easier to implement, and easier to debug.