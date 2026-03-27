## Assignment 1 — ReAct-style agent around the RAG tool

note: the path is hidden for security purposes hehehe

(.venv) PS C:#######> python 01_react_rag_agent.py
Building graph...
Loading weights: 100%|███████████████████████████████████████████████████████████████| 103/103 [00:00<00:00, 1302.14it/s]
BertModel LOAD REPORT from: sentence-transformers/all-MiniLM-L6-v2
Key                     | Status     |  |
------------------------+------------+--+-
embeddings.position_ids | UNEXPECTED |  |

Notes:
- UNEXPECTED    :can be ignored when loading from different task/architecture; not ok if you expect identical arch.       
Batches: 100%|█████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  1.43it/s]
Graph built. Running first question...

First response:
The recommended first step is to increase spacing to improve air movement, prune to improve air movement, avoid excess nitrogen late in the season, and remove heavily infected leaves early.

Second response:
The result of the calculation is 20.

---

## Assignment 2 — Multi-agent supervisor with database specialist

(.venv) PS C:########> python 02_multi_agent_sql_guardrails.py
Loading weights: 100%|███████████████████████████████████████████████████████████████| 103/103 [00:00<00:00, 3835.48it/s]
BertModel LOAD REPORT from: sentence-transformers/all-MiniLM-L6-v2
Key                     | Status     |  | 
------------------------+------------+--+-
embeddings.position_ids | UNEXPECTED |  | 

Notes:
- UNEXPECTED    :can be ignored when loading from different task/architecture; not ok if you expect identical arch.
Batches: 100%|█████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  1.52it/s]
Saved note under tag='diseases'.
To address powdery mildew on zucchini, first, increase spacing and prune to improve air movement, remove heavily infected leaves early, monitor the disease twice per week, and use a simple severity score to track the disease progression.       
[{'id': 1, 'note': 'zucchini plot A likely has powdery mildew', 'tag': 'diseases', 'created_at': '2026-03-23 23:29:52'}]

---

## Assignment 3 — Voice input/output

(.venv) PS C:######> python 03_voice_and_manual_eval.py --mode eval
Loading weights: 100%|████████████████████████████████████████████████| 103/103 [00:00<00:00, 3465.56it/s]
BertModel LOAD REPORT from: sentence-transformers/all-MiniLM-L6-v2
Key                     | Status     |  | 
------------------------+------------+--+-
embeddings.position_ids | UNEXPECTED |  | 

Notes:
- UNEXPECTED    :can be ignored when loading from different task/architecture; not ok if you expect identical arch.
Batches: 100%|██████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  1.51it/s]
Starting evaluation on 15 questions...
Evaluating question 1/15
Evaluating question 2/15
Evaluating question 3/15
Evaluating question 4/15
Evaluating question 5/15
Evaluating question 6/15
Evaluating question 7/15
Evaluating question 8/15
Evaluating question 9/15
Evaluating question 10/15
Evaluating question 11/15
Evaluating question 12/15
Evaluating question 13/15
Evaluating question 14/15
Evaluating question 15/15

Evaluation summary:
- n_questions: 15
- retrieval_hit_rate_at_k: 1
- citation_presence_rate: 1
- avg_keyword_recall_proxy: 0.644
- avg_top1_similarity: 0.549

Saved detailed results to: manual_eval_results.csv

---

(.venv) PS C:#######> python 03_voice_and_manual_eval.py --mode mic --seconds 8 --play
Speak now for 8 seconds...
Recorded microphone audio to: mic_question.wav
Loading weights: 100%|████████████████████████████████████████████████| 103/103 [00:00<00:00, 5179.52it/s]
BertModel LOAD REPORT from: sentence-transformers/all-MiniLM-L6-v2
Key                     | Status     |  | 
------------------------+------------+--+-
embeddings.position_ids | UNEXPECTED |  | 

Notes:
- UNEXPECTED    :can be ignored when loading from different task/architecture; not ok if you expect identical arch.
Batches: 100%|██████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  1.86it/s]
C########\transcribe.py:132: UserWarning: FP16 is not supported on CPU; using FP32 instead
  warnings.warn("FP16 is not supported on CPU; using FP32 instead")

Transcribed question:
Why the kidney leaves have white boundary spots? What should I do first?

Answer:
Based on the provided context [01_crop_diagnostics.txt::chunk1], the symptoms described for powdery mildew on cucurbits include white, powder-like patches on the upper leaf surface.

Given the description of the kidney leaves having white boundary spots, it is likely that the issue is powdery mildew [01_crop_diagnostics.txt::chunk1].

First, inspect the underside of the leaves for mites, eggs, and fungal growth [01_crop_diagnostics.txt::chunk0]. This is a crucial step in the fast field checklist to determine the cause of the issue.

Answer audio file: voice_answer.wav



NOTE
----
NOT KIDNEY! IT IS ZUCCHIINI!!