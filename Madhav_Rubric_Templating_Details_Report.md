
# Science Rubric Templating & Question Generation Report

**Project:** NSD-AI Essay v3 – AI-Driven Auto Scoring Platform  
**Module:** Science Subject Rubric Generation  
**Author:** Madhav Nepal  
**Date:** March 18, 2026  

---

## 1. Introduction
This report documents the work completed on **Science Rubric Templating** and **Question Generation**, detailing the steps taken to create **structured, AI-compatible rubrics** and **automated assessment items**.  

The project objectives were:  
- To convert **teacher inputs** into **machine-readable rubrics** for automated scoring.  
- To generate **AI-based assessment questions** aligned with curriculum standards.  
- To validate that generated questions and rubrics are **structurally consistent and accurate**.  

---

## 2. Task 1 – Rubric Templating

### 2.1 Objective
- Develop **structured Science rubrics** that can support automated scoring.  
- Ensure rubrics follow the **CREQ framework** (Concept Reasoning, Evidence, Quantitative).  
- Align evaluation dimensions with **curriculum standards** and backend scoring logic.  

### 2.2 Methodology
1. **Understanding Existing Rubrics**
   - Reviewed old reports and templates provided by the team.  
   - Studied **teacher-facing rubrics vs backend rubrics**.  
   - Analyzed curriculum standards from the **Korean government**.

2. **CREQ Framework Analysis**
   - Learned how CREQ ensures **structured, accurate, and observable evaluation** in AI-driven education.  
   - Explored how **each dimension can remain independent** for stable scoring:  
     - **Concept** – Scientific correctness and understanding  
     - **Evidence** – Use of provided materials and references  
     - **Reasoning** – Logical explanation of responses  
     - **Quantitative** – Numerical accuracy and calculations  

3. **Rubric Design & Criteria Mapping**
   - Created evaluation **criteria** and mapped them to appropriate **dimensions**.  
   - Identified inconsistencies in **axis-service mapping**.  
   - Defined clear **axis-to-service policies**, aligning:  
     - Reasoning → Reasoning axis  
     - Numeric tools → Quantitative axis  
     - Retrieval-based tools → Evidence axis  

4. **Detail-Level Units & Scoring Groups**
   - Broke down rubrics into **detail-level scoring units** for precise evaluation.  
   - Aggregated details into **scoring groups** to calculate total question scores.  
   - Maintained **dual rubric structure**:  
     - `original` – Teacher-facing  
     - `backend` – Machine-readable scoring logic  

5. **JSON Schema Implementation**
   - Developed structured JSON schema with sections:  
     - `source_info` – Traceability  
     - `original` – Teacher rubric  
     - `backend` – Scoring engine  
     - `materials` – Reference texts  
   - Defined **microservices** for evaluation:  
     - IntDic, RAG Service, Discourse, Connector, Science Numeric Parser  
   - Ensured pipeline flow: Student Answer → Microservices → Metrics → Backend Rubric → Scoring Groups → Final Score  

### 2.3 Challenges
- Overlapping semantics between axes caused **misalignment of evaluation services**.  
- Teacher input variations required careful mapping to **backend scoring logic**.  

### 2.4 Outcomes
- Established **structured rubric pipeline** for Science subject.  
- Achieved **consistent axis mapping** and fine-grained evaluation.  

---

## 3. Task 2 – Question Generation

### 3.1 Objective
- Generate structured assessment questions from teacher inputs, materials, and achievement standards.  
- Ensure generated questions align with curriculum and evaluation criteria.  
- Provide inputs for rubric creation.  

### 3.2 Methodology
1. **Pipeline Review**
   - Studied the question generation pipeline, including JSON schema:  
     - `meta`, `group_material`, `question`, `feedback`  
   - Analyzed how **materials, prompts, and achievement standards** interact.

2. **Test Generation**
   - Generated multiple test questions using different **materials and achievement standards**.  
   - Verified structural consistency and alignment with curriculum standards.

3. **API Integration**
   - Implemented **Question Generator API** across all subject modules.  
   - Verified API processing of inputs and correct structured outputs.  

4. **Preliminary Validation**
   - Designed checks to ensure **material-topic alignment** before generation.  
   - Documented how fields like `group_material.stem`, `question.stem`, `constraint`, and `response_template` affect question clarity and structure.  

### 3.3 Challenges
- Ensuring consistent alignment between **generated questions and curriculum standards**.  
- Managing variations in material inputs to avoid generating irrelevant questions.  

### 3.4 Outcomes
- Question Generator is **fully operational for all subjects**.  
- Generated questions are **structurally consistent and curriculum-aligned**.  
- Provides reliable input for **rubric creation and automated scoring**.  

---

## 4. Task 3 – Validation of Questions & Rubrics

### 4.1 Objective
- Validate that **generated questions** and **rubric mappings** are correct, consistent, and suitable for automated assessment.  

### 4.2 Methodology
1. Reviewed all **generated question files** and **rubric mapping files**.  
2. Verified:
   - Question IDs  
   - Material references  
   - Total scores  
   - Evaluation factors  
   - JSON structural integrity 

### 4.3 Validation Results
| Category | Result |
|---|---|
| Question–Rubric Alignment | ✓ Correct |
| Material References | ✓ Consistent |
| Score Allocation | ✓ Correct |
| Rubric Structure | ✓ Valid |
| JSON Structural Integrity | ✓ Valid |

### 4.4 Observations
- All question and rubric files were **structurally valid**.  
- **Rubric factors accurately corresponded** to expected student responses.  
- **Scores and evaluation dimensions** were consistent.  

### 4.5 Conclusion
- All generated questions and rubrics are **correct, aligned, and ready for automated scoring pipelines**.  
- Validation confirms the **integrity and reliability** of the entire Science assessment system.  

---

## 5. Final Outcome
- Successfully implemented **Science rubric templating**, **question generation**, and **validation pipeline**.  
- Ensured **alignment with curriculum standards** and **AI-driven automated assessment requirements**.  
- Developed **robust JSON schema and scoring logic** suitable for production use.  

---

**End of Report**
