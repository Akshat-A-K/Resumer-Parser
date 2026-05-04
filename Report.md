# T13 - Smart Document Parser: Resume Entity Extraction

## Project Report - Assignment 3

---

## Abstract

This report presents our implementation of a Smart Document Parser for resume entity extraction. The task is T13.1 which is Resume Parser from the assignment. We built a Streamlit web application that takes a resume file as input in PDF or image format and extracts all the important entities like name, email, skills, education, experience and projects into a clean structured JSON output. The user can also edit the extracted output and download it. We used OCR for text extraction and a local LLM (Llama 3.2 via Ollama) for entity extraction with strict JSON schema prompting. We also built an evaluation pipeline that compares our extraction output with the NER annotated dataset and computes precision, recall and F1 score per entity field. On 10 samples the system achieved a macro average precision of 0.585, recall of 0.617 and F1 of 0.575.

---

## I. Introduction

In HR tech companies and startups in India there is a very common problem. When a job opening is posted the company receives hundreds or thousands of resumes. Reading each resume manually and pulling out the candidate details is very time consuming work. To make this process faster we need an automated system that can read a resume file and extract all the relevant information into a structured format.

The assignment T13 asks us to build a Smart Document Parser. The problem statement says that user uploads a resume or invoice or certificate and the app extracts a clean structured JSON from it. For our variant T13.1 we chose the resume parsing task. The specific dataset given is the Kaggle resume entities dataset which has NER annotated resume data.

Our application is a Streamlit based web app where user can upload a PDF or image resume. The app does OCR on the file to get the raw text. Then it sends this text to a Large Language Model with a strict JSON schema prompt. The LLM returns the extracted data in JSON format. We also validate the output using Pydantic schema and show it in an editable form in the UI. The user can review the output, make corrections and download the final JSON.

We also implemented a batch evaluation mode where the system processes multiple labeled samples from the dataset and calculates precision, recall and F1 score for each entity type.

---

## II. Problem Statement

Given a resume document in PDF or image format the system should extract the following entity types into structured JSON output:

- Personal Information: name, email, phone number, location
- Social Links: LinkedIn, GitHub, portfolio website, Twitter
- Professional Details: designation, companies worked at, years of experience, professional summary
- Education: college name, degree, graduation year, structured education details with institution, degree, field of study, CGPA, start year and end year
- Skills: flat skill list and categorized skills grouped by type
- Projects: project names and structured project details with name, tech stack, description, link and date
- Experience: structured experience details with company, role, location, start date, end date, description and tech stack
- Others: certifications, languages, achievements, publications, hobbies, references

The system should be able to handle different resume formats and layouts. It should give clean output even when the resume has icons, special characters or complex formatting.

---

## III. Tools and Technologies Used

### A. Tools We Used

1. **Python 3** - Main programming language for the whole project.

2. **Streamlit** - We used Streamlit for building the web user interface. It provides easy way to create interactive web apps with just Python code. We chose it because it was mentioned in the assignment requirements and it is very simple to use.

3. **PyMuPDF (fitz)** - This is our primary PDF text extraction library. We chose PyMuPDF because it gives much better text quality compared to pypdf. It handles different PDF layouts well and also extracts hyperlink annotations from the PDF which helps us get LinkedIn and GitHub URLs directly.

4. **pypdf** - We kept pypdf as a fallback PDF reader. If PyMuPDF fails for some reason then pypdf is used. It also supports layout mode extraction and can read hyperlink annotations.

5. **EasyOCR** - For image based resumes we used EasyOCR. It supports English text recognition and works without GPU also. When user uploads a PNG or JPG image file we convert it to numpy array using Pillow and pass it to EasyOCR reader.

6. **Pillow and NumPy** - Used together with EasyOCR for image processing. Pillow opens the image and NumPy converts it to array format that EasyOCR expects.

7. **Ollama with Llama 3.2 (3B)** - This is our main LLM for entity extraction. Ollama runs locally on the machine so there is no API cost. We used the llama3.2:3b model. We set temperature to 0 for deterministic output and forced JSON format in the response.

8. **Pydantic** - For schema validation. We defined ResumeEntity model with all the fields and their types. Pydantic automatically validates the data types, converts strings to lists where needed and catches any wrong format. We also defined nested models like EducationEntry, ExperienceEntry and ProjectEntry for structured extraction.

9. **pandas** - Used for creating dataframes of evaluation results and displaying them in the Streamlit UI as tables.

10. **python-dotenv** - For loading environment variables from .env file. We store Ollama host URL in the .env file.

11. **requests** - For making HTTP POST requests to the Ollama API endpoint.

12. **google-generativeai** - We included the Gemini API library in requirements but in our final implementation we only used Ollama. The Gemini provider code is there in the codebase but it is disabled. We wrote the function extract_with_gemini() but the main extract_entities() function only routes to Ollama.

### B. Tools We Did Not Use

1. **PaddleOCR** - The assignment mentioned paddleocr as one option for OCR. We did not use it because EasyOCR was giving us good enough results and was easier to set up on Windows.

2. **Groq API** - The assignment mentioned free Gemini or Groq Llama-3 API. We tried both but finally settled on local Ollama because it does not need internet connection and has no rate limits. Groq has rate limits which cause problems during batch evaluation.

3. **Gemini API** - We wrote the code for Gemini integration but disabled it in the final version. The main reason is that for evaluation we need to process many samples and Gemini free tier has rate limits. Ollama running locally does not have this problem.

4. **Fine tuning** - The assignment mentions no training is needed. We did not do any model fine tuning. We used the base Llama 3.2 model with prompt engineering only.

5. **Batch mode for Tier 2** - The assignment mentions Tier 2 has batch mode for bulk processing. We did not implement bulk file upload batch mode. But our evaluation mode does process multiple samples in a loop which is similar.

---

## IV. System Architecture and Implementation

Our system has 5 main modules. We explain each one below.

### A. OCR and Text Extraction Module (ocr_extractor.py)

This module handles all the file reading and text extraction work. It has separate functions for PDF and image files.

For PDF files we first try PyMuPDF. It opens the PDF from bytes and extracts text page by page using get_text("text") method. It also reads all the hyperlink annotations from each page using get_links() method. This gives us the actual URL targets embedded in the PDF like LinkedIn profile links and GitHub repository links.

If PyMuPDF fails we fall back to pypdf. It uses PdfReader to extract text. It also tries layout mode extraction first for better formatting. For link extraction it reads /Annots from each page and checks for /Link subtypes with /URI actions.

After getting the raw text we clean it. We have a list of common PDF artefact patterns like icon font characters (bullet symbols, phone icons, envelope icons, LinkedIn icons etc.) that appear in resumes with fancy templates. We use regex to replace all these with clean text. We also collapse extra blank lines and spaces.

For image files we use EasyOCR. We open the image with Pillow, convert to RGB, then to numpy array. We create an EasyOCR Reader with English language and run readtext with paragraph mode. This gives us text chunks which we join with newlines.

We also have a URL extraction function that uses regex to find URLs in plain text. It handles full URLs, shorthand domain patterns like linkedin.com/in/username and mailto links. All extracted URLs are normalized to have https:// prefix if missing.

### B. Schema Module (schema.py)

This module defines the data structure for our extracted resume data. We used Pydantic BaseModel for this.

We defined three nested models for structured extraction:
- **EducationEntry** - Has fields for institution, degree, field_of_study, cgpa, location, start_year and end_year. All fields are Optional[str].
- **ExperienceEntry** - Has fields for company, role, location, start_date, end_date, description and tech_stack (list of strings).
- **ProjectEntry** - Has fields for name, tech_stack (list), description, link and date.

The main model is **ResumeEntity** which has 25+ fields covering all entity types. It has flat string fields (name, email, phone etc.), list fields (skills, designation, companies_worked_at etc.), a dict field (skills_categorized) and structured list fields (education_details, experience_details, projects_detailed).

We also defined FIELD_SPECS dictionary that maps each field name to its label and kind (string, list, dict or structured). This is used by the UI to show proper labels and by the parser to build correct prompts.

We added field validators using Pydantic decorator. These validators handle normalization at input time. For list fields if the input is a comma separated string it splits into a list. For structured fields if the input is a list of dicts it converts them to the proper Pydantic model objects.

The **backfill_flat_fields()** method is important. After extraction the LLM fills the structured fields like education_details. But we also need the old flat fields like college_name and degree to be populated for backward compatibility with the evaluation dataset. This method automatically copies data from structured fields to flat fields if they are empty.

### C. LLM Parser Module (llm_parser.py)

This is the core extraction engine. It builds prompts, calls the LLM and processes the response.

**Prompt Building**: We build a detailed prompt that includes the annotated JSON schema, a strict JSON template with all the requested keys, the list of allowed keys, and the resume text. If extracted links are available we inject them into the prompt with classification labels (LinkedIn Profile, GitHub Project etc.) so the LLM can map them correctly.

The system prompt tells the model to act as a precise resume parser. It must extract information only from the provided text and not fabricate values. It must return only valid JSON with no explanations.

**Section-wise Extraction**: Instead of sending the entire resume text in one prompt we split the resume into sections first. We detect headings like Education, Experience, Projects, Skills, Achievements etc. using keyword matching. Then for each section we only send the relevant text chunk with only the relevant fields. For example the education section text is sent with only education related fields. This gives much better accuracy because the LLM can focus on less text and fewer fields at a time.

The section to field mapping is:
- contact section -> name, email, phone, location, linkedin, github, portfolio, twitter, other_links, summary
- education section -> education_details, college_name, degree, graduation_year, certifications
- experience section -> experience_details, designation, companies_worked_at, years_of_experience
- projects section -> projects_detailed, projects
- skills section -> skills, skills_categorized, languages
- achievements section -> achievements, publications, hobbies, references

After getting outputs from all sections we merge them using _merge_section_results(). For list and structured fields it concatenates the lists. For string fields it takes the first non empty value. For dict fields it merges the dictionaries.

**Ollama API Call**: We call the Ollama REST API at /api/chat endpoint. We use chat format with system and user messages. We set temperature to 0, format to json and configure num_predict and num_ctx based on input size. For smaller inputs (under 6000 chars) we use 512 predict tokens and 2048 context. For larger inputs we use 1024 predict and 4096 context.

We have retry logic at multiple levels. If the request times out we retry with minimal generation settings. If the JSON parsing fails we retry with a stricter prompt. If the output is too sparse (only 1 field filled) we retry with a message telling the model to extract more fields. If the output has extra keys not in the schema we retry asking to remove them.

**JSON Parsing and Repair**: The LLM sometimes gives slightly broken JSON. We have robust parsing logic. First we try direct json.loads(). If that fails we try repairing common mistakes like NULL instead of null, None instead of null, NaN instead of null and missing commas between values. We also strip markdown code fences. If all that fails we use regex to extract the JSON object from the text and try again.

**Data Normalization**: After parsing we clean the data. Phone numbers are stripped of non numeric characters except +, -, () and spaces. URLs starting with www. get https:// prefix added. Dates are normalized to YYYY-MM format. Numeric strings like "2020.0" are cleaned to "2020". For structured fields each entry is cleaned individually.

**Link Mapping**: We have special logic to map GitHub project URLs to project entries. If the resume has GitHub repository links and the extracted projects have missing links we try to match them by comparing project names with repository slugs using token overlap. If there is only one project and one GitHub link we map directly.

**Caching**: We cache all extraction results using SHA-256 hash. The hash is computed from the text content, provider name, model name, selected fields and truncation limit. Results are stored as JSON files in the cache/ directory. If the same resume with same settings is processed again we return the cached result instantly. This saves a lot of time during development and testing.

### D. Evaluator Module (evaluator.py)

This module handles the quality evaluation of our extraction system.

**Ground Truth Loading**: The dataset is in newline delimited JSON format. Each line has a content field with the resume text and an annotation field with labeled entities. We load this using json.loads on each line.

**Annotation Conversion**: Each annotation has a label (like "Name", "Email Address", "Skills" etc.) and points with the actual text value. We map these labels to our schema field names using a LABEL_MAP dictionary. For single valued fields we take the first occurrence. For list fields we collect all unique values.

**Evaluation Metrics**: We compute multiple metrics for each field.

For string fields (name, email, location etc.):
- Exact Match: 1 if normalized prediction equals normalized gold, 0 otherwise
- Fuzzy Match: SequenceMatcher ratio between normalized strings
- Token level Precision, Recall, F1: we tokenize both strings and compute set based metrics on tokens

For list fields (skills, designation, companies_worked_at etc.):
- Set based Precision, Recall, F1 with fuzzy matching: we use a threshold of 0.75 on SequenceMatcher ratio to decide if a predicted item matches a gold item
- Jaccard similarity: intersection over union of normalized item sets

All text is normalized before comparison. Normalization includes lowercasing, stripping, collapsing whitespace and removing punctuation. Numeric strings and dates are also normalized.

**Batch Evaluation**: For a batch of samples we compute per field metrics averaged over all samples. We also compute macro average precision, recall and F1 over all fields that have ground truth support.

### E. Streamlit Application (app.py)

The web interface has two modes.

**Parse One Resume Mode**: This is the main user facing mode.
1. Step 1: User selects which fields to extract from a multiselect dropdown. All 25+ fields are available. By default all fields are selected.
2. Step 2: User uploads a PDF or image file. The app extracts text and links and shows a preview.
3. Step 3: User clicks Run Extraction button. The app calls the LLM parser and shows the results.

The results are displayed in multiple ways:
- Categorized skills are shown in a two column grid with category headers
- Structured data (education, experience, projects) is shown in expandable cards with formatted fields. Each card shows relevant details like institution, degree, CGPA, period etc.
- Flat fields are shown in an editable form where user can modify values
- Full JSON output is displayed and user can download it as a JSON file

We use Streamlit session state to persist results across reruns. This prevents losing data when the user interacts with other widgets.

Link classification is done for display purposes. When showing extracted links we label them as LinkedIn Profile, GitHub Profile, GitHub Project, Kaggle Profile, LeetCode Profile etc. based on URL pattern matching.

**Evaluate Quality Mode**: This mode is for testing the system accuracy.
1. User selects number of samples (1 to 220)
2. Clicks Run Evaluation
3. System loads ground truth from the annotation JSON file
4. For each sample it extracts entities using the same pipeline
5. Compares predictions with ground truth
6. Shows a dataframe with per field metrics
7. Saves results as JSON and CSV in the results/ folder with timestamp

---

## V. Dataset

We used the **Resume Entities for NER** dataset from Kaggle (kaggle.com/datasets/dataturks/resume-entities-for-ner). This dataset has 220 annotated resume samples in newline delimited JSON format. Each sample has the resume text content and NER annotations with labeled entity spans.

The annotations cover these entity types: Name, Email Address, Location, Designation, Companies worked at, Skills, College Name, Degree, Graduation Year and Years of Experience.

The dataset does not have annotations for all our schema fields. Fields like phone, linkedin, github, projects, certifications, achievements, publications, hobbies and references do not have ground truth labels. So during evaluation we can only measure accuracy for the 10 mapped entity types.

---

## VI. Results

We ran the evaluation on 10 samples using Ollama with Llama 3.2 (3B) model in section-wise prompting mode.

### Per Field Results

| Entity Type | Precision | Recall | F1 Score | Support |
|---|---|---|---|---|
| Name | 0.900 | 0.900 | 0.900 | 9 |
| Email | 0.700 | 0.600 | 0.633 | 8 |
| Location | 0.500 | 0.900 | 0.630 | 8 |
| Graduation Year | 0.400 | 0.400 | 0.400 | 7 |
| Years of Experience | 0.600 | 0.550 | 0.566 | 2 |
| College Name | 0.783 | 0.725 | 0.745 | 10 |
| Degree | 0.800 | 0.650 | 0.700 | 8 |
| Designation | 0.600 | 0.583 | 0.550 | 8 |
| Companies Worked At | 0.541 | 0.766 | 0.586 | 8 |
| Skills | 0.028 | 0.100 | 0.044 | 10 |

### Macro Average

| Metric | Score |
|---|---|
| Precision | 0.585 |
| Recall | 0.617 |
| F1 Score | 0.575 |

### Analysis of Results

The system performs best on Name extraction with 0.9 F1 score. This makes sense because name is usually at the top of resume and is easy to identify.

College Name and Degree also have good scores (0.745 and 0.700 F1). Education section is usually well formatted in resumes.

Location has good recall (0.9) but low precision (0.5). This means the model finds the location most of the time but sometimes extracts wrong text as location.

Skills has the lowest score (0.044 F1). This is mainly because the ground truth dataset has skills in different format than what our model extracts. Our model categorizes skills and gives individual skill names while the ground truth might have skill groups or slightly different naming. The fuzzy matching threshold of 0.75 is also quite strict for skill matching.

---

## VII. Challenges Faced

1. **JSON Parsing Issues**: The LLM sometimes returns broken JSON with missing commas or trailing commas. We had to write robust parsing logic with multiple fallback strategies.

2. **PDF Artefacts**: Many resume PDFs use icon fonts for phone, email and social media icons. These show up as garbage characters in extracted text. We had to add specific regex patterns to clean them.

3. **Token Limits**: The Llama 3.2 3B model has limited context window. For long resumes the text gets truncated. Section-wise extraction helps with this because each section is shorter.

4. **Skill Matching**: Skills are hard to match between prediction and ground truth because same skill can be written differently (for example "JS" vs "JavaScript" or "ML" vs "Machine Learning").

5. **Rate Limits**: When we tried using cloud APIs (Gemini, Groq) for batch evaluation we kept hitting rate limits. Using local Ollama solved this problem.

---

## VIII. Conclusion

We built a working resume parser application that extracts entities from resume documents into structured JSON. The system uses PyMuPDF and EasyOCR for text extraction and Llama 3.2 via Ollama for LLM based entity extraction. We used section-wise prompting to improve accuracy by sending relevant text chunks with relevant fields only.

The system achieves 0.585 macro precision, 0.617 macro recall and 0.575 macro F1 on the evaluation dataset. It performs well on name, education and degree fields. Skills matching is the weakest area which can be improved with better normalization and looser matching criteria.

The Streamlit interface allows users to upload resumes, see extracted data in structured format, edit the output and download JSON. The evaluation mode helps in measuring extraction quality against labeled data.

For future work we can try using a bigger model (like Llama 3 8B or 70B) for better extraction. We can also add better skill normalization by maintaining a skill synonym dictionary. Another improvement can be supporting more document types like invoices and certificates as mentioned in the T13 assignment.

---

## IX. How to Run

1. Create virtual environment: python -m venv .venv
2. Activate it: .\.venv\Scripts\Activate.ps1 (on PowerShell)
3. Install dependencies: pip install -r requirements.txt
4. Make sure Ollama is running locally with llama3.2:3b model pulled
5. Run the app: streamlit run app.py
6. Open the URL shown in terminal

---

## X. File Structure

| File | Purpose |
|---|---|
| app.py | Streamlit web application with Parse and Evaluate modes |
| llm_parser.py | LLM based entity extraction with Ollama, prompt building, JSON parsing, caching |
| ocr_extractor.py | PDF and image text extraction using PyMuPDF, pypdf and EasyOCR |
| schema.py | Pydantic data models, field specifications, validators |
| evaluator.py | Evaluation metrics computation - precision, recall, F1, jaccard |
| requirements.txt | Python package dependencies |
| Entity Recognition in Resumes.json | Ground truth NER annotated dataset from Kaggle |
| cache/ | Cached extraction results (SHA-256 hashed) |
| results/ | Saved evaluation outputs in JSON and CSV format |

---

## References

[1] Kaggle Resume Entities for NER Dataset. https://kaggle.com/datasets/dataturks/resume-entities-for-ner

[2] Streamlit Documentation. https://docs.streamlit.io

[3] Ollama Documentation. https://ollama.ai

[4] PyMuPDF Documentation. https://pymupdf.readthedocs.io

[5] Pydantic Documentation. https://docs.pydantic.dev

[6] EasyOCR GitHub. https://github.com/JaidedAI/EasyOCR
