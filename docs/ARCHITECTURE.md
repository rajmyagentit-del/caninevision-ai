# CanineVision AI — System Architecture

## 1. Project Overview

CanineVision AI is an AI-powered dog image intelligence system.

The application accepts an image of a dog and attempts to identify the most likely breed or breed candidates. It then retrieves structured, evidence-backed information about the predicted breed and presents the information through a user-friendly web interface.

The system is designed to combine:

- Computer vision
- Fine-grained image classification
- Multimodal AI
- Structured breed knowledge
- Retrieval-Augmented Generation (RAG)
- Evidence-based information retrieval
- A production-style web application

The system must communicate uncertainty and must not claim that a breed prediction is 100% certain when the visual evidence is insufficient.

---

## 2. Project Goals

The primary goals are:

1. Accept a dog image from a user.
2. Determine whether the image contains a dog.
3. Predict the most likely dog breed.
4. Return multiple candidate breeds when appropriate.
5. Provide model confidence information.
6. Detect potentially uncertain or mixed-breed cases.
7. Map predictions to a canonical breed taxonomy.
8. Retrieve reliable breed information.
9. Provide information about breed origin and history.
10. Provide physical characteristics.
11. Provide typical height, weight, and lifespan.
12. Provide temperament information.
13. Provide exercise and grooming information.
14. Provide common health information.
15. Provide general vaccination information.
16. Provide bite-force information only when evidence is available.
17. Provide citations and source references.
18. Provide a polished public web interface.
19. Provide a public HTTPS demonstration when deployment is complete.
20. Provide reproducible evaluation results.

---

# 3. High-Level System Architecture

```text
                         USER
                           |
                           v
                +----------------------+
                |     Web Frontend     |
                |   Upload Dog Image   |
                +----------+-----------+
                           |
                           v
                +----------------------+
                |   Image Validation   |
                | Type / Size / Quality |
                +----------+-----------+
                           |
                           v
                +----------------------+
                |   Vision Pipeline    |
                | Dog Detection /      |
                | Breed Classification |
                +----------+-----------+
                           |
                           v
                +----------------------+
                |   Top-N Predictions  |
                | Confidence Scores    |
                +----------+-----------+
                           |
                           v
                +----------------------+
                |   Breed Taxonomy     |
                | Canonical Breed ID   |
                +----------+-----------+
                           |
                +----------+-----------+
                |                      |
                v                      v
        +---------------+      +---------------+
        | Breed         |      | RAG Retrieval |
        | Knowledge DB  |      | Evidence      |
        +-------+-------+      +-------+-------+
                |                      |
                +----------+-----------+
                           |
                           v
                +----------------------+
                | AI Response Layer    |
                | Evidence-Grounded    |
                | Response Generation  |
                +----------+-----------+
                           |
                           v
                +----------------------+
                | Structured Result    |
                | Prediction + Facts   |
                | + Sources + Warnings  |
                +----------+-----------+
                           |
                           v
                         USER
4. User Request Flow

The expected user experience is:

Upload Dog Image
       |
       v
Image Preview
       |
       v
Click "Analyze"
       |
       v
Image Validation
       |
       v
Dog / Image Analysis
       |
       v
Breed Prediction
       |
       v
Confidence Evaluation
       |
       v
Knowledge Retrieval
       |
       v
Evidence Retrieval
       |
       v
AI Response Generation
       |
       v
Structured Breed Report
       |
       v
Sources and References
5. Image Validation

Before an uploaded image is passed to the AI system, the application should validate it.

Validation should include:

Supported image format
Maximum file size
Image dimensions
Image readability
Corrupted image detection
Basic image quality checks

Potentially supported formats include:

JPEG
PNG
WEBP

The system should reject unsupported or unsafe files.

The application should not expose uploaded images unnecessarily.

6. Computer Vision Pipeline

The computer vision system is responsible for analyzing the visual characteristics of the dog.

The initial model strategy will use transfer learning rather than training a large neural network completely from scratch.

Candidate architectures include:

ResNet
EfficientNet
ConvNeXt
Vision Transformer (ViT)
DINOv2-based visual representations
CLIP/SigLIP-style models where appropriate

The final model will be selected based on actual experiments.

The selection criteria will include:

Accuracy
Top-5 accuracy
Precision
Recall
F1 score
Calibration
Inference latency
Model size
Cloud deployment requirements
Computational cost

No performance metric will be reported until it has been experimentally measured.

7. Breed Prediction

The model should not simply return one breed.

It should return multiple candidates.

Example:

{
  "top_prediction": {
    "breed": "German Shepherd",
    "confidence": 0.82
  },
  "alternatives": [
    {
      "breed": "Belgian Malinois",
      "confidence": 0.09
    },
    {
      "breed": "Dutch Shepherd",
      "confidence": 0.04
    }
  ]
}

The system must distinguish between:

Model confidence
Prediction probability
Actual certainty

A high model score does not guarantee that the prediction is correct.

8. Uncertainty Handling

Uncertainty is a core feature of CanineVision AI.

The system should not force a prediction when the evidence is weak.

Example: High Confidence
Most likely breed:
German Shepherd

Confidence:
91%

Prediction confidence is high.
Example: Moderate Confidence
Most likely breed:
German Shepherd

Confidence:
58%

Other possibilities:
- Belgian Malinois
- Dutch Shepherd

The model has moderate confidence.
Example: Low Confidence
The model could not confidently determine the breed from this image.

Possible candidates:
- German Shepherd
- Belgian Malinois
- Dutch Shepherd

This behavior is preferable to giving the user a misleading definitive answer.

9. Mixed-Breed Handling

Visual breed identification cannot reliably determine whether a dog is genetically purebred.

Therefore, the application should avoid claiming that an animal is definitely purebred based only on an image.

A possible result could be:

Possible mixed breed

Visual similarities:
- German Shepherd
- Belgian Malinois
- Dutch Shepherd

The available image does not provide enough evidence for a reliable purebred classification.

The system should clearly distinguish:

Breed prediction
Visual similarity
Genetic identification

CanineVision AI is primarily a visual analysis system.

10. Breed Taxonomy

Different datasets may use different names for the same breed.

For example:

Golden_Retriever
Golden Retriever
golden retriever
Golden Retriever Dog

should map to a single canonical breed entity.

The taxonomy should contain:

Canonical breed ID
Canonical breed name
Aliases
Dataset-specific names
Breed group
Country of origin
Region of origin
External identifiers where available

Example:

{
  "breed_id": "golden_retriever",
  "canonical_name": "Golden Retriever",
  "aliases": [
    "Golden Retriever",
    "Golden_Retriever"
  ]
}
11. Dataset Strategy

The project will investigate multiple legitimate publicly available datasets.

Potential datasets include:

Stanford Dogs
Oxford-IIIT Pet
ImageWoof
DogFLW
Kaggle dog breed datasets
Hugging Face dog datasets
Other legitimate research datasets

Each dataset must be evaluated independently.

For every dataset we should record:

Dataset name
Official source
URL
Number of images
Number of breeds/classes
Annotation type
License
Redistribution restrictions
Commercial-use restrictions
Academic/research restrictions
Recommended use
Download method

The project must not assume that "free to download" means "open-source."

Dataset licenses must be reviewed before redistribution.

Large datasets should not normally be committed directly to the GitHub repository.

12. Dataset Architecture

Dataset information will be stored separately from the actual image files.

Example:

data/
├── raw/
├── processed/
└── metadata/
    ├── dataset_registry.json
    └── breed_taxonomy.json

The repository should document how authorized users can obtain the datasets.

Large datasets should remain outside GitHub unless their license and repository size make redistribution appropriate.

13. Breed Knowledge Base

The knowledge base will contain structured breed information.

Potential fields include:

breed_id
canonical_name
aliases
origin_country
origin_region
history
original_purpose
height
weight
lifespan
temperament
exercise_requirements
grooming_requirements
trainability
common_health_conditions
vaccination_information
bite_force_information
sources
evidence_quality
last_verified

Factual information should not be scattered throughout application source code.

The knowledge layer should be independently maintainable.

14. Breed Information Categories

The final application should attempt to provide:

Identification
Predicted breed
Confidence
Alternative breeds
Uncertainty
Origin
Country
Region
Historical development
Original purpose
Physical Characteristics
Height
Weight
Coat
Typical appearance
Lifespan
Behavior
Temperament
Trainability
Exercise requirements
Social characteristics
Care
Grooming
Exercise
General care considerations
Health
Common breed-associated health conditions
General health information
Vaccination
General educational information
Core vaccination concepts
Non-core vaccination concepts
Geographic considerations
Lifestyle considerations
Bite Force
Available scientific evidence
Measurement methodology where available
Evidence quality
Explicit indication when reliable standardized data is unavailable
15. Veterinary Information

Vaccination and health information must be handled carefully.

The application should provide general educational information rather than personalized veterinary advice.

The system should distinguish between:

Core vaccines
Non-core vaccines
Geographic considerations
Lifestyle-related considerations
Age-related considerations

The application should display an educational disclaimer such as:

This information is provided for educational purposes only and is not veterinary advice. Vaccination and healthcare decisions should be made with a licensed veterinarian.

The application must not diagnose an individual animal or prescribe personalized treatment.

16. Bite-Force Evidence

Bite-force information is a particularly important evidence-quality problem.

Online sources frequently provide breed-level PSI numbers without clearly explaining how they were obtained.

Therefore, the application should classify bite-force information as:

Scientifically measured
Reported measurement
Estimate
Unsupported claim
Unavailable

The system should preserve the source and evidence quality.

If reliable standardized breed-level evidence cannot be found, the application should say:

Reliable standardized breed-level bite-force data was not found.

It should not invent or confidently repeat unsupported numbers.

17. Retrieval-Augmented Generation

The RAG system connects breed predictions to factual evidence.

The expected pipeline is:

Predicted Breed
      |
      v
Canonical Breed ID
      |
      v
Metadata Filtering
      |
      v
Document Retrieval
      |
      v
Relevant Evidence
      |
      v
Evidence Ranking
      |
      v
Response Generation
      |
      v
Cited Answer

Documents should retain metadata including:

Source
URL
Title
Breed
Topic
Country
Publication date when available
Evidence quality
18. RAG Safety

The language model must not be treated as the source of truth for factual information.

The preferred flow is:

Reliable Source
      |
      v
Document
      |
      v
Retriever
      |
      v
Relevant Evidence
      |
      v
LLM
      |
      v
Human-readable explanation

The LLM should primarily synthesize retrieved information.

If evidence is unavailable, the system should say that information was not found instead of inventing an answer.

19. Evidence and Citations

Factual claims should be associated with supporting sources whenever practical.

The final response should allow the user to understand:

Where the information came from
What source supports the claim
What type of evidence was used

The system should prioritize authoritative sources, scientific literature, veterinary organizations, public-health organizations, and reputable breed organizations where appropriate.

20. Multimodal AI Layer

A multimodal AI model may be used as an additional reasoning or verification layer.

Potential architecture:

Dog Image
    |
    v
Vision Classifier
    |
    v
Breed Candidates
    |
    v
Multimodal Verification
    |
    v
Confidence / Candidate Analysis
    |
    v
Knowledge Retrieval

The multimodal model should not replace quantitative evaluation of the breed classifier.

Its role should be clearly defined and evaluated.

21. Backend API

The backend will expose structured endpoints.

Initial target endpoints:

POST /analyze
POST /predict
GET /breeds
GET /breeds/{breed_id}
GET /health

The API should use:

Request validation
Response schemas
Error handling
Logging
Authentication where required
Rate limiting where appropriate

The backend should separate:

API logic
Model inference
Knowledge retrieval
RAG
Configuration
Data access
22. Frontend

The frontend should provide a polished product experience.

Expected flow:

Open application.
Upload dog image.
Preview image.
Click Analyze.
Display analysis progress.
Display predicted breed.
Display confidence.
Display alternative breeds.
Display uncertainty.
Display origin.
Display physical characteristics.
Display temperament.
Display health information.
Display vaccination information.
Display bite-force evidence.
Display references.
Display limitations.

The interface should be responsive and usable on desktop and mobile browsers.

23. Security

The application should implement:

Image type validation
Image size limits
Request validation
Safe file handling
Secrets management
Environment variables
Rate limiting where appropriate
No API keys in GitHub
No passwords in source code
No cloud credentials in source control

Sensitive values must never be committed to GitHub.

24. Privacy

Uploaded images should be handled with privacy in mind.

The application should avoid permanently storing user images unless there is a clear reason and appropriate user disclosure.

Temporary uploaded images should be deleted when they are no longer required, where technically practical.

The application should not expose one user's uploaded image to another user.

25. Evaluation Strategy

The project must use real experiments and real measurements.

No performance numbers should be fabricated.

Computer Vision Metrics

The model evaluation should include:

Top-1 accuracy
Top-5 accuracy
Precision
Recall
F1 score
Confusion matrix
Calibration
Inference latency
Model size
RAG Metrics

The RAG system should evaluate:

Retrieval precision
Retrieval recall
Source relevance
Citation correctness
Groundedness
Hallucination rate
System Metrics

The production system should evaluate:

API latency
Failure rate
Image validation
Error handling
Low-confidence behavior
Unsupported image behavior
26. Test Cases

The system should eventually test:

Valid dog image
Non-dog image
Corrupted image
Extremely large image
Unsupported file type
Multiple dogs
Low-quality image
Puppy image
Possible mixed breed
Rare breed
Unknown breed
Missing knowledge-base entry
RAG retrieval failure
Model inference failure
API failure
27. Deployment Architecture

The final application should provide a public HTTPS URL when feasible.

Target architecture:

                    PUBLIC HTTPS URL
                           |
                           v
                    +-------------+
                    |  Frontend   |
                    +------+------+
                           |
                           v
                    +-------------+
                    | Backend API |
                    +------+------+
                           |
              +------------+------------+
              |            |            |
              v            v            v
          Vision       Knowledge       RAG
           Model        Database       System
              |            |            |
              +------------+------------+
                           |
                           v
                    Structured Result

The deployment platform will be selected based on:

Free-tier availability
Model requirements
GPU availability
Latency
Reliability
Security
Maintenance requirements
Cost

No paid cloud resource should be created without explicit approval.

28. Cloud Development Requirement

All development should be performed through browser/cloud environments.

The project should not require local installation of:

Python
Git
Docker
Node.js
CUDA
Anaconda
PostgreSQL
IDEs

GitHub Codespaces and other appropriate cloud environments may be used.

29. GitHub Development Workflow

Development will use Git branches.

Example:

main
 |
 +--- feature/foundation
 |
 +--- feature/dataset-pipeline
 |
 +--- feature/model
 |
 +--- feature/rag
 |
 +--- feature/api
 |
 +--- feature/frontend

Changes should be tested before being merged into the stable branch.

Commits should use meaningful messages.

Examples:

feat: add dataset ingestion pipeline
feat: add dog breed classifier
feat: add breed taxonomy
feat: add RAG retrieval
feat: add FastAPI backend
feat: add frontend analysis page
test: add model evaluation
docs: document dataset licensing
ci: add automated testing
30. CI/CD

GitHub Actions should eventually automate tasks such as:

Unit tests
Integration tests
Linting
Type checking
API tests
Build validation

Deployment automation may be added after the application is stable.

31. Project Development Phases

The project will be developed incrementally.

Phase 0 — Planning

Architecture and technology decisions.

Phase 1 — GitHub Foundation

Repository, branching, project structure, and documentation.

Phase 2 — Cloud Development

Configure browser-based development environment.

Phase 3 — Dataset Discovery

Identify legitimate datasets and verify licensing.

Phase 4 — Dataset Preparation

Download authorized datasets and build preprocessing pipelines.

Phase 5 — Breed Taxonomy

Create canonical breed mappings.

Phase 6 — Baseline Computer Vision

Train and evaluate an initial breed classifier.

Phase 7 — Model Improvement

Evaluate stronger architectures and improve performance.

Phase 8 — Knowledge Base

Build structured breed information.

Phase 9 — RAG

Implement evidence retrieval and grounded generation.

Phase 10 — Backend

Build the API.

Phase 11 — Frontend

Build the user interface.

Phase 12 — Integration

Connect image analysis, model, knowledge base, RAG, and UI.

Phase 13 — Testing

Perform unit, integration, and end-to-end testing.

Phase 14 — Deployment

Deploy the application publicly.

Phase 15 — CI/CD

Automate testing and deployment.

Phase 16 — Portfolio Optimization

Improve README, architecture diagrams, screenshots, evaluation results, and documentation.

32. Major Limitations

CanineVision AI should not claim that a photograph can always identify every dog breed in the world.

Accuracy may be affected by:

Image quality
Lighting
Camera angle
Occlusion
Dog age
Puppy appearance
Grooming
Fur length
Crossbreeding
Mixed breeds
Rare breeds
Unrepresented breeds
Similar-looking breeds
Dataset bias

The system should communicate these limitations clearly.

33. Responsible AI Principles

CanineVision AI should:

Communicate uncertainty
Avoid fabricated facts
Provide evidence for factual claims
Distinguish model prediction from retrieved knowledge
Avoid personalized veterinary diagnosis
Respect dataset licenses
Respect source attribution requirements
Protect API credentials
Protect user-uploaded images
Clearly communicate limitations
34. Final Product Goal

The final user experience should look approximately like:

+--------------------------------------------------+
|                CANINEVISION AI                   |
|                                                  |
|        Upload a photo of your dog                |
|                                                  |
|              [ Upload Image ]                    |
|                                                  |
|                [ Analyze ]                       |
|                                                  |
+--------------------------------------------------+

                    ANALYSIS

Most Likely Breed
-----------------
Golden Retriever

Confidence
----------
87%

Other Possibilities
-------------------
Labrador Retriever     6%
Flat-Coated Retriever  3%
Other                  4%

Origin
------
Scotland / United Kingdom

History
-------
Evidence-backed breed history...

Characteristics
---------------
Height:
Weight:
Lifespan:
Temperament:
Exercise:
Grooming:

Health
------
Common breed-associated conditions...

Vaccination
-----------
General educational information...

Bite Force
----------
Evidence quality:
Available measurement / unavailable

Sources
-------
[Source 1]
[Source 2]
[Source 3]

Important:
This AI prediction is not guaranteed to be correct.
Health and vaccination information is educational and
does not replace advice from a licensed veterinarian.
35. Success Criteria

The project will be considered complete only when:

The GitHub repository is organized.
Dataset licensing is documented.
The breed taxonomy is documented.
A real computer-vision model has been evaluated.
Actual model metrics are recorded.
The knowledge base is implemented.
The RAG pipeline is implemented.
Sources are included.
The backend API works.
The frontend works.
Automated tests exist.
End-to-end testing has been performed.
The application is deployed where feasible.
A public HTTPS demo is available where feasible.
The README documents the system.
Limitations are documented.
No secrets are exposed.
No unsupported performance claims are made.
36. Core Principle

The central design principle of CanineVision AI is:

                 VISUAL PREDICTION
                         +
                 VERIFIED KNOWLEDGE
                         +
                    EVIDENCE
                         +
                    UNCERTAINTY
                         =
              TRUSTWORTHY AI EXPERIENCE