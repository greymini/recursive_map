import os

# Google Gemini API Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
MODEL_NAME = "gemini-2.0-flash-exp"  # Updated to use available model

# Text Processing Configuration
MAX_CHUNK_SIZE = 500  # Maximum tokens per chunk
CHUNK_OVERLAP = 50    # Token overlap between chunks

# Extraction Configuration
CONFIDENCE_THRESHOLD = 0.6  # Minimum confidence for extraction
FUZZY_MATCH_THRESHOLD = 0.85  # Entity deduplication threshold

# Entity Types
ENTITY_TYPES = [
    "Person",
    "Organization",
    "Product",
    "Location",
    "Event",
    "Technology"
]

# Relationship Types
RELATIONSHIP_TYPES = [
    "works_at",
    "founded",
    "located_in",
    "develops",
    "uses",
    "collaboratesWith"
]

# Output Directories
DATA_DIR = "data"
SAMPLE_DOCUMENTS_DIR = os.path.join(DATA_DIR, "sample_documents")
REFERENCE_ANNOTATIONS_DIR = os.path.join(DATA_DIR, "reference_annotations")
OUTPUTS_DIR = os.path.join(DATA_DIR, "outputs")

