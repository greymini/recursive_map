# AI-Powered Knowledge Graph Extraction & Evaluation

An AI-based pipeline that extracts semantic entities and relationships from unstructured documents (PDF/TXT) and generates a structured knowledge graph. Uses LangGraph for LLM orchestration, Google Gemini 2.5 Flash as the language model, NetworkX for graph generation, and Streamlit for document upload and visualization. A video demo can be accessed here: https://url-shortener.me/30WF

## Features

- **Document Processing**: Support for PDF and TXT files using PyMuPDF
- **Entity Extraction**: Extract Person, Organization, Product, Location, Event, Technology entities
- **Relationship Extraction**: Extract directional relationships (works_at, founded, located_in, develops, uses, collaboratesWith)
- **Two Extraction Approaches**:
  - Zero-Shot: Direct extraction without examples
  - Few-Shot + Self-Critique: Extraction with examples and LLM self-validation
- **Graph Visualization**: NetworkX-based visualization with color-coding and confidence-based sizing
- **Multi-Document Processing**: Process multiple documents and merge graphs intelligently
- **Evaluation**: Precision, recall, and F1 score calculation
- **Streamlit UI**: Interactive web interface for document upload and visualization

## Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd recursive_map
   ```

2. **Activate the virtual environment:**
   ```bash
   source ../env/bin/activate  # On macOS/Linux
   # or
   ..\env\Scripts\activate  # On Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up Google Gemini API Key:**
   
   Get your API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
   
   Option 1: Set environment variable
   ```bash
   export GOOGLE_API_KEY="your-api-key-here"
   ```
   
   Option 2: Edit `config.py` and set:
   ```python
   GOOGLE_API_KEY = "your-api-key-here"
   ```

5. **Download NLTK data (if not already downloaded):**
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   ```

## Project Structure

```
recursive_map/
├── config.py                     # API keys and configuration parameters
├── document_processors.py        # PyMuPDF extraction and text loading
├── text_preprocessing.py         # Text cleaning, chunking, preprocessing
├── llm_orchestration.py          # LangGraph workflows and agents
├── knowledge_graph.py            # Graph building, merging, and validation
├── visualizer.py                 # NetworkX visualization
├── evaluator.py                  # Precision/recall evaluation
├── data/
│   ├── sample_documents/         # Sample PDF/TXT files
│   ├── reference_annotations/   # Manual reference data
│   └── outputs/                  # Generated JSON graphs and visualizations
├── scripts/
│   ├── run_pipeline.py           # Main execution script
│   └── evaluate_extraction.py    # Evaluation script
├── ui/
│   └── streamlit_app.py          # Streamlit interface
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Usage

### Command Line Interface

#### Extract Knowledge Graph from Document

```bash
# Single document, zero-shot approach
python scripts/run_pipeline.py --document path/to/document.pdf --approach zero_shot

# Single document, few-shot approach
python scripts/run_pipeline.py --document path/to/document.pdf --approach few_shot_critique

# Multiple documents from directory
python scripts/run_pipeline.py --document data/sample_documents/ --approach few_shot_critique

# Custom output directory
python scripts/run_pipeline.py --document document.pdf --output my_outputs/
```

#### Evaluate Extraction Results

```bash
# Evaluate single extraction
python scripts/evaluate_extraction.py --predicted data/outputs/graph.json --reference data/reference_annotations/reference.json

# Compare zero-shot vs few-shot
python scripts/evaluate_extraction.py --zero-shot data/outputs/zero_shot_graph.json --few-shot data/outputs/few_shot_graph.json --reference data/reference_annotations/reference.json
```

### Streamlit Web Interface

```bash
streamlit run ui/streamlit_app.py
```

Then open your browser to `http://localhost:8501`

Features:
- Upload PDF or TXT documents
- Select extraction approach
- View interactive knowledge graph visualization
- Browse entities and relationships in tables
- Export results as JSON, CSV, or PNG
- Evaluate against reference annotations

## Configuration

Edit `config.py` to customize:

- `GOOGLE_API_KEY`: Your Gemini API key
- `MODEL_NAME`: Gemini model (default: "gemini-2.0-flash-exp")
- `MAX_CHUNK_SIZE`: Maximum tokens per chunk (default: 2000)
- `CHUNK_OVERLAP`: Token overlap between chunks (default: 200)
- `CONFIDENCE_THRESHOLD`: Minimum confidence for extraction (default: 0.6)
- `FUZZY_MATCH_THRESHOLD`: Entity deduplication threshold (default: 0.85)

## Output Format

The knowledge graph is saved as JSON with the following structure:

```json
{
  "nodes": [
    {
      "id": "n1",
      "label": "Entity Name",
      "type": "Person",
      "confidence": 0.95,
      "context": "Brief description",
      "source": "document.pdf"
    }
  ],
  "edges": [
    {
      "id": "e1",
      "source": "n1",
      "target": "n2",
      "relationship": "works_at",
      "confidence": 0.87,
      "description": "Sentence describing the relationship",
      "source": "document.pdf"
    }
  ],
  "source_mapping": {
    "n1": "Text snippet supporting this entity",
    "e1": "Sentence describing the relationship"
  },
  "metadata": {
    "num_nodes": 10,
    "num_edges": 15,
    "document_source": "document.pdf"
  }
}
```

## Entity Types

- **Person**: Individuals, people
- **Organization**: Companies, institutions, groups
- **Product**: Products, services, applications
- **Location**: Places, cities, countries, addresses
- **Event**: Events, conferences, meetings
- **Technology**: Technologies, tools, frameworks

## Relationship Types

- **works_at**: Person works at Organization
- **founded**: Person founded Organization
- **located_in**: Entity located in Location
- **develops**: Organization develops Product
- **uses**: Entity uses Technology
- **collaboratesWith**: Entity collaborates with Entity

## Evaluation

To evaluate extraction quality:

1. Create reference annotations JSON file:
```json
{
  "entities": [
    {"name": "John Smith", "type": "Person"},
    {"name": "Apple Inc.", "type": "Organization"}
  ],
  "relationships": [
    {"source": "John Smith", "target": "Apple Inc.", "type": "works_at"}
  ]
}
```

2. Run evaluation:
```bash
python scripts/evaluate_extraction.py --predicted output.json --reference reference.json
```

## Troubleshooting

### API Key Issues
- Ensure `GOOGLE_API_KEY` is set correctly
- Check API key is valid and has sufficient quota

### Import Errors
- Make sure virtual environment is activated
- Install all dependencies: `pip install -r requirements.txt`
- Download NLTK data if needed

### Memory Issues
- Reduce `MAX_CHUNK_SIZE` in `config.py` for large documents
- Process documents one at a time instead of batch

### Visualization Issues
- Ensure matplotlib backend is properly configured
- For headless servers, may need to set `MPLBACKEND=Agg`



