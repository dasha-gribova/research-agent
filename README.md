# Research Agent

A LangGraph-based AI agent designed for scientific literature analysis and tool-assisted querying over a database of scientific papers. The agent specializes in domains such as quantum computing, machine learning, computer vision, and large language model applications.

## Features

- **Paper Discovery**: List and search available scientific papers by topic, keyword, or domain.
- **Deep Analysis**: Read detailed information, extract technical metrics, methods, results, and limitations of specific papers.
- **Trend Analysis**: Perform statistical analysis across collections of papers (e.g., citation counts, publication years, domain distribution).
- **Automated Summarization**: Generate comprehensive, structured research summaries based on gathered findings.
- **Multi-Provider LLM Support**: Seamlessly switch between OpenAI, OpenRouter, and local Ollama models.

## Architecture

The agent is built using LangGraph, enabling stateful, multi-step reasoning and iterative tool execution. It follows a structured workflow:

1. **Understand** the user's research question.
2. **Plan** which tools to use and in what order.
3. **Execute** searches and gather information.
4. **Read** full details of relevant papers.
5. **Analyze** results using statistical tools.
6. **Synthesize** findings into a coherent summary.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/dasha-gribova/research-agent.git
   cd research-agent
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
   *(On Windows, use: `venv\Scripts\activate`)*

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

Create a `.env` file in the root directory to configure your preferred LLM provider.

**For OpenAI:**
```env
LLM_PROVIDER=openai
OPENAI_API_KEY=your_openai_api_key
```

**For OpenRouter:**
```env
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=your_openrouter_api_key
OPENROUTER_MODEL=meta-llama/llama-3.1-8b-instruct
```

**For Ollama (local execution):**
```env
LLM_PROVIDER=ollama
OLLAMA_MODEL=qwen2.5:3b
```

## Usage

The agent can be run in two modes: interactive or command-line.

### Interactive Mode
Run the script without arguments to start an interactive session:
```bash
python agent.py
```
You will be prompted to enter your research question. Type `quit` or `exit` to terminate the session.

### Command-Line Mode
Pass your research question directly as a command-line argument:
```bash
python agent.py "What are the latest trends in Graph RAG and knowledge graphs?"
```

## Available Tools

The agent utilizes the following tools to interact with the paper database:

- `list_available_papers`: Returns metadata (ID, title, year, citations, domain, venue) of all papers in the database.
- `search_papers`: Searches for papers based on a text query, returning ranked results by citation count.
- `read_paper`: Retrieves detailed information (abstract, key contributions, main results) for a specific paper by its ID.
- `get_paper_metrics`: Extracts advanced technical details, including key metrics, methods, limitations, and hardware experiment results.
- `analyze_trends`: Computes statistical aggregations (mean, max, min, total) for citations, publication years, or domain distributions across a specified list of paper IDs.
- `generate_research_summary`: Formats the gathered key findings and referenced papers into a structured markdown summary.

## Project Structure

```text
research-agent/
├── agent.py          # Main entry point, LLM initialization, and execution loop
├── graph.py          # LangGraph state definition and workflow compilation
├── tools.py          # Tool definitions and the embedded scientific paper database
├── requirements.txt  # Python dependencies
├── .env              # Environment variables (create this file)
└── examples/         # Example usage scripts or outputs
```

## License

This project is provided as-is for research and educational purposes.
