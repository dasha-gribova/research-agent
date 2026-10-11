from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import json
from datetime import datetime

PAPERS = [
    {
        "id": "paper_001",
        "title": "3D Gaussian Splatting for Real-Time Radiance Field Rendering",
        "authors": ["Bernhard Kerbl", "Georgios Kopanas", "Thomas Leimkühler", "George Drettakis"],
        "affiliations": ["Université Côte d'Azur", "Inria", "MPII"],
        "year": 2023,
        "citations": 2847,
        "arxiv_id": "2308.04079",
        "venue": "ACM Transactions on Graphics (SIGGRAPH 2023)",
        "domain": "3d_computer_vision",
        "abstract": (
            "Radiance Field methods have recently revolutionized novel-view synthesis of scenes captured "
            "from multiple photos or videos. However, existing radiance field methods achieve high-quality "
            "novel views by relying on computationally expensive neural networks and costly sampling for "
            "training or rendering, preventing fast inference while undesirable sampling artifacts hamper "
            "training. We introduce a new approach that enables efficient, high-quality radiance field "
            "rendering by optimizing anisotropic 3D Gaussians. Our method achieves high-quality novel view "
            "synthesis at unprecedented frame rates."
        ),
        "keywords": [
            "3D Gaussian Splatting",
            "radiance fields",
            "real-time rendering",
            "novel view synthesis",
            "differentiable rendering",
            "scene reconstruction"
        ],
        "key_contributions": [
            "Representation of scenes as anisotropic 3D Gaussians with position, covariance, opacity, and view-dependent color",
            "Differentiable rasterization-based rendering using tile-based Gaussian splatting",
            "Real-time frame rates (often exceeding 100 fps) with state-of-the-art quality",
            "Competitive or better PSNR, SSIM, LPIPS compared to prior radiance field methods"
        ],
        "full_text_summary": (
            "The paper presents a novel approach to radiance field rendering that represents scenes as "
            "collections of anisotropic 3D Gaussians. Each Gaussian is characterized by its mean (position), "
            "covariance matrix (encoding scaling, rotation, and shearing), opacity, and view-dependent color "
            "expressed via spherical harmonics. Unlike NeRF-style methods that rely on ray tracing through "
            "expensive neural networks, this approach uses efficient rasterization-based rendering, enabling "
            "real-time performance. The method is optimized with adaptive density control and achieves "
            "state-of-the-art novel view synthesis quality while maintaining real-time frame rates, making it "
            "particularly well-suited for applications in virtual reality, autonomous driving simulation, and "
            "real-time scene reconstruction."
        )
    },

    {
        "id": "paper_002",
        "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        "authors": ["Patrick Lewis", "Ethan Perez", "Aleksandra Piktus", "Fabio Petroni", "Vladimir Karpukhin", "Naman Goyal", "Heinrich Küttler", "Mike Lewis", "Wen-tau Yih", "Tim Rocktäschel", "Sebastian Riedel", "Douwe Kiela"],
        "affiliations": ["Facebook AI Research", "University College London", "New York University"],
        "year": 2020,
        "citations": 4523,
        "arxiv_id": "2005.11401",
        "venue": "NeurIPS 2020",
        "domain": "nlp",
        "abstract": (
            "Large pre-trained language models have been shown to store factual knowledge in their parameters, "
            "and achieve state-of-the-art results when fine-tuned on downstream NLP tasks. However, their "
            "ability to access and precisely manipulate knowledge is still limited, and hence on "
            "knowledge-intensive tasks, their performance falls behind specialized task-specific models. "
            "We explore a general-purpose fine-tuning recipe for retrieval-augmented generation (RAG) — "
            "models which combine pre-trained parametric and non-parametric memory for language generation."
        ),
        "keywords": [
            "RAG",
            "retrieval-augmented generation",
            "knowledge-intensive NLP",
            "open-domain question answering",
            "dense retrieval",
            "BART",
            "DPR"
        ],
        "key_contributions": [
            "Introduced RAG architecture combining dense retrieval (DPR) with seq2seq generation (BART)",
            "Demonstrated end-to-end differentiability of retrieval and generation components",
            "Achieved state-of-the-art on open-domain QA, fact verification, and knowledge-grounded dialogue",
            "Showed parametric + non-parametric memory outperforms either approach alone"
        ],
        "full_text_summary": (
            "RAG models combine a pre-trained sequence-to-sequence model with a pre-trained neural retriever. "
            "The retrieval component (Dense Passage Retrieval) accesses a large text corpus, while the "
            "generation component (BART) produces answers conditioned on both the input and retrieved "
            "documents. The entire model is end-to-end differentiable, allowing joint fine-tuning. RAG "
            "achieves state-of-the-art performance on knowledge-intensive tasks while requiring significantly "
            "less computation than much larger models. The approach demonstrates that combining parametric "
            "memory (pre-trained generator) with non-parametric memory (retrieval corpus) yields better "
            "results than either approach alone, and enables easy knowledge updates without retraining."
        )
    },

    {
        "id": "paper_003",
        "title": "From Local to Global: A Graph RAG Approach to Query-Focused Summarization",
        "authors": ["Darren Edge", "Ha Trinh", "Newman Cheng", "Joshua Bradley", "Alex Chao", "Apurva Mody", "Steven Truitt", "Jonathan Larson"],
        "affiliations": ["Microsoft Research", "Microsoft"],
        "year": 2024,
        "citations": 312,
        "arxiv_id": "2404.16130",
        "venue": "arXiv preprint (Microsoft Research)",
        "domain": "knowledge_graphs",
        "abstract": (
            "The use of retrieval-augmented generation (RAG) to retrieve relevant information from an "
            "external knowledge source enables large language models (LLMs) to answer questions over "
            "private and/or previously unseen data. However, RAG fails on global questions directed at "
            "an entire text corpus, such as 'What are the main themes in this dataset?', since this is "
            "inherently a query-focused summarization (QFS) task, rather than an explicit retrieval task. "
            "We introduce Graph RAG, which uses an LLM to build a graph-based text index and generate "
            "community summaries for use in a map-reduce global answer generation process."
        ),
        "keywords": [
            "GraphRAG",
            "knowledge graphs",
            "query-focused summarization",
            "community detection",
            "LLM indexing",
            "map-reduce"
        ],
        "key_contributions": [
            "Introduced Graph RAG combining graph-based text indexing with community summarization",
            "Demonstrated superior performance on global sensemaking questions compared to baseline RAG",
            "Used LLM-generated entity graphs with Leiden community detection for hierarchical indexing",
            "Map-reduce approach for generating comprehensive answers to corpus-level questions"
        ],
        "full_text_summary": (
            "Graph RAG addresses a fundamental limitation of traditional RAG: inability to answer global "
            "questions about an entire corpus. The approach uses an LLM to extract entities and relationships "
            "from source texts, building a knowledge graph. The graph is then partitioned into hierarchical "
            "communities using the Leiden algorithm. For each community, the LLM generates a summary report. "
            "When answering a global question, these community summaries are processed in a map-reduce fashion: "
            "each summary is mapped to a partial answer, which are then reduced into a comprehensive global "
            "answer. Evaluation shows Graph RAG substantially outperforms baseline RAG and map-reduce "
            "summarization on both comprehensiveness and diversity of answers, particularly for global "
            "sensemaking questions."
        )
    },

    {
        "id": "paper_004",
        "title": "AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation",
        "authors": ["Qingyun Wu", "Gagan Bansal", "Jieyu Zhang", "Yiran Wu", "Beibin Li", "Erkang Zhu", "Li Jiang", "Xiaoyun Zhang", "Shaokun Zhang", "Jiale Liu", "Ahmed Hassan Awadallah", "Ryen W. White", "Doug Burger", "Chi Wang"],
        "affiliations": ["Microsoft Research", "Pennsylvania State University", "University of Washington"],
        "year": 2024,
        "citations": 892,
        "arxiv_id": "2308.08155",
        "venue": "COLM 2024",
        "domain": "multi_agent",
        "abstract": (
            "We introduce AutoGen, an open-source framework that allows developers to build LLM "
            "applications via multiple agents that can converse with each other to accomplish tasks. "
            "AutoGen agents are customizable, conversable, and can operate in various modes that employ "
            "combinations of LLMs, human inputs, and tools. Using AutoGen, developers can also flexibly "
            "define agent interaction behaviors. Both natural language and computer code can be used to "
            "program flexible conversation patterns for different applications."
        ),
        "keywords": [
            "multi-agent systems",
            "LLM applications",
            "conversational AI",
            "agent orchestration",
            "tool use",
            "human-in-the-loop"
        ],
        "key_contributions": [
            "Open-source framework for building multi-agent LLM applications via conversation",
            "Customizable agents with pluggable LLM backends and tool integration",
            "Flexible conversation patterns: sequential, hierarchical, group chat",
            "Support for human-in-the-loop interactions and code execution"
        ],
        "full_text_summary": (
            "AutoGen is a generic framework for building next-generation LLM applications using multiple "
            "conversable agents. Agents in AutoGen are customizable and can be powered by LLMs, humans, "
            "tools, or combinations thereof. The framework supports diverse conversation patterns including "
            "sequential, hierarchical, and group chat. Key features include: (1) modular agent design with "
            "pluggable backends, (2) flexible conversation orchestration, (3) built-in support for function "
            "calling and external tool integration, (4) robust error handling. AutoGen demonstrates "
            "effectiveness across diverse applications including math problem solving, coding Q&A, decision-making "
            "in text games, supply chain optimization, online decision-making, and dynamic group chat. "
            "The framework shows that multi-agent approaches can outperform single-agent baselines on "
            "complex reasoning tasks while maintaining interpretability through conversation traces."
        )
    },

    {
        "id": "paper_005",
        "title": "How to find expressible and trainable parameterized quantum circuits?",
        "authors": ["Peter Röseler", "Dennis Willsch", "Kristel Michielsen"],
        "affiliations": [
            "Jülich Supercomputing Centre, Forschungszentrum Jülich",
            "Universität zu Köln",
            "University of Applied Sciences Aachen",
            "AIDAS"
        ],
        "year": 2026,
        "date": "March 17, 2026",
        "citations": 0,
        "arxiv_id": "2603.14451v1",
        "venue": "arXiv preprint (quant-ph)",
        "domain": "quantum_computing",
        "abstract": (
            "Whether parameterized quantum circuits (PQCs) can be systematically constructed "
            "to be both trainable and expressive remains an open question. Highly expressive PQCs "
            "often exhibit barren plateaus, while several trainable alternatives admit efficient "
            "classical simulation. We address this question by deriving a finite-sample, "
            "dimension-independent concentration bound for estimating the variance of a PQC cost "
            "function, yielding explicit trainability guarantees. Across commonly used ansätze, "
            "we observe an anticorrelation between trainability and expressibility, consistent "
            "with theoretical insights. Building on this observation, we propose a property-based "
            "ansatz-search framework for identifying circuits that combine trainability and "
            "expressibility. We demonstrate its practical viability on a real quantum computer "
            "and apply it to variational quantum algorithms. We identify quantum neural network "
            "ansätze with improved effective dimension using over 6× fewer parameters, and for "
            "VQE on H2 we achieve UCCSD-like accuracy at substantially reduced circuit complexity."
        ),
        "keywords": [
            "parameterized quantum circuits",
            "barren plateaus",
            "variational quantum algorithms",
            "VQE",
            "QAOA",
            "quantum neural networks",
            "ansatz search",
            "trainability",
            "expressibility",
            "Bayesian optimization",
            "NISQ"
        ],
        "key_contributions": [
            "Finite-sample, dimension-independent concentration bound for estimating gradient variance of PQC cost functions",
            "Operational barren-plateau diagnostic based on gradient-to-noise ratio on NISQ devices",
            "Property-based ansatz-search framework optimizing expressibility, trainability, entanglement, and complexity",
            "Hardware-constrained ansatz discovery on 5-qubit IQM Spark quantum processor",
            "Demonstration of anticorrelation between expressibility and trainability across common ansatz families"
        ],
        "key_metrics": {
            "expressibility": "KL divergence between PQC-induced fidelity distribution and Haar-random reference",
            "trainability": "Gradient variance over noise floor ratio (gradient-to-noise ratio)",
            "entanglement": "Meyer-Wallach measure (average linear entropy of one-qubit subsystems)",
            "complexity": "Normalized sum of parameter count, circuit depth, and gate count"
        },
        "methods": [
            "Bayesian optimization for ansatz search (Qiskit + Optuna)",
            "Monte Carlo sampling for metric estimation (5000 samples for expressibility, 11806 for trainability)",
            "Gradient clipping to enforce bounded samples",
            "Concentration inequality for self-bounding random variables (Maurer-Pontil)",
            "Hierarchical cost functions with primary and secondary objectives"
        ],
        "main_results": [
            "Discovered QNN ansätze with higher effective dimension using over 6x fewer parameters (7 vs 40)",
            "For VQE on H2: UCCSD-like accuracy at substantially reduced circuit complexity",
            "Hardware-constrained search on IQM Spark improves over all benchmark circuits within search space",
            "Strong positive correlation between expressibility and entanglement observed",
            "Anticorrelation between expressibility and trainability confirmed across benchmark families"
        ],
        "limitations": [
            "Metric values depend on the reference state used for evaluation",
            "Metric drift may occur along optimization trajectory",
            "Global Haar-based expressibility not sufficient to guarantee ground-state convergence in VQE",
            "Scaling with qubit number and circuit depth requires further study",
            "Chemically harder molecules (e.g., LiH) require larger optimization budgets"
        ],
        "hardware_experiments": {
            "device": "5-qubit IQM Spark (JIQCER-5) at Jülich Supercomputing Centre",
            "native_gates": ["Rz", "CZ"],
            "topology": "star connectivity",
            "iterations": 50,
            "result": "Discovered ansatz improves over all benchmark circuits within hardware-constrained search space"
        },
        "full_text_summary": (
            "The paper addresses a fundamental open question in variational quantum computing: can parameterized "
            "quantum circuits be systematically designed to be both trainable and expressive? Highly expressive "
            "circuits tend to suffer from barren plateaus (exponentially vanishing gradients), while trainable "
            "circuits often admit efficient classical simulation. The authors derive a dimension-free concentration "
            "bound for estimating gradient variance from finitely many clipped samples, providing an operational "
            "barren-plateau diagnostic. Four metrics are introduced: expressibility (KL divergence to Haar-random "
            "states), trainability (gradient variance over noise floor), entanglement (Meyer-Wallach measure), and "
            "complexity (parameters, depth, gates). Using Bayesian optimization, the framework discovers ansätze "
            "satisfying metric thresholds while substantially reducing complexity. Key findings include the "
            "anticorrelation between expressibility and trainability, strong positive correlation between "
            "expressibility and entanglement, and practical demonstration on IQM Spark hardware. For VQE on H2, "
            "replacing the gate pool with symmetry-preserving excitation operators (as in QEB-ADAPT-VQE) resolves "
            "the mismatch between global expressibility and ground-state convergence."
        )
    }
]


# ============================================================
# ИНСТРУМЕНТЫ
# ============================================================

@tool
def list_available_papers() -> List[Dict[str, Any]]:
    return [
        {
            "id": p["id"],
            "title": p["title"],
            "year": p["year"],
            "citations": p["citations"],
            "domain": p.get("domain", "general"),
            "venue": p.get("venue", "Unknown")
        }
        for p in PAPERS
    ]


@tool
def search_papers(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    query_lower = query.lower()
    results = []

    for paper in PAPERS:
        searchable_text = " ".join([
            paper["title"],
            paper["abstract"],
            " ".join(paper["keywords"])
        ]).lower()

        if any(word in searchable_text for word in query_lower.split()):
            results.append({
                "id": paper["id"],
                "title": paper["title"],
                "authors": paper["authors"],
                "year": paper["year"],
                "citations": paper["citations"],
                "venue": paper["venue"]
            })

    results.sort(key=lambda x: x["citations"], reverse=True)
    return results[:max_results]


@tool
def read_paper(paper_id: str) -> Dict[str, Any]:
    for paper in PAPERS:
        if paper["id"] == paper_id:
            return {
                "id": paper["id"],
                "title": paper["title"],
                "authors": paper["authors"],
                "year": paper["year"],
                "venue": paper["venue"],
                "abstract": paper["abstract"],
                "key_contributions": paper.get("key_contributions", []),
                "main_results": paper.get("main_results", []),
            }

    available_ids = [p["id"] for p in PAPERS]
    return {"error": f"Paper with id {paper_id} not found. Available IDs: {available_ids}"}


@tool
def analyze_trends(papers: List[str], metric: str = "citations") -> Dict[str, Any]:
    relevant_papers = [p for p in PAPERS if p["id"] in papers]

    if not relevant_papers:
        return {"error": "No papers found for given IDs"}

    if metric == "citations":
        values = [p["citations"] for p in relevant_papers]
        return {
            "metric": "citations",
            "count": len(values),
            "mean": sum(values) / len(values),
            "max": max(values),
            "min": min(values),
            "total": sum(values),
            "papers_analyzed": [p["title"] for p in relevant_papers]
        }

    elif metric == "year":
        years = [p["year"] for p in relevant_papers]
        return {
            "metric": "publication_year",
            "count": len(years),
            "earliest": min(years),
            "latest": max(years),
            "most_recent_papers": [
                p["title"] for p in relevant_papers
                if p["year"] == max(years)
            ]
        }

    elif metric == "domain":
        domains = {}
        for p in relevant_papers:
            d = p.get("domain", "general")
            domains[d] = domains.get(d, 0) + 1
        return {
            "metric": "domain_distribution",
            "count": len(relevant_papers),
            "distribution": domains
        }

    return {"error": f"Unknown metric: {metric}. Use 'citations', 'year', or 'domain'."}


@tool
def get_paper_metrics(paper_id: str) -> Dict[str, Any]:
    for paper in PAPERS:
        if paper["id"] == paper_id:
            return {
                "paper_id": paper_id,
                "title": paper["title"],
                "key_metrics": paper.get("key_metrics", {}),
                "methods": paper.get("methods", []),
                "main_results": paper.get("main_results", []),
                "limitations": paper.get("limitations", []),
                "hardware_experiments": paper.get("hardware_experiments", {})
            }

    return {"error": f"Paper with id {paper_id} not found"}


@tool
def generate_research_summary(
        topic: str,
        key_findings: List[str],
        papers_referenced: List[str]
) -> Dict[str, str]:
    summary = f"""
# Research Summary: {topic}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Key Findings
{chr(10).join(f'- {finding}' for finding in key_findings)}

## Referenced Papers ({len(papers_referenced)})
{chr(10).join(f'- {paper}' for paper in papers_referenced)}

## Conclusion
Based on the analysis of {len(papers_referenced)} papers, 
the research landscape shows active development in {topic.lower()}.
"""

    return {
        "summary": summary,
        "papers_count": len(papers_referenced),
        "findings_count": len(key_findings)
    }