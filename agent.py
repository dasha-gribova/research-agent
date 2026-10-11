import os
import sys
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from tools import (
    search_papers,
    read_paper,
    analyze_trends,
    generate_research_summary,
    list_available_papers,
    get_paper_metrics
)

tools = [
    search_papers,
    read_paper,
    analyze_trends,
    generate_research_summary,
    list_available_papers,
    get_paper_metrics
]
from graph import create_agent, AgentState
from langchain_core.messages import HumanMessage


def create_llm():
    provider = os.getenv("LLM_PROVIDER", "openai")

    if provider == "openrouter":
        from langchain_openai import ChatOpenAI
        model_name = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct")
        api_key = os.getenv("OPENROUTER_API_KEY")
        return ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            temperature=0,
            timeout=60,
            max_retries=1,
            max_tokens=1024,
        )

    elif provider == "ollama":
        from langchain_ollama import ChatOllama
        model_name = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
        return ChatOllama(model=model_name, temperature=0)

    else:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-4o-mini", temperature=0)


def run_agent(query: str, verbose: bool = True, max_retries: int = 5):
    import time

    load_dotenv()

    llm = create_llm()
    tools = [search_papers, read_paper, analyze_trends, generate_research_summary, list_available_papers,
             get_paper_metrics]
    agent = create_agent(llm, tools)

    initial_state: AgentState = {
        "messages": [HumanMessage(content=query)],
        "research_goal": query,
        "papers_found": [],
        "analysis_complete": False
    }

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Research Agent")
        print(f"{'=' * 60}")
        print(f"Query: {query}\n")

    config = {"recursion_limit": 25}

    for attempt in range(max_retries):
        try:
            for event in agent.stream(initial_state, config=config):
                for node_name, node_output in event.items():
                    if node_name == "agent":
                        print("\n[stage] Агент думает (запрос к LLM)...")
                    elif node_name == "tools":
                        print("[stage] Инструмент выполнен, результат отправлен обратно в LLM...")

                    if "messages" in node_output:
                        for message in node_output["messages"]:
                            if hasattr(message, "content") and message.content and node_name == "agent":
                                print(f"\nAgent:")
                                print("-" * 40)
                                print(message.content)

                            if hasattr(message, "tool_calls") and message.tool_calls:
                                for tool_call in message.tool_calls:
                                    print(f"\nTool call: {tool_call['name']}")
                                    print(f"   Args: {tool_call['args']}")

            return agent

        except Exception as e:
            if "429" in str(e) and attempt < max_retries - 1:
                wait_time = 15 * (attempt + 1)  # 15, 30, 45, 60, 75 секунд
                print(f"\nRate limit hit (attempt {attempt + 1}/{max_retries}). Waiting {wait_time} seconds...")
                time.sleep(wait_time)
                continue
            else:
                print(f"\nError: {e}")
                raise

    return agent


def interactive_mode():
    print("\nResearch Agent (interactive mode)")
    print("Type 'quit' or 'exit' to stop\n")

    while True:
        try:
            query = input("Your research question: ").strip()

            if query.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            if not query:
                continue

            run_agent(query)
            print("\n" + "=" * 60 + "\n")

        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        run_agent(query)
    else:
        interactive_mode()