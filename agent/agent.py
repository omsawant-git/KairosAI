# agent/agent.py
# ─────────────────────────────────────────────────────────────
# KairosAI AI Agent — LangChain ReAct Agent on Groq
# Model: Llama 3.3 70B (free, 1000 req/day)
# Tools: SQL query, vector search, live price
# Memory: conversation history in session
# ─────────────────────────────────────────────────────────────

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferWindowMemory
from langchain import hub
from langchain.prompts import PromptTemplate
from agent.tools import TOOLS
from agent.prompts import SYSTEM_PROMPT

load_dotenv()


# ── PROMPT ────────────────────────────────────────────────────

REACT_PROMPT = PromptTemplate.from_template("""You are KairosAI, an expert financial analyst assistant.

{system_prompt}

You have access to these tools:
{tools}

Tool names: {tool_names}

Use this format:
Question: the input question
Thought: think about what to do
Action: tool name (one of {tool_names})
Action Input: input for the tool
Observation: tool result
... (repeat Thought/Action/Observation as needed)
Thought: I now have enough information
Final Answer: your complete answer grounded in the data

Previous conversation:
{chat_history}

Question: {input}
Thought: {agent_scratchpad}""")


# ── AGENT ─────────────────────────────────────────────────────

def create_agent():
    # Creates the KairosAI agent with Groq LLM and all tools
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in .env")

    # Initialize Groq LLM
    # Llama 3.3 70B — best free model for tool calling
    llm = ChatGroq(
        api_key     = api_key,
        model_name  = "llama-3.3-70b-versatile",
        temperature = 0.1,      # low temperature = more factual
        max_tokens  = 2048
    )

    # Conversation memory — remembers last 5 exchanges
    memory = ConversationBufferWindowMemory(
        k                  = 5,
        memory_key         = "chat_history",
        return_messages    = False,
        input_key          = "input",
        output_key         = "output"
    )

    # Fill system prompt into React prompt
    prompt = REACT_PROMPT.partial(system_prompt=SYSTEM_PROMPT)

    # Create ReAct agent — reasons and acts in a loop
    agent = create_react_agent(
        llm     = llm,
        tools   = TOOLS,
        prompt  = prompt
    )

    # Agent executor — runs the agent loop with error handling
    executor = AgentExecutor(
        agent           = agent,
        tools           = TOOLS,
        memory          = memory,
        verbose         = True,      # shows reasoning steps
        max_iterations  = 8,         # prevents infinite loops
        handle_parsing_errors = True # recovers from format errors
    )

    return executor


# ── CHAT INTERFACE ────────────────────────────────────────────

def chat(executor, message: str) -> str:
    # Sends a message to the agent and returns the response
    try:
        result = executor.invoke({
            "input": message
        })
        return result.get("output", "No response generated.")
    except Exception as e:
        return f"Agent error: {str(e)}"


# ── TERMINAL CHAT ─────────────────────────────────────────────

def run_terminal():
    # Interactive terminal chat with the agent
    # For testing before wiring to Streamlit dashboard
    print("=" * 60)
    print("KairosAI Agent — Terminal Mode")
    print("Type 'exit' to quit, 'clear' to reset memory")
    print("=" * 60)

    executor = create_agent()
    print("\nAgent ready. Ask me anything about the market.\n")

    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            if user_input.lower() == "exit":
                print("Goodbye.")
                break

            if user_input.lower() == "clear":
                executor.memory.clear()
                print("Memory cleared.\n")
                continue

            print("\nKairosAI: ", end="", flush=True)
            response = chat(executor, user_input)
            print(response)
            print()

        except KeyboardInterrupt:
            print("\nGoodbye.")
            break


def run():
    # Entry point for the agent
    run_terminal()


if __name__ == "__main__":
    run()