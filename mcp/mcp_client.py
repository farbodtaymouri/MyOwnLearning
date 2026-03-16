"""
MCP Client — Azure AI Agent + FastMCP Bridge
=============================================

This script shows how to connect Azure AI Agents to MCP (Model Context Protocol)
tools, demonstrating how agents discover and intelligently select tools based on
user questions.

WORKFLOW
========
    1. User enters a question
    2. Client discovers available tools from MCP server
    3. Azure AI Agent analyzes the question
    4. Agent selects and calls the appropriate tool(s)
    5. Agent generates natural language response

ARCHITECTURE
============
    ┌─────────────────────────────────────────────────────┐
    │  This Script (MCP Bridge)                           │
    │  1. Get user question                               │
    │  2. Discover tools from MCP server                  │
    │  3. Register tools with Azure AI Agent              │
    │  4. Agent decides which tool to call                │
    │  5. Execute tool calls → forward to MCP server      │
    │  6. Return results to agent → final answer          │
    └─────────────────────────────────────────────────────┘
              ↕ MCP Protocol (SSE)    ↕ Azure AI REST API
    ┌──────────────────┐         ┌───────────────────────┐
    │  FastMCP Server  │         │  Azure AI Agent       │
    │  localhost:8000  │         │  (cloud, gpt-4o)      │
    │  - get_weather   │         │  Intelligent tool     │
    │  - get_stock     │         │  selection based on   │
    │  - calculate     │         │  user question        │
    └──────────────────┘         └───────────────────────┘

PREREQUISITES
=============
    1. Start mcp_server.py:  python mcp_server.py
    2. Install: pip install azure-ai-projects azure-identity mcp fastmcp
    3. Authenticate: az login --scope https://ai.azure.com/.default
"""

import asyncio
import json
import time

from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from azure.ai.agents.models import MessageRole, ListSortOrder
from mcp import ClientSession
from mcp.client.sse import sse_client

# ============================================================
# Configuration
# ============================================================
MCP_SERVER_URL = "http://localhost:8000/sse"  # FastMCP SSE endpoint
PROJECT_ENDPOINT = (
    "https://arch-poc-ft-foundry-pro-resource.services.ai.azure.com"
    "/api/projects/arch_poc_ft_foundry_project"
)
MODEL_DEPLOYMENT = "gpt-4o"


# ============================================================
# STEP 1: Discover MCP Tools (async)
# ============================================================
async def discover_mcp_tools() -> list[dict]:
    """
    Connect to MCP server and retrieve available tool schemas.
    
    MCP handshake:
        1. initialize() → establish capabilities
        2. list_tools() → get tool schemas (JSON Schema format)
    
    Returns: List of Azure AI Agent tool definitions
    """
    async with sse_client(url=MCP_SERVER_URL) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()  # Step 1: Capability negotiation
            tools_response = await session.list_tools()  # Step 2: Get tools
            
            # Convert MCP format → Azure AI Agent format
            # (Both use JSON Schema, so it's a direct mapping)
            azure_tools = []
            for tool in tools_response.tools:
                azure_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema,
                    }
                })
                print(f"  ✓ Discovered: {tool.name}")
            
            return azure_tools


# ============================================================
# STEP 2: Execute MCP Tool Call (async)
# ============================================================
async def call_mcp_tool(tool_name: str, tool_args: dict) -> str:
    """
    Forward a tool call to the MCP server and return the result.
    
    Called when the Azure AI Agent requests a tool execution.
    
    Args:
        tool_name: Name of the tool (e.g., "get_weather")
        tool_args: Arguments dict (e.g., {"city": "Sydney"})
    
    Returns: Tool result as a string (JSON)
    """
    async with sse_client(url=MCP_SERVER_URL) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, tool_args)
            
            # Extract text from MCP result
            text_parts = [
                block.text for block in result.content 
                if hasattr(block, "text")
            ]
            return "\n".join(text_parts) if text_parts else "{}"


# ============================================================
# STEP 3: Run Azure AI Agent with MCP Tools
# ============================================================
def run_agent_with_mcp(user_query: str, mcp_tools: list[dict]) -> None:
    """
    Create an Azure AI Agent with MCP tools and process a query.
    
    Flow:
        1. Create agent with MCP tool schemas
        2. Send user query to thread
        3. Poll agent run status
        4. When "requires_action" → execute tool via MCP → submit result
        5. When "completed" → display final answer
    """
    # Authenticate with Azure
    project_client = AIProjectClient(
        credential=DefaultAzureCredential(
            exclude_environment_credential=True,
            exclude_managed_identity_credential=True,
        ),
        endpoint=PROJECT_ENDPOINT,
    )
    agent_client = project_client.agents
    
    # Create agent with MCP tools registered
    print(f"\nCreating Azure AI Agent with {len(mcp_tools)} MCP tool(s)...")
    agent = agent_client.create_agent(
        model=MODEL_DEPLOYMENT,
        name="MCP-Enabled Agent",
        instructions=(
            "You are a helpful assistant with access to external tools via "
            "Model Context Protocol (MCP). Analyze the user's question carefully "
            "and select the most appropriate tool to answer their question. "
            "Available tools include: weather data, stock prices, and calculations. "
            "Always use a tool when the question requires specific data."
        ),
        tools=mcp_tools,  # ← MCP tools registered here
    )
    print(f"✓ Agent: {agent.id}")
    
    # Create conversation thread and send user message
    thread = agent_client.threads.create()
    agent_client.messages.create(
        thread_id=thread.id,
        role=MessageRole.USER,
        content=user_query,
    )
    print(f"\n👤 User Question: \"{user_query}\"\n")
    
    # Start agent run
    run = agent_client.runs.create(thread_id=thread.id, agent_id=agent.id)
    
    # Poll until completion, handling tool calls
    print("⏳ Agent processing...")
    while run.status in ["queued", "in_progress", "requires_action"]:
        time.sleep(0.5)
        run = agent_client.runs.get(thread_id=thread.id, run_id=run.id)
        
        # Handle tool execution requests
        if run.status == "requires_action":
            tool_calls = run.required_action.submit_tool_outputs.tool_calls
            tool_outputs = []
            
            print(f"\n  🎯 Agent selected {len(tool_calls)} tool(s) to answer the question:")
            for tool_call in tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)
                print(f"\n     Tool: {func_name}")
                print(f"     Arguments: {json.dumps(func_args)}")
                print(f"     Executing via MCP server...")
                
                # Execute via MCP
                mcp_result = asyncio.run(call_mcp_tool(func_name, func_args))
                print(f"     ✓ Result: {mcp_result[:100]}...")
                
                tool_outputs.append({
                    "tool_call_id": tool_call.id,
                    "output": mcp_result,
                })
            
            # Submit results back to agent
            run = agent_client.runs.submit_tool_outputs(
                thread_id=thread.id,
                run_id=run.id,
                tool_outputs=tool_outputs,
            )
    
    # Display final answer
    if run.status == "completed":
        print("\n  ✅ Agent processing complete!\n")
        messages = agent_client.messages.list(
            thread_id=thread.id,
            order=ListSortOrder.DESCENDING,
            limit=1,
        )
        for msg in messages:
            if msg.role == MessageRole.AGENT and msg.text_messages:
                print("=" * 60)
                print("  AGENT'S FINAL ANSWER")
                print("=" * 60)
                print(f"\n{msg.text_messages[0].text.value}\n")
                print("=" * 60)
    else:
        print(f"\n❌ Run failed: {run.status}")
    
    # Cleanup
    agent_client.delete_agent(agent.id)
    print("✓ Agent deleted\n")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  MCP + Azure AI Agent Demo")
    print("=" * 60)
    print(f"\nMCP Server: {MCP_SERVER_URL}")
    print("(Ensure mcp_server.py is running)\n")
    
    # STEP 1: Get user input
    print("=" * 60)
    print("  STEP 1: GET USER QUESTION")
    print("=" * 60)
    print("\nExample questions you can ask:")
    print("  • What's the weather in Tokyo?")
    print("  • Get the stock price for MSFT")
    print("  • Calculate 15 * 8 + 42")
    print()
    
    user_question = input("Enter your question: ").strip()
    
    if not user_question:
        print("\n❌ No question provided. Exiting.")
        raise SystemExit(0)
    
    print(f"\n✓ Question received: \"{user_question}\"")
    
    # STEP 2: Discover tools from MCP server
    print("\n" + "=" * 60)
    print("  STEP 2: DISCOVER AVAILABLE TOOLS")
    print("=" * 60)
    print("\nConnecting to MCP server and discovering tools...\n")
    
    try:
        mcp_tools = asyncio.run(discover_mcp_tools())
    except Exception as exc:
        print(f"\n❌ Cannot connect to MCP server: {exc}")
        print("   → Start mcp_server.py first, then retry.")
        raise SystemExit(1)
    
    print(f"\n✓ Found {len(mcp_tools)} tool(s)")
    
    # Display discovered tools
    print("\nAvailable tools:")
    for i, tool in enumerate(mcp_tools, 1):
        print(f"  {i}. {tool['function']['name']}")
        print(f"     {tool['function']['description']}")
    
    # STEP 3: Agent analyzes question and selects the right tool
    print("\n" + "=" * 60)
    print("  STEP 3: AGENT PROCESSES QUESTION & SELECTS TOOL")
    print("=" * 60)
    print("\nThe Azure AI Agent will:")
    print("  1. Analyze your question")
    print("  2. Decide which tool(s) to use")
    print("  3. Execute the appropriate tool(s)")
    print("  4. Generate a natural language response")
    
    run_agent_with_mcp(user_question, mcp_tools)
    
    print("=" * 60)
    print("  Demo complete")
    print("=" * 60)
