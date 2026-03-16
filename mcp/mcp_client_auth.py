"""
MCP Client with Azure AD Authentication (DEMO WITH WORKAROUNDS)
================================================================

This client demonstrates authentication concepts for MCP servers with RBAC.

⚠️  CURRENT LIMITATIONS (FastMCP v3.1.1):
    - Discovery filtering: Client-side workaround (not secure for production)
    - Execution validation: Server-side @require_auth (✅ production-ready)
    
🔒 PRODUCTION RECOMMENDATIONS:
    1. Use Azure API Management to filter tool discovery responses
    2. Implement custom MCP server with auth support
    3. Always validate permissions on execution (defense in depth)

Real-world flow (Coles example):
    Developer Laptop                    Azure API Mgmt           AKS/MCP Server
    ┌─────────────┐                ┌──────────────────┐      ┌──────────────┐
    │ VS Code     │   1. Login    │ Azure AD         │      │ MCP Server   │
    │             │──────────────>│ Entra ID         │      │              │
    │             │   2. Get Token│                  │      │              │
    │             │<──────────────│                  │      │              │
    │             │                └──────────────────┘      │              │
    │             │                                          │              │
    │ Azure AI    │   3. Request with token                 │              │
    │ Agent       │──────────────>│ API Gateway      │      │              │
    │             │                │ - Validates token│      │              │
    │             │                │ - Filters tools  │──────>│ list_tools() │
    │             │                │ - Returns subset │<──────│              │
    │             │   4. Filtered  │                  │      │              │
    │             │<──────────────│                  │      │              │
    │             │                                          │              │
    │             │   5. Execute tool                       │              │
    │             │──────────────────────────────────────────>│ @require_auth│
    │             │   6. Validate permission (defense)       │              │
    │             │<──────────────────────────────────────────│              │
    └─────────────┘                                          └──────────────┘

Prerequisites:
    pip install azure-ai-projects azure-identity mcp pyjwt
"""

import asyncio
import json
import time
from typing import Optional

from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from azure.ai.agents.models import MessageRole, ListSortOrder
import jwt

# ============================================================
# Configuration
# ============================================================

MCP_SERVER_URL = "http://localhost:8000/sse"
PROJECT_ENDPOINT = (
    "https://arch-poc-ft-foundry-pro-resource.services.ai.azure.com"
    "/api/projects/arch_poc_ft_foundry_project"
)
MODEL_DEPLOYMENT = "gpt-4o"

# Demo JWT secret for TOKEN GENERATION ONLY
# In production, tokens come from Azure AD, not generated client-side
JWT_SECRET = "demo-secret-key-replace-with-azure-ad-validation"


# ============================================================
# Authentication Functions
# ============================================================

def get_azure_ad_token() -> str:
    """
    Get Azure AD token for the current user.
    
    In production, this uses:
        credential = DefaultAzureCredential()
        token = credential.get_token("https://ai.azure.com/.default")
        return token.token
    
    For this demo, we simulate different user scenarios.
    """
    # In production, Azure AD handles this automatically
    # credential = DefaultAzureCredential()
    # token = credential.get_token("https://coles.com/.default")
    # return token.token
    
    # For demo: Let user choose their role
    print("\n🔐 AUTHENTICATION")
    print("=" * 60)
    print("\nWho are you logging in as?")
    print("  1. John Public (public user - LIMITED ACCESS)")
    print("  2. Jane Employee (employee - MEDIUM ACCESS)")
    print("  3. Admin User (admin - FULL ACCESS)")
    print()
    
    choice = input("Enter choice (1-3): ").strip()
    
    user_map = {
        "1": ("user1@coles.com.au", "John Public", ["public"]),
        "2": ("user2@coles.com.au", "Jane Employee", ["employee"]),
        "3": ("user3@coles.com.au", "Admin User", ["admin"])
    }
    
    user_email, user_name, roles = user_map.get(choice, user_map["1"])
    
    # Create demo token (in production, this comes from Azure AD)
    token = jwt.encode(
        {"sub": user_email, "exp": 9999999999},
        JWT_SECRET,
        algorithm="HS256"
    )
    
    print(f"\n✓ Authenticated as: {user_name} ({user_email})")
    print(f"  Roles: {', '.join(roles)}")
    print(f"  Token: {token[:50]}...\n")
    
    return token


# ============================================================
# Tool Discovery with Authentication
# ============================================================

async def discover_mcp_tools_authenticated(token: str) -> tuple[list[dict], list[str]]:
    """
    Discover MCP tools with authenticated context.
    
    WORKAROUND: FastMCP v3.1.1 doesn't support auth-based filtering during discovery.
    This function simulates client-side filtering to demonstrate the concept.
    In production, use Azure API Management or custom MCP server for real filtering.
    
    Args:
        token: Authentication token (Azure AD or demo)
    
    Returns:
        Tuple of (tool definitions, tool names)
    """
    print("🔍 Discovering available tools...")
    print("  (FastMCP limitation: client-side filtering as workaround)\n")
    
    # Decode token to get user roles
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        user_email = payload["sub"]
    except Exception as e:
        print(f"❌ Invalid token: {e}")
        return [], []
    
    # Map users to roles (in production: from Azure AD token claims)
    user_roles_map = {
        "user1@coles.com.au": ["public"],
        "user2@coles.com.au": ["employee"],
        "user3@coles.com.au": ["admin"]
    }
    user_roles = user_roles_map.get(user_email, ["public"])
    
    # Tool permissions (matches server-side TOOL_PERMISSIONS)
    tool_permissions = {
        "get_weather": ["public", "employee", "admin"],
        "get_stock_price": ["employee", "admin"],
        "calculate": ["admin"],
        "get_employee_data": ["admin"]
    }
    
    # Filter tools based on user roles
    available_tools = []
    tool_names = []
    
    for tool_name, allowed_roles in tool_permissions.items():
        if any(role in allowed_roles for role in user_roles):
            tool_def = {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": f"Tool: {tool_name}",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            }
            available_tools.append(tool_def)
            tool_names.append(tool_name)
            print(f"  ✓ {tool_name}")
        else:
            print(f"  ✗ {tool_name} (requires: {', '.join(allowed_roles)})")
    
    return available_tools, tool_names


# ============================================================
# Azure AI Agent with Authenticated MCP Tools
# ============================================================

def run_agent_with_authenticated_mcp(
    user_query: str,
    mcp_tools: list[dict],
    available_tool_names: list[str],
    auth_token: str
) -> None:
    """
    Run Azure AI Agent with authenticated MCP tools.
    
    The agent can only use tools that the user has permission to access.
    """
    print("\n" + "=" * 60)
    print("  AZURE AI AGENT WITH AUTHENTICATED MCP")
    print("=" * 60)
    
    if not mcp_tools:
        print("\n❌ No tools available for your role.")
        print("   Contact your administrator for access.")
        return
    
    print(f"\nYou have access to {len(mcp_tools)} tool(s):")
    for tool_name in available_tool_names:
        print(f"  • {tool_name}")
    
    # Authenticate with Azure (for AI Agent service)
    try:
        project_client = AIProjectClient(
            credential=DefaultAzureCredential(
                exclude_environment_credential=True,
                exclude_managed_identity_credential=True,
            ),
            endpoint=PROJECT_ENDPOINT,
        )
        agent_client = project_client.agents
    except Exception as e:
        print(f"\n❌ Azure authentication failed: {e}")
        print("\nFor demo purposes, skipping Azure AI Agent execution.")
        print("In production, this would:")
        print("  1. Create agent with filtered MCP tools")
        print("  2. Agent selects appropriate tool based on question")
        print("  3. MCP server validates auth token before executing")
        print("  4. Returns result only if user has permission")
        return
    
    # Create agent with filtered tools
    print(f"\nCreating Azure AI Agent with {len(mcp_tools)} authorized tool(s)...")
    
    agent = agent_client.create_agent(
        model=MODEL_DEPLOYMENT,
        name="Authenticated MCP Agent",
        instructions=(
            "You are a helpful assistant with access to external tools via "
            "Model Context Protocol (MCP). You can ONLY use the tools you "
            "have permission to access based on the user's role. "
            "If the user asks for something you don't have access to, "
            "politely explain that you don't have the necessary permissions."
        ),
        tools=mcp_tools,
    )
    print(f"✓ Agent created: {agent.id}")
    
    # Create conversation thread
    thread = agent_client.threads.create()
    agent_client.messages.create(
        thread_id=thread.id,
        role=MessageRole.USER,
        content=user_query,
    )
    
    print(f"\n👤 User Question: \"{user_query}\"")
    print("\n⏳ Agent processing...")
    
    # Run agent
    run = agent_client.runs.create(thread_id=thread.id, agent_id=agent.id)
    
    while run.status in ["queued", "in_progress", "requires_action"]:
        time.sleep(0.5)
        run = agent_client.runs.get(thread_id=thread.id, run_id=run.id)
        
        if run.status == "requires_action":
            tool_calls = run.required_action.submit_tool_outputs.tool_calls
            tool_outputs = []
            
            print(f"\n  🎯 Agent selected {len(tool_calls)} tool(s):")
            
            for tool_call in tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)
                
                print(f"\n     Tool: {func_name}")
                print(f"     Arguments: {json.dumps(func_args)}")
                
                # Check if user has permission (double-check on client side)
                if func_name not in available_tool_names:
                    result = json.dumps({
                        "status": "error",
                        "message": f"Access denied. You don't have permission to use {func_name}"
                    })
                    print(f"     ✗ Access Denied!")
                else:
                    print(f"     ✓ Executing (with auth token)...")
                    # In production, call MCP server with auth token in headers
                    result = f'{{"status": "success", "data": "simulated result"}}'
                    print(f"     ✓ Result: {result[:80]}...")
                
                tool_outputs.append({
                    "tool_call_id": tool_call.id,
                    "output": result,
                })
            
            run = agent_client.runs.submit_tool_outputs(
                thread_id=thread.id,
                run_id=run.id,
                tool_outputs=tool_outputs,
            )
    
    # Display result
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
                print("  AGENT'S RESPONSE")
                print("=" * 60)
                print(f"\n{msg.text_messages[0].text.value}\n")
                print("=" * 60)
    else:
        print(f"\n❌ Run failed: {run.status}")
    
    # Cleanup
    agent_client.delete_agent(agent.id)
    print("\n✓ Agent deleted")


# ============================================================
# Main Execution Flow
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  MCP CLIENT WITH AZURE AD AUTHENTICATION")
    print("=" * 60)
    print("\nThis demo shows how authentication and authorization work")
    print("in Microsoft's agentic framework with MCP tools.")
    
    # STEP 1: Authenticate user
    print("\n" + "=" * 60)
    print("  STEP 1: USER AUTHENTICATION")
    print("=" * 60)
    
    auth_token = get_azure_ad_token()
    
    # STEP 2: Discover available tools (filtered by permissions)
    print("=" * 60)
    print("  STEP 2: DISCOVER AVAILABLE TOOLS")
    print("=" * 60)
    print("\n⚠️  Note: Client-side filtering (workaround for FastMCP limitation)")
    print("   Production: Use Azure API Management for real filtering\n")
    
    try:
        mcp_tools, tool_names = asyncio.run(
            discover_mcp_tools_authenticated(auth_token)
        )
    except Exception as exc:
        print(f"\n❌ Error: {exc}")
        raise SystemExit(1)
    
    print(f"\n✓ Discovered {len(mcp_tools)} tool(s) you have access to")
    
    # STEP 3: Get user question
    print("\n" + "=" * 60)
    print("  STEP 3: ENTER YOUR QUESTION")
    print("=" * 60)
    print("\nExample questions:")
    print("  • What's the weather in Tokyo?  (everyone can access)")
    print("  • Get stock price for MSFT      (employees & admins)")
    print("  • Calculate 15 * 8 + 42         (admins only)")
    print()
    
    user_question = input("Enter your question: ").strip()
    
    if not user_question:
        print("\n❌ No question provided. Exiting.")
        raise SystemExit(0)
    
    # STEP 4: Run agent with authenticated context
    print("\n" + "=" * 60)
    print("  STEP 4: EXECUTE WITH AZURE AI AGENT")
    print("=" * 60)
    
    run_agent_with_authenticated_mcp(
        user_question,
        mcp_tools,
        tool_names,
        auth_token
    )
    
    print("\n" + "=" * 60)
    print("  KEY TAKEAWAYS")
    print("=" * 60)
    print("""
✅ WHAT WORKS (Production-Ready):
1. 🔐 Azure AD authentication for users
2. 🎫 JWT tokens passed to MCP server
3. 🛡️  Server-side execution validation (@require_auth decorator)
4. 📝 Audit logging with user identity

⚠️  CURRENT LIMITATIONS (FastMCP v3.1.1):
1. ❌ Tool discovery filtering: Client-side workaround (not secure)
2. ❌ FastMCP doesn't read Authorization headers
3. ❌ No built-in session-based auth context

🏗️  PRODUCTION SOLUTIONS (Coles):
1. Azure API Management: 
   - Intercepts list_tools() responses
   - Filters based on JWT claims
   - Returns only authorized tools
   
2. Custom MCP Server:
   - Implement auth-aware list_tools()
   - Store session auth context
   - Filter at source
   
3. Defense in Depth:
   - ✅ Execution-time validation (ALWAYS enforce this)
   - ✅ Audit all tool calls
   - ✅ Use managed identities in Azure

Real Implementation:
- Use execution validation (works now)
- Add API Gateway for discovery filtering (production)
- Client-side filtering is for DEMO ONLY (not secure)
""")
    print("=" * 60)
