# MCP (Model Context Protocol) Demo

This demo shows how AI agents discover tools and intelligently select which ones to use based on user questions.

## Files

1. **mcp_server.py** - MCP server with 3 tools:
   - `get_weather` - Get weather for cities
   - `get_stock_price` - Get stock prices
   - `calculate` - Evaluate math expressions

2. **mcp_client.py** - Azure AI Agent that:
   - Gets user input
   - Discovers available tools
   - Intelligently selects the right tool
   - Executes and returns results

3. **agent_decision_demo.py** - Standalone demo (no Azure auth needed)
   - Shows tool discovery process
   - Simulates agent decision-making
   - Runs multiple test queries

4. **mcp_discovery_demo.py** - Simple discovery demo
   - Shows MCP handshake process
   - Demonstrates tool calling

## How to Run

### Option 1: With Azure AI Agent (Full Experience)

**Terminal 1 - Start the MCP Server:**
```bash
cd /home/azureuser/cloudfiles/code/Users/Farbod.Taymouri/MyOwnLearning/mcp
python mcp_server.py
```

**Terminal 2 - Run the Client:**
```bash
# First, authenticate with Azure
az login --scope https://ai.azure.com/.default

# Then run the client
cd /home/azureuser/cloudfiles/code/Users/Farbod.Taymouri/MyOwnLearning/mcp
python mcp_client.py
```

You'll be prompted to enter a question like:
- "What's the weather in Tokyo?"
- "Get the stock price for MSFT"
- "Calculate 15 * 8 + 42"

### Option 2: Standalone Demo (No Azure Auth)

**Terminal 1 - Start the MCP Server:**
```bash
cd /home/azureuser/cloudfiles/code/Users/Farbod.Taymouri/MyOwnLearning/mcp
python mcp_server.py
```

**Terminal 2 - Run the Standalone Demo:**
```bash
cd /home/azureuser/cloudfiles/code/Users/Farbod.Taymouri/MyOwnLearning/mcp
python agent_decision_demo.py
```

This will automatically run 5 test queries and show how the agent selects tools.

### Option 3: Basic Discovery Demo

**Terminal 1 - Start the MCP Server:**
```bash
cd /home/azureuser/cloudfiles/code/Users/Farbod.Taymouri/MyOwnLearning/mcp
python mcp_server.py
```

**Terminal 2 - Run Discovery Demo:**
```bash
cd /home/azureuser/cloudfiles/code/Users/Farbod.Taymouri/MyOwnLearning/mcp
python mcp_discovery_demo.py
```

## What You'll See

### 1. Tool Discovery
```
🔍 STEP 1: DISCOVER AVAILABLE TOOLS
✓ Discovered: get_weather
✓ Discovered: get_stock_price
✓ Discovered: calculate
```

### 2. Agent Analyzes Question
```
👤 User Question: "What's the weather in Tokyo?"

🎯 Agent selected 1 tool(s) to answer the question:
     Tool: get_weather
     Arguments: {"city": "Tokyo"}
     Executing via MCP server...
```

### 3. Final Answer
```
✅ AGENT'S FINAL ANSWER
The weather in Tokyo is currently rainy with a temperature of 18°C 
and 85% humidity.
```

## Key Concepts

1. **MCP = USB-C for AI Tools** - Universal protocol for any agent framework
2. **Dynamic Discovery** - Agents learn capabilities at runtime
3. **Intelligent Selection** - Agents analyze questions and pick the right tool
4. **Type Safety** - JSON Schema ensures correct parameters

## Architecture

```
User Question → Agent Discovers Tools → Agent Selects Tool → Tool Executes → Answer
```

The beauty of MCP is that the agent doesn't need to know about tools in advance - it discovers them dynamically!
