"""
FastMCP Server — Simplified Demo
=================================

MCP (Model Context Protocol) is like "USB-C for AI tools": a universal standard
that lets AI agents discover and call external functions, regardless of which
agent framework you use (Azure AI, LangChain, CrewAI, etc.).

How it works:
    1. Client → initialize()    → Server: "I support MCP v1.0"
    2. Client → list_tools()    → Server: "Here are my tools (JSON Schema)"
    3. Client → call_tool(...)  → Server: Executes Python function, returns result
    4. Agent uses the result to answer the user's question

FastMCP makes this easy with an @mcp.tool() decorator (like Flask's @app.route).

Prerequisites: pip install fastmcp
Start server:  python mcp_server.py
"""

from fastmcp import FastMCP

# ============================================================
# STEP 1: Create the MCP server
# ============================================================
mcp = FastMCP(
    name="Weather Demo Server",
    instructions="Returns simulated weather data for major cities."
)


# ============================================================
# STEP 2: Define a tool using the @mcp.tool() decorator
# ============================================================
# This decorator:
#   - Registers the function as an MCP tool
#   - Auto-generates JSON Schema from type hints
#   - Makes it callable by any MCP client/agent
@mcp.tool()
def get_weather(city: str) -> dict:
    """
    Get simulated weather for a city.
    
    Args:
        city: City name (Sydney, London, New York, Tokyo)
    
    Returns:
        Weather data with temperature, condition, humidity
    """
    # Simulated data (in production, call a real API like OpenWeatherMap)
    weather_db = {
        "sydney":   {"temp": 28, "condition": "sunny",   "humidity": 65},
        "london":   {"temp": 12, "condition": "cloudy",  "humidity": 80},
        "new york": {"temp": 22, "condition": "clear",   "humidity": 70},
        "tokyo":    {"temp": 18, "condition": "rainy",   "humidity": 85},
    }
    
    city_key = city.lower().strip()
    if city_key in weather_db:
        data = weather_db[city_key]
        return {
            "city": city,
            "temperature": f"{data['temp']}°C",
            "condition": data["condition"],
            "humidity": f"{data['humidity']}%",
            "status": "success"
        }
    else:
        return {
            "city": city,
            "status": "error",
            "message": f"Unknown city. Try: {', '.join(weather_db.keys())}"
        }


@mcp.tool()
def get_stock_price(symbol: str) -> dict:
    """
    Get simulated stock price for a company.
    
    Args:
        symbol: Stock ticker symbol (AAPL, MSFT, GOOGL, TSLA, AMZN)
    
    Returns:
        Stock data with price, change, and market cap
    """
    # Simulated stock data (in production, call a real API like Alpha Vantage)
    stock_db = {
        "aapl":  {"price": 178.50, "change": +2.3, "market_cap": "2.8T"},
        "msft":  {"price": 420.15, "change": +5.7, "market_cap": "3.1T"},
        "googl": {"price": 141.80, "change": -1.2, "market_cap": "1.8T"},
        "tsla":  {"price": 248.30, "change": +8.9, "market_cap": "790B"},
        "amzn":  {"price": 178.25, "change": +3.4, "market_cap": "1.9T"},
    }
    
    symbol_key = symbol.upper().strip()
    if symbol_key.lower() in stock_db:
        data = stock_db[symbol_key.lower()]
        return {
            "symbol": symbol_key,
            "price": f"${data['price']}",
            "change": f"+{data['change']}%" if data['change'] >= 0 else f"{data['change']}%",
            "market_cap": data['market_cap'],
            "status": "success"
        }
    else:
        return {
            "symbol": symbol_key,
            "status": "error",
            "message": f"Unknown symbol. Try: {', '.join([s.upper() for s in stock_db.keys()])}"
        }


@mcp.tool()
def calculate(expression: str) -> dict:
    """
    Evaluate a mathematical expression safely.
    
    Args:
        expression: Math expression (e.g., "2 + 2", "10 * 5 + 3", "sqrt(16)")
    
    Returns:
        Calculation result
    """
    import math
    
    # Safe evaluation with limited scope
    allowed_names = {
        "abs": abs, "round": round, "min": min, "max": max,
        "sum": sum, "pow": pow,
        "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos,
        "tan": math.tan, "pi": math.pi, "e": math.e,
    }
    
    try:
        # Evaluate safely with restricted scope
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return {
            "expression": expression,
            "result": result,
            "status": "success"
        }
    except Exception as e:
        return {
            "expression": expression,
            "status": "error",
            "message": f"Invalid expression: {str(e)}"
        }


# ============================================================
# STEP 3: Start the server
# ============================================================
if __name__ == "__main__":
    print("=" * 50)
    print("MCP Server Started")
    print("=" * 50)
    print("\nTools available:")
    print("  1. get_weather - Get weather data for cities")
    print("  2. get_stock_price - Get stock prices")
    print("  3. calculate - Evaluate math expressions")
    print("\nEndpoint: http://localhost:8000/sse")
    print("\nWaiting for client connections...\n")
    
    # Run with SSE transport (Server-Sent Events over HTTP)
    # The client will connect to http://localhost:8000/sse
    mcp.run(transport="sse", host="0.0.0.0", port=8000)
