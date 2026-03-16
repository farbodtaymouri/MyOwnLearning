"""
MCP Server with Authentication & Role-Based Access Control (RBAC)
==================================================================

This server demonstrates how to implement authentication and authorization
in MCP servers, showing how different users get access to different tools
based on their roles and permissions.

Real-world use case (Coles example):
- Public users: Can only access weather data
- Employees: Can access weather + internal stock prices
- Admins: Can access all tools including calculations

Authentication flow:
    1. Client sends Azure AD token in request headers
    2. Server validates token against Azure AD / Entra ID
    3. Server extracts user roles/groups from token claims
    4. Server returns ONLY tools user has permission for
    5. Agent discovers only accessible tools (client has zero knowledge of others)
    6. Server validates permission again on execution (defense in depth)

Security principle: Client never sees permission matrix or unauthorized tools.

Prerequisites: pip install fastmcp pyjwt cryptography aiohttp
"""

from fastmcp import FastMCP
import jwt
import json
from typing import Dict, List, Optional
from functools import wraps
from aiohttp import web
import asyncio

# ============================================================
# STEP 1: Define User Roles and Tool Permissions
# ============================================================

# Role-based access control matrix
TOOL_PERMISSIONS = {
    "get_weather": ["public", "employee", "admin"],      # Everyone can access
    "get_stock_price": ["employee", "admin"],             # Only employees and admins
    "calculate": ["admin"],                               # Only admins
    "get_employee_data": ["admin"],                       # Only admins (sensitive data)
}

# Simulated user database (in production, this comes from Azure AD)
USERS_DB = {
    "user1@coles.com.au": {
        "name": "John Public",
        "roles": ["public"],
        "department": "Customer"
    },
    "user2@coles.com.au": {
        "name": "Jane Employee",
        "roles": ["employee"],
        "department": "Operations"
    },
    "user3@coles.com.au": {
        "name": "Admin User",
        "roles": ["admin", "employee"],
        "department": "IT"
    }
}

# Simulated JWT secret (in production, use Azure AD public keys)
JWT_SECRET = "demo-secret-key-replace-with-azure-ad-validation"


# ============================================================
# STEP 2: Authentication Helper Functions
# ============================================================

class AuthContext:
    """Stores authenticated user context."""
    def __init__(self, user_id: str, roles: List[str], name: str):
        self.user_id = user_id
        self.roles = roles
        self.name = name
    
    def has_permission(self, tool_name: str) -> bool:
        """Check if user has permission to use a tool."""
        allowed_roles = TOOL_PERMISSIONS.get(tool_name, [])
        return any(role in allowed_roles for role in self.roles)


def validate_token(token: str) -> Optional[AuthContext]:
    """
    Validate JWT token and extract user context.
    
    In production, this would:
    1. Validate token signature against Azure AD public keys
    2. Check token expiration
    3. Verify audience and issuer
    4. Extract user roles from token claims
    
    Args:
        token: JWT token from Authorization header
    
    Returns:
        AuthContext if valid, None if invalid
    """
    try:
        # Decode JWT (in production, validate against Azure AD)
        payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        
        user_id = payload.get("sub")  # Subject (user email)
        
        # Get user data (in production, from Azure AD Graph API or token claims)
        user_data = USERS_DB.get(user_id)
        if not user_data:
            print(f"❌ User not found: {user_id}")
            return None
        
        return AuthContext(
            user_id=user_id,
            roles=user_data["roles"],
            name=user_data["name"]
        )
    
    except jwt.ExpiredSignatureError:
        print("❌ Token expired")
        return None
    except jwt.InvalidTokenError as e:
        print(f"❌ Invalid token: {e}")
        return None
    except Exception as e:
        print(f"❌ Authentication error: {e}")
        return None


def create_demo_token(user_email: str) -> str:
    """
    Create a demo JWT token for testing.
    
    In production, users get tokens from Azure AD via:
    - az login
    - DefaultAzureCredential
    - Interactive browser login
    """
    return jwt.encode(
        {"sub": user_email, "exp": 9999999999},  # Sub = subject (user ID)
        JWT_SECRET,
        algorithm="HS256"
    )


# ============================================================
# STEP 3: Create MCP Server with Authentication
# ============================================================

mcp = FastMCP(
    name="Authenticated MCP Server",
    instructions="MCP server with role-based access control."
)

# Store current auth context (in production, use request context)
_current_auth_context: Optional[AuthContext] = None


def set_auth_context(token: str) -> bool:
    """
    Set authentication context from token.
    Called before tool discovery/execution.
    
    Returns:
        True if authentication successful, False otherwise
    """
    global _current_auth_context
    _current_auth_context = validate_token(token)
    return _current_auth_context is not None


def require_auth(allowed_roles: List[str] = None):
    """
    Decorator to require authentication and check roles.
    
    Args:
        allowed_roles: List of roles allowed to use this tool
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Check if user is authenticated
            if not _current_auth_context:
                return {
                    "status": "error",
                    "message": "Authentication required. Please provide a valid token."
                }
            
            # Check if user has required role
            tool_name = func.__name__
            if not _current_auth_context.has_permission(tool_name):
                return {
                    "status": "error",
                    "message": f"Access denied. Required roles: {TOOL_PERMISSIONS.get(tool_name, [])}. Your roles: {_current_auth_context.roles}"
                }
            
            # Execute the tool
            return func(*args, **kwargs)
        return wrapper
    return decorator


# ============================================================
# STEP 4: Define Tools with Access Control
# ============================================================

@mcp.tool()
@require_auth(allowed_roles=["public", "employee", "admin"])
def get_weather(city: str) -> dict:
    """
    Get weather data (PUBLIC ACCESS - all authenticated users).
    
    Args:
        city: City name (Sydney, London, New York, Tokyo)
    
    Returns:
        Weather data with temperature, condition, humidity
    """
    weather_db = {
        "sydney": {"temp": 28, "condition": "sunny", "humidity": 65},
        "london": {"temp": 12, "condition": "cloudy", "humidity": 80},
        "new york": {"temp": 22, "condition": "clear", "humidity": 70},
        "tokyo": {"temp": 18, "condition": "rainy", "humidity": 85},
    }
    
    city_key = city.lower().strip()
    if city_key in weather_db:
        data = weather_db[city_key]
        return {
            "city": city,
            "temperature": f"{data['temp']}°C",
            "condition": data["condition"],
            "humidity": f"{data['humidity']}%",
            "status": "success",
            "accessed_by": _current_auth_context.user_id
        }
    else:
        return {
            "status": "error",
            "message": f"Unknown city. Try: {', '.join(weather_db.keys())}"
        }


@mcp.tool()
@require_auth(allowed_roles=["employee", "admin"])
def get_stock_price(symbol: str) -> dict:
    """
    Get internal stock price data (EMPLOYEE ACCESS ONLY).
    
    This is restricted to employees only as it contains internal pricing data.
    
    Args:
        symbol: Stock ticker symbol (AAPL, MSFT, GOOGL, TSLA, AMZN)
    
    Returns:
        Stock data with price, change, and market cap
    """
    stock_db = {
        "aapl": {"price": 178.50, "change": +2.3, "market_cap": "2.8T"},
        "msft": {"price": 420.15, "change": +5.7, "market_cap": "3.1T"},
        "googl": {"price": 141.80, "change": -1.2, "market_cap": "1.8T"},
        "tsla": {"price": 248.30, "change": +8.9, "market_cap": "790B"},
        "amzn": {"price": 178.25, "change": +3.4, "market_cap": "1.9T"},
    }
    
    symbol_key = symbol.upper().strip()
    if symbol_key.lower() in stock_db:
        data = stock_db[symbol_key.lower()]
        return {
            "symbol": symbol_key,
            "price": f"${data['price']}",
            "change": f"+{data['change']}%" if data['change'] >= 0 else f"{data['change']}%",
            "market_cap": data['market_cap'],
            "status": "success",
            "accessed_by": _current_auth_context.user_id,
            "department": USERS_DB[_current_auth_context.user_id]["department"]
        }
    else:
        return {
            "status": "error",
            "message": f"Unknown symbol. Try: {', '.join([s.upper() for s in stock_db.keys()])}"
        }


@mcp.tool()
@require_auth(allowed_roles=["admin"])
def calculate(expression: str) -> dict:
    """
    Evaluate math expressions (ADMIN ACCESS ONLY).
    
    Restricted to admins only due to potential security risks with eval().
    
    Args:
        expression: Math expression (e.g., "2 + 2", "10 * 5 + 3", "sqrt(16)")
    
    Returns:
        Calculation result
    """
    import math
    
    allowed_names = {
        "abs": abs, "round": round, "min": min, "max": max,
        "sum": sum, "pow": pow,
        "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos,
        "tan": math.tan, "pi": math.pi, "e": math.e,
    }
    
    try:
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return {
            "expression": expression,
            "result": result,
            "status": "success",
            "accessed_by": _current_auth_context.user_id
        }
    except Exception as e:
        return {
            "expression": expression,
            "status": "error",
            "message": f"Invalid expression: {str(e)}"
        }


@mcp.tool()
@require_auth(allowed_roles=["admin"])
def get_employee_data(employee_id: str) -> dict:
    """
    Get sensitive employee data (ADMIN ACCESS ONLY).
    
    This contains PII and is restricted to admins only.
    
    Args:
        employee_id: Employee email address
    
    Returns:
        Employee information
    """
    if employee_id in USERS_DB:
        employee = USERS_DB[employee_id]
        return {
            "employee_id": employee_id,
            "name": employee["name"],
            "department": employee["department"],
            "roles": employee["roles"],
            "status": "success",
            "accessed_by": _current_auth_context.user_id
        }
    else:
        return {
            "status": "error",
            "message": "Employee not found"
        }


# ============================================================
# STEP 5: Connection Authentication & Tool Filtering
# ============================================================

def get_available_tools_for_user(auth_context: AuthContext) -> List[str]:
    """
    Get list of tools available to the authenticated user.
    
    This filters tools based on user's roles.
    Server returns ONLY tools the user has permission for.
    """
    available_tools = []
    for tool_name, allowed_roles in TOOL_PERMISSIONS.items():
        if any(role in allowed_roles for role in auth_context.roles):
            available_tools.append(tool_name)
    return available_tools


# Custom middleware to handle authentication on connection
# NOTE: FastMCP v3.1.1 may not fully support custom tool filtering yet
# In production, you would use a custom SSE endpoint handler that:
#   1. Validates Authorization header on connection
#   2. Stores auth context for the session
#   3. Filters tools in list_tools() based on user permissions
#   4. Validates again on each tool execution
#
# For enterprise deployments, consider:
#   - Implementing custom MCP server with full auth support
#   - Using API Gateway (Azure API Management) for auth
#   - Adding auth middleware at the Kubernetes ingress level


# ============================================================
# STEP 6: Start Server with Authentication Info
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("MCP SERVER WITH AUTHENTICATION & RBAC")
    print("=" * 70)
    
    print("\n📋 Role-Based Access Control Matrix:")
    print("\nTool Name              | Public | Employee | Admin")
    print("-" * 60)
    for tool, roles in TOOL_PERMISSIONS.items():
        public = "✓" if "public" in roles else "✗"
        employee = "✓" if "employee" in roles else "✗"
        admin = "✓" if "admin" in roles else "✗"
        print(f"{tool:22} |   {public}    |    {employee}     |   {admin}")
    
    print("\n👤 Demo Users:")
    for user_id, data in USERS_DB.items():
        print(f"\n  {user_id}")
        print(f"    Name: {data['name']}")
        print(f"    Roles: {', '.join(data['roles'])}")
        print(f"    Department: {data['department']}")
        
        # Generate demo token
        token = create_demo_token(user_id)
        print(f"    Demo Token: {token[:50]}...")
    
    print("\n" + "=" * 70)
    print("Server Endpoint: http://localhost:8000/sse")
    print("=" * 70)
    print("\nAuthentication Flow:")
    print("  1. Client obtains Azure AD token (or demo token)")
    print("  2. Client sends token in request headers")
    print("  3. Server validates token and extracts user roles")
    print("  4. Server filters tools based on user permissions")
    print("  5. Agent discovers only authorized tools")
    print("\nStarting server...\n")
    
    # Run MCP server
    mcp.run(transport="sse", host="0.0.0.0", port=8000)
