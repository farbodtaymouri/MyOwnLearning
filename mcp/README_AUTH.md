# MCP Authentication & Authorization Demo

This demo shows how **authentication and role-based access control (RBAC)** work in Microsoft's agentic framework with MCP servers.

## 🎯 Key Concept

**Not all users should see or access all tools.** Based on authentication and user roles, different tools are available:

```
┌─────────────────────────────────────────────────────────┐
│  User Role        │ Available Tools                     │
├───────────────────┼─────────────────────────────────────┤
│  Public User      │ ✓ get_weather                       │
│                   │ ✗ get_stock_price                   │
│                   │ ✗ calculate                         │
│                   │ ✗ get_employee_data                 │
├───────────────────┼─────────────────────────────────────┤
│  Employee         │ ✓ get_weather                       │
│                   │ ✓ get_stock_price                   │
│                   │ ✗ calculate                         │
│                   │ ✗ get_employee_data                 │
├───────────────────┼─────────────────────────────────────┤
│  Admin            │ ✓ get_weather                       │
│                   │ ✓ get_stock_price                   │
│                   │ ✓ calculate                         │
│                   │ ✓ get_employee_data                 │
└─────────────────────────────────────────────────────────┘
```

## 📋 Files

### 1. `mcp_server_auth.py` - Authenticated MCP Server

**Features:**
- JWT token validation
- Role-based access control (RBAC)
- 4 tools with different permission levels
- User context tracking
- Access logging with user identity

**Tools & Permissions:**
```python
TOOL_PERMISSIONS = {
    "get_weather": ["public", "employee", "admin"],     # PUBLIC
    "get_stock_price": ["employee", "admin"],           # EMPLOYEE+
    "calculate": ["admin"],                             # ADMIN ONLY
    "get_employee_data": ["admin"],                     # ADMIN ONLY
}
```

**Demo Users:**
- `user1@coles.com.au` - John Public (public role)
- `user2@coles.com.au` - Jane Employee (employee role)  
- `user3@coles.com.au` - Admin User (admin role)

### 2. `mcp_client_auth.py` - Authenticated MCP Client

**Features:**
- Azure AD authentication simulation
- Token-based MCP connection
- Filtered tool discovery
- Permission-aware Azure AI Agent
- Access denial handling

**Flow:**
1. User authenticates (selects role)
2. Receives JWT token
3. Connects to MCP server with token
4. Discovers only tools they have access to
5. Agent can only call authorized tools

## 🚀 How to Run

### Terminal 1: Start Authenticated MCP Server

```bash
cd /home/azureuser/cloudfiles/code/Users/Farbod.Taymouri/MyOwnLearning/mcp

# Install dependencies
pip install fastmcp pyjwt cryptography

# Start server
python mcp_server_auth.py
```

**Expected output:**
```
======================================================================
MCP SERVER WITH AUTHENTICATION & RBAC
======================================================================

📋 Role-Based Access Control Matrix:

Tool Name              | Public | Employee | Admin
------------------------------------------------------------
get_weather            |   ✓    |    ✓     |   ✓
get_stock_price        |   ✗    |    ✓     |   ✓
calculate              |   ✗    |    ✗     |   ✓
get_employee_data      |   ✗    |    ✗     |   ✓

👤 Demo Users:
  user1@coles.com.au
    Name: John Public
    Roles: public
    Department: Customer
    Demo Token: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...

...
Server Endpoint: http://localhost:8000/sse
```

### Terminal 2: Run Authenticated Client

```bash
cd /home/azureuser/cloudfiles/code/Users/Farbod.Taymouri/MyOwnLearning/mcp

# Install dependencies  
pip install azure-ai-projects azure-identity mcp pyjwt

# Run client
python mcp_client_auth.py
```

**Interactive flow:**
```
🔐 AUTHENTICATION
Who are you logging in as?
  1. John Public (public user - LIMITED ACCESS)
  2. Jane Employee (employee - MEDIUM ACCESS)
  3. Admin User (admin - FULL ACCESS)

Enter choice (1-3): 2

✓ Authenticated as: Jane Employee (user2@coles.com.au)
  Roles: employee

🔍 Discovering available tools...
  ✓ Discovered: get_weather
  ✓ Discovered: get_stock_price
  ✗ No access: calculate (requires: admin)
  ✗ No access: get_employee_data (requires: admin)

You have access to 2 tool(s):
  • get_weather
  • get_stock_price

Enter your question: Get stock price for MSFT
```

## 🏗️ Production Architecture (Coles Example)

### How It Works in Real Deployment

```
┌──────────────────────────────────────────────────────────────┐
│  Developer Laptop / Client                                   │
│                                                               │
│  1. az login (Azure AD authentication)                       │
│     └─> Receives JWT token with user claims                  │
│                                                               │
│  2. VS Code connects to MCP server                           │
│     └─> Sends: Authorization: Bearer <token>                 │
└───────────────────────────┬──────────────────────────────────┘
                            │
                            │ HTTPS + JWT Token
                            ↓
┌──────────────────────────────────────────────────────────────┐
│  AKS Cluster (Azure Kubernetes Service)                      │
│                                                               │
│  ┌────────────────────────────────────────────┐              │
│  │  MCP Server Pod                            │              │
│  │                                             │              │
│  │  1. Validate JWT token                     │              │
│  │     ├─> Check signature (Azure AD keys)    │◄─────────┐   │
│  │     ├─> Verify expiration                  │          │   │
│  │     └─> Extract user claims               │          │   │
│  │                                             │          │   │
│  │  2. Extract roles from token claims        │          │   │
│  │     └─> groups: ["employee", ...]          │          │   │
│  │                                             │          │   │
│  │  3. Filter tools based on roles           │          │   │
│  │     └─> Return only authorized tools       │          │   │
│  │                                             │          │   │
│  │  4. Execute tool (if authorized)           │          │   │
│  │     └─> Log access with user identity      │          │   │
│  └────────────────────────────────────────────┘          │   │
│                                                           │   │
└───────────────────────────────────────────────────────────┼───┘
                                                            │
                    ┌───────────────────────────────────────┘
                    │
                    ↓
        ┌────────────────────────────┐
        │  Azure AD / Entra ID       │
        │                            │
        │  - Token validation        │
        │  - User groups/roles       │
        │  - Claims extraction       │
        └────────────────────────────┘
```

### Azure AD Integration (Production)

**Token Claims Structure:**
```json
{
  "iss": "https://sts.windows.net/{tenant-id}/",
  "sub": "user2@coles.com.au",
  "aud": "api://mcp-server",
  "exp": 1234567890,
  "groups": [
    "sg-coles-employees",
    "sg-product-team"
  ],
  "roles": ["employee"],
  "upn": "user2@coles.com.au",
  "name": "Jane Employee"
}
```

**Server-side validation:**
```python
from azure.identity import DefaultAzureCredential
from microsoft.identity.web import validate_jwt

def validate_azure_ad_token(token: str) -> AuthContext:
    """Validate token against Azure AD."""
    # Get Azure AD public keys for signature validation
    keys = get_azure_ad_public_keys()
    
    # Validate token
    claims = validate_jwt(
        token=token,
        issuer=f"https://sts.windows.net/{TENANT_ID}/",
        audience="api://mcp-server",
        public_keys=keys
    )
    
    # Extract user info
    return AuthContext(
        user_id=claims["sub"],
        roles=claims.get("roles", []),
        groups=claims.get("groups", []),
        name=claims.get("name", "Unknown")
    )
```

## 🔐 Security Best Practices

### 1. **Token Validation**
```python
# Always validate:
✓ Signature (using Azure AD public keys)
✓ Expiration (exp claim)
✓ Audience (aud claim)
✓ Issuer (iss claim)
✓ Not before (nbf claim)
```

### 2. **Permission Checking**
```python
# Double validation:
✓ Client-side: Filter tools during discovery
✓ Server-side: Validate permissions before execution
```

### 3. **Audit Logging**
```python
# Log all access:
log.info(
    f"Tool executed: {tool_name}",
    extra={
        "user_id": auth_context.user_id,
        "roles": auth_context.roles,
        "tool": tool_name,
        "args": tool_args,
        "ip": request.remote_addr,
        "timestamp": datetime.utcnow()
    }
)
```

### 4. **Least Privilege**
```python
# Start with minimal permissions:
DEFAULT_ROLE = "public"  # Most restrictive

# Require explicit grants:
TOOL_PERMISSIONS = {
    "sensitive_data": ["admin"],  # Not ["public", "employee", "admin"]
}
```

## 📊 Comparison: Demo vs Production

| Aspect | Demo | Production (Coles) |
|--------|------|-------------------|
| **Authentication** | Simulated JWT | Azure AD / Entra ID |
| **Token Source** | Hardcoded secret | Azure AD public keys |
| **User Database** | In-memory dict | Azure AD Graph API |
| **Role Source** | Hardcoded | Azure AD groups/claims |
| **Token Validation** | Basic JWT decode | Full validation + signature check |
| **Transport** | HTTP (localhost) | HTTPS with TLS |
| **Audit** | Console logs | Azure Monitor / Log Analytics |
| **Secrets** | Hardcoded | Azure Key Vault |

## 🎓 Key Takeaways

1. **🔐 Authentication**: Users prove who they are (Azure AD token)
2. **🛡️ Authorization**: System determines what they can do (RBAC)
3. **🔍 Discovery Filtering**: Users only see tools they can access
4. **✋ Execution Gating**: Server validates permissions before running tools
5. **📝 Audit Trail**: All access logged with user identity
6. **🎯 Least Privilege**: Start with minimal access, grant as needed

## 🚀 Next Steps

To deploy this in production at Coles:

1. **Replace JWT validation** with Azure AD validation
2. **Add Azure Key Vault** for secrets management
3. **Implement audit logging** to Azure Monitor
4. **Add rate limiting** per user/role
5. **Deploy to AKS** with proper network policies
6. **Configure Private Endpoints** for internal access
7. **Set up Azure AD groups** for role management
8. **Add monitoring & alerts** for access violations

## 📚 Related Resources

- [Azure AD Authentication](https://learn.microsoft.com/en-us/azure/active-directory/develop/)
- [JWT Token Validation](https://learn.microsoft.com/en-us/azure/active-directory/develop/access-tokens)
- [Azure RBAC](https://learn.microsoft.com/en-us/azure/role-based-access-control/)
- [MCP Specification](https://modelcontextprotocol.io/)
- [Azure AI Agent Service](https://learn.microsoft.com/en-us/azure/ai-services/agents/)
