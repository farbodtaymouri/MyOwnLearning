from azure.identity import DefaultAzureCredential
from azure.ai.agents.models import FilePurpose, CodeInterpreterTool, ListSortOrder, MessageRole
import os
from dotenv import load_dotenv
from typing import Any
from pathlib import Path
from datetime import datetime
import concurrent.futures

# Enable tracing - Simple way!
from azure.ai.projects import AIProjectClient
from azure.monitor.opentelemetry import configure_azure_monitor


class AzureAIAgentPatterns:
    """
    A comprehensive class that demonstrates various AI agent design patterns
    using Azure AI Agent Service, including:
    - Sequential Prompt Chain
    - Multi-Agent Workflow
    - Routing Pattern
    - Parallel Execution
    - Reflection Pattern (Generate → Critique → Refine)
    """
    
    def __init__(self, project_endpoint: str = None, model_deployment: str = None):
        """
        Initialize the Azure AI Agent Patterns class.
        
        Args:
            project_endpoint: Azure AI Project endpoint URL
            model_deployment: Model deployment name (e.g., 'gpt-4o')
        """
        # Load environment variables
        load_dotenv()
        
        # Set endpoint and model from parameters or environment
        # self.project_endpoint = project_endpoint or os.getenv("PROJECT_ENDPOINT")
        # self.model_deployment = model_deployment or os.getenv("MODEL_DEPLOYMENT_NAME")

        self.project_endpoint="https://arch-poc-ft-foundry-pro-resource.services.ai.azure.com/api/projects/arch_poc_ft_foundry_project"
        self.model_deployment="gpt-4o"
        
        if not self.project_endpoint or not self.model_deployment:
            raise ValueError("PROJECT_ENDPOINT and MODEL_DEPLOYMENT_NAME must be provided or set in .env file")
        
        # Initialize project client with tracing
        self.project_client = AIProjectClient(
            credential=DefaultAzureCredential(
                exclude_environment_credential=True,
                exclude_managed_identity_credential=True
            ),
            endpoint=self.project_endpoint
        )
        
        # Configure Azure Monitor tracing
        connection_string = self.project_client.telemetry.get_application_insights_connection_string()
        print(f"✓ Application Insights connected: {connection_string[:50]}...")
        configure_azure_monitor(connection_string=connection_string)
        print("✓ Tracing enabled - traces will appear in AI Foundry Portal > Tracing")
        
        # Get the agents client
        self.agent_client = self.project_client.agents
    
    @staticmethod
    def get_current_datetime():
        """Simple function for routing pattern - No agent needed."""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    @staticmethod
    def get_stock_price(ticker: str) -> dict:
        """Simulated stock price lookup tool."""
        prices = {"AAPL": 178.15, "GOOGL": 1750.30, "MSFT": 425.50}
        price = prices.get(ticker.upper())
        if price:
            return {"ticker": ticker.upper(), "price": price}
        else:
            return {"ticker": ticker.upper(), "error": "Price not found"}
    
    @staticmethod
    def git_scan_status(directory: str) -> dict:
        """Scan git repository for uncommitted changes and remote info."""
        import subprocess
        import os
        
        try:
            # Change to the directory
            original_dir = os.getcwd()
            os.chdir(directory)
            
            # Check if it's a git repo
            subprocess.run(["git", "status"], capture_output=True, check=True)
            
            # Get status
            status_result = subprocess.run(
                ["git", "status", "--porcelain"], 
                capture_output=True, text=True, check=True
            )
            
            # Get remote URL
            remote_result = subprocess.run(
                ["git", "remote", "get-url", "origin"],
                capture_output=True, text=True, check=True
            )
            
            # Get current branch
            branch_result = subprocess.run(
                ["git", "branch", "--show-current"],
                capture_output=True, text=True, check=True
            )
            
            os.chdir(original_dir)
            
            has_changes = bool(status_result.stdout.strip())
            changes_list = status_result.stdout.strip().split('\n') if has_changes else []
            
            return {
                "has_changes": has_changes,
                "changes": changes_list,
                "remote_url": remote_result.stdout.strip(),
                "branch": branch_result.stdout.strip(),
                "directory": directory
            }
        except subprocess.CalledProcessError as e:
            return {"error": f"Git command failed: {str(e)}"}
        except Exception as e:
            return {"error": f"Error: {str(e)}"}
    
    @staticmethod
    def git_commit_and_push(directory: str, commit_message: str) -> dict:
        """Stage all changes, commit with message, and push to remote."""
        import subprocess
        import os
        
        try:
            original_dir = os.getcwd()
            os.chdir(directory)
            
            # Stage all changes
            subprocess.run(["git", "add", "."], capture_output=True, check=True)
            
            # Commit
            commit_result = subprocess.run(
                ["git", "commit", "-m", commit_message],
                capture_output=True, text=True, check=True
            )
            
            # Push to remote
            push_result = subprocess.run(
                ["git", "push"],
                capture_output=True, text=True, check=True
            )
            
            os.chdir(original_dir)
            
            return {
                "success": True,
                "commit_output": commit_result.stdout.strip(),
                "push_output": push_result.stdout.strip()
            }
        except subprocess.CalledProcessError as e:
            return {"error": f"Git command failed: {str(e)}", "stderr": e.stderr}
        except Exception as e:
            return {"error": f"Error: {str(e)}"}
    
    def run_prompt_chain(self):
        """
        Sequential prompt chain: Output from one step becomes input to the next step.
        Single agent processing its own outputs in a pipeline.
        """
        
        with self.agent_client:
            # Create an agent
            print("Creating agent...")
            agent = self.agent_client.create_agent(
                model=self.model_deployment,
                name="Sequential Pipeline Agent",
                instructions="You are a task processor. Execute each task independently using only the input provided."
            )
            print(f"Agent created with ID: {agent.id}")
            
            # Define the pipeline tasks
            pipeline_tasks = [
                {
                    "name": "Extract Keywords",
                    "prompt": "Extract 5 key business terms from this text: 'Retail companies need to modernize their infrastructure to handle peak shopping seasons, manage inventory efficiently, and provide personalized customer experiences.'"
                },
                {
                    "name": "Generate Definitions",
                    "prompt": "For each of these terms: {previous_output}\n\nProvide a one-sentence technical definition."
                },
                {
                    "name": "Create Implementation Steps",
                    "prompt": "Based on these definitions: {previous_output}\n\nCreate 3 actionable implementation steps for a retail company."
                }
            ]
            
            print("\n" + "="*80)
            print("SEQUENTIAL PIPELINE EXECUTION")
            print("="*80)
            
            previous_output = ""
            
            for i, task in enumerate(pipeline_tasks, 1):
                # Create a new thread for each step (isolated context)
                thread = self.agent_client.threads.create()
                
                # Inject previous output into the current prompt
                if "{previous_output}" in task["prompt"]:
                    current_prompt = task["prompt"].replace("{previous_output}", previous_output)
                else:
                    current_prompt = task["prompt"]
                
                print(f"\n{'─'*80}")
                print(f"STEP {i}: {task['name']}")
                print(f"{'─'*80}")
                print(f"Input: {current_prompt[:100]}...")
                
                # Send message to agent
                message = self.agent_client.messages.create(
                    thread_id=thread.id,
                    role=MessageRole.USER,
                    content=current_prompt
                )
                
                # Execute
                run = self.agent_client.runs.create_and_process(
                    thread_id=thread.id,
                    agent_id=agent.id
                )
                
                if run.status == "completed":
                    # Get the response
                    messages = self.agent_client.messages.list(
                        thread_id=thread.id,
                        order=ListSortOrder.DESCENDING,
                        limit=1
                    )
                    
                    for msg in messages:
                        if msg.role == MessageRole.AGENT:
                            if msg.text_messages:
                                previous_output = msg.text_messages[0].text.value
                                print(f"\nOutput:\n{previous_output}\n")
                            break
                else:
                    print(f"❌ Task failed with status: {run.status}")
                    break
            
            # Cleanup
            print(f"\n{'='*80}")
            self.agent_client.delete_agent(agent.id)
            print(f"Pipeline completed. Agent deleted.")
            print(f"{'='*80}")
    
    def run_conditional_prompt_chain(self):
        """
        Multi-agent workflow: Output from one agent becomes input to another agent.
        Each agent is specialized for a specific task.
        """
        
        with self.agent_client:
            # Create specialized agents
            print("Creating specialized agents...")
            
            analyzer_agent = self.agent_client.create_agent(
                model=self.model_deployment,
                name="Data Analyzer Agent",
                instructions="You analyze data and extract insights. Be concise and structured."
            )
            
            validator_agent = self.agent_client.create_agent(
                model=self.model_deployment,
                name="Validator Agent",
                instructions="You validate and check for errors or inconsistencies. Provide clear feedback."
            )
            
            formatter_agent = self.agent_client.create_agent(
                model=self.model_deployment,
                name="Formatter Agent",
                instructions="You format data into specific output formats. Follow instructions precisely."
            )
            
            print(f"✓ Analyzer Agent: {analyzer_agent.id}")
            print(f"✓ Validator Agent: {validator_agent.id}")
            print(f"✓ Formatter Agent: {formatter_agent.id}")
            
            print(f"\n{'='*80}")
            print("MULTI-AGENT SEQUENTIAL WORKFLOW")
            print(f"{'='*80}")
            
            # STEP 1: Analyzer Agent processes initial data
            print(f"\n{'─'*80}")
            print("STEP 1: Analyzer Agent - Extract Key Metrics")
            print(f"{'─'*80}")
            
            thread1 = self.agent_client.threads.create()
            initial_data = """
            Sales Report Q4 2025:
            - Revenue: $2.5M (up 15% from Q3)
            - Customer Acquisition: 1,200 new customers
            - Churn Rate: 8%
            - Top Product: Cloud Storage (40% of revenue)
            """
            
            message1 = self.agent_client.messages.create(
                thread_id=thread1.id,
                role=MessageRole.USER,
                content=f"Extract the key metrics from this data and present them as a bullet list:\n{initial_data}"
            )
            
            run1 = self.agent_client.runs.create_and_process(
                thread_id=thread1.id,
                agent_id=analyzer_agent.id
            )
            
            analyzer_output = ""
            if run1.status == "completed":
                messages = self.agent_client.messages.list(
                    thread_id=thread1.id,
                    order=ListSortOrder.DESCENDING,
                    limit=1
                )
                
                for msg in messages:
                    if msg.role == MessageRole.AGENT:
                        if msg.text_messages:
                            analyzer_output = msg.text_messages[0].text.value
                            print(f"Analyzer Output:\n{analyzer_output}\n")
            
            # STEP 2: Validator Agent validates the extracted metrics
            print(f"{'─'*80}")
            print("STEP 2: Validator Agent - Check for Issues")
            print(f"{'─'*80}")
            
            thread2 = self.agent_client.threads.create()
            
            message2 = self.agent_client.messages.create(
                thread_id=thread2.id,
                role=MessageRole.USER,
                content=f"Review these extracted metrics and identify any potential issues or missing information:\n\n{analyzer_output}"
            )
            
            run2 = self.agent_client.runs.create_and_process(
                thread_id=thread2.id,
                agent_id=validator_agent.id
            )
            
            validator_output = ""
            if run2.status == "completed":
                messages = self.agent_client.messages.list(
                    thread_id=thread2.id,
                    order=ListSortOrder.DESCENDING,
                    limit=1
                )
                
                for msg in messages:
                    if msg.role == MessageRole.AGENT:
                        if msg.text_messages:
                            validator_output = msg.text_messages[0].text.value
                            print(f"Validator Output:\n{validator_output}\n")
            
            # STEP 3: Formatter Agent creates final report
            print(f"{'─'*80}")
            print("STEP 3: Formatter Agent - Create Executive Summary")
            print(f"{'─'*80}")
            
            thread3 = self.agent_client.threads.create()
            
            message3 = self.agent_client.messages.create(
                thread_id=thread3.id,
                role=MessageRole.USER,
                content=f"""Create a concise executive summary using these inputs:

METRICS:
{analyzer_output}

VALIDATION NOTES:
{validator_output}

Format as: Executive Summary with 3 key takeaways."""
            )
            
            run3 = self.agent_client.runs.create_and_process(
                thread_id=thread3.id,
                agent_id=formatter_agent.id
            )
            
            if run3.status == "completed":
                messages = self.agent_client.messages.list(
                    thread_id=thread3.id,
                    order=ListSortOrder.DESCENDING,
                    limit=1
                )
                
                for msg in messages:
                    if msg.role == MessageRole.AGENT:
                        if msg.text_messages:
                            final_output = msg.text_messages[0].text.value
                            print(f"Final Report:\n{final_output}\n")
            
            # Cleanup all agents
            print(f"{'='*80}")
            self.agent_client.delete_agent(analyzer_agent.id)
            self.agent_client.delete_agent(validator_agent.id)
            self.agent_client.delete_agent(formatter_agent.id)
            print("Multi-agent workflow completed. All agents deleted.")
            print(f"{'='*80}")
    
    def run_parallel_pattern(self):
        """
        Parallel execution pattern: Multiple agents run simultaneously on different aspects.
        Demonstrates concurrent agent execution (equivalent to LangChain's RunnableParallel).
        """
        
        with self.agent_client:
            topic = "The history of space exploration"
            
            print("\n" + "="*80)
            print("PARALLEL AGENT EXECUTION")
            print("="*80)
            print(f"\nTopic: {topic}\n")
            
            # Create three specialized agents
            print("Creating agents...")
            agents = {
                "summarizer": self.agent_client.create_agent(
                    model=self.model_deployment,
                    name="Summarizer",
                    instructions="Provide a concise 2-3 sentence summary."
                ),
                "questioner": self.agent_client.create_agent(
                    model=self.model_deployment,
                    name="Question Generator",
                    instructions="Generate exactly 3 interesting questions."
                ),
                "term_extractor": self.agent_client.create_agent(
                    model=self.model_deployment,
                    name="Term Extractor",
                    instructions="List 5-7 key terms, comma-separated."
                )
            }
            
            print(f"✓ Created {len(agents)} agents\n")
            
            # Define tasks for each agent
            tasks = [
                ("summarizer", f"Summarize concisely: {topic}"),
                ("questioner", f"Generate 3 questions about: {topic}"),
                ("term_extractor", f"Extract key terms from: {topic}")
            ]
            
            print(f"{'─'*80}")
            print("⚡ Running 3 agents in parallel...")
            print(f"{'─'*80}\n")
            
            # Execute all agents in parallel
            def execute_agent_task(agent_key, prompt):
                """Helper function to run a single agent task"""
                thread = self.agent_client.threads.create()
                self.agent_client.messages.create(
                    thread_id=thread.id,
                    role=MessageRole.USER,
                    content=prompt
                )
                run = self.agent_client.runs.create_and_process(
                    thread_id=thread.id,
                    agent_id=agents[agent_key].id
                )
                
                # Get response
                if run.status == "completed":
                    messages = self.agent_client.messages.list(
                        thread_id=thread.id,
                        order=ListSortOrder.DESCENDING,
                        limit=1
                    )
                    for msg in messages:
                        if msg.role == MessageRole.AGENT and msg.text_messages:
                            return (agent_key, msg.text_messages[0].text.value)
                return (agent_key, "No response")
            
            # Run tasks concurrently
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                # Submit all three tasks to run in parallel
                # Each submit() starts the task immediately in a separate thread
                futures = []
                for agent_key, prompt in tasks:
                    # Start this agent task in the background
                    future = executor.submit(execute_agent_task, agent_key, prompt)
                    futures.append(future)
                
                # Collect results as agents finish (in any order)
                results = {}
                for future in concurrent.futures.as_completed(futures):
                    agent_key, output = future.result()
                    results[agent_key] = output
                    print(f"✓ {agents[agent_key].name} completed")
            
            # Display results
            print(f"\n{'─'*80}")
            print("RESULTS FROM PARALLEL EXECUTION")
            print(f"{'─'*80}\n")
            
            print(f"📝 Summary:\n{results['summarizer']}\n")
            print(f"❓ Questions:\n{results['questioner']}\n")
            print(f"🏷️  Key Terms:\n{results['term_extractor']}\n")
            
            # Cleanup
            print(f"{'='*80}")
            for agent in agents.values():
                self.agent_client.delete_agent(agent.id)
            print(f"Parallel execution completed. All agents deleted.")
            print(f"{'='*80}")
    
    def run_routing_pattern(self):
        """
        Routing pattern: Router LLM classifies intent and routes to specialized agents or functions.
        """
        
        with self.agent_client:
            # Create router and specialized agents
            print("Creating agents...")
            router = self.agent_client.create_agent(
                model=self.model_deployment,
                name="Router Agent",
                instructions="Classify queries into routes: 1=Sales, 2=Technical, 3=General, 4=DateTime. Reply with only the number."
            )
            
            agents = {
                "1": self.agent_client.create_agent(model=self.model_deployment, name="Sales Agent", 
                                              instructions="Analyze sales data and provide business insights."),
                "2": self.agent_client.create_agent(model=self.model_deployment, name="Tech Agent", 
                                              instructions="Help troubleshoot technical problems."),
                "3": self.agent_client.create_agent(model=self.model_deployment, name="General Agent", 
                                              instructions="Answer general questions clearly.")
            }
            
            print("\n" + "="*80)
            print("ROUTING PATTERN DEMO")
            print("="*80)
            
            queries = [
                "What were our Q4 sales numbers?",
                "I'm getting a 500 error deploying my function",
                "Explain machine learning",
                "What is the current time?"
            ]
            
            for i, query in enumerate(queries, 1):
                print(f"\n{'─'*80}\nQUERY {i}: {query}\n{'─'*80}")
                
                # Route the query
                thread = self.agent_client.threads.create()
                self.agent_client.messages.create(thread_id=thread.id, role=MessageRole.USER, 
                                            content=f"Classify: {query}")
                run = self.agent_client.runs.create_and_process(thread_id=thread.id, agent_id=router.id)
                
                route = ""
                if run.status == "completed":
                    msgs = self.agent_client.messages.list(thread_id=thread.id, order=ListSortOrder.DESCENDING, limit=1)
                    for msg in msgs:
                        if msg.role == MessageRole.AGENT and msg.text_messages:
                            route = msg.text_messages[0].text.value.strip()
                            break
                
                print(f"🔀 Route: {route}")
                
                # Execute route
                if route in agents:
                    print(f"📍 Agent: {agents[route].name}")
                    thread = self.agent_client.threads.create()
                    self.agent_client.messages.create(thread_id=thread.id, role=MessageRole.USER, content=query)
                    run = self.agent_client.runs.create_and_process(thread_id=thread.id, agent_id=agents[route].id)
                    
                    if run.status == "completed":
                        msgs = self.agent_client.messages.list(thread_id=thread.id, order=ListSortOrder.DESCENDING, limit=1)
                        for msg in msgs:
                            if msg.role == MessageRole.AGENT and msg.text_messages:
                                print(f"💬 {msg.text_messages[0].text.value}\n")
                                break
                elif route == "4":
                    print("⚡ Function Call")
                    print(f"💬 Current time: {self.get_current_datetime()}\n")
                else:
                    print(f"❌ Unknown route\n")
            
            # Cleanup
            print(f"{'='*80}")
            self.agent_client.delete_agent(router.id)
            for agent in agents.values():
                self.agent_client.delete_agent(agent.id)
            print("Routing demo completed.")
            print(f"{'='*80}")
    
    def _run_agent(self, agent_id, thread_id, prompt):
        """Helper: Send prompt to agent and get response"""
        self.agent_client.messages.create(
            thread_id=thread_id,
            role=MessageRole.USER,
            content=prompt
        )
        run = self.agent_client.runs.create_and_process(thread_id=thread_id, agent_id=agent_id)
        
        if run.status == "completed":
            messages = self.agent_client.messages.list(thread_id=thread_id, order=ListSortOrder.DESCENDING, limit=1)
            for msg in messages:
                if msg.role == MessageRole.AGENT and msg.text_messages:
                    return msg.text_messages[0].text.value
        return None
    
    def run_reflection_pattern(self):
        """
        Reflection Pattern: Generate code → Get critique → Refine → Repeat
        
        How it works:
        1. Generator Agent writes code
        2. Critic Agent reviews it
        3. Generator Agent improves based on feedback
        4. Loop until code is perfect or max iterations reached
        """
        
        with self.agent_client:
            # Create two agents: one generates, one critiques
            generator = self.agent_client.create_agent(
                model=self.model_deployment,
                name="Generator",
                instructions="Generate Python code. When given critiques, improve the code. Output only code."
            )
            
            critic = self.agent_client.create_agent(
                model=self.model_deployment,
                name="Critic",
                instructions="Review code for bugs, edge cases, and best practices. If perfect, say 'CODE_IS_PERFECT'. Otherwise, list issues."
            )
            
            print("\n" + "="*70)
            print("REFLECTION PATTERN: Generate → Critique → Refine")
            print("="*70)
            
            # The task
            task = """Create a Python function `calculate_factorial` that:
- Takes integer n as input
- Returns n! (factorial)
- Handles n=0 (returns 1)
- Raises ValueError for negative numbers
- Has a clear docstring"""
            
            print(f"\n📋 Task: {task}\n")
            
            # State
            code = ""
            generator_thread = self.agent_client.threads.create()
            max_rounds = 3
            
            # Reflection loop
            for round_num in range(1, max_rounds + 1):
                print(f"\n{'='*70}")
                print(f"Round {round_num}/{max_rounds}")
                print(f"{'='*70}")
                
                # Step 1: Generate or refine code
                if round_num == 1:
                    prompt = task
                else:
                    prompt = f"Previous code:\n{code}\n\nFeedback:\n{feedback}\n\nPlease improve the code."
                
                print(f"\n🔧 Generator: {'Creating' if round_num == 1 else 'Refining'} code...")
                code = self._run_agent(generator.id, generator_thread.id, prompt)
                
                if not code:
                    print("❌ Failed to generate code")
                    break
                
                print(f"\n{code}\n")
                
                # Step 2: Get critique
                print("🔍 Critic: Reviewing code...")
                critic_thread = self.agent_client.threads.create()
                feedback = self._run_agent(
                    critic.id, 
                    # critic_thread.id, 
                    generator_thread.id,
                    f"Task:\n{task}\n\nCode:\n{code}\n\nProvide critique."
                )
                
                if not feedback:
                    print("❌ Failed to get critique")
                    break
                
                # Step 3: Check if done
                if "CODE_IS_PERFECT" in feedback:
                    print("✅ Critic: Code is perfect!")
                    break
                else:
                    print(f"💭 Critic feedback:\n{feedback}\n")
            
            # Show final result
            print("\n" + "="*70)
            print("FINAL CODE")
            print("="*70)
            print(f"\n{code}\n")
            
            # # Cleanup
            # self.agent_client.delete_agent(generator.id)
            # self.agent_client.delete_agent(critic.id)
            # print("="*70)
    
    def run_tool_calling_pattern(self):
        """
        Tool Calling Pattern: Agent autonomously calls external functions to retrieve data.
        
        This pattern demonstrates function calling (also called tool use), where:
        1. You define functions the agent can call
        2. The agent decides WHEN and HOW to call them based on user queries
        3. Your code executes the actual function
        4. The agent incorporates results into its final response
        
        This is equivalent to CrewAI's @tool decorator or LangChain's Tool abstraction.
        """
        import json
        import time
        
        with self.agent_client:
            print("\n" + "="*80)
            print("TOOL CALLING PATTERN")
            print("="*80)
            
            # STEP 1: Define the function schema (OpenAI function calling format)
            # This tells the agent WHAT the function does and WHAT parameters it needs
            # The agent uses this to decide when to call the function
            function_def = {
                "type": "function",
                "function": {
                    "name": "get_stock_price",  # Function name (must match your actual Python function)
                    "description": "Get simulated stock price for a ticker (AAPL, GOOGL, MSFT)",  # Clear description helps agent decide when to use it
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "ticker": {
                                "type": "string", 
                                "description": "Stock ticker symbol"  # Describes what this parameter is
                            }
                        },
                        "required": ["ticker"]  # Which parameters are mandatory
                    }
                }
            }
            
            # STEP 2: Create agent with tool access
            # The tools=[function_def] parameter gives the agent access to call this function
            agent = self.agent_client.create_agent(
                model=self.model_deployment,
                name="Financial Analyst",
                instructions="Use get_stock_price tool to look up stock prices. Report exact prices.",
                tools=[function_def]  # This is what enables function calling
            )
            print(f"✓ Agent created with tool access\n")
            
            # STEP 3: Send user query
            query = "What is the current price for Apple stock (AAPL)?"
            print(f"Query: {query}\n")
            
            thread = self.agent_client.threads.create()
            self.agent_client.messages.create(thread_id=thread.id, role=MessageRole.USER, content=query)
            
            # Start the agent run
            run = self.agent_client.runs.create(thread_id=thread.id, agent_id=agent.id)
            
            # STEP 4: Handle function calling loop
            # The agent's run goes through states:
            # - "queued": Waiting to start
            # - "in_progress": Agent is thinking
            # - "requires_action": Agent wants to call a function (YOUR CODE NEEDS TO EXECUTE IT)
            # - "completed": Agent has final answer
            while run.status in ["queued", "in_progress", "requires_action"]:
                time.sleep(0.5)  # Polling interval
                run = self.agent_client.runs.get(thread_id=thread.id, run_id=run.id)
                
                # When status is "requires_action", the agent wants to call a function
                if run.status == "requires_action":
                    # Extract the function calls the agent wants to make
                    tool_calls = run.required_action.submit_tool_outputs.tool_calls
                    tool_outputs = []
                    
                    # Execute each function call
                    for tool_call in tool_calls:
                        func_name = tool_call.function.name
                        args = json.loads(tool_call.function.arguments)  # Agent provides arguments as JSON
                        print(f"🔧 Agent calling: {func_name}({args})")
                        
                        # STEP 5: Execute the actual Python function
                        # This is where YOUR code runs - the agent can't execute functions itself
                        if func_name == "get_stock_price":
                            result = self.get_stock_price(**args)  # Call our Python function
                            print(f"📊 Result: {result}\n")
                            
                            # Package the result to send back to the agent
                            tool_outputs.append({
                                "tool_call_id": tool_call.id,  # Match this to the agent's request
                                "output": json.dumps(result)   # Agent expects JSON string
                            })
                    
                    # STEP 6: Submit function results back to the agent
                    # The agent will use these results to formulate its final response
                    run = self.agent_client.runs.submit_tool_outputs(
                        thread_id=thread.id, 
                        run_id=run.id, 
                        tool_outputs=tool_outputs
                    )
                    # Loop continues - agent may call more functions or complete
            
            # STEP 7: Get the agent's final response (after using tool results)
            if run.status == "completed":
                messages = self.agent_client.messages.list(
                    thread_id=thread.id, 
                    order=ListSortOrder.DESCENDING, 
                    limit=1
                )
                for msg in messages:
                    if msg.role == MessageRole.AGENT and msg.text_messages:
                        print(f"💬 Agent: {msg.text_messages[0].text.value}\n")
            
            print("="*80)
            self.agent_client.delete_agent(agent.id)
            print("Tool calling pattern completed.")
            print("="*80)
    
    def run_planning_pattern(self):
        """Planning Pattern: Break down complex task → Get approval → Execute subtasks."""
        
        with self.agent_client:
            print("\n" + "="*80)
            print("PLANNING PATTERN")
            print("="*80)
            
            # Get query
            query = input("\nEnter query (or Enter for default): ").strip()
            if not query:
                query = "What are the new research trends in Natural Language Processing?"
            print(f"\nQuery: {query}\n")
            
            # Step 1: Create plan
            planner = self.agent_client.create_agent(
                model=self.model_deployment,
                name="Planner",
                instructions="Break queries into 3-4 numbered subtasks. Format: 1. [task]\n2. [task]..."
            )
            
            thread = self.agent_client.threads.create()
            self.agent_client.messages.create(thread_id=thread.id, role=MessageRole.USER,
                                             content=f"Break this into subtasks: {query}")
            run = self.agent_client.runs.create_and_process(thread_id=thread.id, agent_id=planner.id)
            
            plan = ""
            if run.status == "completed":
                msgs = self.agent_client.messages.list(thread_id=thread.id, order=ListSortOrder.DESCENDING, limit=1)
                for msg in msgs:
                    if msg.role == MessageRole.AGENT and msg.text_messages:
                        plan = msg.text_messages[0].text.value
                        break
            
            print(f"📋 Plan:\n{plan}\n")
            
            # Step 2: Get approval
            if input("Approve? (y/n): ").lower() != 'y':
                print("❌ Cancelled")
                self.agent_client.delete_agent(planner.id)
                return
            
            # Step 3: Execute
            print("\n✅ Executing...\n")
            executor = self.agent_client.create_agent(
                model=self.model_deployment,
                name="Executor",
                instructions="Provide brief, factual answers."
            )
            
            subtasks = [line.strip() for line in plan.split('\n') 
                       if line.strip() and line.strip()[0].isdigit()]
            
            for i, task in enumerate(subtasks, 1):
                print(f"Task {i}: {task}")
                t = self.agent_client.threads.create()
                self.agent_client.messages.create(thread_id=t.id, role=MessageRole.USER, content=task)
                r = self.agent_client.runs.create_and_process(thread_id=t.id, agent_id=executor.id)
                
                if r.status == "completed":
                    msgs = self.agent_client.messages.list(thread_id=t.id, order=ListSortOrder.DESCENDING, limit=1)
                    for msg in msgs:
                        if msg.role == MessageRole.AGENT and msg.text_messages:
                            print(f"→ {msg.text_messages[0].text.value}\n")
                            break
            
            print("="*80)
            # self.agent_client.delete_agent(planner.id)
            # self.agent_client.delete_agent(executor.id)
            print("Planning completed.")
            print("="*80)
    
    def run_git_workflow_pattern(self):
        """
        Multi-Agent Git Workflow Pattern - Demonstrates Microsoft's Agentic Workflow
        
        This pattern showcases two specialized agents collaborating:
        1. Scanner Agent: Detects git changes and gathers repository information
        2. Committer Agent: Generates commit messages and pushes to remote
        
        Key Pattern Elements:
        - Agent Specialization: Each agent has a specific role and tool
        - Information Handoff: Agent 1's output becomes Agent 2's input
        - User Approval Gate: Human-in-the-loop before executing changes
        - Tool Calling: Agents autonomously decide when to use their tools
        """
        import json
        import time
        
        with self.agent_client:
            print("\n" + "="*60)
            print("MULTI-AGENT GIT WORKFLOW")
            print("="*60)
            
            # Get directory from user input
            directory = input("\nDirectory (Enter for current): ").strip() or os.path.dirname(os.path.abspath(__file__))
            
            # ==================== STEP 1: Define Agent Tools ====================
            # Each agent gets its own specialized tool to perform its task
            # This follows the principle of separation of concerns
            tools = {
                "scan": {
                    "type": "function",
                    "function": {
                        "name": "git_scan_status",
                        "description": "Scan git repo for changes",
                        "parameters": {
                            "type": "object",
                            "properties": {"directory": {"type": "string"}},
                            "required": ["directory"]
                        }
                    }
                },
                "commit": {
                    "type": "function",
                    "function": {
                        "name": "git_commit_and_push",
                        "description": "Commit and push changes",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "directory": {"type": "string"},
                                "commit_message": {"type": "string"}
                            },
                            "required": ["directory", "commit_message"]
                        }
                    }
                }
            }
            
            # ==================== AGENT 1: Scanner ====================
            # Purpose: Detect uncommitted changes and gather repo metadata
            # Tool: git_scan_status (read-only operation)
            print("\n[Agent 1: Scanner] Checking for changes...")
            scanner = self.agent_client.create_agent(
                model=self.model_deployment,
                name="Scanner",
                instructions="Scan git repo and report changes using git_scan_status tool.",
                tools=[tools["scan"]]  # Only has access to scanning tool
            )
            
            # Create isolated thread for Agent 1
            thread1 = self.agent_client.threads.create()
            self.agent_client.messages.create(thread_id=thread1.id, role=MessageRole.USER, content=f"Scan: {directory}")
            run1 = self.agent_client.runs.create(thread_id=thread1.id, agent_id=scanner.id)
            
            # Tool Calling Loop: Agent decides when to call git_scan_status
            # The agent interprets the user message and autonomously invokes the tool
            scan_result = None
            while run1.status in ["queued", "in_progress", "requires_action"]:
                time.sleep(0.5)
                run1 = self.agent_client.runs.get(thread_id=thread1.id, run_id=run1.id)
                
                # When status = "requires_action", agent wants to call the tool
                if run1.status == "requires_action":
                    for tool_call in run1.required_action.submit_tool_outputs.tool_calls:
                        args = json.loads(tool_call.function.arguments)
                        # Execute the Python function (agent can't execute code itself)
                        scan_result = self.git_scan_status(**args)
                        # Return results to agent so it can formulate a response
                        run1 = self.agent_client.runs.submit_tool_outputs(
                            thread_id=thread1.id, 
                            run_id=run1.id,
                            tool_outputs=[{"tool_call_id": tool_call.id, "output": json.dumps(scan_result)}]
                        )
            
            # Early exit if no changes detected
            if not scan_result or not scan_result.get("has_changes"):
                print("✓ No changes to commit")
                self.agent_client.delete_agent(scanner.id)
                return
            
            # Display scan results to user
            print(f"\n✓ Found {len(scan_result['changes'])} change(s) on branch: {scan_result['branch']}")
            print(f"   Remote: {scan_result.get('remote_url', 'N/A')}")
            print("\nChanged files:")
            for change in scan_result['changes'][:10]:
                print(f"   {change}")
            
            # ==================== HUMAN-IN-THE-LOOP APPROVAL ====================
            # User approval gate before making any changes to git repository
            # This demonstrates the importance of human oversight in agentic workflows
            proceed = input("\n⚠️  Commit and push these changes? (y/n): ").strip().lower()
            if proceed != 'y':
                print("❌ Operation cancelled by user")
                self.agent_client.delete_agent(scanner.id)
                return
            
            # ==================== AGENT 2: Committer ====================
            # Purpose: Generate semantic commit message and push changes
            # Tool: git_commit_and_push (write operation)
            # This agent receives the scan results from Agent 1 as context
            print("\n[Agent 2: Committer] Creating commit...")
            committer = self.agent_client.create_agent(
                model=self.model_deployment,
                name="Committer",
                instructions="Generate commit message and use git_commit_and_push tool.",
                tools=[tools["commit"]]  # Only has access to commit/push tool
            )
            
            # ==================== INFORMATION HANDOFF ====================
            # Agent 1's output (scan_result) is passed to Agent 2 as context
            # This demonstrates inter-agent communication in multi-agent workflows
            thread2 = self.agent_client.threads.create()
            self.agent_client.messages.create(
                thread_id=thread2.id, 
                role=MessageRole.USER,
                content=f"Commit these changes to {directory}:\n{chr(10).join(scan_result['changes'][:5])}"
            )
            run2 = self.agent_client.runs.create(thread_id=thread2.id, agent_id=committer.id)
            
            # Tool Calling Loop: Agent 2 generates commit message and calls git_commit_and_push
            while run2.status in ["queued", "in_progress", "requires_action"]:
                time.sleep(0.5)
                run2 = self.agent_client.runs.get(thread_id=thread2.id, run_id=run2.id)
                
                if run2.status == "requires_action":
                    for tool_call in run2.required_action.submit_tool_outputs.tool_calls:
                        args = json.loads(tool_call.function.arguments)
                        # Agent 2 autonomously generates a descriptive commit message
                        print(f"  Commit: {args['commit_message']}")
                        # Execute the commit and push operation
                        result = self.git_commit_and_push(**args)
                        # Return execution result to agent
                        run2 = self.agent_client.runs.submit_tool_outputs(
                            thread_id=thread2.id, 
                            run_id=run2.id,
                            tool_outputs=[{"tool_call_id": tool_call.id, "output": json.dumps(result)}]
                        )
            
            # ==================== WORKFLOW COMPLETE ====================
            print("✅ Pushed to remote")
            # Clean up both agents
            # self.agent_client.delete_agent(scanner.id)
            # self.agent_client.delete_agent(committer.id)
            print("="*60)
            
            # Key Takeaways:
            # 1. Agent Specialization: Each agent has one clear responsibility
            # 2. Tool Access Control: Agents only have tools they need (security)
            # 3. Information Flow: Output of Agent 1 → Input to Agent 2
            # 4. Human Oversight: User approval before destructive operations
            # 5. Autonomous Decision Making: Agents decide when to call tools
    
    def run_interactive_menu(self):
        """Display menu and run selected pattern"""
        print("\n🤖 Azure AI Agent Design Patterns")
        print("\nAvailable patterns:")
        print("1. Sequential Prompt Chain")
        print("2. Multi-Agent Workflow")
        print("3. Routing Pattern")
        print("4. Parallel Execution")
        print("5. Reflection Pattern")
        print("6. Tool Calling Pattern")
        print("7. Planning Pattern")
        print("8. Multi-Agent Git Workflow")
        
        choice = input("\nWhich pattern do you want to run? (1-8): ").strip()
        
        if choice == '1':
            self.run_prompt_chain()
        elif choice == '2':
            self.run_conditional_prompt_chain()
        elif choice == '3':
            self.run_routing_pattern()
        elif choice == '4':
            self.run_parallel_pattern()
        elif choice == '5':
            self.run_reflection_pattern()
        elif choice == '6':
            self.run_tool_calling_pattern()
        elif choice == '7':
            self.run_planning_pattern()
        elif choice == '8':
            self.run_git_workflow_pattern()
        else:
            print("Invalid choice. Please run the script again and select 1-8.")


# Main entry point
if __name__ == "__main__":
    # Initialize the patterns class
    patterns = AzureAIAgentPatterns()
    
    # Run interactive menu
    patterns.run_interactive_menu()
