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
        self.project_endpoint = project_endpoint or os.getenv("PROJECT_ENDPOINT")
        self.model_deployment = model_deployment or os.getenv("MODEL_DEPLOYMENT_NAME")
        
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
    """
    Sequential prompt chain: Output from one step becomes input to the next step.
    Single agent processing its own outputs in a pipeline.
    """
    
    with agent_client:
        # Create an agent
        print("Creating agent...")
        agent = agent_client.create_agent(
            model=model_deployment,
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
            thread = agent_client.threads.create()
            
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
            message = agent_client.messages.create(
                thread_id=thread.id,
                role=MessageRole.USER,
                content=current_prompt
            )
            
            # Execute
            run = agent_client.runs.create_and_process(
                thread_id=thread.id,
                agent_id=agent.id
            )
            
            if run.status == "completed":
                # Get the response
                messages = agent_client.messages.list(
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
        agent_client.delete_agent(agent.id)
        print(f"Pipeline completed. Agent deleted.")
        print(f"{'='*80}")


# Alternative: Multi-agent sequential workflow
def run_conditional_prompt_chain():
    """
    Multi-agent workflow: Output from one agent becomes input to another agent.
    Each agent is specialized for a specific task.
    """
    
    with agent_client:
        # Create specialized agents
        print("Creating specialized agents...")
        
        analyzer_agent = agent_client.create_agent(
            model=model_deployment,
            name="Data Analyzer Agent",
            instructions="You analyze data and extract insights. Be concise and structured."
        )
        
        validator_agent = agent_client.create_agent(
            model=model_deployment,
            name="Validator Agent",
            instructions="You validate and check for errors or inconsistencies. Provide clear feedback."
        )
        
        formatter_agent = agent_client.create_agent(
            model=model_deployment,
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
        
        thread1 = agent_client.threads.create()
        initial_data = """
        Sales Report Q4 2025:
        - Revenue: $2.5M (up 15% from Q3)
        - Customer Acquisition: 1,200 new customers
        - Churn Rate: 8%
        - Top Product: Cloud Storage (40% of revenue)
        """
        
        message1 = agent_client.messages.create(
            thread_id=thread1.id,
            role=MessageRole.USER,
            content=f"Extract the key metrics from this data and present them as a bullet list:\n{initial_data}"
        )
        
        run1 = agent_client.runs.create_and_process(
            thread_id=thread1.id,
            agent_id=analyzer_agent.id
        )
        
        analyzer_output = ""
        if run1.status == "completed":
            messages = agent_client.messages.list(
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
        
        thread2 = agent_client.threads.create()
        
        message2 = agent_client.messages.create(
            thread_id=thread2.id,
            role=MessageRole.USER,
            content=f"Review these extracted metrics and identify any potential issues or missing information:\n\n{analyzer_output}"
        )
        
        run2 = agent_client.runs.create_and_process(
            thread_id=thread2.id,
            agent_id=validator_agent.id
        )
        
        validator_output = ""
        if run2.status == "completed":
            messages = agent_client.messages.list(
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
        
        thread3 = agent_client.threads.create()
        
        message3 = agent_client.messages.create(
            thread_id=thread3.id,
            role=MessageRole.USER,
            content=f"""Create a concise executive summary using these inputs:

METRICS:
{analyzer_output}

VALIDATION NOTES:
{validator_output}

Format as: Executive Summary with 3 key takeaways."""
        )
        
        run3 = agent_client.runs.create_and_process(
            thread_id=thread3.id,
            agent_id=formatter_agent.id
        )
        
        if run3.status == "completed":
            messages = agent_client.messages.list(
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
        agent_client.delete_agent(analyzer_agent.id)
        agent_client.delete_agent(validator_agent.id)
        agent_client.delete_agent(formatter_agent.id)
        print("Multi-agent workflow completed. All agents deleted.")
        print(f"{'='*80}")


def get_current_datetime():
    """Simple function for Route 4 - No agent needed."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def run_parallel_pattern():
    """
    Parallel execution pattern: Multiple agents run simultaneously on different aspects.
    Demonstrates concurrent agent execution (equivalent to LangChain's RunnableParallel).
    """
    
    with agent_client:
        topic = "The history of space exploration"
        
        print("\n" + "="*80)
        print("PARALLEL AGENT EXECUTION")
        print("="*80)
        print(f"\nTopic: {topic}\n")
        
        # Create three specialized agents
        print("Creating agents...")
        agents = {
            "summarizer": agent_client.create_agent(
                model=model_deployment,
                name="Summarizer",
                instructions="Provide a concise 2-3 sentence summary."
            ),
            "questioner": agent_client.create_agent(
                model=model_deployment,
                name="Question Generator",
                instructions="Generate exactly 3 interesting questions."
            ),
            "term_extractor": agent_client.create_agent(
                model=model_deployment,
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
            thread = agent_client.threads.create()
            agent_client.messages.create(
                thread_id=thread.id,
                role=MessageRole.USER,
                content=prompt
            )
            run = agent_client.runs.create_and_process(
                thread_id=thread.id,
                agent_id=agents[agent_key].id
            )
            
            # Get response
            if run.status == "completed":
                messages = agent_client.messages.list(
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
            agent_client.delete_agent(agent.id)
        print(f"Parallel execution completed. All agents deleted.")
        print(f"{'='*80}")


def run_routing_pattern():
    """
    Routing pattern: Router LLM classifies intent and routes to specialized agents or functions.
    """
    
    with agent_client:
        # Create router and specialized agents
        print("Creating agents...")
        router = agent_client.create_agent(
            model=model_deployment,
            name="Router Agent",
            instructions="Classify queries into routes: 1=Sales, 2=Technical, 3=General, 4=DateTime. Reply with only the number."
        )
        
        agents = {
            "1": agent_client.create_agent(model=model_deployment, name="Sales Agent", 
                                          instructions="Analyze sales data and provide business insights."),
            "2": agent_client.create_agent(model=model_deployment, name="Tech Agent", 
                                          instructions="Help troubleshoot technical problems."),
            "3": agent_client.create_agent(model=model_deployment, name="General Agent", 
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
            thread = agent_client.threads.create()
            agent_client.messages.create(thread_id=thread.id, role=MessageRole.USER, 
                                        content=f"Classify: {query}")
            run = agent_client.runs.create_and_process(thread_id=thread.id, agent_id=router.id)
            
            route = ""
            if run.status == "completed":
                msgs = agent_client.messages.list(thread_id=thread.id, order=ListSortOrder.DESCENDING, limit=1)
                for msg in msgs:
                    if msg.role == MessageRole.AGENT and msg.text_messages:
                        route = msg.text_messages[0].text.value.strip()
                        break
            
            print(f"🔀 Route: {route}")
            
            # Execute route
            if route in agents:
                print(f"📍 Agent: {agents[route].name}")
                thread = agent_client.threads.create()
                agent_client.messages.create(thread_id=thread.id, role=MessageRole.USER, content=query)
                run = agent_client.runs.create_and_process(thread_id=thread.id, agent_id=agents[route].id)
                
                if run.status == "completed":
                    msgs = agent_client.messages.list(thread_id=thread.id, order=ListSortOrder.DESCENDING, limit=1)
                    for msg in msgs:
                        if msg.role == MessageRole.AGENT and msg.text_messages:
                            print(f"💬 {msg.text_messages[0].text.value}\n")
                            break
            elif route == "4":
                print("⚡ Function Call")
                print(f"💬 Current time: {get_current_datetime()}\n")
            else:
                print(f"❌ Unknown route\n")
        
        # Cleanup
        print(f"{'='*80}")
        agent_client.delete_agent(router.id)
        for agent in agents.values():
            agent_client.delete_agent(agent.id)
        print("Routing demo completed.")
        print(f"{'='*80}")


# Main entry point
if __name__ == "__main__":
    print("\n🤖 Azure AI Agent Design Patterns")
    print("\nAvailable patterns:")
    print("1. Sequential Prompt Chain")
    print("2. Multi-Agent Workflow")
    print("3. Routing Pattern")
    print("4. Parallel Execution")
    
    choice = input("\nWhich pattern do you want to run? (1-4): ").strip()
    
    if choice == '1':
        run_prompt_chain()
    elif choice == '2':
        run_conditional_prompt_chain()
    elif choice == '3':
        run_routing_pattern()
    elif choice == '4':
        run_parallel_pattern()
    else:
        print("Invalid choice. Please run the script again and select 1-4.")