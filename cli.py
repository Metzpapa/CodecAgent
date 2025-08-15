# cli.py

import os
import uuid
import shutil
import logging
from pathlib import Path
from dotenv import load_dotenv

# Use a rich console for beautiful, color-coded output
from rich.console import Console
from rich.logging import RichHandler
from rich.prompt import Prompt

# Import the Brain components
from codec.agent import Agent
from codec.state import State
from codec.agent_logging import AgentContextLogger
from codec.tools.finish_job import JobFinishedException

# --- Configuration ---
load_dotenv()
SAMPLE_PROJECT_PATH = os.environ.get("SAMPLE_PROJECT_PATH")
JOBS_BASE_DIR = Path("codec_jobs")

# --- Setup Logging to Console ---
# This will make the AgentContextLogger print beautifully to the terminal
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, markup=True, show_path=False)]
)
log = logging.getLogger("rich")
console = Console()

# --- Main CLI Function ---
def run_cli():
    """
    Sets up a temporary environment and runs an interactive chat session
    with the Codec Agent.
    """
    if not SAMPLE_PROJECT_PATH or not Path(SAMPLE_PROJECT_PATH).is_dir():
        console.print(f"[bold red]Error:[/bold red] SAMPLE_PROJECT_PATH is not set in your .env file or the directory does not exist.")
        console.print(f"Please add `SAMPLE_PROJECT_PATH=/path/to/your/sample/assets` to your .env file.")
        return

    # 1. Setup the temporary job environment
    job_id = f"cli-session-{uuid.uuid4().hex[:8]}"
    job_dir = JOBS_BASE_DIR / job_id
    assets_dir = job_dir / "assets"
    output_dir = job_dir / "output"
    
    session_cleanup_path = job_dir

    try:
        console.print(f"[cyan]Setting up new session: {job_id}[/cyan]")
        shutil.copytree(SAMPLE_PROJECT_PATH, assets_dir)
        output_dir.mkdir(exist_ok=True)
        console.print(f"[green]Copied sample assets from '{SAMPLE_PROJECT_PATH}' to '{assets_dir}'[/green]")

        # 2. Initialize the Brain
        state = State(assets_directory=str(assets_dir))
        # The stream_logger directs the Agent's narrative log to our rich console.
        context_logger = AgentContextLogger(job_id=job_id, stream_logger=log)
        agent = Agent(state=state, context_logger=context_logger)

        # 3. The Interaction Loop
        console.print("\n[bold magenta]Welcome to the Codec Agent CLI.[/bold magenta]")
        console.print("Type your request to the agent. Type 'exit' or 'quit' to end the session.")
        
        while True:
            prompt = Prompt.ask("[bold yellow]You[/bold yellow]")
            if prompt.lower() in ['exit', 'quit']:
                break

            try:
                # This is the core call to the agent's brain for one turn.
                agent.step(prompt)

                # After the agent's turn, check for new multimodal content.
                if state.new_file_ids_for_model:
                    console.print(f"\n[bold cyan]System:[/bold cyan] [italic]Agent's turn complete. The model can now see {len(state.new_file_ids_for_model)} new file(s).[/italic]")
                    # In a more advanced version, you could add logic here to
                    # automatically open the generated images.
                    state.new_file_ids_for_model.clear() # Clear for the next turn

            except JobFinishedException as e:
                # This is the clean exit path when the agent calls 'finish_job'.
                console.print("\n[bold green]JOB FINISHED[/bold green]")
                console.print(f"Final Message: {e.result.get('message')}")
                if e.result.get('output_path'):
                    console.print(f"Output Path: {e.result.get('output_path')}")
                break
            except Exception as e:
                console.print(f"[bold red]An unexpected error occurred in the agent step:[/bold red]")
                log.exception(e) # Print full, rich-formatted traceback

    finally:
        # 4. Cleanup
        if 'session_cleanup_path' in locals() and session_cleanup_path.exists():
            console.print(f"\n[cyan]Cleaning up session directory: {session_cleanup_path}[/cyan]")
            shutil.rmtree(session_cleanup_path)
        console.print("[bold magenta]Session ended.[/bold magenta]")


if __name__ == "__main__":
    run_cli()