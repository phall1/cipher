"""
Interactive CLI for the Cipher RAG chatbot.

This module provides a command-line interface for interacting with
the RAG-powered Cipher chatbot.
"""

import os
import sys
import argparse
import readline  # Enable arrow keys and command history
from pathlib import Path
from typing import Optional, List, Dict, Any

# Add project root to the path if needed
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.append(project_root)

from ingestion.chat.rag import RAGEngine


class Colors:
    """ANSI color codes for terminal output."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    CYAN = "\033[96m"
    MAGENTA = "\033[95m"


class CipherCLI:
    """
    Interactive CLI for the Cipher RAG chatbot.
    
    This class provides a user-friendly command-line interface for
    interacting with the RAG-powered chatbot.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "claude-3-5-sonnet-20240620",
        vector_index_dir: str = "./ingestion/vector/data",
        search_results_count: int = 5,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        verbose: bool = False,
    ):
        """
        Initialize the CLI.
        
        Args:
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            model_name: Claude model to use
            vector_index_dir: Directory containing the FAISS index
            search_results_count: Number of search results to retrieve
            max_tokens: Maximum tokens in the response
            temperature: Temperature for response generation
            verbose: Whether to show debug information like retrieved context
        """
        self.verbose = verbose
        
        # Initialize the RAG engine
        self.engine = RAGEngine(
            api_key=api_key,
            model_name=model_name,
            vector_index_dir=vector_index_dir,
            search_results_count=search_results_count,
            max_tokens=max_tokens,
            temperature=temperature,
            history_dir="./ingestion/chat/history",
        )
        
        # Store configuration for display
        self.config = {
            "model": model_name,
            "index_dir": vector_index_dir,
            "results_count": search_results_count,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        
        # Get available topics for display
        self.topics = self.engine.search.get_topics()
        self.reports = self.engine.search.get_reports()
    
    def print_welcome(self):
        """Print welcome message and configuration details."""
        print(f"\n{Colors.BOLD}{Colors.CYAN}╔════════════════════════════════════════════╗{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}║       THE CIPHER BRIEF INTELLIGENCE       ║{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}╚════════════════════════════════════════════╝{Colors.RESET}")
        print(f"\n{Colors.BOLD}Your elite AI analyst for geopolitical intelligence and national security insights.{Colors.RESET}")
        print(f"Type {Colors.YELLOW}/help{Colors.RESET} for available commands.")
        
        if self.verbose:
            print(f"\n{Colors.CYAN}Configuration:{Colors.RESET}")
            print(f"  • Model: {Colors.YELLOW}{self.config['model']}{Colors.RESET}")
            print(f"  • Vector index: {Colors.YELLOW}{self.config['index_dir']}{Colors.RESET}")
            print(f"  • Results count: {Colors.YELLOW}{self.config['results_count']}{Colors.RESET}")
            print(f"  • Available topics: {Colors.YELLOW}{', '.join(self.topics)}{Colors.RESET}")
            print(f"  • Available reports: {Colors.YELLOW}{', '.join(self.reports)}{Colors.RESET}")
    
    def print_help(self):
        """Print help information about available commands."""
        print(f"\n{Colors.BOLD}{Colors.CYAN}Available Commands:{Colors.RESET}")
        print(f"  {Colors.YELLOW}/help{Colors.RESET} - Show this help message")
        print(f"  {Colors.YELLOW}/exit{Colors.RESET} - Exit the chatbot")
        print(f"  {Colors.YELLOW}/verbose{Colors.RESET} - Toggle showing retrieved context")
        print(f"  {Colors.YELLOW}/clear{Colors.RESET} - Clear the screen")
        print(f"  {Colors.YELLOW}/topics{Colors.RESET} - Show available topics")
        print(f"  {Colors.YELLOW}/reports{Colors.RESET} - Show available reports")
        print(f"  {Colors.YELLOW}/topic <n>{Colors.RESET} - Filter next query by topic")
        print(f"  {Colors.YELLOW}/report <id>{Colors.RESET} - Filter next query by report")
        print(f"  {Colors.YELLOW}/history{Colors.RESET} - Show current chat history file location")
    
    def print_topics(self):
        """Print available topics."""
        print(f"\n{Colors.BOLD}{Colors.CYAN}Available Topics:{Colors.RESET}")
        for topic in sorted(self.topics):
            print(f"  • {Colors.YELLOW}{topic}{Colors.RESET}")
    
    def print_reports(self):
        """Print available reports."""
        print(f"\n{Colors.BOLD}{Colors.CYAN}Available Reports:{Colors.RESET}")
        for report in sorted(self.reports):
            print(f"  • {Colors.YELLOW}{report}{Colors.RESET}")
    
    def print_history_location(self):
        """Print the location of the current intelligence collection file."""
        print(f"\n{Colors.BOLD}{Colors.CYAN}Intelligence Collection:{Colors.RESET}")
        print(f"  Securely archived at: {Colors.YELLOW}{self.engine.history_path}{Colors.RESET}")
        print(f"  Classification: {Colors.YELLOW}UNCLASSIFIED // FOUO{Colors.RESET}")
    
    def print_context(self, results: List[Dict[str, Any]]):
        """
        Print the retrieved intelligence sources.
        
        Args:
            results: Retrieved intelligence sources
        """
        if not results:
            print(f"\n{Colors.YELLOW}No relevant intelligence found.{Colors.RESET}")
            return
        
        # Print header with classification banner
        print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}INTELLIGENCE SOURCES - UNCLASSIFIED // FOUO{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.RESET}")
        
        for i, result in enumerate(results):
            # Calculate confidence score based on distance
            confidence = 1.0 - (result.get('distance', 0.5) / 2)
            confidence_level = "HIGH" if confidence > 0.8 else "MEDIUM" if confidence > 0.6 else "LOW"
            
            # Format each result like an intelligence source with classification marking
            print(f"\n{Colors.BOLD}[SOURCE {i+1}] {Colors.YELLOW}{result['title']}{Colors.RESET}")
            print(f"{Colors.CYAN}Classification:{Colors.RESET} UNCLASSIFIED // FOUO")
            print(f"{Colors.CYAN}Intel Category:{Colors.RESET} {result['topic'].upper()}")
            print(f"{Colors.CYAN}Reference ID:{Colors.RESET} {result['report_id']}")
            print(f"{Colors.CYAN}Confidence:{Colors.RESET} {confidence_level} ({confidence:.2f})")
            
            # Print a snippet of the content with intelligence formatting
            content = result['content']
            if len(content) > 200:
                content = content[:200] + "..."
            print(f"{Colors.CYAN}Content:{Colors.RESET} {content}")
            print(f"{Colors.CYAN}{'-'*60}{Colors.RESET}")
    
    def process_command(self, command: str):
        """
        Process a command entered by the user.
        
        Args:
            command: User command
            
        Returns:
            Either a boolean indicating whether to continue, or
            a tuple of (continue, topic, report) for filter commands
        """
        if command == "/help":
            self.print_help()
            return True
        elif command == "/exit":
            return False
        elif command == "/verbose":
            self.verbose = not self.verbose
            print(f"\n{Colors.CYAN}Verbose mode: {Colors.YELLOW}{'on' if self.verbose else 'off'}{Colors.RESET}")
            return True
        elif command == "/clear":
            os.system('cls' if os.name == 'nt' else 'clear')
            self.print_welcome()
            return True
        elif command == "/topics":
            self.print_topics()
            return True
        elif command == "/reports":
            self.print_reports()
            return True
        elif command == "/history":
            self.print_history_location()
            return True
        elif command.startswith("/topic "):
            topic = command[7:].strip()
            if topic in self.topics:
                print(f"\n{Colors.CYAN}Next query will be filtered by topic: {Colors.YELLOW}{topic}{Colors.RESET}")
                return (True, topic, None)
            else:
                print(f"\n{Colors.RED}Topic not found. Use /topics to see available topics.{Colors.RESET}")
                return True
        elif command.startswith("/report "):
            report = command[8:].strip()
            if report in self.reports:
                print(f"\n{Colors.CYAN}Next query will be filtered by report: {Colors.YELLOW}{report}{Colors.RESET}")
                return (True, None, report)
            else:
                print(f"\n{Colors.RED}Report not found. Use /reports to see available reports.{Colors.RESET}")
                return True
        else:
            print(f"\n{Colors.RED}Unknown command. Type /help for available commands.{Colors.RESET}")
            return True
    
    def run_demo_mode(self):
        """Run the CLI in demo mode (non-interactive)."""
        self.print_welcome()
        
        # Print demo message
        print(f"\n{Colors.BOLD}{Colors.GREEN}Demo Mode:{Colors.RESET} Running a non-interactive intelligence briefing")
        print("\nThis application provides elite intelligence analysis powered by The Cipher Brief.")
        print("It uses advanced neural retrieval paired with Claude 3.5 for high-fidelity intelligence analysis.")
        print(f"Analysis history is encrypted and saved to: {self.engine.history_path}")
        
        # Run a single example query to demonstrate functionality
        demo_query = "What are the current geopolitical implications of Russia's actions in Ukraine?"
        print(f"\n{Colors.BOLD}{Colors.GREEN}Intelligence Query:{Colors.RESET} {demo_query}")
        
        try:
            # Process the demo query
            print(f"\n{Colors.BOLD}{Colors.MAGENTA}Cipher:{Colors.RESET} Thinking...")
            
            # Standard query
            response, results = self.engine.process_query(demo_query)
            
            # Print the response
            print(f"\n{Colors.BOLD}{Colors.MAGENTA}INTELLIGENCE RESPONSE:{Colors.RESET} {response}")
            
            # Show retrieved context
            print(f"\n{Colors.BOLD}{Colors.CYAN}Intelligence Sources:{Colors.RESET}")
            self.print_context(results)
            
        except Exception as e:
            print(f"\n{Colors.RED}Error: {str(e)}{Colors.RESET}")
            
            # Print fallback response
            print(f"\n{Colors.BOLD}{Colors.MAGENTA}INTELLIGENCE RESPONSE:{Colors.RESET} Analysis capabilities compromised. Unable to process intelligence query.")
            print(f"Secure connection to intelligence database required. Contact your security officer for credentials.")
        
        print(f"\n{Colors.BOLD}{Colors.GREEN}Intelligence briefing completed.{Colors.RESET}")
        print("Secure interactive analysis available with proper authentication. Use /help for available intelligence tools.")
    
    def run(self):
        """Run the interactive CLI loop."""
        self.print_welcome()
        
        # Show intelligence collection history location
        print(f"\n{Colors.CYAN}Intelligence collection saved to: {Colors.YELLOW}{self.engine.history_path}{Colors.RESET}")
        
        # Main conversation loop
        active_topic = None
        active_report = None
        
        while True:
            try:
                # Get user input
                if active_topic:
                    print(f"\n{Colors.CYAN}[Topic: {active_topic}]{Colors.RESET}")
                elif active_report:
                    print(f"\n{Colors.CYAN}[Report: {active_report}]{Colors.RESET}")
                    
                user_input = input(f"\n{Colors.BOLD}{Colors.GREEN}ANALYST QUERY:{Colors.RESET} ")
                
                # Handle empty input
                if not user_input.strip():
                    continue
                
                # Handle commands
                if user_input.startswith("/"):
                    if user_input == "/exit":
                        print("\n\nExiting... Goodbye!")
                        break
                        
                    result = self.process_command(user_input)
                    if isinstance(result, tuple):
                        continue_chat, new_topic, new_report = result
                        if new_topic:
                            active_topic = new_topic
                            active_report = None
                        elif new_report:
                            active_report = new_report
                            active_topic = None
                    else:
                        continue_chat = result
                        
                    if not continue_chat:
                        break
                    continue
                
                # Process query with any active filters
                print(f"\n{Colors.BOLD}{Colors.MAGENTA}INTELLIGENCE RESPONSE:{Colors.RESET}")
                print(f"{Colors.CYAN}Analyzing query...{Colors.RESET}")
                
                try:
                    # Check if this might be a general question first
                    is_general = self.engine.is_general_question(user_input)
                    
                    if is_general:
                        print(f"{Colors.CYAN}Detected general question. Processing directly...{Colors.RESET}")
                    
                    if active_topic and not is_general:
                        # Filter by topic
                        print(f"{Colors.CYAN}Searching for sources on topic '{active_topic}'...{Colors.RESET}")
                        results = self.engine.search.search(user_input, filter_topic=active_topic)
                        print(f"{Colors.CYAN}Found {len(results)} sources. Formatting context...{Colors.RESET}")
                        context = self.engine.format_context(results)
                        print(f"{Colors.CYAN}Generating intelligence response...{Colors.RESET}")
                        response = self.engine.generate_response(user_input, context)
                        
                        # Update history manually since we're not using process_query
                        self.engine.history.append({"role": "user", "content": user_input})
                        self.engine.history.append({"role": "assistant", "content": response})
                        self.engine.save_history()
                        
                        # Reset active topic after use
                        active_topic = None
                    elif active_report and not is_general:
                        # Filter by report
                        print(f"{Colors.CYAN}Searching within report '{active_report}'...{Colors.RESET}")
                        results = self.engine.search.search_by_report(user_input, report_id=active_report)
                        print(f"{Colors.CYAN}Found {len(results)} sources. Formatting context...{Colors.RESET}")
                        context = self.engine.format_context(results)
                        print(f"{Colors.CYAN}Generating intelligence response...{Colors.RESET}")
                        response = self.engine.generate_response(user_input, context)
                        
                        # Update history manually since we're not using process_query
                        self.engine.history.append({"role": "user", "content": user_input})
                        self.engine.history.append({"role": "assistant", "content": response})
                        self.engine.save_history()
                        
                        # Reset active report after use
                        active_report = None
                    else:
                        # For general questions or standard RAG queries
                        if not is_general:
                            print(f"{Colors.CYAN}Reformulating query for optimal retrieval...{Colors.RESET}")
                            print(f"{Colors.CYAN}Searching for relevant intelligence sources...{Colors.RESET}")
                            
                        # Perform the query processing with progress indicators
                        import threading
                        import time
                        
                        # Setup a progress indicator thread
                        stop_progress = threading.Event()
                        
                        def show_progress():
                            stages = ["Analyzing", "Retrieving", "Processing", "Synthesizing"]
                            stage_idx = 0
                            while not stop_progress.is_set():
                                print(f"{Colors.CYAN}{stages[stage_idx % len(stages)]} intelligence...{Colors.RESET}", end="\r")
                                stage_idx += 1
                                time.sleep(1.5)
                        
                        # Start progress thread if not in demo mode
                        if not self.engine.demo_mode:
                            progress_thread = threading.Thread(target=show_progress)
                            progress_thread.daemon = True
                            progress_thread.start()
                        
                        try:
                            # Process the query
                            response, results = self.engine.process_query(user_input)
                        finally:
                            # Stop the progress indicator
                            if not self.engine.demo_mode:
                                stop_progress.set()
                                progress_thread.join(timeout=0.5)
                                # Clear the progress line
                                print(" " * 50, end="\r")
                        
                        # Reset any active filters for general questions
                        if is_general:
                            active_topic = None
                            active_report = None
                    
                    # Print the response
                    print(f"\n{Colors.BOLD}{Colors.MAGENTA}INTELLIGENCE RESPONSE:{Colors.RESET} {response}")
                    
                    # Optionally show retrieved context
                    if self.verbose:
                        self.print_context(results)
                except Exception as e:
                    print(f"\n{Colors.RED}Error: {str(e)}{Colors.RESET}")
                    if self.verbose:
                        import traceback
                        traceback.print_exc()
                    
            except KeyboardInterrupt:
                # Handle Ctrl+C gracefully
                print("\n\nExiting... Goodbye!")
                break
            except EOFError:
                # Handle EOF gracefully (happens in some environments)
                print("\n\nEOF detected. Exiting... Goodbye!")
                break
            except Exception as e:
                print(f"\n{Colors.RED}Error: {str(e)}{Colors.RESET}")
                if self.verbose:
                    import traceback
                    traceback.print_exc()



def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(description="Cipher RAG Chatbot CLI")
    parser.add_argument("--api-key", help="Anthropic API key (defaults to ANTHROPIC_API_KEY env var)")
    parser.add_argument("--model", default="claude-3-5-sonnet-20240620", help="Claude model to use")
    parser.add_argument("--index-dir", default="./ingestion/vector/data", help="FAISS index directory")
    parser.add_argument("--results", type=int, default=5, help="Number of search results to retrieve")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Maximum tokens in response")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for response generation")
    parser.add_argument("--verbose", action="store_true", help="Show retrieved context")
    parser.add_argument("--demo", action="store_true", help="Run in demo mode (non-interactive)")
    args = parser.parse_args()
    
    # Initialize the CLI
    cli = CipherCLI(
        api_key=args.api_key,
        model_name=args.model,
        vector_index_dir=args.index_dir,
        search_results_count=args.results,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        verbose=args.verbose,
    )
    
    # Run in either demo or interactive mode
    if args.demo:
        cli.run_demo_mode()
    else:
        cli.run()


if __name__ == "__main__":
    main()