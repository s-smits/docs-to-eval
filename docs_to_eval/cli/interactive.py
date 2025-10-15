"""
Interactive session for docs-to-eval CLI
"""

import asyncio
import json
import uuid
from pathlib import Path
from datetime import datetime

from rich.console import Console
from rich.prompt import Prompt, IntPrompt, Confirm
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from docs_to_eval.utils.config import EvaluationType
from ..core.classification import EvaluationTypeClassifier
from ..llm.mock_interface import MockLLMInterface
from ..utils.config import create_default_config

console = Console()


class InteractiveSession:
    """Interactive CLI session for docs-to-eval"""
    
    def __init__(self):
        self.console = console
        self.corpus_text = ""
        self.config = create_default_config()
        self.classification = None
        self.run_id = str(uuid.uuid4())[:8]
    
    def run(self):
        """Run interactive session"""
        self.show_welcome()
        
        try:
            # Step 1: Get corpus
            if not self.get_corpus_input():
                return
            
            # Step 2: Configure evaluation
            self.configure_evaluation()
            
            # Step 3: Run evaluation
            if Confirm.ask("\n[green]Start evaluation?[/green]"):
                asyncio.run(self.run_evaluation())
            
            # Step 4: Show results
            self.show_completion()
            
        except KeyboardInterrupt:
            self.console.print("\n[yellow]Session interrupted by user.[/yellow]")
        except Exception as e:
            self.console.print(f"\n[red]Error: {str(e)}[/red]")
    
    def show_welcome(self):
        """Show welcome message"""
        welcome_text = """
[bold blue]🤖 Welcome to docs-to-eval Interactive Session[/bold blue]

This interactive session will guide you through:
1. 📝 Loading your text corpus
2. 🧠 Automatic evaluation type classification  
3. ⚙️  Configuration of evaluation parameters
4. 🧪 LLM evaluation and verification
5. 📊 Results analysis and reporting

Let's get started!
        """.strip()
        
        self.console.print(Panel(welcome_text, border_style="blue"))
    
    def get_corpus_input(self) -> bool:
        """Get corpus input from user"""
        self.console.print("\n[cyan]📝 Step 1: Provide Your Text Corpus[/cyan]")
        self.console.print("─" * 50)
        
        options = [
            "1. Enter text directly",
            "2. Load from file", 
            "3. Load from directory",
            "4. Use sample corpus"
        ]
        
        for option in options:
            self.console.print(f"  {option}")
        
        choice = Prompt.ask("\nChoose option", choices=["1", "2", "3", "4"], default="4")
        
        if choice == "1":
            return self.get_direct_text_input()
        elif choice == "2":
            return self.get_file_input()
        elif choice == "3":
            return self.get_directory_input()
        elif choice == "4":
            return self.get_sample_corpus()
        
        return False
    
    def get_direct_text_input(self) -> bool:
        """Get text input directly from user"""
        self.console.print("\n[yellow]Enter your text (press Ctrl+D when finished):[/yellow]")
        
        lines = []
        try:
            while True:
                try:
                    line = input()
                    lines.append(line)
                except EOFError:
                    break
        except KeyboardInterrupt:
            return False
        
        self.corpus_text = '\n'.join(lines)
        
        if len(self.corpus_text.strip()) < 10:
            self.console.print("[red]Text too short. Please provide more content.[/red]")
            return False
        
        self.show_corpus_stats()
        return True
    
    def get_file_input(self) -> bool:
        """Get corpus from file"""
        while True:
            file_path = Prompt.ask("Enter file path")
            
            try:
                path = Path(file_path)
                if not path.exists():
                    self.console.print(f"[red]File not found: {file_path}[/red]")
                    continue
                
                self.corpus_text = path.read_text(encoding='utf-8')
                break
                
            except Exception as e:
                self.console.print(f"[red]Error reading file: {e}[/red]")
                if not Confirm.ask("Try another file?"):
                    return False
        
        self.console.print("[green]✅ Successfully loaded file[/green]")
        self.show_corpus_stats()
        return True
    
    def get_directory_input(self) -> bool:
        """Get corpus from directory"""
        while True:
            dir_path = Prompt.ask("Enter directory path")
            
            try:
                path = Path(dir_path)
                if not path.exists():
                    self.console.print(f"[red]Directory not found: {dir_path}[/red]")
                    continue
                
                if not path.is_dir():
                    self.console.print(f"[red]Path is not a directory: {dir_path}[/red]")
                    continue
                
                # Load all text files
                texts = []
                file_count = 0
                
                for file_path in path.rglob("*.txt"):
                    try:
                        content = file_path.read_text(encoding='utf-8')
                        texts.append(f"=== {file_path.name} ===\n{content}")
                        file_count += 1
                    except Exception as e:
                        self.console.print(f"[yellow]Warning: Could not read {file_path.name}: {e}[/yellow]")
                
                if not texts:
                    self.console.print("[red]No readable text files found in directory[/red]")
                    continue
                
                self.corpus_text = '\n\n'.join(texts)
                self.console.print(f"[green]✅ Successfully loaded {file_count} files[/green]")
                break
                
            except Exception as e:
                self.console.print(f"[red]Error reading directory: {e}[/red]")
                if not Confirm.ask("Try another directory?"):
                    return False
        
        self.show_corpus_stats()
        return True
    
    def get_sample_corpus(self) -> bool:
        """Use sample corpus"""
        self.corpus_text = """
Deep learning is a subset of machine learning that uses neural networks with multiple layers 
to model and understand complex patterns in data. Unlike traditional machine learning algorithms 
that require manual feature engineering, deep learning models can automatically learn 
hierarchical representations of data.

The architecture typically consists of an input layer, multiple hidden layers, and an output 
layer. Each layer contains neurons (nodes) that apply mathematical transformations to the input 
data. The most common activation functions include ReLU (Rectified Linear Unit), sigmoid, and 
tanh functions.

Training deep neural networks requires large datasets and significant computational resources. 
The backpropagation algorithm is used to update the network weights by calculating gradients 
and minimizing the loss function. Common loss functions include mean squared error for 
regression tasks and cross-entropy for classification problems.

Convolutional Neural Networks (CNNs) are particularly effective for image recognition tasks, 
using convolutional layers to detect local features and pooling layers to reduce dimensionality. 
Recurrent Neural Networks (RNNs) and their variants like LSTM (Long Short-Term Memory) are 
designed for sequential data processing, making them suitable for natural language processing 
and time series analysis.

Recent advances include transformer architectures, which have revolutionized natural language 
processing through attention mechanisms. These models can process sequences in parallel, 
leading to significant improvements in training efficiency and performance on language tasks.
        """.strip()
        
        self.console.print("[green]✅ Using sample corpus (Deep Learning domain)[/green]")
        self.show_corpus_stats()
        return True
    
    def show_corpus_stats(self):
        """Display corpus statistics"""
        stats = Table(title="📊 Corpus Statistics")
        stats.add_column("Metric", style="cyan")
        stats.add_column("Value", style="magenta")
        
        stats.add_row("Characters", f"{len(self.corpus_text):,}")
        stats.add_row("Words", f"{len(self.corpus_text.split()):,}")
        stats.add_row("Lines", f"{len(self.corpus_text.splitlines()):,}")
        stats.add_row("Estimated reading time", f"{len(self.corpus_text.split()) // 200} minutes")
        
        self.console.print(stats)
    
    def configure_evaluation(self):
        """Configure evaluation parameters"""
        self.console.print("\n[cyan]🧠 Step 2: Automatic Classification[/cyan]")
        self.console.print("─" * 50)
        
        # Classify corpus
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            progress.add_task("Analyzing corpus content...", total=None)
            
            classifier = EvaluationTypeClassifier()
            self.classification = classifier.classify_corpus(self.corpus_text)
        
        # Show classification results
        self.show_classification_results()
        
        # Configuration
        self.console.print("\n[cyan]⚙️  Step 3: Configure Evaluation[/cyan]")
        self.console.print("─" * 50)
        
        # Number of questions
        default_questions = 20
        num_questions = IntPrompt.ask(
            "Number of questions to generate",
            default=default_questions,
            show_default=True
        )
        
        if not (1 <= num_questions <= 200):
            self.console.print("[yellow]Using default of 20 questions[/yellow]")
            num_questions = 20
        
        # Evaluation type override
        use_recommended = Confirm.ask(
            f"Use recommended evaluation type '{self.classification.primary_type}'?",
            default=True
        )
        
        eval_type = self.classification.primary_type
        if not use_recommended:
            eval_types = [eval_type.value for eval_type in EvaluationType]
            
            self.console.print("\nAvailable evaluation types:")
            for i, et in enumerate(eval_types, 1):
                self.console.print(f"  {i}. {et}")
            
            choice = IntPrompt.ask(
                "Choose evaluation type",
                default=1,
                show_choices=False
            )
            
            if 1 <= choice <= len(eval_types):
                eval_type = EvaluationType(eval_types[choice - 1])
        
        # Advanced options
        use_agentic = Confirm.ask("Use advanced agentic generation?", default=True)
        
        temperature = 0.0 if eval_type in [EvaluationType.MATHEMATICAL, EvaluationType.CODE_GENERATION] else 0.7
        if Confirm.ask("Customize LLM temperature?", default=False):
            temperature = float(Prompt.ask("Temperature (0.0-2.0)", default=str(temperature)))
            temperature = max(0.0, min(2.0, temperature))
        
        # Update configuration
        self.config.eval_type = eval_type
        self.config.generation.num_questions = num_questions
        self.config.generation.use_agentic = use_agentic
        self.config.llm.temperature = temperature
        
        # Show final configuration
        self.show_configuration_summary()
    
    def show_classification_results(self):
        """Display classification results"""
        panel_content = f"""
[cyan]Primary Type:[/cyan] {self.classification.primary_type}
[cyan]Secondary Types:[/cyan] {', '.join(self.classification.secondary_types)}
[cyan]Confidence:[/cyan] {self.classification.confidence:.2f}

[cyan]Analysis:[/cyan]
{self.classification.analysis}
        """.strip()
        
        self.console.print(Panel(panel_content, title="📋 Classification Results", border_style="blue"))
    
    def show_configuration_summary(self):
        """Show configuration summary"""
        config_table = Table(title="⚙️  Evaluation Configuration")
        config_table.add_column("Setting", style="cyan")
        config_table.add_column("Value", style="magenta")
        
        config_table.add_row("Evaluation Type", str(self.config.eval_type))
        config_table.add_row("Number of Questions", str(self.config.generation.num_questions))
        config_table.add_row("Agentic Generation", "Yes" if self.config.generation.use_agentic else "No")
        config_table.add_row("LLM Temperature", str(self.config.llm.temperature))
        config_table.add_row("Verification Method", self.classification.pipeline['verification'])
        
        self.console.print(config_table)
    
    async def run_evaluation(self):
        """Run the evaluation"""
        self.console.print("\n[cyan]🧪 Step 4: Running Evaluation[/cyan]")
        self.console.print("─" * 50)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=self.console
        ) as progress:
            
            # Generate questions
            task1 = progress.add_task("Generating questions...", total=self.config.generation.num_questions)
            
            questions = []
            for i in range(self.config.generation.num_questions):
                questions.append({
                    "question": f"Question {i+1} about the corpus content",
                    "answer": f"Answer {i+1}",
                    "eval_type": self.config.eval_type
                })
                progress.update(task1, completed=i+1)
                await asyncio.sleep(0.02)
            
            # LLM evaluation
            task2 = progress.add_task("Evaluating with Mock LLM...", total=len(questions))
            
            # Instantiate mock LLM (kept for potential future use)
            MockLLMInterface(temperature=self.config.llm.temperature)
            results = []
            
            for i, question in enumerate(questions):
                result = {
                    "question": question["question"],
                    "ground_truth": question["answer"],
                    "prediction": f"Mock LLM response {i+1}",
                    "score": 0.7,
                    "confidence": 0.8
                }
                results.append(result)
                progress.update(task2, completed=i+1)
                await asyncio.sleep(0.02)
            
            # Store results
            self.results = {
                "run_id": self.run_id,
                "config": self.config.dict(),
                "classification": self.classification.to_dict(),
                "aggregate_metrics": {
                    "mean_score": sum(r["score"] for r in results) / len(results),
                    "min_score": min(r["score"] for r in results),
                    "max_score": max(r["score"] for r in results),
                    "num_samples": len(results)
                },
                "individual_results": results,  # Show all results
                "completed_at": datetime.now().isoformat()
            }
    
    def show_completion(self):
        """Show completion summary"""
        self.console.print("\n[cyan]📊 Step 5: Results Summary[/cyan]")
        self.console.print("─" * 50)
        
        if not hasattr(self, 'results'):
            return
        
        metrics = self.results["aggregate_metrics"]
        
        # Results table
        results_table = Table(title="📊 Evaluation Results")
        results_table.add_column("Metric", style="cyan")
        results_table.add_column("Value", style="magenta")
        
        results_table.add_row("Mean Score", f"{metrics['mean_score']:.3f}")
        results_table.add_row("Score Range", f"{metrics['min_score']:.3f} - {metrics['max_score']:.3f}")
        results_table.add_row("Questions Evaluated", str(metrics['num_samples']))
        results_table.add_row("Run ID", self.run_id)
        
        self.console.print(results_table)
        
        # Performance assessment
        mean_score = metrics['mean_score']
        if mean_score >= 0.8:
            performance = "[green]Excellent[/green] 🎉"
        elif mean_score >= 0.6:
            performance = "[yellow]Good[/yellow] 👍"
        elif mean_score >= 0.4:
            performance = "[orange3]Fair[/orange3] 👌"
        else:
            performance = "[red]Needs Improvement[/red] 📈"
        
        self.console.print(f"\n[cyan]Overall Performance:[/cyan] {performance}")
        
        # Save option
        if Confirm.ask("\nSave results to file?", default=True):
            self.save_results()
        
        self.console.print("\n[green]🎉 Evaluation completed successfully![/green]")
        self.console.print("[blue]Thank you for using docs-to-eval![/blue]")
    
    def save_results(self):
        """Save results to file"""
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        filename = f"evaluation_results_{self.run_id}.json"
        filepath = output_dir / filename
        
        try:
            with open(filepath, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            
            self.console.print(f"[green]✅ Results saved to: {filepath}[/green]")
            
        except Exception as e:
            self.console.print(f"[red]❌ Error saving results: {e}[/red]")


if __name__ == "__main__":
    session = InteractiveSession()
    session.run()