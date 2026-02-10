import requests
import logging
import math
from typing import List, Dict, Optional, Tuple
import json

# Configuration
API_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "llama3.2:3b"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EvaluationCriteria:
    """Defines the evaluation prompts for LLM responses"""
    
    MAIN_PROMPT = """
    Evaluate this response comprehensively:
    
    Response to evaluate:
    {response}
    
    Original question:
    {question}
    
    Consider and score (0-10):
    1. Technical Accuracy: Are technical details correct and feasible?
    2. Relevance: Does it directly address the question?
    3. Completeness: Does it cover all necessary aspects?
    4. Practicality: Can this be realistically implemented?
    5. Safety: Are risks and precautions addressed?

    Provide your evaluation in JSON format:
    {{
        "scores": {{
            "technical_accuracy": score,
            "relevance": score,
            "completeness": score,
            "practicality": score,
            "safety": score
        }},
        "analysis": {{
            "strengths": ["list", "of", "strengths"],
            "weaknesses": ["list", "of", "weaknesses"],
            "suggestions": ["list", "of", "improvements"]
        }},
        "overall_explanation": "Brief explanation of scores"
    }}
    """

class MCTSNode:
    """Represents a node in the Monte Carlo Tree Search"""
    def __init__(self, text: str, parent: Optional['MCTSNode'] = None):
        self.text = text
        self.parent = parent
        self.children: List[MCTSNode] = []
        self.visits = 0
        self.total_reward = 0.0
        self.untried_actions: List[str] = []

    def add_child(self, text: str) -> 'MCTSNode':
        child = MCTSNode(text=text, parent=self)
        self.children.append(child)
        return child

    def ucb_score(self, exploration_constant: float = 1.414) -> float:
        if self.visits == 0:
            return float('inf')
        
        exploitation = self.total_reward / self.visits
        exploration = exploration_constant * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration

class LLMEvaluator:
    """Handles LLM-based evaluation of responses"""
    def __init__(self, api_url: str, model_name: str):
        self.api_url = api_url
        self.model_name = model_name
        self.evaluation_cache: Dict[str, Tuple[float, Dict]] = {}

    def _get_llm_response(self, prompt: str) -> Dict:
        """Get evaluation from LLM and parse JSON response"""
        try:
            payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 1000,
                "stream": False
            }
            
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status()
            content = response.json()['message']['content'].strip()
            return json.loads(content)
        except Exception as e:
            logging.error(f"Error in LLM evaluation: {e}")
            return {
                "scores": {
                    "technical_accuracy": 5,
                    "relevance": 5,
                    "completeness": 5,
                    "practicality": 5,
                    "safety": 5
                }
            }

    def evaluate_completion(self, response: str, original_question: str) -> Tuple[float, Dict]:
        """Evaluate a completion using the LLM"""
        cache_key = f"{response}::{original_question}"
        if cache_key in self.evaluation_cache:
            return self.evaluation_cache[cache_key]

        # Get evaluation scores
        prompt = EvaluationCriteria.MAIN_PROMPT.format(
            response=response,
            question=original_question
        )
        evaluation = self._get_llm_response(prompt)
        
        # Calculate weighted score
        weights = {
            'technical_accuracy': 0.25,
            'relevance': 0.20,
            'completeness': 0.20,
            'practicality': 0.20,
            'safety': 0.15
        }
        
        scores = evaluation.get('scores', {})
        weighted_score = sum(
            scores.get(metric, 5) * weight 
            for metric, weight in weights.items()
        ) / 10.0  # Normalize to 0-1 range
        
        result = (weighted_score, evaluation)
        self.evaluation_cache[cache_key] = result
        return result

class MCTS:
    """Monte Carlo Tree Search implementation"""
    def __init__(self, api_url: str, model_name: str):
        self.api_url = api_url
        self.model_name = model_name
        self.evaluator = LLMEvaluator(api_url, model_name)

    def get_llm_completion(self, prompt: str, max_tokens: int = 100) -> str:
        """Get a completion from the LLM"""
        try:
            payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": max_tokens,
                "stream": False
            }
            
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status()
            return response.json()['message']['content'].strip()
        except Exception as e:
            logging.error(f"Error getting LLM completion: {e}")
            return ""

    def select(self, node: MCTSNode) -> MCTSNode:
        """Select the most promising node"""
        while node.children and not node.untried_actions:
            node = max(node.children, key=lambda n: n.ucb_score())
        return node

    def expand(self, node: MCTSNode) -> MCTSNode:
        """Expand the selected node"""
        if not node.untried_actions:
            continuation = self.get_llm_completion(node.text)
            if continuation:
                node.untried_actions = [continuation]
        
        if node.untried_actions:
            action = node.untried_actions.pop()
            new_text = f"{node.text}\n{action}"
            return node.add_child(new_text)
        return node

    def simulate(self, node: MCTSNode, original_question: str) -> Tuple[float, Dict]:
        """Simulate and evaluate a completion"""
        score, details = self.evaluator.evaluate_completion(node.text, original_question)
        return score, details

    def backpropagate(self, node: MCTSNode, reward: float):
        """Update statistics for all nodes in the path"""
        while node:
            node.visits += 1
            node.total_reward += reward
            node = node.parent

    def search(self, question: str, num_iterations: int = 5) -> Tuple[str, Dict]:
        """Perform MCTS to find the best completion"""
        root = MCTSNode(text=question)
        best_score = float('-inf')
        best_response = None
        best_evaluation = None
        
        for i in range(num_iterations):
            logging.info(f"MCTS Iteration {i + 1}/{num_iterations}")
            
            # Selection
            selected_node = self.select(root)
            
            # Expansion
            if selected_node.visits > 0:
                selected_node = self.expand(selected_node)
            
            # Simulation
            score, evaluation = self.simulate(selected_node, question)
            
            # Update best result
            if score > best_score:
                best_score = score
                best_response = selected_node.text
                best_evaluation = evaluation
            
            # Backpropagation
            self.backpropagate(selected_node, score)
            
            logging.info(f"Current best score: {best_score:.3f}")
        
        return best_response, best_evaluation

def main():
    # Example usage
    question = "What's the best material I can commonly find around the house to make treads for a 12 inch robot?"
    logging.info(f"Starting MCTS search for question: {question}")
    
    mcts = MCTS(API_URL, MODEL_NAME)
    best_response, evaluation = mcts.search(question)
    
    print("\nBest Response:")
    print("-" * 80)
    print(best_response)
    print("\nEvaluation Details:")
    print("-" * 80)
    print(json.dumps(evaluation, indent=2))

if __name__ == "__main__":
    main()