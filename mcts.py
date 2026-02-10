import math
import random
import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import numpy as np
from config import Config

class MCTSNode:
    def __init__(self, parent=None, temperature: float = 0.7, 
                 response: str = None, prompt_addition: str = ""):
        self.parent = parent
        self.children: List['MCTSNode'] = []
        self.visits: int = 0
        self.value: float = 0.0
        self.temperature = temperature
        self.response = response
        self.prompt_addition = prompt_addition
    
    def ucb_score(self, exploration_constant: float) -> float:
        if self.visits == 0:
            return float('inf')
        
        exploitation = self.value / max(self.visits, 1)  # Prevent division by zero
        
        if self.parent and self.parent.visits > 0:
            exploration = exploration_constant * math.sqrt(
                math.log(self.parent.visits) / max(self.visits, 1)
            )
        else:
            exploration = float('inf')
            
        return exploitation + exploration
    
    def is_leaf(self) -> bool:
        return len(self.children) == 0
    
    def best_child(self) -> Optional['MCTSNode']:
        """Return the child with the best average value."""
        if not self.children:
            return None
            
        def score(node: 'MCTSNode') -> float:
            if node.visits == 0:
                return float('-inf')
            return node.value / node.visits
            
        return max(self.children, key=score)

class MCTS:
    def __init__(self, llm_client: 'LLMClient', config: Config):
        self.llm_client = llm_client
        self.config = config
    
    def search(self, question: str, system_prompt: str) -> str:
        root = MCTSNode()
        
        for i in range(self.config.mcts_iterations):
            node = self._select(root)
            
            if node.visits > 0:
                node = self._expand(node, question, system_prompt)
            
            value = self._simulate(node, question, system_prompt)
            self._backpropagate(node, value)
            
            logging.info(f"Completed MCTS iteration {i+1}/{self.config.mcts_iterations}")
        
        best_child = root.best_child()
        if best_child and best_child.response:
            return best_child.response
        elif root.children and root.children[0].response:  # Fallback to first child if exists
            return root.children[0].response
        else:
            # Last resort: generate a new response
            return self.llm_client.ask_question(
                question=question,
                temperature=0.7,
                system_prompt=system_prompt
            ) or "Unable to generate a response."
    
    def _select(self, node: MCTSNode) -> MCTSNode:
        current = node
        while not current.is_leaf() and current.children:
            current = max(current.children, 
                         key=lambda c: c.ucb_score(self.config.exploration_constant))
        return current
    
    def _expand(self, node: MCTSNode, question: str, system_prompt: str) -> MCTSNode:
        successful_expansions = 0
        max_attempts = self.config.beam_width * 2  # Allow for some failed attempts
        
        for _ in range(max_attempts):
            if successful_expansions >= self.config.beam_width:
                break
                
            temp = random.uniform(self.config.min_temperature, 
                                self.config.max_temperature)
            
            prompt_addition = random.choice([
                "\nBe comprehensive and detailed.",
                "\nFocus on practical implementation.",
                "\nConsider edge cases.",
                "\nProvide specific examples.",
                ""
            ])
            
            response = self.llm_client.ask_question(
                question=question,
                temperature=temp,
                system_prompt=system_prompt + prompt_addition
            )
            
            if response:
                child = MCTSNode(
                    parent=node,
                    temperature=temp,
                    response=response,
                    prompt_addition=prompt_addition
                )
                node.children.append(child)
                successful_expansions += 1
        
        if node.children:
            return random.choice(node.children)
        return node  # Return original node if no children were created
    
    def _simulate(self, node: MCTSNode, question: str, system_prompt: str) -> float:
        if not node.response:
            return 0.0
        
        # Evaluate response quality
        eval_prompt = f"""
        Evaluate this response to the question:
        Question: {question}
        Response: {node.response}
        
        Rate from 0-10 on:
        1. Relevance
        2. Completeness
        3. Accuracy
        4. Clarity
        5. Practicality
        
        Provide only numbers separated by commas.
        """
        
        try:
            eval_result = self.llm_client.ask_question(
                eval_prompt,
                temperature=0.1
            )
            
            if eval_result:
                scores = [float(x.strip()) for x in eval_result.split(',')]
                return sum(scores) / (len(scores) * 10)
        except Exception as e:
            logging.error(f"Error in simulation: {str(e)}")
        
        return 0.0
    
    def _backpropagate(self, node: MCTSNode, value: float) -> None:
        current = node
        while current is not None:
            current.visits += 1
            current.value += value
            current = current.parent