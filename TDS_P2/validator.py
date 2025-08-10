from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import re
import logging
from urllib.parse import urlparse

from config import config
from models import ExecutionPlan, Action, ActionType

logger = logging.getLogger(__name__)

class ValidationResult(BaseModel):
    is_valid: bool
    errors: List[str] = []
    warnings: List[str] = []

class PlanValidator:
    def __init__(self):
        self.max_actions = 20
        self.max_time_per_action = 120  # Increased to 2 minutes for scraping
        
    async def validate_plan(self, plan: ExecutionPlan) -> ValidationResult:
        """Validate an execution plan for safety and feasibility"""
        errors = []
        warnings = []
        
        try:
            # Basic plan validation
            basic_errors = self._validate_basic_plan(plan)
            errors.extend(basic_errors)
            
            # Time validation
            time_errors = self._validate_time_constraints(plan)
            errors.extend(time_errors)
            
            # Action validation
            for action in plan.actions:
                action_errors, action_warnings = self._validate_action(action)
                errors.extend(action_errors)
                warnings.extend(action_warnings)
            
            # Dependency validation
            dep_errors = self._validate_dependencies(plan)
            errors.extend(dep_errors)
            
            # Resource validation
            resource_warnings = self._validate_resources(plan)
            warnings.extend(resource_warnings)
            
            is_valid = len(errors) == 0
            
            if not is_valid:
                logger.warning(f"Plan validation failed with {len(errors)} errors")
            
            return ValidationResult(
                is_valid=is_valid,
                errors=errors,
                warnings=warnings
            )
            
        except Exception as e:
            logger.error(f"Validation error: {str(e)}")
            return ValidationResult(
                is_valid=False,
                errors=[f"Validation process failed: {str(e)}"]
            )
    
    def _validate_basic_plan(self, plan: ExecutionPlan) -> List[str]:
        """Validate basic plan structure"""
        errors = []
        
        if not plan.plan_id:
            errors.append("Plan must have a valid plan_id")
        
        if not plan.actions:
            errors.append("Plan must contain at least one action")
        
        if len(plan.actions) > self.max_actions:
            errors.append(f"Plan contains too many actions (max: {self.max_actions})")
        
        # Check for duplicate action IDs
        action_ids = [action.action_id for action in plan.actions]
        if len(action_ids) != len(set(action_ids)):
            errors.append("Duplicate action IDs found in plan")
        
        return errors
    
    def _validate_time_constraints(self, plan: ExecutionPlan) -> List[str]:
        """Validate time constraints"""
        errors = []
        
        if plan.estimated_total_time > config.MAX_EXECUTION_TIME:
            errors.append(f"Plan exceeds maximum execution time ({config.MAX_EXECUTION_TIME}s)")
        
        for action in plan.actions:
            if action.estimated_time and action.estimated_time > self.max_time_per_action:
                errors.append(f"Action {action.action_id} exceeds maximum time per action ({self.max_time_per_action}s)")
        
        return errors
    
    def _validate_action(self, action: Action) -> tuple[List[str], List[str]]:
        """Validate individual action"""
        errors = []
        warnings = []
        
        # Validate action type
        if action.type not in ActionType:
            errors.append(f"Invalid action type: {action.type}")
        
        # Type-specific validation
        if action.type == ActionType.SCRAPE:
            scrape_errors = self._validate_scrape_action(action)
            errors.extend(scrape_errors)
        
        elif action.type == ActionType.API_CALL:
            api_errors = self._validate_api_action(action)
            errors.extend(api_errors)
        
        elif action.type == ActionType.PLOT:
            plot_warnings = self._validate_plot_action(action)
            warnings.extend(plot_warnings)
        
        elif action.type == ActionType.SQL:
            sql_errors = self._validate_sql_action(action)
            errors.extend(sql_errors)
        
        # Check for dangerous parameters
        dangerous_params = self._check_dangerous_parameters(action)
        errors.extend(dangerous_params)
        
        return errors, warnings
    
    def _validate_scrape_action(self, action: Action) -> List[str]:
        """Validate scraping action"""
        errors = []
        
        url = action.parameters.get("url")
        if not url:
            errors.append(f"Scrape action {action.action_id} missing URL parameter")
            return errors
        
        # Validate URL format
        try:
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                errors.append(f"Invalid URL format: {url}")
        except Exception:
            errors.append(f"Malformed URL: {url}")
        
        # Check allowed domains
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            
            # Remove www. prefix for checking
            if domain.startswith('www.'):
                domain = domain[4:]
            
            allowed = any(domain.endswith(allowed_domain) for allowed_domain in config.ALLOWED_DOMAINS)
            if not allowed:
                errors.append(f"Domain not allowed for scraping: {domain}")
        except Exception:
            errors.append(f"Could not validate domain for URL: {url}")
        
        return errors
    
    def _validate_api_action(self, action: Action) -> List[str]:
        """Validate API call action"""
        errors = []
        
        url = action.parameters.get("url")
        if not url:
            errors.append(f"API action {action.action_id} missing URL parameter")
            return errors
        
        # Validate URL format
        try:
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                errors.append(f"Invalid API URL format: {url}")
        except Exception:
            errors.append(f"Malformed API URL: {url}")
        
        return errors
    
    def _validate_plot_action(self, action: Action) -> List[str]:
        """Validate plot action (returns warnings)"""
        warnings = []
        
        plot_type = action.parameters.get("plot_type")
        if not plot_type:
            warnings.append(f"Plot action {action.action_id} missing plot_type parameter")
        
        # Check if output format is specified
        output_format = action.parameters.get("format", "png")
        if output_format not in ["png", "jpg", "svg", "pdf"]:
            warnings.append(f"Unusual plot format: {output_format}")
        
        return warnings
    
    def _validate_sql_action(self, action: Action) -> List[str]:
        """Validate SQL action"""
        errors = []
        
        query = action.parameters.get("query", "")
        if not query:
            errors.append(f"SQL action {action.action_id} missing query parameter")
            return errors
        
        # Check for dangerous SQL patterns
        dangerous_patterns = [
            r'\bDROP\s+TABLE\b',
            r'\bDELETE\s+FROM\b',
            r'\bTRUNCATE\b',
            r'\bALTER\s+TABLE\b',
            r'\bCREATE\s+TABLE\b',
            r'\bINSERT\s+INTO\b',
            r'\bUPDATE\s+.*\s+SET\b'
        ]
        
        query_upper = query.upper()
        for pattern in dangerous_patterns:
            if re.search(pattern, query_upper):
                errors.append(f"Dangerous SQL operation detected in {action.action_id}: {pattern}")
        
        return errors
    
    def _check_dangerous_parameters(self, action: Action) -> List[str]:
        """Check for dangerous parameters across all action types"""
        errors = []
        
        # Convert parameters to string for pattern checking
        params_str = str(action.parameters).lower()
        
        # Check for dangerous patterns
        dangerous_patterns = [
            ('system', 'system command execution'),
            ('exec', 'code execution'),
            ('eval', 'code evaluation'),
            ('import os', 'OS module import'),
            ('subprocess', 'subprocess execution'),
            ('__import__', 'dynamic imports'),
            ('file://', 'local file access'),
            ('../', 'directory traversal')
        ]
        
        for pattern, description in dangerous_patterns:
            if pattern in params_str:
                errors.append(f"Dangerous parameter detected in {action.action_id}: {description}")
        
        return errors
    
    def _validate_dependencies(self, plan: ExecutionPlan) -> List[str]:
        """Validate action dependencies"""
        errors = []
        
        action_ids = {action.action_id for action in plan.actions}
        
        for action in plan.actions:
            for dep in action.dependencies:
                if dep not in action_ids:
                    errors.append(f"Action {action.action_id} depends on non-existent action: {dep}")
        
        # Check for circular dependencies (simplified check)
        if self._has_circular_dependencies(plan):
            errors.append("Circular dependencies detected in plan")
        
        return errors
    
    def _has_circular_dependencies(self, plan: ExecutionPlan) -> bool:
        """Check for circular dependencies"""
        # Build dependency graph
        deps = {}
        for action in plan.actions:
            deps[action.action_id] = action.dependencies
        
        # Simple cycle detection using DFS
        visited = set()
        rec_stack = set()
        
        def has_cycle(node):
            if node in rec_stack:
                return True
            if node in visited:
                return False
            
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in deps.get(node, []):
                if has_cycle(neighbor):
                    return True
            
            rec_stack.remove(node)
            return False
        
        for action_id in deps:
            if has_cycle(action_id):
                return True
        
        return False
    
    def _validate_resources(self, plan: ExecutionPlan) -> List[str]:
        """Validate resource usage (returns warnings)"""
        warnings = []
        
        # Count different types of actions
        action_counts = {}
        for action in plan.actions:
            action_type = action.type
            action_counts[action_type] = action_counts.get(action_type, 0) + 1
        
        # Warn about potentially resource-intensive combinations
        if action_counts.get(ActionType.SCRAPE, 0) > 3:
            warnings.append("Plan contains many scraping actions - may be slow")
        
        if action_counts.get(ActionType.PLOT, 0) > 5:
            warnings.append("Plan contains many plot actions - may consume significant memory")
        
        return warnings
