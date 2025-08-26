#!/usr/bin/env python3
"""
ACIS Trading Platform - Codebase Analysis Tool
Analyzes codebase for redundancy, consolidation opportunities, and EOD script dependencies
"""

import os
import ast
import re
from collections import defaultdict, Counter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CodebaseAnalyzer:
    def __init__(self, root_path):
        """Initialize codebase analyzer"""
        self.root_path = root_path
        self.python_files = []
        self.import_dependencies = defaultdict(set)
        self.function_definitions = defaultdict(list)
        self.similar_functions = defaultdict(list)
        self.table_references = defaultdict(set)
        self.eod_script_dependencies = defaultdict(set)
        
    def scan_codebase(self):
        """Scan entire codebase for analysis"""
        print("\n[CODEBASE SCAN] Analyzing ACIS Trading Platform")
        print("=" * 80)
        
        # Find all Python files (excluding archived)
        for root, dirs, files in os.walk(self.root_path):
            # Skip archived directories
            if 'archive' in root or '__pycache__' in root:
                continue
                
            for file in files:
                if file.endswith('.py'):
                    filepath = os.path.join(root, file)
                    self.python_files.append(filepath)
        
        print(f"Found {len(self.python_files)} active Python files")
        
        # Analyze each file
        for filepath in self.python_files:
            self.analyze_file(filepath)
        
        return len(self.python_files)
    
    def analyze_file(self, filepath):
        """Analyze individual Python file"""
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Parse AST
            try:
                tree = ast.parse(content)
                self.extract_functions(filepath, tree)
                self.extract_imports(filepath, tree)
            except SyntaxError:
                pass  # Skip files with syntax errors
            
            # Extract table references
            self.extract_table_references(filepath, content)
            
            # Check for EOD script patterns
            self.check_eod_patterns(filepath, content)
            
        except Exception as e:
            logger.warning(f"Error analyzing {filepath}: {e}")
    
    def extract_functions(self, filepath, tree):
        """Extract function definitions from AST"""
        class FunctionVisitor(ast.NodeVisitor):
            def __init__(self, analyzer):
                self.analyzer = analyzer
                self.filepath = filepath
            
            def visit_FunctionDef(self, node):
                func_info = {
                    'name': node.name,
                    'args': [arg.arg for arg in node.args.args],
                    'file': self.filepath,
                    'line': node.lineno
                }
                self.analyzer.function_definitions[node.name].append(func_info)
                self.generic_visit(node)
        
        visitor = FunctionVisitor(self)
        visitor.visit(tree)
    
    def extract_imports(self, filepath, tree):
        """Extract import dependencies from AST"""
        class ImportVisitor(ast.NodeVisitor):
            def __init__(self, analyzer):
                self.analyzer = analyzer
                self.filepath = filepath
            
            def visit_Import(self, node):
                for alias in node.names:
                    self.analyzer.import_dependencies[self.filepath].add(alias.name)
            
            def visit_ImportFrom(self, node):
                if node.module:
                    self.analyzer.import_dependencies[self.filepath].add(node.module)
        
        visitor = ImportVisitor(self)
        visitor.visit(tree)
    
    def extract_table_references(self, filepath, content):
        """Extract database table references"""
        # Common table patterns in ACIS
        table_patterns = [
            r"CREATE TABLE\s+(\w+)",
            r"INSERT INTO\s+(\w+)",
            r"SELECT.*FROM\s+(\w+)",
            r"UPDATE\s+(\w+)",
            r"DELETE FROM\s+(\w+)",
            r"DROP TABLE\s+(\w+)",
            r"'(\w*portfolio\w*)'",
            r"'(\w*stock\w*)'",
            r"'(\w*price\w*)'",
            r"'(\w*fundamental\w*)'",
            r"'(\w*dividend\w*)'",
            r"'(\w*score\w*)'",
        ]
        
        for pattern in table_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                table_name = match.group(1).lower()
                if table_name and len(table_name) > 2:  # Filter out very short names
                    self.table_references[table_name].add(filepath)
    
    def check_eod_patterns(self, filepath, content):
        """Check for End-of-Day script patterns"""
        eod_indicators = [
            'run_eod',
            'eod_pipeline',
            'fetch_prices',
            'compute_forward_returns',
            'train_ai_',
            'score_ai_',
            'create_portfolios',
            'backtest'
        ]
        
        filename = os.path.basename(filepath)
        for indicator in eod_indicators:
            if indicator in filename.lower() or indicator in content.lower():
                self.eod_script_dependencies[indicator].add(filepath)
    
    def identify_redundant_scripts(self):
        """Identify potentially redundant scripts"""
        print("\n[REDUNDANCY ANALYSIS] Identifying Duplicate/Similar Scripts")
        print("=" * 80)
        
        # Group scripts by similar functionality
        functionality_groups = defaultdict(list)
        
        for filepath in self.python_files:
            filename = os.path.basename(filepath)
            
            # Categorize by functionality keywords
            if 'backtest' in filename:
                functionality_groups['backtesting'].append(filepath)
            elif 'train_ai' in filename:
                functionality_groups['ai_training'].append(filepath)
            elif 'score_ai' in filename or 'score' in filename:
                functionality_groups['ai_scoring'].append(filepath)
            elif 'create_' in filename and 'portfolio' in filename:
                functionality_groups['portfolio_creation'].append(filepath)
            elif 'fetch' in filename:
                functionality_groups['data_fetching'].append(filepath)
            elif 'strategy' in filename:
                functionality_groups['strategy_related'].append(filepath)
            elif 'schwab' in filename:
                functionality_groups['broker_integration'].append(filepath)
            elif 'test_' in filename:
                functionality_groups['testing'].append(filepath)
        
        # Identify potential redundancies
        redundancy_candidates = {}
        
        print("POTENTIAL REDUNDANT SCRIPT GROUPS:")
        for group, scripts in functionality_groups.items():
            if len(scripts) > 3:  # Groups with many similar scripts
                print(f"\n{group.upper().replace('_', ' ')} ({len(scripts)} scripts):")
                for script in scripts:
                    filename = os.path.basename(script)
                    print(f"  {filename}")
                
                redundancy_candidates[group] = scripts
        
        return redundancy_candidates
    
    def identify_similar_functions(self):
        """Identify functions with similar names/purposes"""
        print("\n[FUNCTION ANALYSIS] Similar/Duplicate Functions")
        print("=" * 80)
        
        # Find functions with similar names
        similar_groups = defaultdict(list)
        
        for func_name, func_list in self.function_definitions.items():
            if len(func_list) > 1:  # Functions defined in multiple files
                similar_groups[func_name].extend(func_list)
        
        # Also check for similar naming patterns
        function_names = list(self.function_definitions.keys())
        for i, name1 in enumerate(function_names):
            for name2 in function_names[i+1:]:
                # Check for similar names (same root)
                if len(name1) > 5 and len(name2) > 5:
                    # Simple similarity check
                    root1 = name1.replace('_', '').lower()
                    root2 = name2.replace('_', '').lower()
                    
                    if root1 in root2 or root2 in root1:
                        similar_groups[f"{name1}_vs_{name2}"] = [
                            *self.function_definitions[name1],
                            *self.function_definitions[name2]
                        ]
        
        print("DUPLICATE/SIMILAR FUNCTIONS:")
        consolidation_opportunities = {}
        
        for group_name, functions in similar_groups.items():
            if len(functions) > 1:
                print(f"\n{group_name}:")
                files_involved = set()
                for func in functions:
                    filename = os.path.basename(func['file'])
                    files_involved.add(filename)
                    print(f"  {func['name']} in {filename} (line {func['line']})")
                
                consolidation_opportunities[group_name] = {
                    'functions': functions,
                    'files_involved': list(files_involved)
                }
        
        return consolidation_opportunities
    
    def analyze_table_usage(self):
        """Analyze database table usage patterns"""
        print("\n[TABLE ANALYSIS] Database Table Usage")
        print("=" * 80)
        
        # Sort tables by usage frequency
        table_usage = sorted(self.table_references.items(), 
                           key=lambda x: len(x[1]), reverse=True)
        
        print("TABLE USAGE FREQUENCY:")
        print("Table Name                   Files Using   Files")
        print("-" * 70)
        
        heavily_used_tables = []
        lightly_used_tables = []
        
        for table_name, files in table_usage:
            file_count = len(files)
            filenames = [os.path.basename(f) for f in list(files)[:3]]  # Show first 3
            files_str = ", ".join(filenames)
            if file_count > 3:
                files_str += f" (+{file_count-3} more)"
            
            print(f"{table_name:<30} {file_count:>5}        {files_str}")
            
            if file_count >= 5:
                heavily_used_tables.append((table_name, file_count))
            elif file_count == 1:
                lightly_used_tables.append((table_name, files))
        
        print(f"\nTABLE CONSOLIDATION OPPORTUNITIES:")
        print(f"  Heavily Used Tables: {len(heavily_used_tables)} (keep as core)")
        print(f"  Lightly Used Tables: {len(lightly_used_tables)} (candidates for cleanup)")
        
        if lightly_used_tables:
            print(f"\nTables used by only 1 script (cleanup candidates):")
            for table_name, files in lightly_used_tables[:10]:  # Show top 10
                filename = os.path.basename(list(files)[0])
                print(f"  {table_name:<25} -> {filename}")
        
        return heavily_used_tables, lightly_used_tables
    
    def analyze_eod_dependencies(self):
        """Analyze End-of-Day script dependencies"""
        print("\n[EOD ANALYSIS] End-of-Day Script Dependencies")
        print("=" * 80)
        
        print("EOD SCRIPT CATEGORIES:")
        for category, scripts in self.eod_script_dependencies.items():
            print(f"\n{category.upper().replace('_', ' ')} ({len(scripts)} scripts):")
            for script in sorted(scripts):
                filename = os.path.basename(script)
                print(f"  {filename}")
        
        # Check for main EOD orchestrator
        main_eod_scripts = []
        for script in self.python_files:
            filename = os.path.basename(script)
            if 'run_eod' in filename.lower() or 'eod_full_pipeline' in filename.lower():
                main_eod_scripts.append(script)
        
        print(f"\nMAIN EOD ORCHESTRATORS:")
        for script in main_eod_scripts:
            filename = os.path.basename(script)
            print(f"  {filename}")
            
            # Check what this script calls
            try:
                with open(script, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                # Look for subprocess calls, imports, or function calls
                calls = []
                for other_script in self.python_files:
                    other_filename = os.path.basename(other_script)
                    if other_filename.replace('.py', '') in content:
                        calls.append(other_filename)
                
                if calls:
                    print(f"    Calls: {', '.join(calls[:5])}{'...' if len(calls) > 5 else ''}")
                
            except Exception as e:
                print(f"    Could not analyze dependencies: {e}")
        
        return main_eod_scripts
    
    def generate_consolidation_recommendations(self, redundancy_candidates, 
                                            consolidation_opportunities, 
                                            lightly_used_tables, main_eod_scripts):
        """Generate specific consolidation recommendations"""
        print("\n[CONSOLIDATION RECOMMENDATIONS] Cleanup & Optimization Plan")
        print("=" * 80)
        
        recommendations = []
        
        # Script consolidation recommendations
        print("SCRIPT CONSOLIDATION OPPORTUNITIES:")
        
        priority_consolidations = {
            'backtesting': {
                'target': 'unified_backtest_engine.py',
                'description': 'Consolidate all backtesting scripts into single engine'
            },
            'ai_training': {
                'target': 'ai_model_trainer.py', 
                'description': 'Combine AI training scripts for value/growth/dividend/momentum'
            },
            'ai_scoring': {
                'target': 'ai_model_scorer.py',
                'description': 'Unify AI scoring scripts into single scoring engine'
            },
            'portfolio_creation': {
                'target': 'portfolio_generator.py',
                'description': 'Combine portfolio creation scripts (8-strategy, 12-strategy, etc.)'
            },
            'data_fetching': {
                'target': 'data_fetcher.py',
                'description': 'Consolidate price, fundamental, and dividend fetching'
            }
        }
        
        for group, config in priority_consolidations.items():
            if group in redundancy_candidates:
                script_count = len(redundancy_candidates[group])
                print(f"\n{group.upper()}:")
                print(f"  Current: {script_count} separate scripts")
                print(f"  Recommended: Consolidate into {config['target']}")
                print(f"  Benefit: {config['description']}")
                
                recommendations.append({
                    'type': 'script_consolidation',
                    'group': group,
                    'current_count': script_count,
                    'target': config['target'],
                    'priority': 'high' if script_count > 5 else 'medium'
                })
        
        # Table cleanup recommendations
        print(f"\nTABLE CLEANUP OPPORTUNITIES:")
        if lightly_used_tables:
            print(f"  Tables to review: {len(lightly_used_tables)} single-use tables")
            print(f"  Potential cleanup: Remove unused/temporary tables")
            print(f"  Estimated storage savings: Significant")
            
            recommendations.append({
                'type': 'table_cleanup',
                'tables_count': len(lightly_used_tables),
                'priority': 'medium'
            })
        
        # EOD optimization recommendations
        print(f"\nEOD SCRIPT OPTIMIZATION:")
        if len(main_eod_scripts) > 1:
            print(f"  Current: {len(main_eod_scripts)} EOD orchestrators")
            print(f"  Recommended: Single master EOD script")
            print(f"  Benefit: Eliminate redundant executions")
            
            recommendations.append({
                'type': 'eod_consolidation',
                'current_count': len(main_eod_scripts),
                'priority': 'high'
            })
        
        # Calculate estimated benefits
        total_scripts = len(self.python_files)
        consolidatable_scripts = sum(len(scripts) for scripts in redundancy_candidates.values())
        potential_reduction = min(50, consolidatable_scripts * 0.6)  # Conservative estimate
        
        print(f"\nESTIMATED CONSOLIDATION BENEFITS:")
        print(f"  Current Scripts: {total_scripts}")
        print(f"  Consolidatable: {consolidatable_scripts}")
        print(f"  Potential Reduction: {potential_reduction:.0f} scripts")
        print(f"  Maintenance Reduction: {potential_reduction/total_scripts:.0%}")
        
        return recommendations
    
    def check_eod_execution_patterns(self):
        """Check if EOD scripts run during end-to-end tests"""
        print("\n[EOD EXECUTION ANALYSIS] EOD Scripts in End-to-End Runs")
        print("=" * 80)
        
        # Check our recent AI systems for EOD dependencies
        recent_ai_scripts = [
            'ai_discovery_engine.py',
            'ai_regime_detector.py', 
            'ai_weight_optimizer.py',
            'ai_ensemble_framework.py',
            'enhanced_ai_ensemble.py',
            'dividend_optimized_acis.py',
            'complete_backtesting_engine.py'
        ]
        
        eod_dependencies_found = {}
        
        for script_name in recent_ai_scripts:
            script_path = os.path.join(self.root_path, script_name)
            if os.path.exists(script_path):
                try:
                    with open(script_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    # Check for EOD-related calls
                    eod_calls = []
                    eod_patterns = [
                        'fetch_prices',
                        'fetch_fundamentals', 
                        'run_eod',
                        'compute_forward_returns',
                        'train_ai_',
                        'CREATE TABLE',
                        'subprocess',
                        'os.system'
                    ]
                    
                    for pattern in eod_patterns:
                        if pattern in content:
                            eod_calls.append(pattern)
                    
                    if eod_calls:
                        eod_dependencies_found[script_name] = eod_calls
                
                except Exception as e:
                    logger.warning(f"Could not analyze {script_name}: {e}")
        
        print("EOD DEPENDENCIES IN AI SYSTEMS:")
        if eod_dependencies_found:
            for script, dependencies in eod_dependencies_found.items():
                print(f"\n{script}:")
                for dep in dependencies:
                    print(f"  - {dep}")
        else:
            print("* No direct EOD dependencies found in recent AI systems")
            print("* AI systems use synthetic/simulated data - good architecture!")
        
        # Check if our backtesting runs real EOD processes
        backtesting_scripts = [script for script in self.python_files if 'backtest' in script.lower()]
        
        print(f"\nBACKTESTING SCRIPT ANALYSIS:")
        print(f"Found {len(backtesting_scripts)} backtesting scripts")
        
        synthetic_data_scripts = []
        real_data_scripts = []
        
        for script in backtesting_scripts:
            try:
                with open(script, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                if 'np.random' in content or 'simulate' in content or 'generate' in content:
                    synthetic_data_scripts.append(os.path.basename(script))
                elif 'fetch_prices' in content or 'database' in content:
                    real_data_scripts.append(os.path.basename(script))
                    
            except Exception:
                pass
        
        print(f"\nData Usage Patterns:")
        print(f"  Synthetic Data Scripts: {len(synthetic_data_scripts)}")
        for script in synthetic_data_scripts[:5]:
            print(f"    {script}")
        
        print(f"  Real Data Scripts: {len(real_data_scripts)}")  
        for script in real_data_scripts[:5]:
            print(f"    {script}")
        
        return eod_dependencies_found, synthetic_data_scripts, real_data_scripts

def main():
    """Run complete codebase analysis"""
    print("\n[LAUNCH] ACIS Codebase Analysis & Consolidation Tool")
    print("Analyzing for redundancy, consolidation opportunities, and EOD dependencies")
    
    root_path = r"C:\Users\frank\PycharmProjects\PythonProject\acis-trading-platform"
    analyzer = CodebaseAnalyzer(root_path)
    
    # Scan codebase
    file_count = analyzer.scan_codebase()
    
    # Identify redundancies
    redundancy_candidates = analyzer.identify_redundant_scripts()
    
    # Analyze functions
    consolidation_opportunities = analyzer.identify_similar_functions()
    
    # Analyze tables
    heavily_used_tables, lightly_used_tables = analyzer.analyze_table_usage()
    
    # Analyze EOD dependencies
    main_eod_scripts = analyzer.analyze_eod_dependencies()
    
    # Check EOD execution patterns
    eod_deps, synthetic_scripts, real_scripts = analyzer.check_eod_execution_patterns()
    
    # Generate recommendations
    recommendations = analyzer.generate_consolidation_recommendations(
        redundancy_candidates, consolidation_opportunities, 
        lightly_used_tables, main_eod_scripts
    )
    
    print(f"\n[SUCCESS] Codebase Analysis Complete!")
    print(f"Analyzed {file_count} Python files")
    print(f"Found {len(redundancy_candidates)} consolidation opportunities")
    print(f"Identified {len(lightly_used_tables)} cleanup candidates")
    print(f"Generated {len(recommendations)} optimization recommendations")
    
    return analyzer, recommendations

if __name__ == "__main__":
    main()