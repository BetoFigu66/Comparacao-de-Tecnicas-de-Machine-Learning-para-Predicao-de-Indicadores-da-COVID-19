"""
IBGE Municipal Data Explorer for COVID-19 Prediction Models
Analyzes Base_MUNIC_2020.xlsx to identify relevant variables for COVID-19 prediction

Author: TCC Analysis
Date: 2025-09-27
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class IBGEDataExplorer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.xl = pd.ExcelFile(file_path)
        self.sheets_data = {}
        self.relevant_variables = {}
        
    def load_all_sheets(self):
        """Load all sheets from the Excel file"""
        print("Loading all sheets from IBGE data...")
        for sheet_name in self.xl.sheet_names:
            if sheet_name != 'Dicionário':  # Skip dictionary for now
                try:
                    df = pd.read_excel(self.file_path, sheet_name=sheet_name)
                    self.sheets_data[sheet_name] = df
                    print(f"[OK] Loaded {sheet_name}: {df.shape}")
                except Exception as e:
                    print(f"[ERROR] Error loading {sheet_name}: {e}")
    
    def analyze_data_quality(self):
        """Analyze data quality across all sheets"""
        print("\n=== DATA QUALITY ANALYSIS ===")
        quality_report = {}
        
        for sheet_name, df in self.sheets_data.items():
            # Find municipality code column
            mun_col = None
            for col in df.columns:
                if 'CodMun' in col or ('Cod' in col and 'Mun' in col):
                    mun_col = col
                    break
            
            if mun_col is None:
                print(f"[WARNING] {sheet_name}: No municipality code column found")
                continue
                
            # Calculate missing data percentages
            missing_pct = (df.isnull().sum() / len(df)) * 100
            
            # Identify numeric columns (potential features)
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            # Identify categorical columns with reasonable number of categories
            categorical_cols = []
            for col in df.columns:
                if col not in numeric_cols and col not in [mun_col, 'UF', 'Cod UF', 'Mun', 'Desc Mun', 'Sigla UF', 'Regiao']:
                    unique_vals = df[col].nunique()
                    if 2 <= unique_vals <= 20:  # Reasonable number of categories
                        categorical_cols.append(col)
            
            quality_report[sheet_name] = {
                'total_columns': len(df.columns),
                'numeric_columns': len(numeric_cols),
                'categorical_columns': len(categorical_cols),
                'high_quality_cols': len(missing_pct[missing_pct < 30]),  # Less than 30% missing
                'municipality_code': mun_col,
                'missing_data_summary': {
                    'low_missing': len(missing_pct[missing_pct < 10]),
                    'medium_missing': len(missing_pct[(missing_pct >= 10) & (missing_pct < 30)]),
                    'high_missing': len(missing_pct[missing_pct >= 30])
                }
            }
            
            print(f"\n{sheet_name}:")
            print(f"  Total columns: {quality_report[sheet_name]['total_columns']}")
            print(f"  Numeric columns: {quality_report[sheet_name]['numeric_columns']}")
            print(f"  Categorical columns: {quality_report[sheet_name]['categorical_columns']}")
            print(f"  High quality columns (<30% missing): {quality_report[sheet_name]['high_quality_cols']}")
            print(f"  Missing data: Low(<10%): {quality_report[sheet_name]['missing_data_summary']['low_missing']}, "
                  f"Medium(10-30%): {quality_report[sheet_name]['missing_data_summary']['medium_missing']}, "
                  f"High(>30%): {quality_report[sheet_name]['missing_data_summary']['high_missing']}")
        
        return quality_report
    
    def identify_covid_relevant_variables(self):
        """Identify variables most relevant for COVID-19 prediction"""
        print("\n=== COVID-19 RELEVANT VARIABLES ANALYSIS ===")
        
        # Define relevance categories based on COVID-19 literature
        relevance_categories = {
            'health_infrastructure': {
                'keywords': ['saude', 'medico', 'enfermeiro', 'hospital', 'ubs', 'posto', 'profissional'],
                'description': 'Health infrastructure and human resources'
            },
            'social_vulnerability': {
                'keywords': ['social', 'vulnerab', 'pobreza', 'renda', 'bolsa', 'auxilio', 'beneficio'],
                'description': 'Social vulnerability and assistance programs'
            },
            'demographics': {
                'keywords': ['populacao', 'idade', 'idoso', 'crianca', 'densidade'],
                'description': 'Demographic characteristics'
            },
            'food_security': {
                'keywords': ['alimentar', 'nutricao', 'fome', 'seguranca', 'agricultura'],
                'description': 'Food security and nutrition programs'
            },
            'public_safety': {
                'keywords': ['seguranca', 'policia', 'violencia', 'crime'],
                'description': 'Public safety indicators'
            },
            'education': {
                'keywords': ['educacao', 'escola', 'ensino', 'professor'],
                'description': 'Education infrastructure and programs'
            },
            'economic': {
                'keywords': ['trabalho', 'emprego', 'renda', 'economia', 'produtivo'],
                'description': 'Economic and employment indicators'
            }
        }
        
        for sheet_name, df in self.sheets_data.items():
            print(f"\n--- {sheet_name} ---")
            sheet_relevant = {}
            
            # Find municipality code column
            mun_col = None
            for col in df.columns:
                if 'CodMun' in col or ('Cod' in col and 'Mun' in col):
                    mun_col = col
                    break
            
            if mun_col is None:
                continue
                
            # Analyze each column for relevance
            for col in df.columns:
                if col in [mun_col, 'UF', 'Cod UF', 'Mun', 'Desc Mun', 'Sigla UF', 'Regiao', 'PopMun', 'Faixa_pop']:
                    continue  # Skip basic identification columns
                
                # Calculate data quality
                missing_pct = (df[col].isnull().sum() / len(df)) * 100
                
                if missing_pct > 50:  # Skip columns with too much missing data
                    continue
                
                # Check relevance based on column name patterns
                col_lower = col.lower()
                relevance_scores = {}
                
                for category, info in relevance_categories.items():
                    score = 0
                    for keyword in info['keywords']:
                        if keyword in col_lower:
                            score += 1
                    relevance_scores[category] = score
                
                max_relevance = max(relevance_scores.values())
                if max_relevance > 0:
                    best_category = max(relevance_scores, key=relevance_scores.get)
                    
                    if best_category not in sheet_relevant:
                        sheet_relevant[best_category] = []
                    
                    # Get basic statistics
                    if df[col].dtype in ['object', 'string']:
                        unique_vals = df[col].nunique()
                        stats = f"Categorical ({unique_vals} categories)"
                    else:
                        stats = f"Numeric (mean: {df[col].mean():.2f})" if not df[col].isnull().all() else "Numeric (all null)"
                    
                    sheet_relevant[best_category].append({
                        'column': col,
                        'missing_pct': missing_pct,
                        'stats': stats,
                        'relevance_score': max_relevance
                    })
            
            # Display results for this sheet
            if sheet_relevant:
                for category, variables in sheet_relevant.items():
                    print(f"  {relevance_categories[category]['description']}:")
                    for var in sorted(variables, key=lambda x: x['relevance_score'], reverse=True)[:5]:  # Top 5
                        print(f"    • {var['column']} - Missing: {var['missing_pct']:.1f}% - {var['stats']}")
            
            self.relevant_variables[sheet_name] = sheet_relevant
    
    def create_integration_recommendations(self):
        """Create recommendations for integrating IBGE data with COVID models"""
        print("\n=== INTEGRATION RECOMMENDATIONS ===")
        
        # Prioritize variables based on COVID-19 relevance and data quality
        priority_variables = []
        
        for sheet_name, categories in self.relevant_variables.items():
            for category, variables in categories.items():
                for var in variables:
                    if var['missing_pct'] < 30:  # Good data quality
                        priority_score = var['relevance_score']
                        if var['missing_pct'] < 10:  # Excellent data quality
                            priority_score += 2
                        if category in ['health_infrastructure', 'social_vulnerability', 'demographics']:
                            priority_score += 1  # High COVID relevance
                        
                        priority_variables.append({
                            'sheet': sheet_name,
                            'category': category,
                            'variable': var['column'],
                            'missing_pct': var['missing_pct'],
                            'stats': var['stats'],
                            'priority_score': priority_score
                        })
        
        # Sort by priority score
        priority_variables.sort(key=lambda x: x['priority_score'], reverse=True)
        
        print("TOP 20 RECOMMENDED VARIABLES FOR COVID-19 MODELS:")
        print("-" * 80)
        for i, var in enumerate(priority_variables[:20], 1):
            print(f"{i:2d}. {var['variable']} ({var['sheet']})")
            print(f"    Category: {var['category']}")
            print(f"    Missing data: {var['missing_pct']:.1f}%")
            print(f"    Stats: {var['stats']}")
            print(f"    Priority score: {var['priority_score']}")
            print()
        
        return priority_variables[:20]
    
    def generate_integration_script(self, top_variables):
        """Generate a script to integrate IBGE data with existing COVID models"""
        script_content = '''"""
IBGE Data Integration Script for COVID-19 Models
Integrates selected IBGE municipal variables with existing COVID-19 datasets

Generated automatically by IBGEDataExplorer
"""

import pandas as pd
import numpy as np
from pathlib import Path

def load_ibge_variables():
    """Load selected IBGE variables for COVID-19 prediction"""
    
    file_path = r"C:\\Beto\\Profissional\\MBA_UspEsalq\\TCC\\Projeto_TCC_Jun_25\\DataSources\\IBGE\\Base_MUNIC_2023.xlsx"
    
    # Selected variables based on COVID-19 relevance and data quality
    selected_vars = {
'''
        
        # Add selected variables to script
        for var in top_variables:
            script_content += f'        "{var["variable"]}": "{var["sheet"]}",  # {var["category"]}\n'
        
        script_content += '''    }
    
    # Load and combine selected variables
    combined_data = None
    
    for var_name, sheet_name in selected_vars.items():
        try:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            
            # Find municipality code column
            mun_col = None
            for col in df.columns:
                if 'CodMun' in col or ('Cod' in col and 'Mun' in col):
                    mun_col = col
                    break
            
            if mun_col is None:
                print(f"Warning: No municipality code found in {sheet_name}")
                continue
            
            # Select relevant columns
            if var_name in df.columns:
                var_data = df[[mun_col, var_name]].copy()
                var_data = var_data.rename(columns={mun_col: 'CodMun', var_name: f'IBGE_{var_name}'})
                
                if combined_data is None:
                    combined_data = var_data
                else:
                    combined_data = combined_data.merge(var_data, on='CodMun', how='outer')
                    
                print(f"[OK] Added {var_name} from {sheet_name}")
            else:
                print(f"Warning: {var_name} not found in {sheet_name}")
                
        except Exception as e:
            print(f"Error loading {var_name} from {sheet_name}: {e}")
    
    return combined_data

def integrate_with_covid_data(covid_df, ibge_df):
    """Integrate IBGE data with existing COVID-19 dataset"""
    
    # Ensure municipality codes are consistent
    if 'CodMun' not in covid_df.columns:
        print("Error: COVID data must have 'CodMun' column")
        return covid_df
    
    # Merge datasets
    integrated_df = covid_df.merge(ibge_df, on='CodMun', how='left')
    
    print(f"Integration complete:")
    print(f"  Original COVID data: {covid_df.shape}")
    print(f"  IBGE data: {ibge_df.shape}")
    print(f"  Integrated data: {integrated_df.shape}")
    print(f"  New IBGE features: {len([col for col in integrated_df.columns if col.startswith('IBGE_')])}")
    
    return integrated_df

if __name__ == "__main__":
    # Load IBGE variables
    ibge_data = load_ibge_variables()
    
    if ibge_data is not None:
        print(f"\\nIBGE data loaded successfully: {ibge_data.shape}")
        print("Available IBGE variables:")
        for col in ibge_data.columns:
            if col.startswith('IBGE_'):
                print(f"  - {col}")
        
        # Save processed IBGE data
        output_path = Path("Data/Processed/ibge_selected_variables.parquet")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        ibge_data.to_parquet(output_path, index=False)
        print(f"\\nIBGE data saved to: {output_path}")
    else:
        print("Failed to load IBGE data")
'''
        
        return script_content
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        print("Starting IBGE Data Analysis for COVID-19 Prediction Models")
        print("=" * 70)
        
        # Load all data
        self.load_all_sheets()
        
        # Analyze data quality
        quality_report = self.analyze_data_quality()
        
        # Identify relevant variables
        self.identify_covid_relevant_variables()
        
        # Create recommendations
        top_variables = self.create_integration_recommendations()
        
        # Generate integration script
        integration_script = self.generate_integration_script(top_variables)
        
        # Save integration script
        script_path = Path("Scripts/ibge_integration.py")
        script_path.parent.mkdir(parents=True, exist_ok=True)
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(integration_script)
        
        print(f"Integration script saved to: {script_path}")
        print("\nNext steps:")
        print("1. Review the recommended variables above")
        print("2. Run the generated integration script to add IBGE data to your models")
        print("3. Update your model training pipeline to include the new features")
        
        return top_variables, quality_report

def main():
    # Initialize explorer
    file_path = r"C:\Beto\Profissional\MBA_UspEsalq\TCC\Projeto_TCC_Jun_25\DataSources\IBGE\Base_MUNIC_2020.xlsx"
    explorer = IBGEDataExplorer(file_path)
    
    # Run complete analysis
    top_variables, quality_report = explorer.run_complete_analysis()
    
    return explorer, top_variables, quality_report

if __name__ == "__main__":
    explorer, top_vars, quality = main()
