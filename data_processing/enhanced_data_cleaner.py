"""
Enhanced Data Cleaner with subsection header removal and improved currency detection.
Handles financial data with subsection headers and blank rows.
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Any, Tuple
import logging

logger = logging.getLogger(__name__)

class EnhancedDataCleaner:
    """Advanced data cleaner with subsection detection and removal."""
    
    def __init__(self):
        self.cleaning_report = {
            'operations_performed': [],
            'warnings': [],
            'errors': [],
            'subsection_headers_removed': [],
            'blank_rows_removed': 0,
            'currency_columns_detected': 0
        }
    
    def clean_financial_data(self, df: pd.DataFrame, 
                           type_inference: Dict[str, Any] = None) -> pd.DataFrame:
        """
        Clean financial data with subsection header removal and currency handling.
        
        Args:
            df: Input DataFrame
            type_inference: Type inference results for intelligent cleaning
            
        Returns:
            Cleaned DataFrame with subsection headers and blank rows removed
        """
        cleaned_df = df.copy()
        
        # Step 1: Identify and remove subsection headers
        subsection_rows = self._identify_subsection_headers(cleaned_df)
        if subsection_rows:
            cleaned_df = cleaned_df.drop(subsection_rows)
            self.cleaning_report['subsection_headers_removed'] = [
                cleaned_df.iloc[i].tolist() for i in subsection_rows
            ]
            logger.info(f"Removed {len(subsection_rows)} subsection header rows")
        
        # Step 2: Remove completely blank rows
        blank_rows = self._identify_blank_rows(cleaned_df)
        if blank_rows:
            cleaned_df = cleaned_df.drop(blank_rows)
            self.cleaning_report['blank_rows_removed'] = len(blank_rows)
            logger.info(f"Removed {len(blank_rows)} blank rows")
        
        # Step 3: Enhanced currency detection and cleaning
        currency_columns = self._detect_currency_columns(cleaned_df)
        for col in currency_columns:
            cleaned_df[col] = self._clean_currency_column(cleaned_df[col])
            self.cleaning_report['currency_columns_detected'] += 1
            self.cleaning_report['operations_performed'].append({
                'column': col,
                'operation': 'currency_cleaning',
                'values_cleaned': len(cleaned_df[col])
            })
        
        # Step 4: Numeric column cleaning
        numeric_columns = self._detect_numeric_columns(cleaned_df)
        for col in numeric_columns:
            cleaned_df[col] = self._clean_numeric_column(cleaned_df[col])
            self.cleaning_report['operations_performed'].append({
                'column': col,
                'operation': 'numeric_cleaning',
                'values_cleaned': len(cleaned_df[col])
            })
        
        # Step 5: Reset index after row removal
        cleaned_df = cleaned_df.reset_index(drop=True)
        
        return cleaned_df
    
    def _identify_subsection_headers(self, df: pd.DataFrame) -> List[int]:
        """
        Identify subsection header rows (rows with blank data columns).
        
        Args:
            df: Input DataFrame
            
        Returns:
            List of row indices to remove
        """
        subsection_rows = []
        
        for idx, row in df.iterrows():
            # Check if first column contains text (subsection header)
            first_col = str(row.iloc[0]).strip()
            
            # Check if all other columns are blank or NaN
            other_cols = row.iloc[1:]
            all_blank = all(pd.isna(val) or str(val).strip() == '' 
                           or str(val).strip() == '0' 
                           for val in other_cols)
            
            if first_col and all_blank:
                subsection_rows.append(idx)
                logger.debug(f"Identified subsection header: '{first_col}' at row {idx}")
        
        return subsection_rows
    
    def _identify_blank_rows(self, df: pd.DataFrame) -> List[int]:
        """
        Identify completely blank rows.
        
        Args:
            df: Input DataFrame
            
        Returns:
            List of row indices to remove
        """
        blank_rows = []
        
        for idx, row in df.iterrows():
            # Check if all columns are blank or NaN
            all_blank = all(pd.isna(val) or str(val).strip() == '' 
                           or str(val).strip() == '0'
                           for val in row)
            
            if all_blank:
                blank_rows.append(idx)
                logger.debug(f"Identified blank row at index {idx}")
        
        return blank_rows
    
    def _detect_currency_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Detect columns that contain currency values.
        
        Args:
            df: Input DataFrame
            
        Returns:
            List of column names containing currency data
        """
        currency_columns = []
        
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check for currency patterns
                currency_pattern = r'\$[\d,]+(?:\.\d{2})?'
                currency_count = sum(1 for val in df[col].dropna() 
                                   if re.search(currency_pattern, str(val)))
                
                total_count = len(df[col].dropna())
                
                if currency_count > 0 and currency_count / total_count > 0.5:
                    currency_columns.append(col)
                    logger.info(f"Detected currency column: {col} ({currency_count}/{total_count} values)")
        
        return currency_columns
    
    def _detect_numeric_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Detect columns that should be numeric.
        
        Args:
            df: Input DataFrame
            
        Returns:
            List of column names that should be numeric
        """
        numeric_columns = []
        
        for col in df.columns:
            if df[col].dtype == 'object':
                # Try to parse as numeric after cleaning
                try:
                    cleaned_values = []
                    for val in df[col].dropna():
                        cleaned_val = str(val).replace('$', '').replace(',', '')
                        if cleaned_val.strip():
                            float(cleaned_val)
                            cleaned_values.append(cleaned_val)
                    
                    if len(cleaned_values) > len(df[col].dropna()) * 0.8:
                        numeric_columns.append(col)
                        logger.info(f"Detected numeric column: {col}")
                        
                except (ValueError, TypeError):
                    continue
        
        return numeric_columns
    
    def _clean_currency_column(self, series: pd.Series) -> pd.Series:
        """
        Clean currency column by removing symbols and converting to numeric.
        
        Args:
            series: Input pandas Series
            
        Returns:
            Cleaned Series with numeric values
        """
        cleaned = series.astype(str).copy()
        
        # Remove currency symbols
        cleaned = cleaned.str.replace('$', '', regex=False)
        
        # Remove commas
        cleaned = cleaned.str.replace(',', '', regex=False)
        
        # Convert to numeric
        cleaned = pd.to_numeric(cleaned, errors='coerce')
        
        return cleaned
    
    def _clean_numeric_column(self, series: pd.Series) -> pd.Series:
        """
        Clean numeric column by removing commas and converting to numeric.
        
        Args:
            series: Input pandas Series
            
        Returns:
            Cleaned Series with numeric values
        """
        cleaned = series.astype(str).copy()
        
        # Remove commas
        cleaned = cleaned.str.replace(',', '', regex=False)
        
        # Convert to numeric
        cleaned = pd.to_numeric(cleaned, errors='coerce')
        
        return cleaned
    
    def get_cleaning_report(self) -> Dict[str, Any]:
        """Return comprehensive cleaning report."""
        return self.cleaning_report
    
    def validate_cleaned_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate cleaned data for quality assurance.
        
        Args:
            df: Cleaned DataFrame
            
        Returns:
            Validation report
        """
        validation_report = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'null_values': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.to_dict(),
            'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
            'subsection_headers_removed': len(self.cleaning_report['subsection_headers_removed']),
            'blank_rows_removed': self.cleaning_report['blank_rows_removed']
        }
        
        # Check for any remaining non-numeric data in numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                validation_report['warnings'] = validation_report.get('warnings', [])
                validation_report['warnings'].append(
                    f"Column '{col}' has {null_count} null values after cleaning"
                )
        
        return validation_report

# Example usage and testing
if __name__ == "__main__":
    import pandas as pd
    
    # Create sample financial data
    sample_data = {
        'Account': ['Assets', 'Cash & Equivalents', 'Short-Term Investments', 
                   'Receivables', 'Inventory', 'Total Curr. Assets'],
        '2022': ['', '$1,990', '19,218', '$4,650', '$2,605', '$28,829'],
        '2023': ['', '$3,389', '9,907', '$3,827', '$5,159', '$23,073'],
        '2024': ['', '$7,280', '18,704', '$9,999', '$5,282', '$44,345'],
        '2025': ['', '$8,589', '34,621', '$23,065', '$10,080', '$80,126']
    }
    
    df = pd.DataFrame(sample_data)
    
    # Clean the data
    cleaner = EnhancedDataCleaner()
    cleaned_df = cleaner.clean_financial_data(df)
    
    print("Original DataFrame:")
    print(df)
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    print("\nCleaning Report:")
    print(json.dumps(cleaner.get_cleaning_report(), indent=2))
