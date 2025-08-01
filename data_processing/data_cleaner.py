import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import re
import logging
from datetime import datetime

class DataCleaner:
    """
    Type-aware data cleaning engine with configurable strategies.
    Handles various data quality issues with best-effort recovery.
    """
    
    def __init__(self):
        self.logger = logging.getLogger('data_processing')
        self.cleaning_report = {
            'operations_performed': [],
            'issues_found': [],
            'corrections_made': [],
            'warnings': []
        }
        
    def clean_dataframe(self, df: pd.DataFrame, type_info: Dict[str, Any]) -> pd.DataFrame:
        """Main cleaning method with type-aware strategies."""
        
        cleaned_df = df.copy()
        
        for column in cleaned_df.columns:
            if column in type_info:
                column_type = type_info[column]
                cleaned_df[column] = self._clean_column(
                    cleaned_df[column], 
                    column_type,
                    column
                )
        
        return cleaned_df
    
    def _clean_column(self, series: pd.Series, type_info: Dict[str, Any], 
                     column_name: str) -> pd.Series:
        """Clean individual column based on detected type."""
        
        detected_type = type_info.get('detected_type', 'text')
        
        cleaning_strategies = {
            'currency': self._clean_currency,
            'percentage': self._clean_percentage,
            'date': self._clean_date,
            'id': self._clean_id,
            'integer': self._clean_numeric,
            'float': self._clean_numeric,
            'categorical': self._clean_categorical,
            'text': self._clean_text
        }
        
        cleaner = cleaning_strategies.get(detected_type, self._clean_text)
        return cleaner(series, type_info, column_name)
    
    def _clean_currency(self, series: pd.Series, type_info: Dict[str, Any], 
                       column_name: str) -> pd.Series:
        """Clean currency values."""
        
        def extract_numeric(value):
            if pd.isna(value):
                return np.nan
            
            value_str = str(value).strip()
            
            # Remove currency symbols and commas
            cleaned = re.sub(r'[$€£¥₹₽₩₪,]', '', value_str)
            cleaned = re.sub(r'\s+', '', cleaned)
            
            try:
                return float(cleaned)
            except (ValueError, TypeError):
                self.cleaning_report['warnings'].append({
                    'column': column_name,
                    'issue': f"Cannot convert '{value}' to numeric",
                    'action': 'set_to_nan'
                })
                return np.nan
        
        cleaned_series = series.apply(extract_numeric)
        
        self.cleaning_report['operations_performed'].append({
            'column': column_name,
            'operation': 'currency_cleaning',
            'values_cleaned': len(series) - cleaned_series.isna().sum()
        })
        
        return cleaned_series
    
    def _clean_percentage(self, series: pd.Series, type_info: Dict[str, Any], 
                         column_name: str) -> pd.Series:
        """Clean percentage values."""
        
        def extract_percentage(value):
            if pd.isna(value):
                return np.nan
            
            value_str = str(value).strip()
            
            # Remove percentage symbols and convert to decimal
            cleaned = re.sub(r'[%\s]', '', value_str)
            
            try:
                return float(cleaned) / 100
            except (ValueError, TypeError):
                self.cleaning_report['warnings'].append({
                    'column': column_name,
                    'issue': f"Cannot convert '{value}' to percentage",
                    'action': 'set_to_nan'
                })
                return np.nan
        
        cleaned_series = series.apply(extract_percentage)
        
        self.cleaning_report['operations_performed'].append({
            'column': column_name,
            'operation': 'percentage_cleaning',
            'values_cleaned': len(series) - cleaned_series.isna().sum()
        })
        
        return cleaned_series
    
    def _clean_date(self, series: pd.Series, type_info: Dict[str, Any], 
                   column_name: str) -> pd.Series:
        """Clean date values with format detection."""
        
        def parse_date(value):
            if pd.isna(value):
                return pd.NaT
            
            value_str = str(value).strip()
            
            # Try pandas to_datetime
            try:
                parsed_date = pd.to_datetime(value_str, errors='coerce')
                if pd.notna(parsed_date):
                    return parsed_date
            except:
                pass
            
            # Try common date formats
            date_formats = [
                '%Y-%m-%d',
                '%d/%m/%Y',
                '%m/%d/%Y',
                '%d-%m-%Y',
                '%m-%d-%Y',
                '%Y.%m.%d',
                '%d.%m.%Y',
                '%d %b %Y',
                '%b %d, %Y',
                '%Y%m%d',
                '%d%m%Y'
            ]
            
            for fmt in date_formats:
                try:
                    return pd.to_datetime(value_str, format=fmt)
                except:
                    continue
            
            self.cleaning_report['warnings'].append({
                'column': column_name,
                'issue': f"Cannot parse date '{value}'",
                'action': 'set_to_nat'
            })
            return pd.NaT
        
        cleaned_series = series.apply(parse_date)
        
        self.cleaning_report['operations_performed'].append({
            'column': column_name,
            'operation': 'date_cleaning',
            'values_cleaned': len(series) - cleaned_series.isna().sum()
        })
        
        return cleaned_series
    
    def _clean_id(self, series: pd.Series, type_info: Dict[str, Any], 
                 column_name: str) -> pd.Series:
        """Clean ID values - minimal cleaning, preserve format."""
        
        def clean_id(value):
            if pd.isna(value):
                return np.nan
            
            value_str = str(value).strip()
            
            # Remove extra whitespace
            cleaned = re.sub(r'\s+', ' ', value_str).strip()
            
            return cleaned
        
        cleaned_series = series.apply(clean_id)
        
        self.cleaning_report['operations_performed'].append({
            'column': column_name,
            'operation': 'id_cleaning',
            'values_cleaned': len(series)
        })
        
        return cleaned_series
    
    def _clean_numeric(self, series: pd.Series, type_info: Dict[str, Any], 
                      column_name: str) -> pd.Series:
        """Clean numeric values."""
        
        def extract_numeric(value):
            if pd.isna(value):
                return np.nan
            
            value_str = str(value).strip()
            
            # Remove common formatting
            cleaned = re.sub(r'[,\s]', '', value_str)
            
            try:
                if type_info.get('detected_type') == 'integer':
                    return int(float(cleaned))
                else:
                    return float(cleaned)
            except (ValueError, TypeError):
                self.cleaning_report['warnings'].append({
                    'column': column_name,
                    'issue': f"Cannot convert '{value}' to numeric",
                    'action': 'set_to_nan'
                })
                return np.nan
        
        cleaned_series = series.apply(extract_numeric)
        
        self.cleaning_report['operations_performed'].append({
            'column': column_name,
            'operation': 'numeric_cleaning',
            'values_cleaned': len(series) - cleaned_series.isna().sum()
        })
        
        return cleaned_series
    
    def _clean_categorical(self, series: pd.Series, type_info: Dict[str, Any], 
                          column_name: str) -> pd.Series:
        """Clean categorical values."""
        
        def clean_category(value):
            if pd.isna(value):
                return np.nan
            
            value_str = str(value).strip()
            
            # Standardize case
            cleaned = value_str.title()
            
            # Remove extra spaces
            cleaned = re.sub(r'\s+', ' ', cleaned).strip()
            
            return cleaned
        
        cleaned_series = series.apply(clean_category)
        
        # Standardize categories
        unique_categories = cleaned_series.dropna().unique()
        category_mapping = {cat: cat.strip() for cat in unique_categories}
        
        cleaned_series = cleaned_series.map(category_mapping)
        
        self.cleaning_report['operations_performed'].append({
            'column': column_name,
            'operation': 'categorical_cleaning',
            'values_cleaned': len(series)
        })
        
        return cleaned_series
    
    def _clean_text(self, series: pd.Series, type_info: Dict[str, Any], 
                   column_name: str) -> pd.Series:
        """Clean text values."""
        
        def clean_text(value):
            if pd.isna(value):
                return np.nan
            
            value_str = str(value).strip()
            
            # Remove HTML tags
            cleaned = re.sub(r'<[^>]+>', '', value_str)
            
            # Normalize unicode
            cleaned = cleaned.encode('utf-8').decode('utf-8', 'ignore')
            
            # Remove extra whitespace
            cleaned = re.sub(r'\s+', ' ', cleaned).strip()
            
            return cleaned
        
        cleaned_series = series.apply(clean_text)
        
        self.cleaning_report['operations_performed'].append({
            'column': column_name,
            'operation': 'text_cleaning',
            'values_cleaned': len(series)
        })
        
        return cleaned_series
    
    def get_cleaning_report(self) -> Dict[str, Any]:
        """Get comprehensive cleaning report."""
        
        report = self.cleaning_report.copy()
        
        # Calculate summary statistics
        total_operations = len(report['operations_performed'])
        total_warnings = len(report['warnings'])
        
        report['summary'] = {
            'total_operations': total_operations,
            'total_warnings': total_warnings,
            'cleaning_success_rate': (total_operations - total_warnings) / total_operations if total_operations > 0 else 1.0
        }
        
        return report
