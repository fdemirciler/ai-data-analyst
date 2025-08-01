import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from scipy import stats
import logging
from datetime import datetime

class DataProfiler:
    """
    Comprehensive data profiling engine with quality metrics.
    Provides statistical analysis and quality scoring.
    """
    
    def __init__(self):
        self.logger = logging.getLogger('data_processing')
        
    def profile_dataframe(self, df: pd.DataFrame, type_info: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive profile for dataframe."""
        
        profile = {
            'dataset_info': self._get_dataset_info(df),
            'columns': {},
            'quality_metrics': {},
            'correlations': {},
            'outliers': {}
        }
        
        for column in df.columns:
            if column in type_info:
                profile['columns'][column] = self._profile_column(
                    df[column], 
                    type_info[column]
                )
        
        profile['quality_metrics'] = self._calculate_quality_metrics(df, type_info)
        profile['correlations'] = self._calculate_correlations(df)
        profile['outliers'] = self._detect_outliers(df)
        
        return profile
    
    def _get_dataset_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get basic dataset information."""
        
        return {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
            'dtypes': df.dtypes.to_dict(),
            'null_counts': df.isnull().sum().to_dict(),
            'null_percentages': (df.isnull().sum() / len(df) * 100).to_dict()
        }
    
    def _profile_column(self, series: pd.Series, type_info: Dict[str, Any]) -> Dict[str, Any]:
        """Generate detailed profile for single column."""
        
        profile = {
            'name': series.name,
            'detected_type': type_info.get('detected_type', 'unknown'),
            'original_dtype': str(series.dtype),
            'null_count': series.isnull().sum(),
            'null_percentage': series.isnull().sum() / len(series) * 100,
            'unique_count': series.nunique(),
            'unique_percentage': series.nunique() / len(series) * 100,
            'most_frequent_values': self._get_most_frequent_values(series),
            'completeness_score': (len(series) - series.isnull().sum()) / len(series) * 100,
            'uniqueness_score': self._calculate_uniqueness_score(series)
        }
        
        # Type-specific statistics
        detected_type = type_info.get('detected_type', 'unknown')
        
        if detected_type in ['integer', 'float', 'currency', 'percentage']:
            profile['statistics'] = self._get_numeric_statistics(series)
        elif detected_type == 'date':
            profile['statistics'] = self._get_date_statistics(series)
        elif detected_type == 'categorical':
            profile['statistics'] = self._get_categorical_statistics(series)
        else:
            profile['statistics'] = self._get_text_statistics(series)
        
        return profile
    
    def _get_most_frequent_values(self, series: pd.Series, top_n: int = 5) -> List[Dict[str, Any]]:
        """Get most frequent values in series."""
        
        value_counts = series.value_counts()
        
        return [
            {
                'value': str(value),
                'count': int(count),
                'percentage': float(count / len(series) * 100)
            }
            for value, count in value_counts.head(top_n).items()
        ]
    
    def _calculate_uniqueness_score(self, series: pd.Series) -> float:
        """Calculate uniqueness score for series."""
        
        unique_ratio = series.nunique() / len(series)
        
        # Score based on uniqueness ratio
        if unique_ratio == 1.0:
            return 100.0  # Perfectly unique
        elif unique_ratio >= 0.9:
            return 90.0
        elif unique_ratio >= 0.7:
            return 70.0
        elif unique_ratio >= 0.5:
            return 50.0
        else:
            return unique_ratio * 100  # Scale lower values
    
    def _get_numeric_statistics(self, series: pd.Series) -> Dict[str, Any]:
        """Get numeric statistics for series."""
        
        clean_series = series.dropna()
        
        if len(clean_series) == 0:
            return {'error': 'No valid numeric values'}
        
        stats = {
            'min': float(clean_series.min()),
            'max': float(clean_series.max()),
            'mean': float(clean_series.mean()),
            'median': float(clean_series.median()),
            'mode': float(clean_series.mode().iloc[0]) if not clean_series.mode().empty else None,
            'std': float(clean_series.std()),
            'variance': float(clean_series.var()),
            'q1': float(clean_series.quantile(0.25)),
            'q3': float(clean_series.quantile(0.75)),
            'iqr': float(clean_series.quantile(0.75) - clean_series.quantile(0.25)),
            'skewness': float(clean_series.skew()),
            'kurtosis': float(clean_series.kurtosis())
        }
        
        return stats
    
    def _get_date_statistics(self, series: pd.Series) -> Dict[str, Any]:
        """Get date statistics for series."""
        
        clean_series = series.dropna()
        
        if len(clean_series) == 0:
            return {'error': 'No valid date values'}
        
        stats = {
            'min_date': str(clean_series.min()),
            'max_date': str(clean_series.max()),
            'date_range_days': int((clean_series.max() - clean_series.min()).days),
            'most_recent': str(clean_series.max()),
            'oldest': str(clean_series.min()),
            'unique_dates': int(clean_series.nunique())
        }
        
        return stats
    
    def _get_categorical_statistics(self, series: pd.Series) -> Dict[str, Any]:
        """Get categorical statistics for series."""
        
        clean_series = series.dropna()
        
        if len(clean_series) == 0:
            return {'error': 'No valid categorical values'}
        
        value_counts = clean_series.value_counts()
        
        stats = {
            'categories': len(value_counts),
            'most_common': str(value_counts.index[0]) if len(value_counts) > 0 else None,
            'least_common': str(value_counts.index[-1]) if len(value_counts) > 0 else None,
            'imbalance_ratio': float(value_counts.iloc[0] / value_counts.iloc[-1]) if len(value_counts) > 1 else 1.0,
            'categories_list': value_counts.index.tolist()
        }
        
        return stats
    
    def _get_text_statistics(self, series: pd.Series) -> Dict[str, Any]:
        """Get text statistics for series."""
        
        clean_series = series.dropna()
        
        if len(clean_series) == 0:
            return {'error': 'No valid text values'}
        
        # Text analysis
        text_lengths = clean_series.astype(str).str.len()
        
        stats = {
            'min_length': int(text_lengths.min()),
            'max_length': int(text_lengths.max()),
            'mean_length': float(text_lengths.mean()),
            'median_length': float(text_lengths.median()),
            'total_characters': int(text_lengths.sum()),
            'empty_strings': int((clean_series.astype(str) == '').sum())
        }
        
        return stats
    
    def _calculate_quality_metrics(self, df: pd.DataFrame, type_info: Dict[str, Any]) -> Dict[str, float]:
        """Calculate comprehensive quality metrics."""
        
        metrics = {
            'completeness_score': 0.0,
            'uniqueness_score': 0.0,
            'validity_score': 0.0,
            'consistency_score': 0.0,
            'overall_score': 0.0
        }
        
        # Completeness score
        total_cells = df.size
        null_cells = df.isnull().sum().sum()
        metrics['completeness_score'] = (total_cells - null_cells) / total_cells * 100
        
        # Uniqueness score
        uniqueness_scores = []
        for column in df.columns:
            if column in type_info:
                uniqueness_scores.append(
                    self._calculate_uniqueness_score(df[column])
                )
        
        if uniqueness_scores:
            metrics['uniqueness_score'] = np.mean(uniqueness_scores)
        
        # Validity score (based on type detection confidence)
        validity_scores = []
        for column, type_data in type_info.items():
            if column in df.columns:
                confidence = type_data.get('confidence_score', 0.5)
                validity_scores.append(confidence * 100)
        
        if validity_scores:
            metrics['validity_score'] = np.mean(validity_scores)
        
        # Consistency score (based on data consistency)
        metrics['consistency_score'] = self._calculate_consistency_score(df)
        
        # Overall score (weighted average)
        weights = {
            'completeness': 0.3,
            'uniqueness': 0.2,
            'validity': 0.3,
            'consistency': 0.2
        }
        
        overall_score = (
            metrics['completeness_score'] * weights['completeness'] +
            metrics['uniqueness_score'] * weights['uniqueness'] +
            metrics['validity_score'] * weights['validity'] +
            metrics['consistency_score'] * weights['consistency']
        )
        
        metrics['overall_score'] = max(0, min(100, overall_score))
        
        return metrics
    
    def _calculate_consistency_score(self, df: pd.DataFrame) -> float:
        """Calculate data consistency score."""
        
        consistency_issues = 0
        total_checks = 0
        
        # Check for consistent data types across columns
        for column in df.columns:
            total_checks += 1
            
            # Check for mixed data types
            if df[column].dtype == 'object':
                types = df[column].dropna().apply(type).unique()
                if len(types) > 1:
                    consistency_issues += 1
        
        consistency_score = max(0, 100 - (consistency_issues / max(total_checks, 1) * 100))
        
        return consistency_score
    
    def _calculate_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate correlation matrix for numeric columns."""
        
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            return {'error': 'No numeric columns for correlation analysis'}
        
        correlation_matrix = numeric_df.corr()
        
        # Find strong correlations
        strong_correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) > 0.7:  # Strong correlation threshold
                    strong_correlations.append({
                        'column1': correlation_matrix.columns[i],
                        'column2': correlation_matrix.columns[j],
                        'correlation': float(corr_value)
                    })
        
        return {
            'correlation_matrix': correlation_matrix.to_dict(),
            'strong_correlations': strong_correlations,
            'high_correlation_count': len(strong_correlations)
        }
    
    def _detect_outliers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect outliers in numeric columns."""
        
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            return {'error': 'No numeric columns for outlier detection'}
        
        outliers = {}
        
        for column in numeric_df.columns:
            series = numeric_df[column]
            
            # IQR method
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outlier_mask = (series < lower_bound) | (series > upper_bound)
            outlier_count = outlier_mask.sum()
            
            if outlier_count > 0:
                outliers[column] = {
                    'outlier_count': int(outlier_count),
                    'outlier_percentage': float(outlier_count / len(series) * 100),
                    'lower_bound': float(lower_bound),
                    'upper_bound': float(upper_bound),
                    'outlier_values': series[outlier_mask].tolist()
                }
        
        return {
            'outlier_columns': outliers,
            'total_outliers': sum(col['outlier_count'] for col in outliers.values()),
            'columns_with_outliers': len(outliers)
        }
    
    def calculate_quality_scores(self, cleaned_data: Dict[str, Any], 
                               cleaning_report: Dict[str, Any]) -> Dict[str, float]:
        """Calculate final quality scores based on profiling and cleaning."""
        
        scores = {
            'completeness_score': 0.0,
            'uniqueness_score': 0.0,
            'validity_score': 0.0,
            'consistency_score': 0.0,
            'overall_score': 0.0
        }
        
        # Base scores from data quality
        if isinstance(cleaned_data, pd.DataFrame):
            scores['completeness_score'] = (1 - cleaned_data.isnull().sum().sum() / cleaned_data.size) * 100
            scores['uniqueness_score'] = np.mean([cleaned_data[col].nunique() / len(cleaned_data) * 100 
                                                for col in cleaned_data.columns])
        
        # Adjust scores based on cleaning report
        cleaning_warnings = cleaning_report.get('warnings', [])
        if cleaning_warnings:
            penalty = min(len(cleaning_warnings) * 2, 20)  # Max 20% penalty
            scores['validity_score'] = max(0, 100 - penalty)
        else:
            scores['validity_score'] = 100.0
        
        # Calculate overall weighted score
        weights = {
            'completeness': 0.3,
            'uniqueness': 0.2,
            'validity': 0.3,
            'consistency': 0.2
        }
        
        overall_score = (
            scores['completeness_score'] * weights['completeness'] +
            scores['uniqueness_score'] * weights['uniqueness'] +
            scores['validity_score'] * weights['validity'] +
            scores['consistency_score'] * weights['consistency']
        )
        
        scores['overall_score'] = max(0, min(100, overall_score))
        
        return scores
