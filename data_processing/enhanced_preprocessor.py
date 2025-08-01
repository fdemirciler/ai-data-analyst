import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import chardet
import io
from pathlib import Path
import json
import logging
from datetime import datetime
import re
from .type_inference import TypeInferencer
from .layout_detector import LayoutDetector
from .data_cleaner import DataCleaner
from .data_profiler import DataProfiler
from ..monitoring.processing_monitor import ProcessingMonitor

class EnhancedDataPreprocessor:
    """
    Production-ready data preprocessing engine with real-time processing,
    comprehensive monitoring, and best-effort error handling.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger('data_processing')
        self.type_inferencer = TypeInferencer()
        self.layout_detector = LayoutDetector()
        self.data_cleaner = DataCleaner()
        self.data_profiler = DataProfiler()
        
    def process_file(self, file_path: str, file_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main entry point for real-time file processing.
        Processes file immediately upon upload with comprehensive monitoring.
        """
        monitor = ProcessingMonitor()
        
        try:
            # Initialize processing
            metadata = monitor.start_processing(file_info)
            start_time = time.time()
            
            # Step 1: File ingestion and validation
            ingestion_result = self._ingest_file(file_path, file_info, monitor)
            
            # Step 2: CSV/Excel specific processing
            if file_info.get('file_type') == 'csv':
                processed_data = self._process_csv(ingestion_result, monitor)
            elif file_info.get('file_type') in ['xlsx', 'xls']:
                processed_data = self._process_excel(ingestion_result, monitor)
            else:
                raise ValueError(f"Unsupported file type: {file_info.get('file_type')}")
            
            # Step 3: Data profiling and quality assessment
            profiled_data = self._profile_data(processed_data, monitor)
            
            # Step 4: Data cleaning and standardization
            cleaned_data = self._clean_data(profiled_data, monitor)
            
            # Step 5: Final quality assessment
            quality_scores = self._assess_quality(cleaned_data, monitor)
            
            # Finalize processing
            final_report = monitor.finalize_processing(start_time, quality_scores)
            
            return {
                'status': 'success',
                'data': cleaned_data,
                'metadata': final_report['metadata'],
                'quality': quality_scores,
                'audit': final_report['audit']
            }
            
        except Exception as e:
            self.logger.error(f"Processing failed: {str(e)}")
            return {
                'status': 'failed',
                'error': str(e),
                'metadata': metadata,
                'audit': monitor.finalize_processing(start_time, {'overall_score': 0})
            }
    
    def _ingest_file(self, file_path: str, file_info: Dict[str, Any], 
                    monitor: ProcessingMonitor) -> Dict[str, Any]:
        """File ingestion with validation and encoding detection."""
        
        node_data = monitor.start_node('file_ingestion', file_info)
        
        try:
            file_path = Path(file_path)
            
            # File validation
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            file_size = file_path.stat().st_size
            if file_size > 50 * 1024 * 1024:  # 50MB limit
                raise ValueError(f"File too large: {file_size / 1024 / 1024:.1f}MB")
            
            # Encoding detection for CSV files
            encoding = 'utf-8'
            if file_info.get('file_type') == 'csv':
                encoding = self._detect_encoding(file_path)
            
            result = {
                'file_path': str(file_path),
                'encoding': encoding,
                'file_size': file_size,
                'validation_passed': True
            }
            
            monitor.complete_node('file_ingestion', node_data, result)
            return result
            
        except Exception as e:
            monitor.complete_node('file_ingestion', node_data, error=e)
            raise
    
    def _detect_encoding(self, file_path: Path) -> str:
        """Detect file encoding using chardet."""
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read(10000)  # Read first 10KB
                result = chardet.detect(raw_data)
                return result.get('encoding', 'utf-8')
        except Exception:
            return 'utf-8'
    
    def _process_csv(self, ingestion_result: Dict[str, Any], 
                    monitor: ProcessingMonitor) -> Dict[str, Any]:
        """Enhanced CSV processing with delimiter detection and type inference."""
        
        node_data = monitor.start_node('csv_processing', ingestion_result)
        
        try:
            file_path = ingestion_result['file_path']
            encoding = ingestion_result['encoding']
            
            # Delimiter detection
            delimiter = self._detect_delimiter(file_path, encoding)
            
            # Read CSV with detected parameters
            df = pd.read_csv(
                file_path,
                delimiter=delimiter,
                encoding=encoding,
                low_memory=False,
                on_bad_lines='warn'
            )
            
            # Type inference
            type_inference = self.type_inferencer.infer_types(df)
            
            # Layout detection
            layout_info = self.layout_detector.detect_layout(df)
            
            result = {
                'data': df,
                'delimiter': delimiter,
                'encoding': encoding,
                'type_inference': type_inference,
                'layout_info': layout_info,
                'original_shape': df.shape
            }
            
            monitor.complete_node('csv_processing', node_data, result)
            return result
            
        except Exception as e:
            monitor.complete_node('csv_processing', node_data, error=e)
            raise
    
    def _process_excel(self, ingestion_result: Dict[str, Any], 
                      monitor: ProcessingMonitor) -> Dict[str, Any]:
        """Enhanced Excel processing with multi-sheet support."""
        
        node_data = monitor.start_node('excel_processing', ingestion_result)
        
        try:
            file_path = ingestion_result['file_path']
            
            # Read Excel file
            excel_file = pd.ExcelFile(file_path)
            
            # Process all sheets
            sheets_data = {}
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                
                # Type inference for each sheet
                type_inference = self.type_inferencer.infer_types(df)
                
                # Layout detection
                layout_info = self.layout_detector.detect_layout(df)
                
                sheets_data[sheet_name] = {
                    'data': df,
                    'type_inference': type_inference,
                    'layout_info': layout_info,
                    'original_shape': df.shape
                }
            
            result = {
                'sheets_data': sheets_data,
                'sheet_names': excel_file.sheet_names,
                'primary_sheet': excel_file.sheet_names[0] if sheets_data else None
            }
            
            monitor.complete_node('excel_processing', node_data, result)
            return result
            
        except Exception as e:
            monitor.complete_node('excel_processing', node_data, error=e)
            raise
    
    def _detect_delimiter(self, file_path: str, encoding: str) -> str:
        """Detect CSV delimiter with fallback options."""
        delimiters = [',', ';', '\t', '|', ' ']
        
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                sample = f.read(1024)
                
            # Count occurrences of each delimiter
            delimiter_counts = {}
            for delim in delimiters:
                count = sample.count(delim)
                if count > 0:
                    delimiter_counts[delim] = count
            
            # Return most common delimiter
            if delimiter_counts:
                return max(delimiter_counts, key=delimiter_counts.get)
            
            return ','  # Default fallback
            
        except Exception:
            return ','
    
    def _profile_data(self, processed_data: Dict[str, Any], 
                     monitor: ProcessingMonitor) -> Dict[str, Any]:
        """Comprehensive data profiling with quality metrics."""
        
        node_data = monitor.start_node('data_profiling', processed_data)
        
        try:
            if 'sheets_data' in processed_data:
                # Excel multi-sheet profiling
                profiles = {}
                for sheet_name, sheet_data in processed_data['sheets_data'].items():
                    profile = self.data_profiler.profile_dataframe(
                        sheet_data['data'],
                        sheet_data['type_inference']
                    )
                    profiles[sheet_name] = profile
                    
                result = {
                    'profiles': profiles,
                    'primary_profile': profiles.get(processed_data['primary_sheet'])
                }
            else:
                # Single CSV profiling
                profile = self.data_profiler.profile_dataframe(
                    processed_data['data'],
                    processed_data['type_inference']
                )
                result = {'profile': profile}
            
            monitor.complete_node('data_profiling', node_data, result)
            return result
            
        except Exception as e:
            monitor.complete_node('data_profiling', node_data, error=e)
            raise
    
    def _clean_data(self, profiled_data: Dict[str, Any], 
                   monitor: ProcessingMonitor) -> Dict[str, Any]:
        """Type-aware data cleaning with best-effort error handling."""
        
        node_data = monitor.start_node('data_cleaning', profiled_data)
        
        try:
            cleaned_data = {}
            
            if 'sheets_data' in profiled_data:
                # Multi-sheet cleaning
                for sheet_name, data_info in profiled_data['sheets_data'].items():
                    if 'data' in data_info:
                        cleaned_df = self.data_cleaner.clean_dataframe(
                            data_info['data'],
                            data_info.get('type_inference', {})
                        )
                        cleaned_data[sheet_name] = cleaned_df
            else:
                # Single CSV cleaning
                if 'data' in profiled_data:
                    cleaned_df = self.data_cleaner.clean_dataframe(
                        profiled_data['data'],
                        profiled_data.get('type_inference', {})
                    )
                    cleaned_data = cleaned_df
            
            result = {
                'cleaned_data': cleaned_data,
                'cleaning_report': self.data_cleaner.get_cleaning_report()
            }
            
            monitor.complete_node('data_cleaning', node_data, result)
            return result
            
        except Exception as e:
            monitor.complete_node('data_cleaning', node_data, error=e)
            raise
    
    def _assess_quality(self, cleaned_data: Dict[str, Any], 
                       monitor: ProcessingMonitor) -> Dict[str, float]:
        """Final quality assessment with scoring."""
        
        node_data = monitor.start_node('quality_assessment', cleaned_data)
        
        try:
            quality_scores = self.data_profiler.calculate_quality_scores(
                cleaned_data.get('cleaned_data', {}),
                cleaned_data.get('cleaning_report', {})
            )
            
            monitor.complete_node('quality_assessment', node_data, quality_scores)
            return quality_scores
            
        except Exception as e:
            monitor.complete_node('quality_assessment', node_data, error=e)
            return {
                'completeness_score': 0,
                'uniqueness_score': 0,
                'validity_score': 0,
                'consistency_score': 0,
                'overall_score': 0
            }

# Global preprocessor instance
preprocessor = EnhancedDataPreprocessor()

def process_file_realtime(file_path: str, file_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Real-time file processing entry point.
    Called immediately when file is uploaded.
    """
    return preprocessor.process_file(file_path, file_info)
