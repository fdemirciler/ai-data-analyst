# Copied from original location
import pandas as pd
import numpy as np
from typing import Dict, Any, List
import re


class TypeInferencer:
    """Type inference with confidence scoring."""

    def infer_types(self, df: pd.DataFrame) -> Dict[str, Any]:
        type_info = {}
        for column in df.columns:
            column_data = df[column]
            type_info[column] = self._infer_column_type(column_data, column)
        return type_info

    def _infer_column_type(self, series: pd.Series, column_name: str) -> Dict[str, Any]:
        clean_series = series.dropna()
        if len(clean_series) == 0:
            return {
                "detected_type": "empty",
                "confidence_score": 1.0,
                "original_dtype": str(series.dtype),
                "null_percentage": 100.0,
                "sample_values": [],
                "validation_rules": [],
            }
        null_percentage = (len(series) - len(clean_series)) / len(series) * 100
        type_detectors = [
            self._detect_currency,
            self._detect_percentage,
            self._detect_date,
            self._detect_id,
            self._detect_numeric,
            self._detect_categorical,
            self._detect_text,
        ]
        for detector in type_detectors:
            result = detector(clean_series, column_name)
            if result["confidence_score"] > 0.7:
                result.update(
                    {
                        "original_dtype": str(series.dtype),
                        "null_percentage": null_percentage,
                        "sample_values": clean_series.head().tolist(),
                        "validation_rules": self._get_validation_rules(
                            result["detected_type"]
                        ),
                    }
                )
                return result
        return {
            "detected_type": "text",
            "confidence_score": 0.8,
            "original_dtype": str(series.dtype),
            "null_percentage": null_percentage,
            "sample_values": clean_series.head().tolist(),
            "validation_rules": self._get_validation_rules("text"),
        }

    def _detect_currency(self, series: pd.Series, column_name: str) -> Dict[str, Any]:
        patterns = [
            r"^[$€£¥₹₽₩₪]?\d+(?:,\d{3})*(?:\.\d{2})?$",
            r"^\d+(?:,\d{3})*(?:\.\d{2})?[$€£¥₹₽₩₪]$",
            r"^[$€£¥₹₽₩₪]\s?\d+(?:,\d{3})*(?:\.\d{2})?$",
            r"^\d+(?:,\d{3})*(?:\.\d{2})?\s?[$€£¥₹₽₩₪]$",
        ]
        currency_keywords = [
            "price",
            "cost",
            "amount",
            "salary",
            "revenue",
            "budget",
            "currency",
        ]
        matches = 0
        total = len(series)
        for value in series.astype(str):
            value_str = str(value).strip()
            for pattern in patterns:
                if re.match(pattern, value_str):
                    matches += 1
                    break
        confidence = matches / total if total > 0 else 0
        name_lower = column_name.lower()
        if any(keyword in name_lower for keyword in currency_keywords):
            confidence = min(confidence + 0.2, 1.0)
        return {
            "detected_type": "currency",
            "confidence_score": confidence,
            "format": "currency",
            "currency_symbol": self._extract_currency_symbol(series),
        }

    def _detect_percentage(self, series: pd.Series, column_name: str) -> Dict[str, Any]:
        patterns = [
            r"^\d+(?:\.\d+)?%$",
            r"^\d+(?:\.\d+)?\s*%$",
            r"^\d+(?:\.\d+)?\s*percent$",
            r"^\d+(?:\.\d+)?\s*per\s*cent$",
        ]
        percentage_keywords = [
            "percentage",
            "rate",
            "ratio",
            "discount",
            "tax",
            "interest",
        ]
        matches = 0
        total = len(series)
        for value in series.astype(str):
            value_str = str(value).strip()
            for pattern in patterns:
                if re.match(pattern, value_str.lower()):
                    matches += 1
                    break
        confidence = matches / total if total > 0 else 0
        name_lower = column_name.lower()
        if any(keyword in name_lower for keyword in percentage_keywords):
            confidence = min(confidence + 0.2, 1.0)
        return {
            "detected_type": "percentage",
            "confidence_score": confidence,
            "format": "percentage",
        }

    def _detect_date(self, series: pd.Series, column_name: str) -> Dict[str, Any]:
        date_keywords = [
            "date",
            "time",
            "day",
            "month",
            "year",
            "timestamp",
            "created",
            "updated",
        ]
        try:
            if series.astype(str).str.fullmatch(r"[+-]?\d+(?:\.\d+)?").mean() == 1.0:
                return {"detected_type": "unknown", "confidence_score": 0.0}
            parsed_dates = pd.to_datetime(series, errors="coerce")
            valid_dates = parsed_dates.notna()
            if valid_dates.sum() > len(series) * 0.8:
                return {
                    "detected_type": "date",
                    "confidence_score": valid_dates.sum() / len(series),
                    "date_format": self._detect_date_format(series),
                    "timezone": "UTC",
                }
        except Exception:
            pass
        date_patterns = [
            r"^\d{4}-\d{2}-\d{2}$",
            r"^\d{2}/\d{2}/\d{4}$",
            r"^\d{2}-\d{2}-\d{4}$",
            r"^\d{2}\.\d{2}\.\d{4}$",
            r"^\d{1,2}\s+[A-Za-z]{3}\s+\d{4}$",
            r"^[A-Za-z]{3}\s+\d{1,2},\s+\d{4}$",
        ]
        matches = 0
        for value in series.astype(str):
            value_str = str(value).strip()
            for pattern in date_patterns:
                if re.match(pattern, value_str):
                    matches += 1
                    break
        confidence = matches / len(series) if len(series) > 0 else 0
        name_lower = column_name.lower()
        if any(keyword in name_lower for keyword in date_keywords):
            confidence = min(confidence + 0.3, 1.0)
        if confidence > 0.7:
            return {
                "detected_type": "date",
                "confidence_score": confidence,
                "date_format": "detected",
            }
        return {"detected_type": "unknown", "confidence_score": 0.0}

    def _detect_id(self, series: pd.Series, column_name: str) -> Dict[str, Any]:
        id_keywords = ["id", "key", "code", "identifier", "uuid", "index", "number"]
        unique_ratio = len(series.unique()) / len(series)
        id_patterns = [
            r"^\d+$",
            r"^[A-Z0-9]+$",
            r"^[a-z0-9]+$",
            r"^[A-Fa-f0-9]{8}-[A-Fa-f0-9]{4}-[A-Fa-f0-9]{4}-[A-Fa-f0-9]{4}-[A-Fa-f0-9]{12}$",
            r"^\d{4}-\d{4}-\d{4}-\d{4}$",
        ]
        matches = 0
        for value in series.astype(str):
            value_str = str(value).strip()
            for pattern in id_patterns:
                if re.match(pattern, value_str):
                    matches += 1
                    break
        confidence = 0.0
        if unique_ratio > 0.9 and any(
            keyword in column_name.lower() for keyword in id_keywords
        ):
            confidence = 0.9
        elif matches > len(series) * 0.8:
            confidence = 0.8
        elif unique_ratio > 0.8:
            confidence = 0.7
        return {
            "detected_type": "id",
            "confidence_score": confidence,
            "unique_ratio": unique_ratio,
            "is_primary_key": unique_ratio == 1.0,
        }

    def _detect_numeric(self, series: pd.Series, column_name: str) -> Dict[str, Any]:
        try:
            numeric_series = pd.to_numeric(series, errors="coerce")
            valid_numeric = numeric_series.notna()
            if valid_numeric.sum() > len(series) * 0.8:
                is_integer = (
                    numeric_series.dropna() == numeric_series.dropna().astype(int)
                ).all()
                return {
                    "detected_type": "integer" if is_integer else "float",
                    "confidence_score": valid_numeric.sum() / len(series),
                    "numeric_type": "integer" if is_integer else "float",
                    "min_value": float(numeric_series.min()),
                    "max_value": float(numeric_series.max()),
                    "mean_value": float(numeric_series.mean()),
                }
        except Exception:
            pass
        return {"detected_type": "unknown", "confidence_score": 0.0}

    def _detect_categorical(
        self, series: pd.Series, column_name: str
    ) -> Dict[str, Any]:
        unique_ratio = len(series.unique()) / len(series)
        if unique_ratio <= 0.1 and len(series.unique()) <= 50:
            return {
                "detected_type": "categorical",
                "confidence_score": 0.9,
                "unique_values": len(series.unique()),
                "categories": series.unique().tolist(),
            }
        return {"detected_type": "unknown", "confidence_score": 0.0}

    def _detect_text(self, series: pd.Series, column_name: str) -> Dict[str, Any]:
        return {
            "detected_type": "text",
            "confidence_score": 0.8,
            "text_length_mean": series.astype(str).str.len().mean(),
            "text_length_max": series.astype(str).str.len().max(),
        }

    def _extract_currency_symbol(self, series: pd.Series) -> str:
        currency_symbols = ["$", "€", "£", "¥", "₹", "₽", "₩", "₪"]
        for value in series.astype(str):
            for symbol in currency_symbols:
                if symbol in str(value):
                    return symbol
        return "$"

    def _detect_date_format(self, series: pd.Series) -> str:
        date_formats = [
            "%Y-%m-%d",
            "%d/%m/%Y",
            "%m/%d/%Y",
            "%d-%m-%Y",
            "%m-%d-%Y",
            "%Y.%m.%d",
            "%d.%m.%Y",
            "%d %b %Y",
            "%b %d, %Y",
        ]
        for fmt in date_formats:
            try:
                pd.to_datetime(series, format=fmt, errors="raise")
                return fmt
            except:
                continue
        return "auto-detect"

    def _get_validation_rules(self, detected_type: str) -> List[str]:
        rules_map = {
            "currency": ["numeric_format", "currency_symbol"],
            "percentage": ["numeric_format", "percentage_symbol"],
            "date": ["date_format", "chronological_order"],
            "id": ["unique_values", "format_pattern"],
            "integer": ["numeric_format", "integer_constraint"],
            "float": ["numeric_format"],
            "categorical": ["category_values", "no_nulls"],
            "text": ["string_format", "length_constraints"],
        }
        return rules_map.get(detected_type, ["basic_validation"])
