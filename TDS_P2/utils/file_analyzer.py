import pandas as pd
import json
from pathlib import Path
from typing import Dict, Any, List
import logging
import mimetypes

logger = logging.getLogger(__name__)

class FileAnalyzer:
    """Analyze uploaded files to understand their structure and content"""
    
    def __init__(self):
        self.max_preview_rows = 5
        self.max_preview_size = 1000  # characters
    
    async def analyze(self, file_path: Path) -> Dict[str, Any]:
        """Analyze a file and return metadata"""
        try:
            metadata = {
                "filename": file_path.name,
                "size": file_path.stat().st_size,
                "type": self._detect_file_type(file_path),
                "mime_type": mimetypes.guess_type(str(file_path))[0]
            }
            
            # Type-specific analysis
            file_type = metadata["type"]
            
            if file_type == "csv":
                csv_data = await self._analyze_csv(file_path)
                metadata.update(csv_data)
            
            elif file_type == "json":
                json_data = await self._analyze_json(file_path)
                metadata.update(json_data)
            
            elif file_type == "excel":
                excel_data = await self._analyze_excel(file_path)
                metadata.update(excel_data)
            
            elif file_type == "text":
                text_data = await self._analyze_text(file_path)
                metadata.update(text_data)
            
            elif file_type == "parquet":
                parquet_data = await self._analyze_parquet(file_path)
                metadata.update(parquet_data)
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error analyzing file {file_path}: {str(e)}")
            return {
                "filename": file_path.name,
                "size": file_path.stat().st_size if file_path.exists() else 0,
                "type": "unknown",
                "error": str(e)
            }
    
    def _detect_file_type(self, file_path: Path) -> str:
        """Detect file type from extension"""
        suffix = file_path.suffix.lower()
        
        type_mapping = {
            '.csv': 'csv',
            '.json': 'json',
            '.xlsx': 'excel',
            '.xls': 'excel',
            '.parquet': 'parquet',
            '.txt': 'text',
            '.md': 'text',
            '.png': 'image',
            '.jpg': 'image',
            '.jpeg': 'image',
            '.pdf': 'pdf',
            '.html': 'html',
            '.xml': 'xml'
        }
        
        return type_mapping.get(suffix, 'unknown')
    
    async def _analyze_csv(self, file_path: Path) -> Dict[str, Any]:
        """Analyze CSV file"""
        try:
            # Read first few rows to understand structure
            df = pd.read_csv(file_path, nrows=100)
            
            preview_df = df.head(self.max_preview_rows)
            preview = preview_df.to_string(max_cols=10)
            
            return {
                "rows": len(df),
                "columns": list(df.columns),
                "column_count": len(df.columns),
                "dtypes": df.dtypes.astype(str).to_dict(),
                "preview": preview[:self.max_preview_size],
                "missing_values": df.isnull().sum().to_dict(),
                "memory_usage": df.memory_usage(deep=True).sum()
            }
            
        except Exception as e:
            return {"analysis_error": str(e)}
    
    async def _analyze_json(self, file_path: Path) -> Dict[str, Any]:
        """Analyze JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            preview = json.dumps(data, indent=2)[:self.max_preview_size]
            
            metadata = {
                "preview": preview,
                "structure": self._analyze_json_structure(data)
            }
            
            if isinstance(data, list):
                metadata["array_length"] = len(data)
                if data and isinstance(data[0], dict):
                    metadata["keys"] = list(data[0].keys())
            
            elif isinstance(data, dict):
                metadata["keys"] = list(data.keys())
            
            return metadata
            
        except Exception as e:
            return {"analysis_error": str(e)}
    
    async def _analyze_excel(self, file_path: Path) -> Dict[str, Any]:
        """Analyze Excel file"""
        try:
            # Get sheet names
            xl_file = pd.ExcelFile(file_path)
            sheet_names = xl_file.sheet_names
            
            # Analyze first sheet
            df = pd.read_excel(file_path, sheet_name=sheet_names[0], nrows=100)
            
            preview_df = df.head(self.max_preview_rows)
            preview = preview_df.to_string(max_cols=10)
            
            return {
                "sheet_names": sheet_names,
                "active_sheet": sheet_names[0],
                "rows": len(df),
                "columns": list(df.columns),
                "column_count": len(df.columns),
                "dtypes": df.dtypes.astype(str).to_dict(),
                "preview": preview[:self.max_preview_size]
            }
            
        except Exception as e:
            return {"analysis_error": str(e)}
    
    async def _analyze_text(self, file_path: Path) -> Dict[str, Any]:
        """Analyze text file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.split('\n')
            
            return {
                "line_count": len(lines),
                "character_count": len(content),
                "preview": content[:self.max_preview_size],
                "encoding": "utf-8"
            }
            
        except Exception as e:
            return {"analysis_error": str(e)}
    
    async def _analyze_parquet(self, file_path: Path) -> Dict[str, Any]:
        """Analyze Parquet file"""
        try:
            df = pd.read_parquet(file_path)
            
            preview_df = df.head(self.max_preview_rows)
            preview = preview_df.to_string(max_cols=10)
            
            return {
                "rows": len(df),
                "columns": list(df.columns),
                "column_count": len(df.columns),
                "dtypes": df.dtypes.astype(str).to_dict(),
                "preview": preview[:self.max_preview_size],
                "memory_usage": df.memory_usage(deep=True).sum()
            }
            
        except Exception as e:
            return {"analysis_error": str(e)}
    
    def _analyze_json_structure(self, data: Any, max_depth: int = 3, current_depth: int = 0) -> str:
        """Analyze JSON structure recursively"""
        if current_depth >= max_depth:
            return "..."
        
        if isinstance(data, dict):
            if not data:
                return "{}"
            
            keys = list(data.keys())[:5]  # First 5 keys
            structure_parts = []
            
            for key in keys:
                value_type = type(data[key]).__name__
                if isinstance(data[key], (dict, list)):
                    nested = self._analyze_json_structure(data[key], max_depth, current_depth + 1)
                    structure_parts.append(f'"{key}": {nested}')
                else:
                    structure_parts.append(f'"{key}": {value_type}')
            
            if len(data) > 5:
                structure_parts.append("...")
            
            return "{" + ", ".join(structure_parts) + "}"
        
        elif isinstance(data, list):
            if not data:
                return "[]"
            
            first_item_structure = self._analyze_json_structure(data[0], max_depth, current_depth + 1)
            return f"[{first_item_structure}, ...]" if len(data) > 1 else f"[{first_item_structure}]"
        
        else:
            return type(data).__name__
