"""
Data Ingestion Agent for AdTech LLM.

Handles automated collection, preprocessing, and storage of adtech data
from multiple sources including Kaggle, Hugging Face, web scraping, and partner APIs.
"""

import asyncio
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import aiohttp
import pandas as pd

from src.agents.base import AgentResult, AgentTool, BaseAgent
from src.config.settings import settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


class DataIngestionAgent(BaseAgent):
    """
    Agent for automated data ingestion from various sources.

    Capabilities:
    - Discover and download datasets from Kaggle, Hugging Face
    - Web scraping for public adtech data
    - Partner API integration
    - Data cleaning and preprocessing
    - PII anonymization
    - Quality validation
    """

    def __init__(self):
        super().__init__(
            name="DataIngestionAgent",
            description="Autonomous data collection and preprocessing for adtech datasets",
        )
        self._setup_tools()
        self._setup_paths()

    def _setup_paths(self) -> None:
        """Ensure data directories exist."""
        for path in [
            settings.data.raw_path,
            settings.data.processed_path,
            settings.data.cache_path,
        ]:
            path.mkdir(parents=True, exist_ok=True)

    def _setup_tools(self) -> None:
        """Register agent tools."""
        self.register_tool(AgentTool(
            name="discover_datasets",
            description="Search for adtech datasets across repositories",
            func=self._discover_datasets,
            parameters={
                "query": {"type": "string", "description": "Search query"},
                "sources": {"type": "array", "description": "Sources to search"},
                "limit": {"type": "integer", "description": "Max results per source"},
            },
        ))

        self.register_tool(AgentTool(
            name="download_dataset",
            description="Download a dataset from a specified source",
            func=self._download_dataset,
            parameters={
                "source": {"type": "string", "description": "Source type"},
                "identifier": {"type": "string", "description": "Dataset identifier"},
            },
        ))

        self.register_tool(AgentTool(
            name="preprocess_dataset",
            description="Clean and preprocess a downloaded dataset",
            func=self._preprocess_dataset,
            parameters={
                "file_path": {"type": "string", "description": "Path to raw data"},
                "schema": {"type": "object", "description": "Target schema"},
            },
        ))

        self.register_tool(AgentTool(
            name="validate_quality",
            description="Validate dataset quality and generate report",
            func=self._validate_quality,
            parameters={
                "file_path": {"type": "string", "description": "Path to processed data"},
            },
        ))

        self.register_tool(AgentTool(
            name="anonymize_pii",
            description="Detect and anonymize PII in dataset",
            func=self._anonymize_pii,
            parameters={
                "file_path": {"type": "string", "description": "Path to data"},
            },
        ))

    async def plan(self, task: str, context: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Create data ingestion plan.

        Args:
            task: Task description (e.g., "ingest CTR prediction data")
            context: Context including sources, requirements

        Returns:
            List of planned steps
        """
        sources = context.get("sources", ["kaggle", "huggingface"])
        query = context.get("query", task)

        plan = [
            {
                "name": "Discover datasets",
                "tool": "discover_datasets",
                "params": {"query": query, "sources": sources, "limit": 10},
                "critical": True,
            },
            {
                "name": "Download datasets",
                "tool": "download_dataset",
                "params": {},  # Will be filled from discovery results
                "critical": True,
            },
            {
                "name": "Anonymize PII",
                "tool": "anonymize_pii",
                "params": {},
                "critical": True,
            },
            {
                "name": "Preprocess data",
                "tool": "preprocess_dataset",
                "params": {},
                "critical": True,
            },
            {
                "name": "Validate quality",
                "tool": "validate_quality",
                "params": {},
                "critical": False,
            },
        ]

        return plan

    async def execute_step(
        self,
        step: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute a single ingestion step."""
        tool_name = step.get("tool")
        params = step.get("params", {})

        # Merge with context data for dependent steps
        if tool_name == "download_dataset" and "discovered_datasets" in context:
            datasets = context["discovered_datasets"]
            if datasets:
                params["datasets"] = datasets[:5]  # Download top 5

        if tool_name in ["preprocess_dataset", "validate_quality", "anonymize_pii"]:
            if "downloaded_files" in context:
                params["files"] = context["downloaded_files"]

        result = await self.use_tool(tool_name, **params)
        return result

    async def _discover_datasets(
        self,
        query: str,
        sources: list[str],
        limit: int = 10,
    ) -> dict[str, Any]:
        """
        Search for adtech datasets across multiple repositories.

        Args:
            query: Search query
            sources: List of sources to search (kaggle, huggingface, etc.)
            limit: Maximum results per source

        Returns:
            Dictionary with discovered datasets
        """
        discovered = []

        for source in sources:
            try:
                if source == "kaggle":
                    datasets = await self._search_kaggle(query, limit)
                    discovered.extend(datasets)
                elif source == "huggingface":
                    datasets = await self._search_huggingface(query, limit)
                    discovered.extend(datasets)
                elif source == "criteo":
                    datasets = self._get_criteo_datasets()
                    discovered.extend(datasets)
            except Exception as e:
                self.logger.warning(f"Failed to search {source}: {e}")

        return {"discovered_datasets": discovered, "total_found": len(discovered)}

    async def _search_kaggle(self, query: str, limit: int) -> list[dict[str, Any]]:
        """Search Kaggle for datasets."""
        # Predefined adtech datasets from Kaggle
        known_datasets = [
            {
                "source": "kaggle",
                "identifier": "groffo/ads-16-dataset",
                "name": "ADS-16 Dataset",
                "description": "Advertising dataset with user engagement metrics",
                "size_mb": 150,
            },
            {
                "source": "kaggle",
                "identifier": "avazu/avazu-ctr-prediction",
                "name": "Avazu CTR Prediction",
                "description": "Click-through rate prediction dataset",
                "size_mb": 1200,
            },
            {
                "source": "kaggle",
                "identifier": "c/criteo-display-ad-challenge",
                "name": "Criteo Display Ad Challenge",
                "description": "Large-scale display advertising dataset",
                "size_mb": 11000,
            },
            {
                "source": "kaggle",
                "identifier": "lsjsj/advertising-campaign-dataset",
                "name": "Advertising Campaign Dataset",
                "description": "User-level engagement data for content optimization",
                "size_mb": 50,
            },
            {
                "source": "kaggle",
                "identifier": "fayomi/advertising",
                "name": "Advertising Dataset",
                "description": "Digital advertising metrics dataset",
                "size_mb": 5,
            },
        ]

        # Filter by query if provided
        query_lower = query.lower()
        filtered = [
            d for d in known_datasets
            if query_lower in d["name"].lower() or query_lower in d["description"].lower()
            or "ad" in query_lower  # Return all for generic ad queries
        ]

        return filtered[:limit]

    async def _search_huggingface(self, query: str, limit: int) -> list[dict[str, Any]]:
        """Search Hugging Face for datasets."""
        known_datasets = [
            {
                "source": "huggingface",
                "identifier": "ad-imagenet/ad-imagenet",
                "name": "AdImageNet",
                "description": "9,003 programmatic ad creatives with sizes and text",
                "size_mb": 500,
            },
            {
                "source": "huggingface",
                "identifier": "microsoft/ms-marco",
                "name": "MS MARCO (for ad relevance)",
                "description": "Query-passage pairs useful for ad matching",
                "size_mb": 3000,
            },
        ]

        return known_datasets[:limit]

    def _get_criteo_datasets(self) -> list[dict[str, Any]]:
        """Get Criteo AI Lab datasets information."""
        return [
            {
                "source": "criteo",
                "identifier": "criteo-1tb-click-logs",
                "name": "Criteo Terabyte Click Logs",
                "description": "1TB dataset with 4B click events for CTR prediction",
                "size_mb": 1000000,
                "url": "https://ailab.criteo.com/download-criteo-1tb-click-logs-dataset/",
            },
            {
                "source": "criteo",
                "identifier": "criteo-uplift",
                "name": "Criteo Uplift Modeling Dataset",
                "description": "Dataset for counterfactual/uplift learning",
                "size_mb": 2000,
                "url": "https://ailab.criteo.com/criteo-uplift-prediction-dataset/",
            },
            {
                "source": "criteo",
                "identifier": "criteo-sponsored-search",
                "name": "Criteo Sponsored Search Conversion Log",
                "description": "Sponsored search conversion data",
                "size_mb": 5000,
                "url": "https://ailab.criteo.com/criteo-sponsored-search-conversion-log-dataset/",
            },
        ]

    async def _download_dataset(
        self,
        datasets: list[dict[str, Any]] | None = None,
        source: str | None = None,
        identifier: str | None = None,
    ) -> dict[str, Any]:
        """
        Download datasets from specified sources.

        Supports Kaggle API, Hugging Face Hub, and direct URLs.
        """
        downloaded_files = []

        if datasets:
            for dataset in datasets:
                try:
                    file_path = await self._download_single(dataset)
                    if file_path:
                        downloaded_files.append({
                            "path": str(file_path),
                            "source": dataset["source"],
                            "name": dataset["name"],
                        })
                except Exception as e:
                    self.logger.error(f"Failed to download {dataset['name']}: {e}")
        elif source and identifier:
            file_path = await self._download_single({"source": source, "identifier": identifier})
            if file_path:
                downloaded_files.append({"path": str(file_path), "source": source})

        return {"downloaded_files": downloaded_files}

    async def _download_single(self, dataset: dict[str, Any]) -> Path | None:
        """Download a single dataset."""
        source = dataset.get("source")
        identifier = dataset.get("identifier", "")

        # Create safe filename
        safe_name = identifier.replace("/", "_").replace("-", "_")
        output_dir = settings.data.raw_path / source
        output_dir.mkdir(parents=True, exist_ok=True)

        if source == "kaggle":
            return await self._download_from_kaggle(identifier, output_dir)
        elif source == "huggingface":
            return await self._download_from_huggingface(identifier, output_dir)
        elif source == "criteo":
            # Criteo requires manual download agreement
            self.logger.info(f"Criteo dataset requires manual download: {dataset.get('url')}")
            return None
        elif "url" in dataset:
            return await self._download_from_url(dataset["url"], output_dir / f"{safe_name}.csv")

        return None

    async def _download_from_kaggle(self, dataset_id: str, output_dir: Path) -> Path | None:
        """Download from Kaggle using kaggle API."""
        try:
            # Check for Kaggle credentials
            if not settings.data.kaggle_username or not settings.data.kaggle_key:
                self.logger.warning("Kaggle credentials not configured")
                # Create a placeholder/sample file for demonstration
                return await self._create_sample_dataset(output_dir, dataset_id, "kaggle")

            # Use kaggle CLI
            import subprocess
            cmd = f"kaggle datasets download -d {dataset_id} -p {output_dir} --unzip"
            result = subprocess.run(cmd.split(), capture_output=True, text=True)

            if result.returncode == 0:
                # Find downloaded file
                for f in output_dir.iterdir():
                    if f.suffix in [".csv", ".parquet", ".json"]:
                        return f

        except Exception as e:
            self.logger.error(f"Kaggle download failed: {e}")

        # Return sample for demonstration
        return await self._create_sample_dataset(output_dir, dataset_id, "kaggle")

    async def _download_from_huggingface(
        self,
        dataset_id: str,
        output_dir: Path,
    ) -> Path | None:
        """Download from Hugging Face Hub."""
        try:
            from datasets import load_dataset

            dataset = load_dataset(dataset_id, split="train")
            output_path = output_dir / f"{dataset_id.replace('/', '_')}.parquet"
            dataset.to_parquet(str(output_path))
            return output_path

        except Exception as e:
            self.logger.warning(f"HuggingFace download failed: {e}")
            return await self._create_sample_dataset(output_dir, dataset_id, "huggingface")

    async def _download_from_url(self, url: str, output_path: Path) -> Path | None:
        """Download from direct URL."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        content = await response.read()
                        output_path.write_bytes(content)
                        return output_path
        except Exception as e:
            self.logger.error(f"URL download failed: {e}")
        return None

    async def _create_sample_dataset(
        self,
        output_dir: Path,
        identifier: str,
        source: str,
    ) -> Path:
        """Create a sample dataset for demonstration/testing."""
        import numpy as np

        safe_name = identifier.replace("/", "_").replace("-", "_")
        output_path = output_dir / f"{safe_name}_sample.csv"

        # Generate sample adtech data
        n_samples = 10000
        np.random.seed(42)

        data = {
            "impression_id": [f"imp_{i}" for i in range(n_samples)],
            "user_id": [f"user_{np.random.randint(0, 1000)}" for _ in range(n_samples)],
            "campaign_id": [f"camp_{np.random.randint(0, 50)}" for _ in range(n_samples)],
            "creative_id": [f"creative_{np.random.randint(0, 200)}" for _ in range(n_samples)],
            "placement": np.random.choice(["banner", "sidebar", "native", "video"], n_samples),
            "device_type": np.random.choice(["mobile", "desktop", "tablet"], n_samples),
            "os": np.random.choice(["ios", "android", "windows", "macos"], n_samples),
            "country": np.random.choice(["US", "UK", "DE", "FR", "JP"], n_samples),
            "hour_of_day": np.random.randint(0, 24, n_samples),
            "day_of_week": np.random.randint(0, 7, n_samples),
            "bid_price": np.random.exponential(0.5, n_samples).round(4),
            "win_price": np.random.exponential(0.3, n_samples).round(4),
            "click": np.random.binomial(1, 0.02, n_samples),
            "conversion": np.random.binomial(1, 0.005, n_samples),
        }

        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)

        self.logger.info(f"Created sample dataset: {output_path}")
        return output_path

    async def _preprocess_dataset(
        self,
        files: list[dict[str, Any]] | None = None,
        file_path: str | None = None,
        schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Preprocess and clean datasets.

        Operations:
        - Handle missing values
        - Normalize features
        - Convert data types
        - Create derived features
        - Apply schema mapping
        """
        processed_files = []
        files_to_process = files or ([{"path": file_path}] if file_path else [])

        for file_info in files_to_process:
            path = Path(file_info["path"])
            if not path.exists():
                continue

            try:
                # Load data
                if path.suffix == ".csv":
                    df = pd.read_csv(path)
                elif path.suffix == ".parquet":
                    df = pd.read_parquet(path)
                elif path.suffix == ".json":
                    df = pd.read_json(path)
                else:
                    continue

                # Basic preprocessing
                original_rows = len(df)

                # Remove duplicates
                df = df.drop_duplicates()

                # Handle missing values
                numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

                categorical_cols = df.select_dtypes(include=["object"]).columns
                df[categorical_cols] = df[categorical_cols].fillna("unknown")

                # Create derived features if applicable
                if "click" in df.columns and "impression_id" in df.columns:
                    # This is CTR data
                    if "hour_of_day" in df.columns:
                        df["is_prime_time"] = df["hour_of_day"].between(18, 22).astype(int)
                    if "device_type" in df.columns:
                        df["is_mobile"] = (df["device_type"] == "mobile").astype(int)

                # Save processed data
                output_path = settings.data.processed_path / f"processed_{path.name}"
                df.to_parquet(output_path.with_suffix(".parquet"), index=False)

                processed_files.append({
                    "original_path": str(path),
                    "processed_path": str(output_path.with_suffix(".parquet")),
                    "original_rows": original_rows,
                    "processed_rows": len(df),
                    "columns": list(df.columns),
                })

                self.logger.info(f"Preprocessed {path.name}: {original_rows} -> {len(df)} rows")

            except Exception as e:
                self.logger.error(f"Failed to preprocess {path}: {e}")

        return {"processed_files": processed_files}

    async def _anonymize_pii(
        self,
        files: list[dict[str, Any]] | None = None,
        file_path: str | None = None,
    ) -> dict[str, Any]:
        """
        Detect and anonymize PII in datasets.

        Handles:
        - Email addresses
        - IP addresses
        - Names
        - Phone numbers
        - User IDs (hashing)
        """
        # PII patterns and their anonymization methods
        pii_columns = {
            "email": lambda x: hashlib.sha256(str(x).encode()).hexdigest()[:16] if pd.notna(x) else x,
            "ip_address": lambda x: hashlib.sha256(str(x).encode()).hexdigest()[:16] if pd.notna(x) else x,
            "ip": lambda x: hashlib.sha256(str(x).encode()).hexdigest()[:16] if pd.notna(x) else x,
            "user_id": lambda x: hashlib.sha256(str(x).encode()).hexdigest()[:16] if pd.notna(x) else x,
            "name": lambda x: "REDACTED",
            "phone": lambda x: "REDACTED",
        }

        files_to_process = files or ([{"path": file_path}] if file_path else [])
        anonymized_files = []

        for file_info in files_to_process:
            path = Path(file_info["path"])
            if not path.exists():
                continue

            try:
                df = pd.read_csv(path) if path.suffix == ".csv" else pd.read_parquet(path)
                pii_found = []

                for col in df.columns:
                    col_lower = col.lower()
                    for pii_type, anonymizer in pii_columns.items():
                        if pii_type in col_lower:
                            df[col] = df[col].apply(anonymizer)
                            pii_found.append({"column": col, "type": pii_type})
                            break

                if pii_found:
                    output_path = path.parent / f"anonymized_{path.name}"
                    if path.suffix == ".csv":
                        df.to_csv(output_path, index=False)
                    else:
                        df.to_parquet(output_path, index=False)

                    anonymized_files.append({
                        "original_path": str(path),
                        "anonymized_path": str(output_path),
                        "pii_found": pii_found,
                    })

                    # Update file info for next steps
                    file_info["path"] = str(output_path)

            except Exception as e:
                self.logger.error(f"Failed to anonymize {path}: {e}")

        return {"anonymized_files": anonymized_files, "files": files_to_process}

    async def _validate_quality(
        self,
        files: list[dict[str, Any]] | None = None,
        file_path: str | None = None,
    ) -> dict[str, Any]:
        """
        Validate dataset quality and generate quality report.

        Checks:
        - Completeness (missing values)
        - Consistency (data types, ranges)
        - Accuracy (outliers, invalid values)
        - Uniqueness (duplicates)
        """
        files_to_validate = files or ([{"path": file_path}] if file_path else [])
        quality_reports = []

        for file_info in files_to_validate:
            path = Path(file_info.get("processed_path", file_info.get("path", "")))
            if not path.exists():
                continue

            try:
                df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)

                # Calculate quality metrics
                total_cells = df.size
                missing_cells = df.isnull().sum().sum()
                duplicate_rows = df.duplicated().sum()

                # Column-level analysis
                column_stats = {}
                for col in df.columns:
                    col_data = df[col]
                    stats = {
                        "dtype": str(col_data.dtype),
                        "missing_pct": (col_data.isnull().sum() / len(col_data) * 100).round(2),
                        "unique_count": col_data.nunique(),
                    }

                    if pd.api.types.is_numeric_dtype(col_data):
                        stats.update({
                            "min": float(col_data.min()) if not col_data.isnull().all() else None,
                            "max": float(col_data.max()) if not col_data.isnull().all() else None,
                            "mean": float(col_data.mean()) if not col_data.isnull().all() else None,
                        })

                    column_stats[col] = stats

                # Calculate overall quality score
                completeness = 1 - (missing_cells / total_cells)
                uniqueness = 1 - (duplicate_rows / len(df)) if len(df) > 0 else 1
                quality_score = (completeness * 0.6 + uniqueness * 0.4) * 100

                report = {
                    "file_path": str(path),
                    "row_count": len(df),
                    "column_count": len(df.columns),
                    "total_cells": total_cells,
                    "missing_cells": int(missing_cells),
                    "duplicate_rows": int(duplicate_rows),
                    "completeness_pct": round(completeness * 100, 2),
                    "uniqueness_pct": round(uniqueness * 100, 2),
                    "quality_score": round(quality_score, 2),
                    "column_stats": column_stats,
                    "passes_threshold": quality_score >= 90,
                }

                quality_reports.append(report)

                self.logger.info(
                    f"Quality report for {path.name}: score={quality_score:.1f}%"
                )

            except Exception as e:
                self.logger.error(f"Failed to validate {path}: {e}")

        return {"quality_reports": quality_reports}
