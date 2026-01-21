"""
Structured JSONL logging utilities.

Per R18: Every script emits JSONL logs with standard keys:
- script_name, run_id, config_digest, inputs, outputs, row_counts, NA rates,
  CRS info, join distance summaries, pixel counts/nodata summaries, versions.
"""

import json
import logging
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from sleep_esi.paths import LOGS_DIR


def generate_run_id() -> str:
    """Generate a unique run ID for this execution."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    short_uuid = uuid.uuid4().hex[:8]
    return f"{timestamp}_{short_uuid}"


def get_versions() -> dict[str, str]:
    """Get versions of key libraries for reproducibility logging."""
    versions = {"python": sys.version.split()[0]}
    
    # Try to get versions of key geospatial libraries
    try:
        import geopandas
        versions["geopandas"] = geopandas.__version__
    except ImportError:
        pass
    
    try:
        import pandas
        versions["pandas"] = pandas.__version__
    except ImportError:
        pass
    
    try:
        import numpy
        versions["numpy"] = numpy.__version__
    except ImportError:
        pass
    
    try:
        import rasterio
        versions["rasterio"] = rasterio.__version__
    except ImportError:
        pass
    
    try:
        import pyproj
        versions["pyproj"] = pyproj.__version__
    except ImportError:
        pass
    
    try:
        import shapely
        versions["shapely"] = shapely.__version__
    except ImportError:
        pass
    
    return versions


class JSONLLogger:
    """
    Structured JSONL logger for pipeline scripts.
    
    Usage:
        logger = JSONLLogger("my_script")
        logger.info("Starting processing", extra={"input_file": "data.csv"})
        logger.log_metrics({"row_count": 1000, "na_rate": 0.05})
        logger.close()
    """
    
    def __init__(
        self,
        script_name: str,
        run_id: Optional[str] = None,
        log_dir: Optional[Path] = None,
    ):
        """
        Initialize the JSONL logger.
        
        Args:
            script_name: Name of the script (used in log filename)
            run_id: Unique run identifier. Auto-generated if not provided.
            log_dir: Directory for log files. Defaults to LOGS_DIR.
        """
        self.script_name = script_name
        self.run_id = run_id or generate_run_id()
        self.log_dir = log_dir or LOGS_DIR
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_file = self.log_dir / f"{script_name}_{self.run_id}.jsonl"
        self._file_handle = open(self.log_file, "a", encoding="utf-8")
        
        # Also set up console logging
        self._console_handler = logging.StreamHandler(sys.stdout)
        self._console_handler.setLevel(logging.INFO)
        self._console_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        
        self._logger = logging.getLogger(f"sleep_esi.{script_name}")
        self._logger.setLevel(logging.DEBUG)
        self._logger.addHandler(self._console_handler)
        
        # Log initialization
        self._write_record(
            level="INFO",
            message="Logger initialized",
            extra={
                "script_name": script_name,
                "run_id": self.run_id,
                "log_file": str(self.log_file),
                "versions": get_versions(),
            },
        )
    
    def _write_record(
        self,
        level: str,
        message: str,
        extra: Optional[dict[str, Any]] = None,
    ) -> None:
        """Write a single JSONL record."""
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "script_name": self.script_name,
            "run_id": self.run_id,
            "level": level,
            "message": message,
        }
        if extra:
            record["extra"] = extra
        
        self._file_handle.write(json.dumps(record, default=str) + "\n")
        self._file_handle.flush()
    
    def debug(self, message: str, extra: Optional[dict[str, Any]] = None) -> None:
        """Log a debug message."""
        self._write_record("DEBUG", message, extra)
        self._logger.debug(message)
    
    def info(self, message: str, extra: Optional[dict[str, Any]] = None) -> None:
        """Log an info message."""
        self._write_record("INFO", message, extra)
        self._logger.info(message)
    
    def warning(self, message: str, extra: Optional[dict[str, Any]] = None) -> None:
        """Log a warning message."""
        self._write_record("WARNING", message, extra)
        self._logger.warning(message)
    
    def error(self, message: str, extra: Optional[dict[str, Any]] = None) -> None:
        """Log an error message."""
        self._write_record("ERROR", message, extra)
        self._logger.error(message)
    
    def log_config(self, config: dict[str, Any], config_digest: Optional[str] = None) -> None:
        """Log the configuration used for this run."""
        self._write_record(
            "INFO",
            "Configuration loaded",
            extra={"config": config, "config_digest": config_digest},
        )
    
    def log_inputs(self, inputs: dict[str, str]) -> None:
        """Log input files/paths."""
        self._write_record("INFO", "Inputs registered", extra={"inputs": inputs})
    
    def log_outputs(self, outputs: dict[str, str]) -> None:
        """Log output files/paths."""
        self._write_record("INFO", "Outputs registered", extra={"outputs": outputs})
    
    def log_metrics(self, metrics: dict[str, Any]) -> None:
        """Log metrics (row counts, NA rates, etc.)."""
        self._write_record("INFO", "Metrics recorded", extra={"metrics": metrics})
    
    def log_crs_info(self, crs_info: dict[str, Any]) -> None:
        """Log CRS information for spatial data."""
        self._write_record("INFO", "CRS info recorded", extra={"crs_info": crs_info})
    
    def log_join_stats(self, join_stats: dict[str, Any]) -> None:
        """Log spatial join statistics."""
        self._write_record("INFO", "Join stats recorded", extra={"join_stats": join_stats})
    
    def log_raster_stats(self, raster_stats: dict[str, Any]) -> None:
        """Log raster processing statistics."""
        self._write_record("INFO", "Raster stats recorded", extra={"raster_stats": raster_stats})
    
    def close(self) -> None:
        """Close the log file handle."""
        self._write_record("INFO", "Logger closing", extra={"run_id": self.run_id})
        self._file_handle.close()
        self._logger.removeHandler(self._console_handler)
    
    def __enter__(self) -> "JSONLLogger":
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type is not None:
            self.error(
                f"Exception occurred: {exc_type.__name__}: {exc_val}",
                extra={"traceback": str(exc_tb)},
            )
        self.close()


def get_logger(script_name: str, run_id: Optional[str] = None) -> JSONLLogger:
    """
    Convenience function to get a configured logger.
    
    Args:
        script_name: Name of the script
        run_id: Optional run ID (auto-generated if not provided)
    
    Returns:
        Configured JSONLLogger instance
    """
    return JSONLLogger(script_name=script_name, run_id=run_id)

