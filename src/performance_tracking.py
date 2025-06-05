import pandas as pd
import json
import os
from datetime import datetime


class PerformanceTracker:
    """Tracks RAG system performance and provides analysis metrics"""

    def __init__(self, log_path="./logs/conversations"):
        self.log_path = log_path
        os.makedirs(log_path, exist_ok=True)
        # Use conversation logs instead of separate performance logs
        self.current_log_file = os.path.join(log_path, f"conv_log_{datetime.now().strftime('%Y%m%d')}.jsonl")

    def log_performance(self, metrics):
        """Records performance metrics of a query - now integrated with conversation logs"""
        # This method is deprecated - performance data is now stored in conversation logs
        pass

    def track_query(
        self,
        question,
        retrieval_time,
        generation_time,
        num_docs_retrieved,
        embedding_time=None,
        total_time=None,
        memory_usage=None,
    ):
        """Records performance metrics of a complete query"""
        metrics = {
            "question": question,
            "retrieval_time_ms": retrieval_time,
            "generation_time_ms": generation_time,
            "num_docs_retrieved": num_docs_retrieved,
            "total_time_ms": total_time or (retrieval_time + generation_time),
            "embedding_time_ms": embedding_time,
            "memory_usage_mb": memory_usage,
        }

        self.log_performance(metrics)
        return metrics

    def get_performance_logs(self, days=7):
        """Retrieves performance logs from conversation logs"""
        logs = []

        # Find all conversation log files in the specified period
        for filename in os.listdir(self.log_path):
            if filename.startswith("conv_log_") and filename.endswith(".jsonl"):
                file_path = os.path.join(self.log_path, filename)
                with open(file_path, "r") as f:
                    for line in f:
                        try:
                            log_entry = json.loads(line.strip())
                            log_date = datetime.fromisoformat(log_entry["timestamp"])
                            days_ago = (datetime.now() - log_date).days
                            if days_ago <= days:
                                # Convert conversation log to performance format
                                perf_entry = {
                                    "timestamp": log_entry["timestamp"],
                                    "question": log_entry.get("question", ""),
                                    "total_time_ms": log_entry.get("response_time_ms", 0),
                                    "retrieval_time_ms": log_entry.get("response_time_ms", 0) * 0.3,  # Approximation
                                    "generation_time_ms": log_entry.get("response_time_ms", 0) * 0.7,  # Approximation
                                    "num_docs_retrieved": log_entry.get("sources_count", 0),
                                }
                                logs.append(perf_entry)
                        except json.JSONDecodeError:
                            continue

        return logs

    def analyze_performance(self, logs=None):
        """Analyzes performance metrics"""
        if logs is None:
            logs = self.get_performance_logs()

        if not logs:
            return {"message": "No data available for analysis"}

        # Convert to DataFrame to facilitate analysis
        df = pd.DataFrame(logs)

        # Add date/time for temporal analysis
        df["datetime"] = pd.to_datetime(df["timestamp"])
        df["date"] = df["datetime"].dt.date
        df["hour"] = df["datetime"].dt.hour

        # Global metrics
        metrics = {
            "total_queries": len(df),
            "avg_total_time_ms": df["total_time_ms"].mean(),
            "avg_retrieval_time_ms": df["retrieval_time_ms"].mean(),
            "avg_generation_time_ms": df["generation_time_ms"].mean(),
            "avg_docs_retrieved": df["num_docs_retrieved"].mean(),
            "p95_total_time_ms": df["total_time_ms"].quantile(0.95),
            "max_total_time_ms": df["total_time_ms"].max(),
            "min_total_time_ms": df["total_time_ms"].min(),
        }

        # Temporal trends (by day)
        daily_metrics = (
            df.groupby("date")
            .agg(
                {
                    "total_time_ms": "mean",
                    "retrieval_time_ms": "mean",
                    "generation_time_ms": "mean",
                    "timestamp": "count",  # Number of queries per day
                }
            )
            .rename(columns={"timestamp": "query_count"})
        )

        # Trends by hour
        hourly_metrics = (
            df.groupby("hour")
            .agg(
                {
                    "total_time_ms": "mean",
                    "timestamp": "count",  # Number of queries per hour
                }
            )
            .rename(columns={"timestamp": "query_count"})
        )

        return {
            "metrics": metrics,
            "daily_metrics": daily_metrics.reset_index().to_dict("records"),
            "hourly_metrics": hourly_metrics.reset_index().to_dict("records"),
        }

    # Dashboard rendering method moved to src/dashboard.py for centralization


# Function removed - performance tracking is now integrated through FeaturesManager
# to avoid multiple wrapping of the ask_question method
