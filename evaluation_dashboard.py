#!/usr/bin/env python3
"""
Streamlit Dashboard for Isschat Evaluation Results
Provides interactive visualization and comparison of evaluation results
"""

import streamlit as st
import pandas as pd
import json
import plotly.express as px
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# Page configuration
st.set_page_config(
    page_title="Isschat Evaluation Dashboard", page_icon="📊", layout="wide", initial_sidebar_state="expanded"
)

# Enhanced CSS for better styling inspired by HTML report
st.markdown(
    """
<style>
    .test-separator {
        margin: 30px 0;
        padding: 10px 0;
        border-top: 2px solid #dee2e6;
        text-align: right;
    }
    
    .status-badge {
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85em;
        font-weight: bold;
        text-transform: uppercase;
    }
    
    .status-passed {
        background: #d4edda;
        color: #155724;
    }
    
    .status-failed {
        background: #f8d7da;
        color: #721c24;
    }
    
    .status-error {
        background: #fff3cd;
        color: #856404;
    }
    
    .status-measured {
        background: #d1ecf1;
        color: #0c5460;
    }
    
    .question-section {
        margin-bottom: 20px;
    }
    
    .response-section {
        margin-bottom: 20px;
    }
    
    .expected-section {
        margin-bottom: 20px;
    }
    
    .section-label {
        font-weight: bold;
        color: #495057;
        margin-bottom: 8px;
        display: block;
        font-size: 0.95em;
    }
    
    .question-text {
        background: #e3f2fd;
        padding: 15px;
        border-radius: 5px;
        border-left: 4px solid #2196f3;
        font-style: italic;
        color: #1565c0;
    }
    
    .response-text {
        padding: 15px;
        border-radius: 5px;
        white-space: pre-wrap;
        line-height: 1.5;
    }
    
    .response-passed {
        background: #f1f8e9;
        border-left: 4px solid #4caf50;
        color: #2e7d32;
    }
    
    .response-failed {
        background: #ffebee;
        border-left: 4px solid #f44336;
        color: #c62828;
    }
    
    .expected-text {
        background: #fff3e0;
        padding: 15px;
        border-radius: 5px;
        border-left: 4px solid #ff9800;
        font-size: 0.9em;
        color: #f57c00;
    }
    
    .evaluation-details {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 5px;
        margin-top: 15px;
        border: 1px solid #e9ecef;
    }
    
    .score-display {
        display: flex;
        align-items: center;
        gap: 15px;
        margin-bottom: 10px;
    }
    
    .score-value {
        font-size: 1.2em;
        font-weight: bold;
        color: #007bff;
    }
    
    .reasoning {
        font-style: italic;
        color: #666;
        background: #ffffff;
        padding: 10px;
        border-radius: 4px;
        border-left: 3px solid #007bff;
        margin-top: 10px;
    }
    
    .metadata {
        display: flex;
        gap: 15px;
        margin-top: 15px;
        flex-wrap: wrap;
    }
    
    .metadata-item {
        background: #e9ecef;
        padding: 5px 10px;
        border-radius: 15px;
        font-size: 0.85em;
        color: #495057;
    }
    
    .sources {
        margin-top: 15px;
    }
    
    .source-item {
        background: #f8f9fa;
        padding: 8px 12px;
        margin: 5px 0;
        border-radius: 4px;
        font-size: 0.9em;
        border-left: 3px solid #6c757d;
        color: #495057;
    }
    
    .error-message {
        background: #f8d7da;
        color: #721c24;
        padding: 15px;
        border-radius: 5px;
        border-left: 4px solid #dc3545;
        margin-top: 10px;
    }
    
    .category-description {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 25px;
        border-left: 4px solid #007bff;
    }
    
    .category-description h3 {
        margin-top: 0;
        font-size: 1.5em;
        margin-bottom: 10px;
        color: #007bff;
    }
    
    .category-description p {
        margin-bottom: 0;
        font-size: 1.1em;
        line-height: 1.4;
        color: #495057;
    }
    
    .no-data {
        text-align: center;
        color: #6c757d;
        font-style: italic;
        padding: 40px;
        background: #f8f9fa;
        border-radius: 8px;
        border: 1px solid #dee2e6;
    }
</style>
""",
    unsafe_allow_html=True,
)


class EvaluationDashboard:
    def __init__(self):
        self.evaluation_results = {}
        self.available_files = []
        self.load_available_files()

        # Category descriptions based on evaluator analysis
        self.category_descriptions = {
            "robustness": {
                "title": "Robustness Tests",
                "description": "Evaluates Isschat's ability to handle edge cases: invalid data validation, confidentiality protection, out-of-context question handling, and internal knowledge verification.",
            },
            "generation": {
                "title": "Conversational Generation Tests",
                "description": "Tests Isschat's conversational capabilities: context continuity, conversational references, memory recall, clarifications, topic transitions, and French linguistic coherence.",
            },
            "retrieval": {
                "title": "Document Retrieval Metrics",
                "description": "Measures document retrieval performance with advanced metrics: precision, recall, F1-score, Precision@K, MRR, MAP, and NDCG to evaluate ranking quality.",
            },
            "business_value": {
                "title": "Business Value",
                "description": "Evaluates Isschat's business impact by comparing response quality, processing time against human estimates, and efficiency across different complexity levels.",
            },
            "feedback": {
                "title": "User Feedback Analysis",
                "description": "Automatic analysis of user feedback with CamemBERT classification by themes to identify system strengths and weaknesses based on real user feedback.",
            },
        }

    def load_available_files(self):
        """Load available evaluation result files"""
        results_dir = Path("evaluation_results")
        if results_dir.exists():
            self.available_files = list(results_dir.glob("*.json"))
            self.available_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    def load_evaluation_file(self, file_path: Path) -> Dict[str, Any]:
        """Load evaluation results from JSON file"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Error loading file {file_path}: {str(e)}")
            return {}

    def get_file_display_name(self, file_path: Path) -> str:
        """Get display name for file"""
        try:
            data = self.load_evaluation_file(file_path)
            timestamp = data.get("timestamp", "")
            if timestamp:
                dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                return f"{dt.strftime('%Y-%m-%d %H:%M:%S')} - {file_path.name}"
            return file_path.name
        except:
            return file_path.name

    def render_sidebar(self):
        """Render sidebar with file selection"""
        st.sidebar.title("Evaluation Dashboard")

        # File selection
        st.sidebar.header("Results Selection")

        if not self.available_files:
            st.sidebar.error("No result files found in evaluation_results/")
            return []

        # Single file selection
        file_options = {self.get_file_display_name(f): f for f in self.available_files}

        selected_file = st.sidebar.selectbox(
            "Choose a file to analyze:",
            options=list(file_options.keys()),
            index=0 if file_options else None,
            help="Select an evaluation file to analyze",
        )

        # Add comparison button
        st.sidebar.markdown("---")
        if st.sidebar.button("Compare Evaluations", type="primary"):
            st.session_state.show_comparison = True
            st.rerun()

        return [file_options[selected_file]] if selected_file else []

    def render_overview_metrics(self, results: Dict[str, Dict[str, Any]]):
        """Render overview metrics for selected files"""
        st.header("Metrics Overview")

        # Always single file view now
        file_name = list(results.keys())[0]
        data = results[file_name]
        self.render_single_file_metrics(data, file_name)

    def render_single_file_metrics(self, data: Dict[str, Any], file_name: str):
        """Render metrics for a single file"""
        overall_stats = data.get("overall_stats", {})

        # Main metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Tests", overall_stats.get("total_tests", 0))

        with col2:
            st.metric("Passed", overall_stats.get("total_passed", 0))

        with col3:
            st.metric("Failed", overall_stats.get("total_failed", 0))

        with col4:
            pass_rate = overall_stats.get("overall_pass_rate", 0)
            st.metric("Pass Rate", f"{pass_rate:.1%}")

        # Category breakdown
        st.subheader("Category Breakdown")
        category_results = overall_stats.get("category_results", {})

        if category_results:
            category_data = []
            for category, stats in category_results.items():
                if stats:  # Skip empty categories
                    category_data.append(
                        {
                            "Category": self.category_descriptions.get(category, {}).get(
                                "title", category.replace("_", " ").title()
                            ),
                            "Total Tests": stats.get("total_tests", 0),
                            "Passed": stats.get("passed", 0),
                            "Failed": stats.get("failed", 0),
                            "Measured": stats.get("measured", 0),
                            "Pass Rate": f"{stats.get('pass_rate', 0):.1%}",
                            "Avg Score": f"{stats.get('average_score', 0):.3f}",
                            "Avg Time (s)": f"{stats.get('average_response_time', 0):.2f}",
                        }
                    )

            if category_data:
                df = pd.DataFrame(category_data)
                st.dataframe(df, use_container_width=True, hide_index=True)

    def render_category_details(self, results: Dict[str, Dict[str, Any]]):
        """Render detailed category analysis"""
        st.header("Category Analysis")

        # Get all categories from all files
        all_categories = set()
        for data in results.values():
            category_results = data.get("category_results", {})
            all_categories.update(category_results.keys())

        if not all_categories:
            st.info("No category data available")
            return

        # Category selector
        category_options = {
            self.category_descriptions.get(cat, {}).get("title", cat.replace("_", " ").title()): cat
            for cat in sorted(all_categories)
        }

        selected_category_display = st.selectbox("Select a category to analyze:", list(category_options.keys()))

        selected_category = category_options[selected_category_display]

        if selected_category:
            self.render_category_analysis(results, selected_category)

    def render_category_analysis(self, results: Dict[str, Dict[str, Any]], category: str):
        """Render analysis for a specific category"""
        # Display category description
        if category in self.category_descriptions:
            desc = self.category_descriptions[category]
            st.markdown(
                f"""
            <div class="category-description">
                <h3>{desc["title"]}</h3>
                <p>{desc["description"]}</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

        category_data = []
        for file_name, data in results.items():
            category_results = data.get("category_results", {})
            cat_data = category_results.get(category, {})

            if cat_data:
                category_data.append({"File": file_name, **cat_data})

        if not category_data:
            st.info(f"No data available for category {category}")
            return

        # Metrics display
        cols = st.columns(len(category_data))
        for i, (file_name, data) in enumerate([(d["File"], d) for d in category_data]):
            with cols[i]:
                st.markdown(f"**{file_name}**")

                # Key metrics
                if "total_tests" in data:
                    st.metric("Total Tests", data["total_tests"])
                if "pass_rate" in data:
                    st.metric("Pass Rate", f"{data['pass_rate']:.1%}")
                if "average_score" in data:
                    st.metric("Avg Score", f"{data['average_score']:.3f}")
                if "average_response_time" in data:
                    st.metric("Avg Time (s)", f"{data['average_response_time']:.2f}")

        # Detailed results for category
        self.render_detailed_results(results, category)

    def render_detailed_results(self, results: Dict[str, Dict[str, Any]], category: str):
        """Render detailed test results for a category using HTML-like cards"""
        st.subheader(
            f"Detailed Results - {self.category_descriptions.get(category, {}).get('title', category.replace('_', ' ').title())}"
        )

        # Collect all test results for this category
        all_results = []
        for file_name, data in results.items():
            category_results = data.get("category_results", {})
            cat_data = category_results.get(category, {})

            if "results" in cat_data:
                for result in cat_data["results"]:
                    all_results.append({"file": file_name, "result": result})

        if not all_results:
            st.markdown(
                '<div class="no-data">No detailed results available for this category</div>', unsafe_allow_html=True
            )
            return

        # Add filters
        col1, col2 = st.columns(2)

        with col1:
            all_statuses = list(set(r["result"].get("status", "N/A") for r in all_results))
            status_filter = st.multiselect("Filter by Status:", options=all_statuses, default=all_statuses)

        with col2:
            # Single file mode, no file filter needed
            file_filter = [list(results.keys())[0]]

        # Apply filters
        filtered_results = [
            r for r in all_results if r["result"].get("status", "N/A") in status_filter and r["file"] in file_filter
        ]

        # Display results as cards
        for item in filtered_results:
            result = item["result"]
            file_name = item["file"]

            # Create test case card
            test_id = result.get("test_id", "N/A")
            status = result.get("status", "N/A").lower()

            # Test separator with status badge only
            st.markdown(
                f"""
                <div class="test-separator">
                    <span class="status-badge status-{status}">
                        {status.upper()}
                    </span>
                </div>
            """,
                unsafe_allow_html=True,
            )

            # Question section
            question = result.get("question", "N/A")
            if question != "N/A":
                st.markdown(
                    f"""
                <div class="question-section">
                    <span class="section-label">Question:</span>
                    <div class="question-text">{question}</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            # Response section with conditional coloring
            response = result.get("response", "")
            error_message = result.get("error_message", "")

            # Determine response class based on criteria
            eval_details = result.get("evaluation_details", {})
            passes_criteria = eval_details.get("passes_criteria")
            response_class = "response-passed" if passes_criteria else "response-failed"

            if error_message:
                st.markdown(
                    f"""
                <div class="error-message">
                    <strong>Error:</strong> {error_message}
                </div>
                """,
                    unsafe_allow_html=True,
                )
            elif response:
                st.markdown(
                    f"""
                <div class="response-section">
                    <span class="section-label">Isschat Response:</span>
                    <div class="response-text {response_class}">{response}</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            # Expected behavior section
            expected = result.get("expected_behavior", "")
            if expected:
                st.markdown(
                    f"""
                <div class="expected-section">
                    <span class="section-label">Expected Behavior:</span>
                    <div class="expected-text">{expected}</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            # Evaluation details
            eval_details = result.get("evaluation_details", {})
            score = result.get("score", 0)
            category = result.get("category", "")

            if eval_details or score:
                reasoning = eval_details.get("reasoning", "")
                passes_criteria = eval_details.get("passes_criteria")

                st.markdown(
                    f"""
                <div class="evaluation-details">
                    <div class="score-display">
                        <span class="section-label">Score:</span>
                        <span class="score-value">{score:.3f}</span>
                """,
                    unsafe_allow_html=True,
                )

                # For business_value and generation categories, show pass/fail below score
                if category in ["business_value", "generation"] and passes_criteria is not None:
                    status_text = "PASSED" if passes_criteria else "FAILED"
                    status_class = "passed" if passes_criteria else "failed"
                    st.markdown(
                        f"""
                    </div>
                    <div style="margin-top: 10px;">
                        <span class="status-badge status-{status_class}">
                            {status_text}
                        </span>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )
                else:
                    # For other categories, show criteria as before but remove text
                    if passes_criteria is not None and category not in ["business_value", "generation"]:
                        criteria_class = "passed" if passes_criteria else "failed"
                        criteria_text = "Criteria Met" if passes_criteria else "Criteria Not Met"
                        st.markdown(
                            f"""
                            <span class="status-badge status-{criteria_class}">
                                {criteria_text}
                            </span>
                        """,
                            unsafe_allow_html=True,
                        )
                    st.markdown("</div>", unsafe_allow_html=True)

                if reasoning:
                    st.markdown(
                        f"""
                    <div class="reasoning">
                        <strong>Reasoning:</strong> {reasoning}
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

                st.markdown("</div>", unsafe_allow_html=True)

            # Metadata
            metadata = result.get("metadata", {})
            response_time = result.get("response_time", 0)

            if metadata or response_time:
                st.markdown('<div class="metadata">', unsafe_allow_html=True)

                if response_time:
                    st.markdown(
                        f'<span class="metadata-item"><strong>Response Time:</strong> {response_time:.2f}s</span>',
                        unsafe_allow_html=True,
                    )

                for key, value in metadata.items():
                    if key not in ["response_time"]:  # Avoid duplicating response_time
                        st.markdown(
                            f'<span class="metadata-item"><strong>{key.replace("_", " ").title()}:</strong> {value}</span>',
                            unsafe_allow_html=True,
                        )

                st.markdown("</div>", unsafe_allow_html=True)

            # Sources
            sources = result.get("sources", [])
            if sources:
                st.markdown(
                    """
                <div class="sources">
                    <span class="section-label">Sources:</span>
                """,
                    unsafe_allow_html=True,
                )
                for source in sources:
                    st.markdown(f'<div class="source-item">{source}</div>', unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

            # Add spacing after each test case
            st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)

    def render_retrieval_metrics(self, results: Dict[str, Dict[str, Any]]):
        """Render specialized retrieval metrics"""
        st.header("Retrieval Metrics")

        # Check if any file has retrieval data
        has_retrieval = any("retrieval" in data.get("category_results", {}) for data in results.values())

        if not has_retrieval:
            st.info("No retrieval metrics available")
            return

        # Display category description
        if "retrieval" in self.category_descriptions:
            desc = self.category_descriptions["retrieval"]
            st.markdown(
                f"""
            <div class="category-description">
                <h3>{desc["title"]}</h3>
                <p>{desc["description"]}</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

        # Display retrieval metrics
        retrieval_data = []
        for file_name, data in results.items():
            category_results = data.get("category_results", {})
            retrieval_stats = category_results.get("retrieval", {})

            if retrieval_stats:
                retrieval_data.append(
                    {
                        "File": file_name,
                        "Precision": retrieval_stats.get("average_precision", 0),
                        "Recall": retrieval_stats.get("average_recall", 0),
                        "F1-Score": retrieval_stats.get("average_f1_score", 0),
                        "P@1": retrieval_stats.get("average_precision_at_1", 0),
                        "P@3": retrieval_stats.get("average_precision_at_3", 0),
                        "P@5": retrieval_stats.get("average_precision_at_5", 0),
                        "MRR": retrieval_stats.get("average_mrr", 0),
                        "MAP": retrieval_stats.get("average_map", 0),
                        "NDCG@5": retrieval_stats.get("average_ndcg_at_5", 0),
                        "Avg Time (ms)": retrieval_stats.get("average_retrieval_time_ms", 0),
                    }
                )

        if retrieval_data:
            df = pd.DataFrame(retrieval_data)

            # Display metrics table
            st.dataframe(df, use_container_width=True, hide_index=True)

    def show_comparison_modal(self):
        """Show comparison of all evaluation files"""
        st.subheader("All Evaluations Comparison")

        # Load all available files
        all_results = {}
        for file_path in self.available_files:
            display_name = self.get_file_display_name(file_path)
            data = self.load_evaluation_file(file_path)
            if data:
                all_results[display_name] = data

        if len(all_results) < 2:
            st.warning("At least 2 evaluation files are required to perform a comparison.")
            return

        # Create comparison dataframe
        comparison_data = []
        for file_name, data in all_results.items():
            overall_stats = data.get("overall_stats", {})
            comparison_data.append(
                {
                    "File": file_name,
                    "Total Tests": overall_stats.get("total_tests", 0),
                    "Passed": overall_stats.get("total_passed", 0),
                    "Failed": overall_stats.get("total_failed", 0),
                    "Measured": overall_stats.get("total_measured", 0),
                    "Pass Rate": f"{overall_stats.get('overall_pass_rate', 0):.1%}",
                    "Timestamp": data.get("timestamp", ""),
                }
            )

        df = pd.DataFrame(comparison_data)

        # Display comparison table
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Visualizations
        col1, col2 = st.columns(2)

        with col1:
            # Test count comparison
            fig_pass_rate = px.bar(
                df,
                x="File",
                y="Total Tests",
                title="Test Count Comparison",
                labels={"Total Tests": "Number of Tests", "File": "Evaluation"},
            )
            fig_pass_rate.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_pass_rate, use_container_width=True)

        with col2:
            # Test distribution
            test_data = []
            for _, row in df.iterrows():
                test_data.extend(
                    [
                        {"File": row["File"], "Type": "Passed", "Count": row["Passed"]},
                        {"File": row["File"], "Type": "Failed", "Count": row["Failed"]},
                        {"File": row["File"], "Type": "Measured", "Count": row["Measured"]},
                    ]
                )
            test_df = pd.DataFrame(test_data)

            fig_tests = px.bar(
                test_df,
                x="File",
                y="Count",
                color="Type",
                title="Results Distribution by Type",
                labels={"Count": "Number of Tests", "File": "Evaluation"},
            )
            fig_tests.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_tests, use_container_width=True)

        # Category comparison if available
        st.subheader("Category Comparison")

        # Get all categories across all files
        all_categories = set()
        for data in all_results.values():
            category_results = data.get("category_results", {})
            all_categories.update(category_results.keys())

        if all_categories:
            selected_category = st.selectbox(
                "Select a category for comparison:",
                sorted(all_categories),
                format_func=lambda x: self.category_descriptions.get(x, {}).get("title", x.replace("_", " ").title()),
            )

            if selected_category:
                category_comparison_data = []
                for file_name, data in all_results.items():
                    category_results = data.get("category_results", {})
                    cat_data = category_results.get(selected_category, {})

                    if cat_data:
                        category_comparison_data.append(
                            {
                                "File": file_name,
                                "Tests": cat_data.get("total_tests", 0),
                                "Pass Rate": f"{cat_data.get('pass_rate', 0):.1%}",
                                "Avg Score": f"{cat_data.get('average_score', 0):.3f}",
                                "Avg Time (s)": f"{cat_data.get('average_response_time', 0):.2f}",
                            }
                        )

                if category_comparison_data:
                    cat_df = pd.DataFrame(category_comparison_data)
                    st.dataframe(cat_df, use_container_width=True, hide_index=True)

    def run(self):
        """Run the dashboard"""
        st.title("Isschat Evaluation Dashboard")

        # Check if we're in comparison mode
        if "show_comparison" in st.session_state and st.session_state.show_comparison:
            self.show_comparison_modal()
            if st.button("Back to Individual Analysis"):
                st.session_state.show_comparison = False
                st.rerun()
            return

        # Sidebar
        selected_files = self.render_sidebar()

        if not selected_files:
            st.info("Please select an evaluation file from the sidebar")
            return

        # Load selected files
        results = {}
        for file_path in selected_files:
            display_name = self.get_file_display_name(file_path)
            data = self.load_evaluation_file(file_path)
            if data:
                results[display_name] = data

        if not results:
            st.error("No valid evaluation data found")
            return

        # Main content tabs (removed Retrieval Metrics tab)
        tab1, tab2 = st.tabs(["Overview", "Categories"])

        with tab1:
            self.render_overview_metrics(results)

        with tab2:
            self.render_category_details(results)

        # Footer
        st.markdown("---")
        st.markdown(f"*Dashboard generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")


if __name__ == "__main__":
    dashboard = EvaluationDashboard()
    dashboard.run()
