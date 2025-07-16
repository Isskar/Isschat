#!/usr/bin/env python3
"""
Streamlit Dashboard for Isschat Evaluation Results
Provides interactive visualization and comparison of evaluation results
"""

import streamlit as st
import pandas as pd
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# Page configuration
st.set_page_config(page_title="Evaluation Dashboard", layout="wide", initial_sidebar_state="expanded")

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

        # Load evaluation config once at initialization
        self.evaluation_config = None
        self.available_categories = []
        self._load_evaluation_config()

        # Category descriptions based on evaluator configuration
        self.category_descriptions = {
            "robustness": {
                "title": "Robustness",
                "description": "Tests for model knowledge, data validation, and context handling",
            },
            "generation": {
                "title": "Generation",
                "description": "Tests for conversational generation capabilities and context handling",
            },
            "retrieval": {
                "title": "Retrieval performance",
                "description": "Tests for document retrieval accuracy and ranking quality",
            },
            "business_value": {
                "title": "Business value",
                "description": "Tests for measuring Isschat's business impact and efficiency",
            },
            "feedback": {
                "title": "User feedback analysis",
                "description": (
                    "Analyzes user feedback using CamemBERT classification to identify strengths and weaknesses"
                ),
            },
        }

    def _load_evaluation_config(self):
        """Load evaluation configuration once at initialization"""
        try:
            # Import here to avoid circular dependency
            sys.path.append(str(Path(__file__).parent.parent))
            from rag_evaluation.config import EvaluationConfig

            self.evaluation_config = EvaluationConfig()
            self.available_categories = self.evaluation_config.get_all_categories()
        except Exception as e:
            print(f"Error loading evaluation config: {str(e)}")
            self.evaluation_config = None
            self.available_categories = []

    def load_available_files(self):
        """Load available evaluation result files"""
        results_dir = Path(__file__).parent.parent / "evaluation_results"
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
        except Exception:
            return file_path.name

    def _extract_page_id(self, url: str) -> str:
        """Extract page ID from URL"""
        if not url:
            return ""

        # Extract pages/number from URL
        import re

        match = re.search(r"pages/(\d+)", url)
        return match.group(1) if match else url

    def _format_retrieved_documents(self, sources: list) -> str:
        """Format retrieved documents for display"""
        if not sources:
            return "No documents retrieved"

        formatted_docs = []
        for i, source in enumerate(sources[:5], 1):  # Show top 5
            formatted_docs.append(f"{i}. {source}")

        return "<br>".join(formatted_docs)

    def _urls_match_flexibly(self, url1: str, url2: str) -> bool:
        """Check if two URLs match using flexible criteria"""
        if not url1 or not url2:
            return False

        # Exact match
        if url1 == url2:
            return True

        # One URL contains the other (handles cases like /pages/123 vs /pages/123/Title)
        if url1 in url2 or url2 in url1:
            return True

        # Same page ID (extract page ID and compare)
        page_id1 = self._extract_page_id(url1)
        page_id2 = self._extract_page_id(url2)

        if page_id1 and page_id2 and page_id1 == page_id2:
            return True

        return False

    def _find_matching_retrieved_url(self, expected_url: str, retrieved_sources: list) -> str | None:
        """Find a retrieved URL that matches the expected URL using flexible matching"""
        for retrieved_url in retrieved_sources:
            if self._urls_match_flexibly(expected_url, retrieved_url):
                return retrieved_url
        return None

    def _create_document_comparison(self, expected_docs: list, retrieved_sources: list) -> str:
        """Create HTML comparison between expected and retrieved documents using flexible URL matching"""
        comparison_html = []
        comparison_html.append('<div style="margin-top: 10px;">')

        matched_retrieved_urls = set()

        # Show expected documents section (without green background)
        if expected_docs:
            comparison_html.append(
                '<div style="margin-bottom: 15px; padding: 15px; border-left: 4px solid #6c757d; background: #f8f9fa;">'
            )
            comparison_html.append('<strong style="color: #495057;">Expected Documents:</strong>')
            for doc in expected_docs:
                expected_url = doc.get("url", "")
                title = doc.get("title", "Untitled Document")

                # Find matching retrieved URL using flexible matching
                matching_retrieved_url = self._find_matching_retrieved_url(expected_url, retrieved_sources)

                if matching_retrieved_url:
                    matched_retrieved_urls.add(matching_retrieved_url)

                comparison_html.append(
                    f'<div style="padding: 8px; margin: 5px 0; border-radius: 3px; '
                    f'background: white; border: 1px solid #dee2e6;">'
                    f'{title}<br><small style="color: #6c757d;">{expected_url}</small></div>'
                )
            comparison_html.append("</div>")

        # Show retrieved documents section with color coding
        comparison_html.append(
            '<div style="margin-bottom: 15px; padding: 15px; border-left: 4px solid #6c757d; background: #f8f9fa;">'
        )
        comparison_html.append('<strong style="color: #495057;">Retrieved Documents:</strong>')

        if not retrieved_sources:
            comparison_html.append(
                '<div style="color: #666; font-style: italic; padding: 8px;">No documents retrieved</div>'
            )
        else:
            for url in retrieved_sources:
                # Check if this retrieved document matches any expected document
                is_expected = url in matched_retrieved_urls

                if is_expected:
                    # Green for expected documents
                    comparison_html.append(
                        f'<div style="color: #155724; background: #d4edda; padding: 8px; margin: 5px 0; '
                        f'border-radius: 3px; border: 1px solid #c3e6cb;">✓ Expected document<br>'
                        f"<small>{url}</small></div>"
                    )
                else:
                    # Yellow for unexpected documents
                    comparison_html.append(
                        f'<div style="color: #856404; background: #fff3cd; padding: 8px; margin: 5px 0; '
                        f'border-radius: 3px; border: 1px solid #ffeaa7;">⚠ Unexpected document<br>'
                        f"<small>{url}</small></div>"
                    )

        comparison_html.append("</div>")
        comparison_html.append("</div>")
        return "".join(comparison_html)

    def render_feedback_analysis(self, result: Dict[str, Any], eval_details: Dict[str, Any]):
        """Render feedback analysis with custom dashboard layout"""
        feedback_metrics = eval_details.get("feedback_metrics", {})

        if not feedback_metrics or feedback_metrics.get("total_feedbacks", 0) == 0:
            st.markdown(
                '<div class="evaluation-details">'
                '<div style="text-align: center; color: #666; padding: 20px;">'
                "<h3>Aucun feedback disponible</h3>"
                "<p>Aucune donnée de feedback n'a été trouvée pour l'analyse.</p>"
                "</div></div>",
                unsafe_allow_html=True,
            )
            return

        # Main metrics overview
        total_feedbacks = feedback_metrics.get("total_feedbacks", 0)
        overall_satisfaction = feedback_metrics.get("overall_satisfaction", 0)

        st.markdown(
            f"""
            <div class="evaluation-details">
                <div style="text-align: center; margin-bottom: 20px;">
                    <h3 style="color: #1f77b4; margin-bottom: 10px;">Analyse des feedbacks utilisateurs (in french)</h3>
                    <div style="display: flex; justify-content: center; gap: 40px; margin-bottom: 20px;">
                        <div style="text-align: center;">
                            <div style="font-size: 2em; font-weight: bold; color: #1f77b4;">{total_feedbacks}</div>
                            <div style="color: #666; font-size: 0.9em;">Total Feedbacks</div>
                        </div>
                        <div style="text-align: center;">
                            <div style="font-size: 2em; font-weight: bold; color: #28a745;">
                                {overall_satisfaction:.0%}
                            </div>
                            <div style="color: #666; font-size: 0.9em;">Satisfaction Globale</div>
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Topic breakdown
        topic_breakdown = feedback_metrics.get("topic_breakdown", {})
        if topic_breakdown:
            st.markdown(
                '<div class="evaluation-details" style="margin-top: 15px;">'
                '<span class="section-label">Répartition par Thème (Classification CamemBERT):</span>',
                unsafe_allow_html=True,
            )

            # Sort topics by count
            sorted_topics = sorted(topic_breakdown.items(), key=lambda x: x[1]["total_count"], reverse=True)

            for topic_id, topic_data in sorted_topics:
                topic_name = topic_data["topic_name"]
                total_count = topic_data["total_count"]
                satisfaction_rate = topic_data["satisfaction_rate"]
                positive_count = topic_data["positive_count"]
                negative_count = topic_data["negative_count"]

                # Color based on satisfaction rate
                if satisfaction_rate >= 0.7:
                    color = "#28a745"  # Green
                elif satisfaction_rate <= 0.4:
                    color = "#dc3545"  # Red
                else:
                    color = "#ffc107"  # Yellow

                topic_style = (
                    f"margin: 10px 0; padding: 15px; border-left: 4px solid {color}; "
                    f"background: #f8f9fa; border-radius: 5px;"
                )
                feedback_details = (
                    f"{total_count} feedback(s) • {positive_count} positif(s) • {negative_count} négatif(s)"
                )
                satisfaction_style = f"font-size: 1.2em; font-weight: bold; color: {color};"

                st.markdown(
                    f"""
                    <div style="{topic_style}">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <span style="font-weight: bold; color: #333;">{topic_name}</span>
                                <div style="font-size: 0.9em; color: #666; margin-top: 5px;">
                                    {feedback_details}
                                </div>
                            </div>
                            <div style="text-align: right;">
                                <div style="{satisfaction_style}">{satisfaction_rate:.0%}</div>
                                <div style="font-size: 0.8em; color: #666;">satisfaction</div>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("</div>", unsafe_allow_html=True)

        # Strengths and weaknesses - fix logic to avoid duplicates
        topic_breakdown = feedback_metrics.get("topic_breakdown", {})

        if topic_breakdown:
            # Create proper strengths and weaknesses from topic breakdown
            strengths = []
            weaknesses = []

            for topic_id, topic_data in topic_breakdown.items():
                satisfaction_rate = topic_data["satisfaction_rate"]

                if satisfaction_rate >= 0.5:
                    strengths.append(topic_data)
                elif satisfaction_rate <= 0.5:
                    weaknesses.append(topic_data)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown(
                    '<div class="evaluation-details"><span class="section-label">Points Forts:</span>',
                    unsafe_allow_html=True,
                )

                if strengths:
                    for strength in strengths:
                        strength_style = (
                            "margin: 8px 0; padding: 10px; background: #d4edda; "
                            "border-radius: 5px; border-left: 3px solid #28a745;"
                        )
                        strength_details = (
                            f"{strength['total_count']} feedback(s) • {strength['satisfaction_rate']:.0%} satisfaction"
                        )

                        st.markdown(
                            f"""
                            <div style="{strength_style}">
                                <div style="font-weight: bold; color: #155724;">
                                    {strength["topic_name"]}
                                </div>
                                <div style="font-size: 0.9em; color: #155724;">
                                    {strength_details}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                else:
                    st.markdown(
                        '<div style="font-style: italic; color: #666;">Aucun point fort identifié</div>',
                        unsafe_allow_html=True,
                    )

                st.markdown("</div>", unsafe_allow_html=True)

            with col2:
                st.markdown(
                    '<div class="evaluation-details"><span class="section-label">Points d\'Amélioration:</span>',
                    unsafe_allow_html=True,
                )

                if weaknesses:
                    for weakness in weaknesses:
                        weakness_style = (
                            "margin: 8px 0; padding: 10px; background: #f8d7da; "
                            "border-radius: 5px; border-left: 3px solid #dc3545;"
                        )
                        weakness_details = (
                            f"{weakness['total_count']} feedback(s) • {weakness['satisfaction_rate']:.0%} satisfaction"
                        )

                        st.markdown(
                            f"""
                            <div style="{weakness_style}">
                                <div style="font-weight: bold; color: #721c24;">
                                    {weakness["topic_name"]}
                                </div>
                                <div style="font-size: 0.9em; color: #721c24;">
                                    {weakness_details}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                else:
                    st.markdown(
                        '<div style="font-style: italic; color: #666;">'
                        "Aucun point d'amélioration critique identifié"
                        "</div>",
                        unsafe_allow_html=True,
                    )

                st.markdown("</div>", unsafe_allow_html=True)

    def render_sidebar(self):
        """Render sidebar with file selection"""
        # File selection
        st.sidebar.header("Results selection")

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

        st.sidebar.markdown("---")
        if st.sidebar.button("Dashboard", use_container_width=True):
            # Reset all page states to go back to main dashboard
            st.session_state.show_new_evaluation = False
            st.session_state.show_comparison = False
            st.session_state.show_evaluation_launcher = False
            st.session_state.evaluation_running = False
            st.session_state.evaluation_result = None
            st.rerun()

        if st.sidebar.button("New evaluation", use_container_width=True):
            # Reset other states and set new evaluation
            st.session_state.show_comparison = False
            st.session_state.show_evaluation_launcher = False
            st.session_state.evaluation_running = False
            st.session_state.evaluation_result = None
            st.session_state.show_new_evaluation = True
            st.rerun()

        if st.sidebar.button("Compare evaluations", use_container_width=True):
            # Reset other states and set comparison
            st.session_state.show_new_evaluation = False
            st.session_state.show_evaluation_launcher = False
            st.session_state.evaluation_running = False
            st.session_state.evaluation_result = None
            st.session_state.show_comparison = True
            st.rerun()

        return [file_options[selected_file]] if selected_file else []

    def render_overview_metrics(self, results: Dict[str, Dict[str, Any]]):
        """Render overview metrics for selected files"""
        st.header("Metrics overview")

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
        st.subheader("Category breakdown")
        category_results = overall_stats.get("category_results", {})

        if category_results:
            category_data = []
            for category, stats in category_results.items():
                if stats:
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
        st.header("Category analysis")

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
            <div style="margin-bottom: 15px; padding: 15px; border-left: 4px solid #6c757d; background: #f8f9fa;">
                <strong style="color: #495057;">{desc["title"]}:</strong>
                <div style="margin-top: 8px; color: #495057;">{desc["description"]}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        category_data = []
        for file_name, data in results.items():
            category_results = data.get("category_results", {})
            cat_data = category_results.get(category, {})

            if cat_data:
                category_data.append(cat_data)  # Remove file name from display

        if not category_data:
            st.info(f"No data available for category {category}")
            return

        # Metrics display without file names
        data = category_data[0]  # Single file mode
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if "total_tests" in data:
                st.metric("Total Tests", data["total_tests"])
        with col2:
            if "pass_rate" in data:
                st.metric("Pass Rate", f"{data['pass_rate']:.1%}")
        with col3:
            if "average_score" in data:
                st.metric("Avg Score", f"{data['average_score']:.3f}")
        with col4:
            if "average_response_time" in data:
                st.metric("Avg Time (s)", f"{data['average_response_time']:.2f}")

        # Detailed results for category
        self.render_detailed_results(results, category)

    def render_detailed_results(self, results: Dict[str, Dict[str, Any]], category: str):
        """Render detailed test results for a category using HTML-like cards"""
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

        # Add filters (skip for feedback category)
        if category != "feedback":
            col1, col2 = st.columns(2)

            with col1:
                all_statuses = list(set(r["result"].get("status", "N/A") for r in all_results))
                status_filter = st.multiselect("Filter by status:", options=all_statuses, default=all_statuses)

            with col2:
                # Single file mode, no file filter needed
                file_filter = [list(results.keys())[0]]

            # Apply filters
            filtered_results = [
                r for r in all_results if r["result"].get("status", "N/A") in status_filter and r["file"] in file_filter
            ]
        else:
            # For feedback category, no filters needed
            filtered_results = all_results

        # Display results as cards with question numbering and selective status removal
        question_counter = 1

        for item in filtered_results:
            result = item["result"]
            file_name = item["file"]
            category = result.get("category", "")

            # Get evaluation details early for use in multiple sections
            eval_details = result.get("evaluation_details", {})
            score = result.get("score", 0)

            # Create test case card
            status = result.get("status", "N/A").lower()

            # Hide status badge for business_value, robustness, generation, feedback categories
            if category not in ["business_value", "robustness", "generation", "feedback"]:
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
            else:
                # Just add spacing for categories without status badges
                st.markdown(
                    '<div style="margin: 30px 0; padding: 10px 0;"></div>',
                    unsafe_allow_html=True,
                )

            # Question section with numbering (skip for feedback category)
            question = result.get("question", "N/A")
            if question != "N/A" and category != "feedback":
                st.markdown(
                    f"""
                <div class="question-section">
                    <span class="section-label">Question {question_counter}:</span>
                    <div class="question-text">{question}</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )
                question_counter += 1

            # Special layout for retrieval category: Question → Document Comparison → Score → Status → Metrics
            if category == "retrieval":
                sources = result.get("sources", [])
                expected_docs = result.get("metadata", {}).get("expected_documents", [])

                # Document comparison right after question
                st.markdown(
                    '<div class="evaluation-details"><span class="section-label">Document Comparison:</span>',
                    unsafe_allow_html=True,
                )

                comparison_html = self._create_document_comparison(expected_docs, sources)
                st.markdown(comparison_html, unsafe_allow_html=True)

                # Score
                st.markdown(
                    f'<div style="margin-top: 15px;"><span class="section-label">Score:</span> '
                    f'<span class="score-value">{score:.3f}</span></div>',
                    unsafe_allow_html=True,
                )

                # Status
                passes_criteria = eval_details.get("passes_criteria")
                if passes_criteria is not None:
                    status_text = "PASSED" if passes_criteria else "FAILED"
                    status_class = "passed" if passes_criteria else "failed"
                    st.markdown(
                        f'<div style="margin-top: 10px;"><span class="status-badge '
                        f'status-{status_class}">{status_text}</span></div>',
                        unsafe_allow_html=True,
                    )

                # Metrics (response time only for retrieval)
                response_time = result.get("response_time", 0)
                if response_time:
                    st.markdown(
                        f'<div style="margin-top: 10px;"><span class="metadata-item">'
                        f"<strong>Response Time:</strong> {response_time:.2f}s</span></div>",
                        unsafe_allow_html=True,
                    )

                st.markdown("</div>", unsafe_allow_html=True)

            # Special layout for feedback category: Display feedback metrics dashboard
            elif category == "feedback":
                self.render_feedback_analysis(result, eval_details)

            else:
                # For other categories, keep original structure
                response = result.get("response", "")
                error_message = result.get("error_message", "")

                # Determine response class based on criteria
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

                # Expected behavior section - only for non-retrieval categories
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

                if category in ["robustness", "generation"] and (eval_details or score):
                    reasoning = eval_details.get("reasoning", "")
                    passes_criteria = eval_details.get("passes_criteria")

                    st.markdown(
                        '<div class="evaluation-details"><span class="section-label">'
                        "Evaluation of Isschat answer:</span>",
                        unsafe_allow_html=True,
                    )

                    # Score first
                    st.markdown(
                        f'<div class="score-display"><span class="section-label">Score:</span>'
                        f'<span class="score-value">{score:.3f}</span></div>',
                        unsafe_allow_html=True,
                    )

                    # Reasoning second
                    if reasoning:
                        st.markdown(
                            f'<div class="reasoning"><strong>Reasoning:</strong> {reasoning}</div>',
                            unsafe_allow_html=True,
                        )

                    # Status third (like retrieval)
                    if passes_criteria is not None:
                        status_text = "PASSED" if passes_criteria else "FAILED"
                        status_class = "passed" if passes_criteria else "failed"
                        st.markdown(
                            f'<div style="margin-top: 10px;"><span class="status-badge '
                            f'status-{status_class}">{status_text}</span></div>',
                            unsafe_allow_html=True,
                        )

                    st.markdown("</div>", unsafe_allow_html=True)

                # Only show evaluation here for other categories (not robustness/generation/retrieval/business_value)
                elif category not in ["robustness", "generation", "retrieval", "business_value"] and (
                    eval_details or score
                ):
                    reasoning = eval_details.get("reasoning", "")
                    passes_criteria = eval_details.get("passes_criteria")

                    st.markdown(
                        '<div class="evaluation-details"><span class="section-label">'
                        "Evaluation of Isschat answer:</span>",
                        unsafe_allow_html=True,
                    )

                    # Score first
                    st.markdown(
                        f'<div class="score-display"><span class="section-label">Score:</span>'
                        f'<span class="score-value">{score:.3f}</span></div>',
                        unsafe_allow_html=True,
                    )

                    # Reasoning second
                    if reasoning:
                        st.markdown(
                            f'<div class="reasoning"><strong>Reasoning:</strong> {reasoning}</div>',
                            unsafe_allow_html=True,
                        )

                    # Passed/Failed status last
                    if passes_criteria is not None:
                        status_text = "PASSED" if passes_criteria else "FAILED"
                        status_class = "passed" if passes_criteria else "failed"
                        st.markdown(
                            f'<div style="margin-top: 10px;">'
                            f'<span class="status-badge status-{status_class}">{status_text}</span></div>',
                            unsafe_allow_html=True,
                        )

                    st.markdown("</div>", unsafe_allow_html=True)

                # Show evaluation for business_value with status badge in evaluation details
                elif category == "business_value" and (eval_details or score):
                    reasoning = eval_details.get("reasoning", "")
                    passes_criteria = eval_details.get("passes_criteria")

                    st.markdown(
                        '<div class="evaluation-details"><span class="section-label">'
                        "Evaluation of Isschat answer:</span>",
                        unsafe_allow_html=True,
                    )

                    # Score first
                    st.markdown(
                        f'<div class="score-display"><span class="section-label">Score:</span>'
                        f'<span class="score-value">{score:.3f}</span></div>',
                        unsafe_allow_html=True,
                    )

                    # Reasoning second
                    if reasoning:
                        st.markdown(
                            f'<div class="reasoning"><strong>Reasoning:</strong> {reasoning}</div>',
                            unsafe_allow_html=True,
                        )

                    # Status third (like retrieval)
                    if passes_criteria is not None:
                        status_text = "PASSED" if passes_criteria else "FAILED"
                        status_class = "passed" if passes_criteria else "failed"
                        st.markdown(
                            f'<div style="margin-top: 10px;"><span class="status-badge '
                            f'status-{status_class}">{status_text}</span></div>',
                            unsafe_allow_html=True,
                        )

                    st.markdown("</div>", unsafe_allow_html=True)

                # Metadata - only for non-retrieval categories to avoid showing Expected Documents
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
                        if key not in [
                            "response_time",
                            "expected_documents",
                        ]:  # Avoid duplicating and hide expected_documents
                            st.markdown(
                                f'<span class="metadata-item"><strong>'
                                f"{key.replace('_', ' ').title()}:</strong> {value}</span>",
                                unsafe_allow_html=True,
                            )

                    st.markdown("</div>", unsafe_allow_html=True)

            # Sources (only show for non-retrieval categories since retrieval shows them as Retrieved Documents)
            if category != "retrieval":
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
            <div style="margin-bottom: 15px; padding: 15px; border-left: 4px solid #6c757d; background: #f8f9fa;">
                <strong style="color: #495057;">{desc["title"]}:</strong>
                <div style="margin-top: 8px; color: #495057;">{desc["description"]}</div>
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
        """Show comparison of two selected evaluation files"""
        st.subheader("Evaluation Comparison")

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

        # Selection of two files to compare
        col1, col2 = st.columns(2)

        file_options = list(all_results.keys())

        with col1:
            file1 = st.selectbox("Select first evaluation:", options=file_options, index=0, key="comparison_file1")

        with col2:
            # Filter out the first selected file from second selection
            file2_options = [f for f in file_options if f != file1]
            file2 = (
                st.selectbox("Select second evaluation:", options=file2_options, index=0, key="comparison_file2")
                if file2_options
                else None
            )

        if not file2:
            st.warning("Please select two different evaluation files.")
            return

        # Create comparison dataframe for selected files only
        selected_results = {file1: all_results[file1], file2: all_results[file2]}

        comparison_data = []
        for file_name, data in selected_results.items():
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
        st.subheader("Overall Comparison")
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Category comparison for selected files
        st.subheader("Category Comparison")

        # Get all categories across selected files
        all_categories = set()
        for data in selected_results.values():
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
                for file_name, data in selected_results.items():
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
                else:
                    st.info(f"No data available for category '{selected_category}' in the selected files.")
        else:
            st.info("No categories available for comparison in the selected files.")

    def show_new_evaluation_page(self):
        """Show the new evaluation configuration page"""
        st.header("Run New Evaluation")

        # Check if evaluation config is loaded
        if not self.evaluation_config or not self.available_categories:
            st.error("Error loading evaluation configuration. Please refresh the page.")
            return

        # Category selection
        st.subheader("Select Categories:")
        selected_categories = st.multiselect(
            "Choose which evaluation categories to run:",
            options=self.available_categories,
            default=self.available_categories,
            help="Select one or more categories to evaluate",
        )

        # CI mode toggle
        ci_mode = st.checkbox("CI Mode", value=False, help="Run in CI mode with pass/fail thresholds")

        # Output file name
        st.subheader("Output Configuration:")
        output_name = st.text_input(
            "Output File Name (optional):",
            placeholder="evaluation_results_custom.json",
            help="Custom name for output file (leave empty for auto-generated timestamp)",
        )

        # Action button
        if st.button("Run Evaluation", type="primary"):
            if selected_categories:
                st.session_state.show_evaluation_launcher = True
                st.session_state.evaluation_params = {
                    "categories": selected_categories,
                    "ci_mode": ci_mode,
                    "output_name": output_name,
                }
                st.rerun()
            else:
                st.error("Please select at least one category")

    def run_evaluation_launcher(self):
        """Show evaluation execution page and run evaluation"""
        st.header("Evaluation Execution")

        params = st.session_state.evaluation_params

        # Display selected parameters
        st.subheader("Selected Parameters")
        col1, col2 = st.columns(2)

        with col1:
            st.write(f"**Categories:** {', '.join(params['categories'])}")
            st.write(f"**CI Mode:** {'Yes' if params['ci_mode'] else 'No'}")

        with col2:
            output_file = params["output_name"] if params["output_name"] else "Auto-generated timestamp"
            st.write(f"**Output File:** {output_file}")

        # Progress and status
        if "evaluation_running" not in st.session_state:
            st.session_state.evaluation_running = False

        if "evaluation_result" not in st.session_state:
            st.session_state.evaluation_result = None

        # Action buttons
        col1, col2 = st.columns(2)

        with col1:
            if st.button("Start Evaluation", type="primary") and not st.session_state.evaluation_running:
                st.session_state.evaluation_running = True
                st.session_state.evaluation_result = None
                st.rerun()

        with col2:
            if st.button("Back to Configuration"):
                st.session_state.show_evaluation_launcher = False
                st.session_state.show_new_evaluation = True
                st.session_state.evaluation_running = False
                st.session_state.evaluation_result = None
                st.rerun()

        # Run evaluation if started
        if st.session_state.evaluation_running:
            self.run_evaluation_process(params)

        # Show results if evaluation completed
        if st.session_state.evaluation_result:
            self.show_evaluation_results(st.session_state.evaluation_result)

    def run_evaluation_process(self, params):
        """Run the evaluation process"""
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()

        output_lines = []
        process = None

        try:
            # Build command
            cmd = [sys.executable, "-m", "rag_evaluation.main"]

            # Add categories
            if params["categories"]:
                cmd.extend(["--categories"] + params["categories"])

            # Add CI mode
            if params["ci_mode"]:
                cmd.append("--ci")

            # Add output file
            if params["output_name"]:
                # Use absolute path to ensure it goes to the correct directory
                output_path = Path(__file__).parent.parent / "evaluation_results" / params["output_name"]
                cmd.extend(["--output", str(output_path)])

            status_text.text("Starting evaluation process...")
            progress_bar.progress(10)

            # Run evaluation
            status_text.text("Running evaluation... This may take several minutes.")
            progress_bar.progress(30)

            # Execute the command
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                cwd=Path(__file__).parent.parent,
            )

            # Show output in real-time
            output_container = st.empty()

            # Read output line by line
            while True:
                line = process.stdout.readline()
                if not line:
                    break

                output_lines.append(line.rstrip())
                # Show last 10 lines
                recent_output = "\n".join(output_lines[-10:])
                output_container.code(recent_output)

                # Update progress based on output
                if "Starting" in line:
                    progress_bar.progress(40)
                elif "evaluation" in line.lower() and "completed" in line.lower():
                    progress_bar.progress(80)
                elif "Results saved" in line:
                    progress_bar.progress(90)

            # Wait for process to complete
            return_code = process.wait()

            if return_code == 0:
                status_text.text("Evaluation completed successfully!")
                progress_bar.progress(100)

                # Find the most recent results file
                results_dir = Path(__file__).parent.parent / "evaluation_results"
                if results_dir.exists():
                    result_files = list(results_dir.glob("*.json"))
                    if result_files:
                        latest_file = max(result_files, key=lambda x: x.stat().st_mtime)
                        st.session_state.evaluation_result = {
                            "success": True,
                            "output_file": latest_file,
                            "output": "\n".join(output_lines),
                        }
                    else:
                        st.session_state.evaluation_result = {
                            "success": False,
                            "error": "No result files found",
                            "output": "\n".join(output_lines),
                        }
                else:
                    st.session_state.evaluation_result = {
                        "success": False,
                        "error": "Results directory not found",
                        "output": "\n".join(output_lines),
                    }
            else:
                status_text.text("Evaluation failed!")
                st.session_state.evaluation_result = {
                    "success": False,
                    "error": f"Process exited with code {return_code}",
                    "output": "\n".join(output_lines),
                }

        except Exception as e:
            status_text.text(f"Error running evaluation: {str(e)}")
            st.session_state.evaluation_result = {
                "success": False,
                "error": str(e),
                "output": output_lines if output_lines else [],
            }

        finally:
            # Ensure process and file handles are properly closed
            if process and process.stdout:
                process.stdout.close()
            if process and process.poll() is None:
                process.terminate()
                process.wait()

            st.session_state.evaluation_running = False
            st.rerun()

    def show_evaluation_results(self, result):
        """Show evaluation results"""
        st.markdown("---")
        st.subheader("Evaluation Results")

        if result["success"]:
            st.success("Evaluation completed successfully!")

            if "output_file" in result:
                st.info(f"Results saved to: {result['output_file']}")

                # Add button to view results
                if st.button("View Results"):
                    # Reload available files and switch to results view
                    self.load_available_files()
                    st.session_state.show_evaluation_launcher = False
                    st.session_state.show_new_evaluation = False
                    st.session_state.evaluation_result = None
                    st.rerun()
        else:
            st.error(f"Evaluation failed: {result['error']}")

        # Show output log
        if result["output"]:
            with st.expander("Show Evaluation Log"):
                st.code(result["output"])

    def run(self):
        """Run the dashboard"""
        st.title("Isschat evaluation dashboard")

        # Always render sidebar
        selected_files = self.render_sidebar()

        # Check if we're in evaluation launcher mode
        if "show_evaluation_launcher" in st.session_state and st.session_state.show_evaluation_launcher:
            self.run_evaluation_launcher()
            return

        # Check if we're in new evaluation configuration mode
        if "show_new_evaluation" in st.session_state and st.session_state.show_new_evaluation:
            self.show_new_evaluation_page()
            return

        # Check if we're in comparison mode
        if "show_comparison" in st.session_state and st.session_state.show_comparison:
            self.show_comparison_modal()
            return

        # Main dashboard content
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
