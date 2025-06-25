#!/usr/bin/env python3
"""
HTML Report Generator for Isschat Evaluation System
Generates comprehensive HTML reports with questions, responses, and evaluation results
Includes specialized tabs for different evaluation categories
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from jinja2 import Template


class HTMLReportGenerator:
    """Generates HTML reports from evaluation results"""

    def __init__(self):
        """Initialize the HTML report generator"""
        # Only exclude conversational, keep retrieval for metrics tab
        self.excluded_categories = {"conversational"}

    def generate_report(
        self, results: Dict[str, Any], output_path: Optional[Path] = None, title: str = "Rapport d'Évaluation Isschat"
    ) -> Path:
        """
        Generate HTML report from evaluation results

        Args:
            results: Evaluation results dictionary
            output_path: Optional output file path
            title: Report title

        Returns:
            Path to generated HTML file
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = Path(f"rapport_evaluation_{timestamp}.html")

        # Filter out excluded categories for main tabs
        filtered_results = self._filter_results(results)

        # Generate HTML content with tabs
        html_content = self._generate_html_content(filtered_results, results, title)

        # Write to file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        return output_path

    def _filter_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Filter out excluded categories from results"""
        filtered_results = results.copy()

        # Filter category results
        if "category_results" in filtered_results:
            filtered_category_results = {}
            for category, category_data in filtered_results["category_results"].items():
                if category not in self.excluded_categories:
                    filtered_category_results[category] = category_data
            filtered_results["category_results"] = filtered_category_results

        # Update overall stats to exclude filtered categories
        if "overall_stats" in filtered_results:
            overall_stats = filtered_results["overall_stats"].copy()
            category_results = overall_stats.get("category_results", {})

            # Recalculate stats excluding filtered categories
            total_tests = 0
            total_passed = 0
            total_failed = 0
            total_errors = 0
            total_measured = 0

            filtered_category_stats = {}
            for category, stats in category_results.items():
                if category not in self.excluded_categories:
                    filtered_category_stats[category] = stats
                    total_tests += stats.get("total_tests", 0)
                    total_passed += stats.get("passed", 0)
                    total_failed += stats.get("failed", 0)
                    total_errors += stats.get("errors", 0)
                    total_measured += stats.get("measured", 0)

            overall_stats.update(
                {
                    "total_tests": total_tests,
                    "total_passed": total_passed,
                    "total_failed": total_failed,
                    "total_errors": total_errors,
                    "total_measured": total_measured,
                    "category_results": filtered_category_stats,
                    "overall_pass_rate": total_passed / total_tests if total_tests > 0 else 0.0,
                }
            )

            filtered_results["overall_stats"] = overall_stats

        return filtered_results

    def _generate_html_content(
        self, filtered_results: Dict[str, Any], original_results: Dict[str, Any], title: str
    ) -> str:
        """Generate HTML content with tabs for different categories"""
        template = Template(self._get_html_template())

        # Prepare data for template
        template_data = {
            "title": title,
            "timestamp": filtered_results.get("timestamp", datetime.now().isoformat()),
            "overall_stats": filtered_results.get("overall_stats", {}),
            "category_results": filtered_results.get("category_results", {}),
            "config": filtered_results.get("config", {}),
            "retrieval_data": original_results.get("category_results", {}).get("retrieval", {}),
        }

        return template.render(**template_data)

    def _get_html_template(self) -> str:
        """Get the HTML template with tabs for the report"""
        return """<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
            color: #333;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }
        
        .header {
            text-align: center;
            margin-bottom: 40px;
            padding-bottom: 20px;
            border-bottom: 3px solid #007bff;
        }
        
        .header h1 {
            color: #007bff;
            margin: 0;
            font-size: 2.5em;
        }
        
        .header .timestamp {
            color: #666;
            font-size: 1.1em;
            margin-top: 10px;
        }
        
        /* Tab Navigation */
        .tab-navigation {
            display: flex;
            border-bottom: 2px solid #dee2e6;
            margin-bottom: 30px;
            overflow-x: auto;
        }
        
        .tab-button {
            background: none;
            border: none;
            padding: 15px 25px;
            cursor: pointer;
            font-size: 1.1em;
            font-weight: 500;
            color: #6c757d;
            border-bottom: 3px solid transparent;
            transition: all 0.3s ease;
            white-space: nowrap;
        }
        
        .tab-button:hover {
            color: #007bff;
            background-color: #f8f9fa;
        }
        
        .tab-button.active {
            color: #007bff;
            border-bottom-color: #007bff;
            background-color: #f8f9fa;
        }
        
        .tab-content {
            display: none;
        }
        
        .tab-content.active {
            display: block;
        }
        
        /* Summary Section */
        .summary-section {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            border-radius: 10px;
            margin-bottom: 30px;
        }
        
        .summary-section h2 {
            margin-top: 0;
            font-size: 1.8em;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        
        .stat-card {
            background: rgba(255,255,255,0.1);
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }
        
        .stat-number {
            font-size: 2em;
            font-weight: bold;
            display: block;
        }
        
        .stat-label {
            font-size: 0.9em;
            opacity: 0.9;
        }
        
        /* Metrics Grid for Retrieval */
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .metric-card {
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 20px;
        }
        
        .metric-card h4 {
            margin-top: 0;
            color: #007bff;
            font-size: 1.2em;
            border-bottom: 2px solid #007bff;
            padding-bottom: 10px;
        }
        
        .metric-value {
            font-size: 1.5em;
            font-weight: bold;
            color: #28a745;
            margin: 10px 0;
        }
        
        .metric-description {
            color: #6c757d;
            font-size: 0.9em;
            line-height: 1.4;
        }
        
        .metric-explanation {
            background: #e3f2fd;
            border-left: 4px solid #2196f3;
            padding: 15px;
            margin-top: 15px;
            border-radius: 0 5px 5px 0;
        }
        
        .metric-explanation h5 {
            margin: 0 0 10px 0;
            color: #1976d2;
        }
        
        /* Category Section */
        .category-section {
            margin-bottom: 40px;
        }
        
        .category-header {
            background: #f8f9fa;
            padding: 20px;
            border-left: 5px solid #007bff;
            margin-bottom: 20px;
        }
        
        .category-header h3 {
            margin: 0;
            color: #007bff;
            font-size: 1.5em;
        }
        
        .category-stats {
            display: flex;
            gap: 20px;
            margin-top: 10px;
            flex-wrap: wrap;
        }
        
        .category-stat {
            background: white;
            padding: 10px 15px;
            border-radius: 5px;
            border: 1px solid #dee2e6;
        }
        
        .test-case {
            background: #fff;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            margin-bottom: 20px;
            overflow: hidden;
        }
        
        .test-header {
            background: #f8f9fa;
            padding: 15px 20px;
            border-bottom: 1px solid #dee2e6;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .test-id {
            font-weight: bold;
            color: #495057;
        }
        
        .test-name {
            color: #007bff;
            font-weight: 500;
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
        
        .test-content {
            padding: 20px;
        }
        
        .question-section, .response-section, .expected-section {
            margin-bottom: 20px;
        }
        
        .section-label {
            font-weight: bold;
            color: #495057;
            margin-bottom: 8px;
            display: block;
        }
        
        .question-text {
            background: #e3f2fd;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #2196f3;
            font-style: italic;
        }
        
        .response-text {
            background: #f1f8e9;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #4caf50;
            white-space: pre-wrap;
        }
        
        .expected-text {
            background: #fff3e0;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #ff9800;
            font-size: 0.9em;
        }
        
        .evaluation-details {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            margin-top: 15px;
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
        }
        
        .error-message {
            background: #f8d7da;
            color: #721c24;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #dc3545;
        }
        
        .no-data {
            text-align: center;
            color: #6c757d;
            font-style: italic;
            padding: 40px;
        }
        
        @media (max-width: 768px) {
            .container {
                padding: 15px;
            }
            
            .stats-grid, .metrics-grid {
                grid-template-columns: 1fr;
            }
            
            .category-stats {
                flex-direction: column;
            }
            
            .test-header {
                flex-direction: column;
                align-items: flex-start;
                gap: 10px;
            }
            
            .tab-navigation {
                flex-direction: column;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{{ title }}</h1>
            <div class="timestamp">
                Généré le {{ timestamp[:19].replace('T', ' à ') }}
            </div>
        </div>
        
        <!-- Tab Navigation -->
        <div class="tab-navigation">
            <button class="tab-button active" onclick="showTab('summary')">📊 Résumé</button>
            {% for cat in category_results.keys() if cat != 'retrieval' %}
            <button class="tab-button" onclick="showTab('{{ cat }}')">{{ cat|replace('_',' ')|title }}</button>
            {% endfor %}
            {% if retrieval_data %}
            <button class="tab-button" onclick="showTab('retrieval')">🔍 Métriques Retrieval</button>
            {% endif %}
        </div>
        
        <!-- Summary Tab -->
        <div id="summary" class="tab-content active">
            <div class="summary-section">
                <h2>📊 Résumé Global</h2>
                <div class="stats-grid">
                    <div class="stat-card">
                        <span class="stat-number">{{ overall_stats.total_tests or 0 }}</span>
                        <span class="stat-label">Tests Total</span>
                    </div>
                    <div class="stat-card">
                        <span class="stat-number">{{ overall_stats.total_passed or 0 }}</span>
                        <span class="stat-label">Réussis</span>
                    </div>
                    <div class="stat-card">
                        <span class="stat-number">{{ overall_stats.total_failed or 0 }}</span>
                        <span class="stat-label">Échoués</span>
                    </div>
                    <div class="stat-card">
                        <span class="stat-number">{{ overall_stats.total_measured or 0 }}</span>
                        <span class="stat-label">Mesurés</span>
                    </div>
                    {% if overall_stats.total_errors %}
                    <div class="stat-card">
                        <span class="stat-number">{{ overall_stats.total_errors }}</span>
                        <span class="stat-label">Erreurs</span>
                    </div>
                    {% endif %}
                    {% if overall_stats.overall_pass_rate is defined %}
                    <div class="stat-card">
                        <span class="stat-number">{{ "%.1f"|format(overall_stats.overall_pass_rate * 100) }}%</span>
                        <span class="stat-label">Taux de Réussite</span>
                    </div>
                    {% endif %}
                </div>
            </div>
            
            <!-- Category Summary -->
            {% for category, category_data in category_results.items() %}
            <div class="category-section">
                <div class="category-header">
                    <h3>{{ category.title() }}</h3>
                    <div class="category-stats">
                        {% set summary = category_data.summary %}
                        <div class="category-stat">
                            <strong>Tests:</strong> {{ summary.total_tests or 0 }}
                        </div>
                        <div class="category-stat">
                            <strong>Réussis:</strong> {{ summary.passed or 0 }}
                        </div>
                        {% if summary.failed %}
                        <div class="category-stat">
                            <strong>Échoués:</strong> {{ summary.failed }}
                        </div>
                        {% endif %}
                        {% if summary.measured %}
                        <div class="category-stat">
                            <strong>Mesurés:</strong> {{ summary.measured }}
                        </div>
                        {% endif %}
                        {% if summary.average_score is defined %}
                        <div class="category-stat">
                            <strong>Score Moyen:</strong> {{ "%.3f"|format(summary.average_score) }}
                        </div>
                        {% endif %}
                        {% if summary.pass_rate is defined %}
                        <div class="category-stat">
                            <strong>Taux de Réussite:</strong> {{ "%.1f"|format(summary.pass_rate * 100) }}%
                        </div>
                        {% endif %}
                        {% if summary.average_response_time is defined %}
                        <div class="category-stat">
                            <strong>Temps Moyen de Réponse:</strong> {{ "%.2f"|format(summary.average_response_time) }}s
                        </div>
                        {% endif %}
                    </div>
                </div>
            </div>
            {% endfor %}
        </div>
        
        <!-- Category Tabs -->
        {% for category, category_data in category_results.items() if category != 'retrieval' %}
        <div id="{{ category }}" class="tab-content">
            <div class="category-section">
                <div class="category-header">
                    <h3>{{ category|replace('_',' ')|title }}</h3>
                    <div class="category-stats">
                        {% set summary = category_data.summary %}
                        <div class="category-stat">
                            <strong>Tests:</strong> {{ summary.total_tests or 0 }}
                        </div>
                        <div class="category-stat">
                            <strong>Réussis:</strong> {{ summary.passed or 0 }}
                        </div>
                        {% if summary.failed %}
                        <div class="category-stat">
                            <strong>Échoués:</strong> {{ summary.failed }}
                        </div>
                        {% endif %}
                        {% if summary.average_score is defined %}
                        <div class="category-stat">
                            <strong>Score Moyen:</strong> {{ "%.3f"|format(summary.average_score) }}
                        </div>
                        {% endif %}
                        {% if summary.pass_rate is defined %}
                        <div class="category-stat">
                            <strong>Taux de Réussite:</strong> {{ "%.1f"|format(summary.pass_rate * 100) }}%
                        </div>
                        {% endif %}
                        {% if summary.average_response_time is defined %}
                        <div class="category-stat">
                            <strong>Temps Moyen de Réponse:</strong> {{ "%.2f"|format(summary.average_response_time) }}s
                        </div>
                        {% endif %}
                    </div>
                </div>
                
                {% if category_data.results %}
                    {% for result in category_data.results %}
                    <div class="test-case">
                        <div class="test-header">
                            <div>
                                <span class="test-id">{{ result.test_id }}</span>
                                <span class="test-name">{{ result.test_name }}</span>
                            </div>
                            <span class="status-badge status-{{ result.status.lower() }}">
                                {{ result.status }}
                            </span>
                        </div>
                        
                        <div class="test-content">
                            <div class="question-section">
                                <span class="section-label">❓ Question:</span>
                                <div class="question-text">{{ result.question }}</div>
                            </div>
                            
                            {% if result.response and not result.error_message %}
                            <div class="response-section">
                                <span class="section-label">💬 Réponse d'Isschat:</span>
                                <div class="response-text">{{ result.response }}</div>
                            </div>
                            {% endif %}
                            
                            {% if result.error_message %}
                            <div class="error-message">
                                <strong>❌ Erreur:</strong> {{ result.error_message }}
                            </div>
                            {% endif %}
                            
                            <div class="expected-section">
                                <span class="section-label">🎯 Comportement Attendu:</span>
                                <div class="expected-text">{{ result.expected_behavior }}</div>
                            </div>
                            
                            {% if result.evaluation_details %}
                            <div class="evaluation-details">
                                <div class="score-display">
                                    <span class="section-label">📈 Score:</span>
                                    <span class="score-value">{{ "%.3f"|format(result.score) }}</span>
                                    {% if result.evaluation_details.passes_criteria is defined %}
                                    <span class="status-badge status-{{ 'passed' if result.evaluation_details.passes_criteria else 'failed' }}">
                                        {{ 'Critères Respectés' if result.evaluation_details.passes_criteria else 'Critères Non Respectés' }}
                                    </span>
                                    {% endif %}
                                </div>
                                {% if result.evaluation_details.reasoning %}
                                <div class="reasoning">
                                    <strong>💭 Raisonnement:</strong> {{ result.evaluation_details.reasoning }}
                                </div>
                                {% endif %}
                            </div>
                            {% endif %}
                            
                            {% if result.metadata %}
                            <div class="metadata">
                                {% for key, value in result.metadata.items() %}
                                <span class="metadata-item">
                                    <strong>{{ key.replace('_', ' ').title() }}:</strong> {{ value }}
                                </span>
                                {% endfor %}
                            </div>
                            {% endif %}
                            
                            {% if result.sources %}
                            <div class="sources">
                                <span class="section-label">📚 Sources:</span>
                                {% for source in result.sources %}
                                <div class="source-item">{{ source }}</div>
                                {% endfor %}
                            </div>
                            {% endif %}
                            
                            {% if result.response_time %}
                            <div class="metadata">
                                <span class="metadata-item">
                                    <strong>⏱️ Temps de Réponse:</strong> {{ "%.2f"|format(result.response_time) }}s
                                </span>
                            </div>
                            {% endif %}
                        </div>
                    </div>
                    {% endfor %}
                {% else %}
                    <div class="no-data">
                        Aucun résultat disponible pour cette catégorie.
                    </div>
                {% endif %}
            </div>
        </div>
        {% endfor %}
        
        <!-- Retrieval Metrics Tab -->
        {% if retrieval_data %}
        <div id="retrieval" class="tab-content">
            <div class="category-section">
                <div class="category-header">
                    <h3>🔍 Métriques de Performance du Retrieval</h3>
                    {% set summary = retrieval_data.summary %}
                    <div class="category-stats">
                        <div class="category-stat">
                            <strong>Tests:</strong> {{ summary.total_tests or 0 }}
                        </div>
                        <div class="category-stat">
                            <strong>Score Moyen:</strong> {{ "%.3f"|format(summary.average_score or 0) }}
                        </div>
                        {% if summary.average_retrieval_time_ms %}
                        <div class="category-stat">
                            <strong>Temps Moyen:</strong> {{ "%.1f"|format(summary.average_retrieval_time_ms) }}ms
                        </div>
                        {% endif %}
                    </div>
                </div>
                
                <!-- Metrics Grid -->
                <div class="metrics-grid">
                    <!-- Basic Metrics -->
                    <div class="metric-card">
                        <h4>📊 Métriques de Base</h4>
                        {% set summary = retrieval_data.summary %}
                        <div class="metric-value">
                            Precision: {{ "%.3f"|format(summary.average_precision or 0) }}
                        </div>
                        <div class="metric-value">
                            Recall: {{ "%.3f"|format(summary.average_recall or 0) }}
                        </div>
                        <div class="metric-value">
                            F1-Score: {{ "%.3f"|format(summary.average_f1_score or 0) }}
                        </div>
                        <div class="metric-explanation">
                            <h5>📝 Explication</h5>
                            <strong>Precision:</strong> Proportion de documents récupérés qui sont pertinents.<br>
                            <strong>Recall:</strong> Proportion de documents pertinents qui ont été récupérés.<br>
                            <strong>F1-Score:</strong> Moyenne harmonique entre precision et recall.
                        </div>
                    </div>
                    
                    <!-- Precision@K Metrics -->
                    <div class="metric-card">
                        <h4>🎯 Precision@K</h4>
                        <div class="metric-value">
                            P@1: {{ "%.3f"|format(summary.average_precision_at_1 or 0) }}
                        </div>
                        <div class="metric-value">
                            P@3: {{ "%.3f"|format(summary.average_precision_at_3 or 0) }}
                        </div>
                        <div class="metric-value">
                            P@5: {{ "%.3f"|format(summary.average_precision_at_5 or 0) }}
                        </div>
                        <div class="metric-explanation">
                            <h5>📝 Explication</h5>
                            <strong>Precision@K:</strong> Proportion de documents pertinents parmi les K premiers résultats retournés. Mesure la qualité des premiers résultats.
                        </div>
                    </div>
                    
                    <!-- Recall@K Metrics -->
                    <div class="metric-card">
                        <h4>🔄 Recall@K</h4>
                        <div class="metric-value">
                            R@1: {{ "%.3f"|format(summary.average_recall_at_1 or 0) }}
                        </div>
                        <div class="metric-value">
                            R@3: {{ "%.3f"|format(summary.average_recall_at_3 or 0) }}
                        </div>
                        <div class="metric-value">
                            R@5: {{ "%.3f"|format(summary.average_recall_at_5 or 0) }}
                        </div>
                        <div class="metric-explanation">
                            <h5>📝 Explication</h5>
                            <strong>Recall@K:</strong> Proportion de documents pertinents trouvés dans les K premiers résultats. Mesure la capacité à retrouver les documents importants rapidement.
                        </div>
                    </div>
                    
                    <!-- Advanced Metrics -->
                    <div class="metric-card">
                        <h4>🚀 Métriques Avancées</h4>
                        <div class="metric-value">
                            MRR: {{ "%.3f"|format(summary.average_mrr or 0) }}
                        </div>
                        <div class="metric-value">
                            MAP: {{ "%.3f"|format(summary.average_map or 0) }}
                        </div>
                        <div class="metric-value">
                            NDCG@5: {{ "%.3f"|format(summary.average_ndcg_at_5 or 0) }}
                        </div>
                        <div class="metric-explanation">
                            <h5>📝 Explication</h5>
                            <strong>MRR:</strong> Mean Reciprocal Rank - moyenne de l'inverse du rang du premier document pertinent.<br>
                            <strong>MAP:</strong> Mean Average Precision - moyenne des précisions à chaque document pertinent trouvé.<br>
                            <strong>NDCG:</strong> Normalized Discounted Cumulative Gain - mesure la qualité du classement en tenant compte de la position.
                        </div>
                    </div>
                </div>
                
                <!-- Individual Test Results -->
                {% if retrieval_data.results %}
                <h3>📋 Résultats Détaillés des Tests</h3>
                {% for result in retrieval_data.results %}
                <div class="test-case">
                    <div class="test-header">
                        <div>
                            <span class="test-id">{{ result.test_id }}</span>
                            <span class="test-name">{{ result.test_name }}</span>
                        </div>
                        <span class="status-badge status-{{ result.status.lower() }}">
                            {{ result.status }}
                        </span>
                    </div>
                    
                    <div class="test-content">
                        <div class="question-section">
                            <span class="section-label">❓ Question:</span>
                            <div class="question-text">{{ result.question }}</div>
                        </div>
                        
                        {% if result.evaluation_details %}
                        <div class="evaluation-details">
                            <div class="score-display">
                                <span class="section-label">📈 Score Global:</span>
                                <span class="score-value">{{ "%.3f"|format(result.score) }}</span>
                            </div>
                            
                            <!-- Detailed Metrics -->
                            <div class="metrics-grid">
                                <div class="metric-card">
                                    <h4>Métriques de Base</h4>
                                    <div>Precision: {{ "%.3f"|format(result.evaluation_details.precision or 0) }}</div>
                                    <div>Recall: {{ "%.3f"|format(result.evaluation_details.recall or 0) }}</div>
                                    <div>F1-Score: {{ "%.3f"|format(result.evaluation_details.f1_score or 0) }}</div>
                                </div>
                                
                                <div class="metric-card">
                                    <h4>Precision@K</h4>
                                    <div>P@1: {{ "%.3f"|format(result.evaluation_details.precision_at_1 or 0) }}</div>
                                    <div>P@3: {{ "%.3f"|format(result.evaluation_details.precision_at_3 or 0) }}</div>
                                    <div>P@5: {{ "%.3f"|format(result.evaluation_details.precision_at_5 or 0) }}</div>
                                </div>
                                
                                <div class="metric-card">
                                    <h4>Recall@K</h4>
                                    <div>R@1: {{ "%.3f"|format(result.evaluation_details.recall_at_1 or 0) }}</div>
                                    <div>R@3: {{ "%.3f"|format(result.evaluation_details.recall_at_3 or 0) }}</div>
                                    <div>R@5: {{ "%.3f"|format(result.evaluation_details.recall_at_5 or 0) }}</div>
                                </div>
                                
                                <div class="metric-card">
                                    <h4>Métriques Avancées</h4>
                                    <div>MRR: {{ "%.3f"|format(result.evaluation_details.mrr or 0) }}</div>
                                    <div>MAP: {{ "%.3f"|format(result.evaluation_details.map or 0) }}</div>
                                    <div>NDCG@5: {{ "%.3f"|format(result.evaluation_details.ndcg_at_5 or 0) }}</div>
                                </div>
                            </div>
                        </div>
                        {% endif %}
                        
                        {% if result.sources %}
                        <div class="sources">
                            <span class="section-label">📚 Documents Récupérés:</span>
                            {% for source in result.sources %}
                            <div class="source-item">{{ source }}</div>
                            {% endfor %}
                        </div>
                        {% endif %}
                        
                        {% if result.response_time %}
                        <div class="metadata">
                            <span class="metadata-item">
                                <strong>⏱️ Temps de Retrieval:</strong> {{ "%.2f"|format(result.response_time) }}ms
                            </span>
                        </div>
                        {% endif %}
                    </div>
                </div>
                {% endfor %}
                {% else %}
                <div class="no-data">
                    Aucun résultat de retrieval disponible.
                </div>
                {% endif %}
            </div>
        </div>
        {% endif %}
        
        {% if not category_results and not retrieval_data %}
        <div class="no-data">
            <h3>Aucune donnée d'évaluation disponible</h3>
            <p>Aucun résultat d'évaluation trouvé.</p>
        </div>
        {% endif %}
    </div>
    
    <script>
        function showTab(tabName) {
            // Hide all tab contents
            const tabContents = document.querySelectorAll('.tab-content');
            tabContents.forEach(content => {
                content.classList.remove('active');
            });
            
            // Remove active class from all tab buttons
            const tabButtons = document.querySelectorAll('.tab-button');
            tabButtons.forEach(button => {
                button.classList.remove('active');
            });
            
            // Show selected tab content
            document.getElementById(tabName).classList.add('active');
            
            // Add active class to clicked button
            event.target.classList.add('active');
        }
    </script>
</body>
</html>"""


def generate_html_report(
    results_file: Path, output_file: Optional[Path] = None, title: str = "Rapport d'Évaluation Isschat"
) -> Path:
    """
    Generate HTML report from JSON results file

    Args:
        results_file: Path to JSON results file
        output_file: Optional output HTML file path
        title: Report title

    Returns:
        Path to generated HTML file
    """
    # Load results
    with open(results_file, "r", encoding="utf-8") as f:
        results = json.load(f)

    # Generate report
    generator = HTMLReportGenerator()
    return generator.generate_report(results, output_file, title)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate HTML report from evaluation results")
    parser.add_argument("results_file", type=Path, help="Path to JSON results file")
    parser.add_argument("--output", "-o", type=Path, help="Output HTML file path")
    parser.add_argument("--title", "-t", default="Rapport d'Évaluation Isschat", help="Report title")

    args = parser.parse_args()

    if not args.results_file.exists():
        print(f"❌ Results file not found: {args.results_file}")
        exit(1)

    try:
        output_path = generate_html_report(args.results_file, args.output, args.title)
        print(f"✅ HTML report generated: {output_path}")
    except Exception as e:
        print(f"❌ Error generating report: {e}")
        exit(1)
