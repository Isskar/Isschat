"""
Feedback evaluator using CamemBERT for multi-class classification
Analyzes user feedback to identify strengths and weaknesses by topic
"""

import logging
import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from rag_evaluation.core.base_evaluator import BaseEvaluator, TestCase, EvaluationResult, EvaluationStatus
from src.core.data_manager import get_data_manager, JSONLDataStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeedbackSentiment(Enum):
    """Binary feedback sentiment"""

    POSITIVE = "positive"  # 👍
    NEGATIVE = "negative"  # 👎


@dataclass
class Topic:
    """Feedback classification topic"""

    id: str
    name: str


@dataclass
class FeedbackEntry:
    """Single feedback entry"""

    question: str
    answer: str
    sentiment: FeedbackSentiment
    comment: Optional[str]
    timestamp: datetime


class FeedbackClassifier:
    """CamemBERT-based feedback classifier"""

    def __init__(self, **kwargs):
        """Initialize classifier with topics and CamemBERT model"""
        self.topics = self._load_topics()
        self._init_camembert(model_name=kwargs.get("model_name", "mtheo/camembert-base-xnli"))

    def _load_topics(self) -> List[Topic]:
        """Load topics from configuration file"""
        config_path = os.path.join(os.path.dirname(__file__), "..", "config", "test_datasets", "feedback_topics.json")

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)

            topics = []
            for category_data in config["categories"].values():
                for topic_data in category_data["topics"]:
                    topics.append(Topic(topic_data["id"], topic_data["name"]))

            logger.info(f"Loaded {len(topics)} topics from configuration")
            return topics

        except Exception as e:
            logger.error(f"Error loading topics configuration: {e}")
            # Fallback to default topics
            return [
                Topic("technical_response_quality", "Qualité des réponses techniques"),
                Topic("information_accuracy", "Précision des informations"),
                Topic("overall_satisfaction", "Satisfaction utilisateur globale"),
            ]

    def _init_camembert(self, model_name: str):
        """Initialize CamemBERT model"""
        from transformers import AutoTokenizer, pipeline

        # Use a lightweight French model for classification
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Initialize classification pipeline once during initialization
        self.classifier_pipeline = pipeline("zero-shot-classification", model=self.model_name, tokenizer=self.tokenizer)

        # TODO: Fine-tune CamemBERT on feedback data
        logger.info("CamemBERT classifier initialized")

    def classify(self, feedback: FeedbackEntry) -> str:
        """Classify feedback into topic"""
        return self._classify_camembert(feedback)

    def _classify_camembert(self, feedback: FeedbackEntry) -> str:
        """CamemBERT-based classification"""
        # Prepare candidate labels
        candidate_labels = [topic.name for topic in self.topics]

        # Combine question and comment for classification
        text_to_classify = ""
        if feedback.question:
            text_to_classify += f"Question: {feedback.question}"
        if feedback.comment:
            if text_to_classify:
                text_to_classify += f" | Commentaire: {feedback.comment}"
            else:
                text_to_classify = f"Commentaire: {feedback.comment}"

        # Classify feedback text using pre-initialized pipeline
        if text_to_classify:
            result = self.classifier_pipeline(text_to_classify, candidate_labels)
            # Get the topic name with highest score
            best_topic_name = max(zip(result["labels"], result["scores"]), key=lambda x: x[1])[0]

            # Find the corresponding topic ID
            for topic in self.topics:
                if topic.name == best_topic_name:
                    return topic.id

            return "unknown"
        else:
            return "unknown"


class FeedbackDataLoader:
    """Loads feedback data from data/logs/feedback directory"""

    def __init__(self, limit: Optional[int] = None):
        """Initialize data loader"""
        self.feedback_data_store: JSONLDataStore = get_data_manager().feedback_store
        self.limit = limit

    def load_feedbacks(self) -> List[FeedbackEntry]:
        """Load feedback entries from data store and convert to evaluator format"""
        try:
            # Load raw feedback data from data manager
            raw_feedbacks = self.feedback_data_store.load_entries(limit=self.limit)

            # Convert to evaluator format
            converted_feedbacks = []
            for raw_feedback in raw_feedbacks:
                # Extract question and answer from metadata
                metadata = raw_feedback.get("metadata", {})
                question = metadata.get("question", "")
                answer = metadata.get("answer", "")

                # Map data manager format to evaluator format
                sentiment = (
                    FeedbackSentiment.POSITIVE if raw_feedback.get("rating", 0) >= 3 else FeedbackSentiment.NEGATIVE
                )

                converted_feedback = FeedbackEntry(
                    question=question,
                    answer=answer,
                    sentiment=sentiment,
                    comment=raw_feedback.get("comment", ""),
                    timestamp=datetime.fromisoformat(raw_feedback.get("timestamp", datetime.now().isoformat())),
                )
                converted_feedbacks.append(converted_feedback)

            return converted_feedbacks

        except Exception as e:
            logger.error(f"Error loading feedback data: {e}")
            return []


@dataclass
class TopicAnalysis:
    """Analysis results for a topic"""

    topic_name: str
    total_count: int
    positive_count: int
    negative_count: int
    satisfaction_rate: float
    examples: List[str]


@dataclass
class FeedbackAnalysis:
    """Complete feedback analysis"""

    total_feedbacks: int
    overall_satisfaction: float
    topic_analyses: Dict[str, TopicAnalysis]
    strengths: List[str]
    weaknesses: List[str]


class FeedbackEvaluator(BaseEvaluator):
    """Feedback evaluator using CamemBERT classification"""

    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.limit = kwargs.get("limit", None)

        # Initialize our specific components
        self.data_loader = FeedbackDataLoader()
        classifier_params = kwargs.get("classifier_params", {})
        self.classifier = FeedbackClassifier(**classifier_params)

    def get_category(self) -> str:
        return "feedback"

    def requires_test_cases(self) -> bool:
        """Feedback evaluator doesn't need test cases - it reads feedback data directly"""
        return False

    def evaluate_single(self, test_case: TestCase) -> EvaluationResult:
        """Custom evaluation method that doesn't query the RAG system"""
        try:
            start_time = datetime.now()

            # Analyze feedback data directly
            analysis = self.analyze_feedback()
            response = self._format_analysis(analysis)
            response_time = (datetime.now() - start_time).total_seconds()

            # Create result
            result = EvaluationResult(
                test_id=test_case.test_id,
                category=test_case.category,
                test_name=test_case.test_name,
                question=test_case.question,
                response=response,
                expected_behavior=test_case.expected_behavior,
                status=EvaluationStatus.MEASURED,
                score=0.0,
                evaluation_details={},
                response_time=response_time,
                sources=[],
                metadata=test_case.metadata,
            )

            self.results.append(result)
            return result

        except Exception as e:
            logger.error(f"Error evaluating feedback: {e}")
            error_result = EvaluationResult(
                test_id=test_case.test_id,
                category=test_case.category,
                test_name=test_case.test_name,
                question=test_case.question,
                response="",
                expected_behavior=test_case.expected_behavior,
                status=EvaluationStatus.ERROR,
                score=0.0,
                error_message=str(e),
                metadata=test_case.metadata,
            )
            self.results.append(error_result)
            return error_result

    def _query_system(self, test_case: TestCase) -> Tuple[str, float, List[str]]:
        """Not used in feedback evaluator"""
        raise NotImplementedError("Feedback evaluator doesn't query the RAG system")

    def _evaluate_semantically(self, test_case: TestCase, response: str) -> Dict[str, Any]:
        """Not used in feedback evaluator"""
        raise NotImplementedError("Feedback evaluator uses custom evaluation logic")

    def analyze_feedback(self) -> FeedbackAnalysis:
        """Main analysis method"""
        # Load feedback data
        feedbacks = self.data_loader.load_feedbacks()

        # Classify feedbacks by topic
        topic_feedbacks = {}
        for feedback in feedbacks:
            topic_id = self.classifier.classify(feedback)
            if topic_id not in topic_feedbacks:
                topic_feedbacks[topic_id] = []
            topic_feedbacks[topic_id].append(feedback)

        # Analyze each topic
        topic_analyses = {}
        for topic_id, topic_feedbacks_list in topic_feedbacks.items():
            topic = next((t for t in self.classifier.topics if t.id == topic_id), None)
            if topic is None:
                logger.warning(f"Topic ID '{topic_id}' not found in classifier topics")
                continue
            analysis = self._analyze_topic(topic, topic_feedbacks_list)
            topic_analyses[topic_id] = analysis

        # Calculate overall metrics
        total_positive = sum(1 for f in feedbacks if f.sentiment == FeedbackSentiment.POSITIVE)
        overall_satisfaction = total_positive / len(feedbacks) if len(feedbacks) > 0 else 0.0

        # Identify strengths and weaknesses
        strengths, weaknesses = self._identify_strengths_weaknesses(topic_analyses)

        analysis = FeedbackAnalysis(
            total_feedbacks=len(feedbacks),
            overall_satisfaction=overall_satisfaction,
            topic_analyses=topic_analyses,
            strengths=strengths,
            weaknesses=weaknesses,
        )

        self.last_analysis = analysis
        return analysis

    def _analyze_topic(self, topic: Topic, feedbacks: List[FeedbackEntry]) -> TopicAnalysis:
        """Analyze feedbacks for a specific topic"""
        positive_count = sum(1 for f in feedbacks if f.sentiment == FeedbackSentiment.POSITIVE)
        negative_count = len(feedbacks) - positive_count
        satisfaction_rate = positive_count / len(feedbacks) if feedbacks else 0

        # Collect text examples
        examples = []
        for feedback in feedbacks[:3]:  # Max 3 examples
            if feedback.comment:
                sentiment_icon = "👍" if feedback.sentiment == FeedbackSentiment.POSITIVE else "👎"
                examples.append(f"{sentiment_icon} {feedback.comment}")

        return TopicAnalysis(
            topic_name=topic.name,
            total_count=len(feedbacks),
            positive_count=positive_count,
            negative_count=negative_count,
            satisfaction_rate=satisfaction_rate,
            examples=examples,
        )

    def _identify_strengths_weaknesses(self, topic_analyses: Dict[str, TopicAnalysis]) -> Tuple[List[str], List[str]]:
        """Identify strengths and weaknesses with actionable insights"""
        strengths = []
        weaknesses = []

        # Load insights from configuration
        topic_insights = self._load_topic_insights()

        # Process all topics, even with low volume, to show CamemBERT classification
        for topic_id, analysis in topic_analyses.items():
            insights = topic_insights.get(
                topic_id,
                {
                    "strength": f"{analysis.topic_name}: {analysis.satisfaction_rate:.0%} satisfaction",
                    "weakness": f"{analysis.topic_name}: {analysis.satisfaction_rate:.0%} satisfaction",
                },
            )

            # Format with CamemBERT classification details
            topic_detail = f"[CamemBERT: {analysis.topic_name}] {analysis.total_count} feedback(s) - {analysis.satisfaction_rate:.0%} satisfaction"  # noqa : E501

            if analysis.satisfaction_rate >= 0.7:
                strengths.append(f"{insights['strength']} ({topic_detail})")
            elif analysis.satisfaction_rate <= 0.4:
                weaknesses.append(f"{insights['weakness']} ({topic_detail})")
            else:
                # Neutral topics - show in both sections for visibility
                strengths.append(f"Topic neutre: {topic_detail}")

        if not strengths:
            strengths.append("Données insuffisantes pour identifier les points forts")
        if not weaknesses:
            weaknesses.append("Aucune faiblesse majeure détectée")

        return strengths, weaknesses

    def _load_topic_insights(self) -> Dict[str, Dict[str, str]]:
        """Load topic insights from configuration file"""
        config = self._load_config()
        insights = {}

        for category_data in config.get("categories", {}).values():
            for topic_data in category_data.get("topics", []):
                topic_id = topic_data["id"]
                insights[topic_id] = {"strength": topic_data["strength"], "weakness": topic_data["weakness"]}

        return insights

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        config_path = os.path.join(os.path.dirname(__file__), "..", "config", "test_datasets", "feedback_topics.json")

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            return {"categories": {}}

    def _empty_analysis(self) -> FeedbackAnalysis:
        """Return empty analysis when no data"""
        return FeedbackAnalysis(
            total_feedbacks=0,
            overall_satisfaction=0.0,
            topic_analyses={},
            strengths=[],
            weaknesses=["Aucun feedback disponible"],
        )

    def _format_analysis(self, analysis: FeedbackAnalysis) -> str:
        """Format analysis as text summary"""
        if analysis.total_feedbacks == 0:
            return "Aucun feedback disponible pour l'analyse"

        lines = [
            "=== ANALYSE FEEDBACK ===",
            f"Total: {analysis.total_feedbacks} feedbacks",
            f"Satisfaction globale: {analysis.overall_satisfaction:.0%}",
            "",
            "RÉPARTITION PAR THÈME:",
        ]

        # Sort topics by count
        sorted_topics = sorted(analysis.topic_analyses.items(), key=lambda x: x[1].total_count, reverse=True)

        for topic_id, topic_analysis in sorted_topics:
            lines.append(
                f"- {topic_analysis.topic_name}: {topic_analysis.total_count} feedbacks ({topic_analysis.satisfaction_rate:.0%})"  # noqa : E501
            )

        lines.extend(
            [
                "",
                "POINTS FORTS:",
            ]
        )
        for strength in analysis.strengths:
            lines.append(f"✓ {strength}")

        lines.extend(
            [
                "",
                "POINTS D'AMÉLIORATION:",
            ]
        )
        for weakness in analysis.weaknesses:
            lines.append(f"⚠ {weakness}")

        return "\n".join(lines)

    def format_detailed_summary(self) -> str:
        """Detailed summary for reporting"""
        if not self.last_analysis:
            return "Aucune analyse disponible"

        analysis = self.last_analysis
        lines = [
            "📊 RAPPORT FEEDBACK CAMEMBERT",
            f"Total: {analysis.total_feedbacks} | Satisfaction: {analysis.overall_satisfaction:.0%}",
            "",
            "🤖 TOPICS CLASSIFIÉS PAR CAMEMBERT:",
        ]

        for topic_id, topic_analysis in analysis.topic_analyses.items():
            if topic_analysis.total_count > 0:
                lines.extend(
                    [
                        f"🏷️ {topic_analysis.topic_name} (ID: {topic_id}) - {topic_analysis.total_count} feedback(s)",
                        f"   👍 {topic_analysis.positive_count} | 👎 {topic_analysis.negative_count} | {topic_analysis.satisfaction_rate:.0%} satisfaction",  # noqa : E501
                    ]
                )

                for example in topic_analysis.examples[:2]:
                    lines.append(f"   💬 {example}")
                lines.append("")

        # Add strengths and weaknesses
        lines.extend(
            [
                "✅ POINTS FORTS IDENTIFIÉS:",
            ]
        )
        for strength in analysis.strengths:
            lines.append(f"   ✓ {strength}")

        lines.extend(
            [
                "",
                "⚠️ POINTS D'AMÉLIORATION:",
            ]
        )
        for weakness in analysis.weaknesses:
            lines.append(f"   ⚠ {weakness}")

        return "\n".join(lines)
