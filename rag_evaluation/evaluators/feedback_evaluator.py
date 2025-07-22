"""
Feedback evaluator using CamemBERT for multi-class classification
Analyzes user feedback to identify strengths and weaknesses by topic

Configuration:
- Uses deployed app feedback data from Azure Blob Storage by default
- Requires USE_AZURE_STORAGE=true in .env for production feedback analysis
- Falls back to local storage for development/testing
"""

import logging
import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from rag_evaluation.core.base_evaluator import BaseEvaluator, TestCase, EvaluationResult, EvaluationStatus
from src.storage.data_manager import get_data_manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress Azure SDK logs to reduce noise
logging.getLogger("azure").setLevel(logging.WARNING)
logging.getLogger("azure.core").setLevel(logging.WARNING)
logging.getLogger("azure.storage").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)


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
                Topic("technical_response_quality", "Technical Response Quality"),
                Topic("information_accuracy", "Information Accuracy"),
                Topic("overall_satisfaction", "Overall User Satisfaction"),
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

    def _classify_camembert(self, feedback: FeedbackEntry) -> Optional[str]:
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

            return None


class FeedbackDataLoader:
    def __init__(self, limit: Optional[int] = None):
        """Initialize data loader"""
        data_manager = get_data_manager()
        logger.info(f"DataManager instance: {data_manager}")
        logger.info(f"DataManager storage type: {type(data_manager.storage).__name__}")
        logger.info(f"DataManager storage: {data_manager.storage}")

        self.feedback_data_store = data_manager.get_feedback_data
        self.limit = limit
        logger.info(f"FeedbackDataLoader initialized with limit: {limit}")

    def load_feedbacks(self) -> Optional[List[FeedbackEntry]]:
        logger.info("=== FEEDBACK LOADING START ===")
        logger.info(f"FeedbackDataLoader limit: {self.limit}")
        logger.info(f"feedback_data_store function: {self.feedback_data_store}")

        try:
            # Load raw feedback data from data manager
            logger.info("Loading feedback data from data manager...")
            # Handle None limit - DataManager expects an integer
            effective_limit = self.limit if self.limit is not None else 100
            logger.info(f"Calling feedback_data_store with limit={effective_limit} (original: {self.limit})")
            raw_feedbacks = self.feedback_data_store(limit=effective_limit)
            logger.info(f"Raw feedbacks result type: {type(raw_feedbacks)}")

            if not raw_feedbacks:
                logger.warning("No feedback data loaded from data manager - empty result")
                logger.info(f"Raw feedbacks value: {raw_feedbacks}")
                return None

            logger.info(f"Loaded {len(raw_feedbacks)} raw feedback entries")
            logger.info(f"Sample raw feedback (first entry): {raw_feedbacks[0] if raw_feedbacks else 'None'}")

            # Convert to evaluator format
            converted_feedbacks = []
            logger.info(f"Starting conversion of {len(raw_feedbacks)} raw feedback entries")

            for i, raw_feedback in enumerate(raw_feedbacks):
                try:
                    logger.debug(f"Processing raw feedback {i + 1}: {raw_feedback}")

                    # Extract question and answer from metadata
                    metadata = raw_feedback.get("metadata", {})
                    question = metadata.get("question", "")
                    answer = metadata.get("answer", "")

                    # Map data manager format to evaluator format
                    rating = raw_feedback.get("rating", 0)
                    sentiment = FeedbackSentiment.POSITIVE if rating >= 3 else FeedbackSentiment.NEGATIVE
                    logger.debug(f"Feedback {i + 1}: rating={rating}, sentiment={sentiment}")

                    # Handle timestamp conversion safely
                    timestamp_str = raw_feedback.get("timestamp", datetime.now().isoformat())
                    try:
                        timestamp = datetime.fromisoformat(timestamp_str)
                    except Exception as ts_e:
                        logger.warning(
                            f"Invalid timestamp '{timestamp_str}' in feedback {i + 1}, using current time: {ts_e}"
                        )
                        timestamp = datetime.now()

                    converted_feedback = FeedbackEntry(
                        question=question,
                        answer=answer,
                        sentiment=sentiment,
                        comment=raw_feedback.get("comment", ""),
                        timestamp=timestamp,
                    )
                    converted_feedbacks.append(converted_feedback)
                    logger.debug(f"Successfully converted feedback {i + 1}")

                except Exception as conv_e:
                    logger.error(f"Error converting feedback entry {i + 1}: {conv_e}")
                    logger.error(f"Raw feedback data: {raw_feedback}")
                    continue  # Skip this entry but continue with others

            logger.info(f"Converted {len(converted_feedbacks)} out of {len(raw_feedbacks)} feedback entries")

            if len(converted_feedbacks) == 0:
                logger.warning("No feedback entries were successfully converted")
                return None

            return converted_feedbacks

        except Exception as e:
            logger.error(f"Error loading feedback data: {e}")
            import traceback

            logger.error(f"Full traceback: {traceback.format_exc()}")
            return None


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

        logger.info("=== FEEDBACK EVALUATOR INITIALIZATION ===")
        logger.info(f"Config: {config}")
        logger.info(f"Kwargs: {kwargs}")
        logger.info(f"Limit parameter: {self.limit}")

        # Initialize our specific components
        logger.info("Initializing FeedbackDataLoader...")
        self.data_loader = FeedbackDataLoader(limit=self.limit)
        logger.info(f"FeedbackDataLoader created: {self.data_loader}")

        classifier_params = kwargs.get("classifier_params", {})
        logger.info(f"Classifier params: {classifier_params}")
        logger.info("Initializing FeedbackClassifier...")
        self.classifier = FeedbackClassifier(**classifier_params)
        logger.info("FeedbackEvaluator initialization completed")

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

            # Create detailed evaluation_details with feedback metrics
            evaluation_details = self._build_evaluation_details(analysis)

            # Create result
            result = EvaluationResult(
                test_id=test_case.test_id,
                category=test_case.category,
                test_name=test_case.test_name,
                question=test_case.question,
                response=response,
                expected_behavior=test_case.expected_behavior,
                status=EvaluationStatus.MEASURED,
                score=analysis.overall_satisfaction,
                evaluation_details=evaluation_details,
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
        logger.info("=== ANALYZE_FEEDBACK START ===")
        logger.info(f"DataLoader instance: {self.data_loader}")
        logger.info(f"DataLoader limit: {self.data_loader.limit}")

        # Load feedback data
        logger.info("About to call data_loader.load_feedbacks()...")
        feedbacks = self.data_loader.load_feedbacks()
        logger.info(
            f"load_feedbacks() returned: {type(feedbacks)} with length: {len(feedbacks) if feedbacks else 'None'}"
        )

        if feedbacks is None or len(feedbacks) == 0:
            logger.warning("No feedback data available for analysis - returning empty analysis")
            empty_result = self._empty_analysis()
            logger.info(f"Empty analysis: total_feedbacks={empty_result.total_feedbacks}")
            return empty_result

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
            f"Overall Satisfaction: {analysis.overall_satisfaction:.0%}",
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
                "AREAS FOR IMPROVEMENT:",
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
                "⚠️ AREAS FOR IMPROVEMENT:",
            ]
        )
        for weakness in analysis.weaknesses:
            lines.append(f"   ⚠ {weakness}")

        return "\n".join(lines)

    def _build_evaluation_details(self, analysis: FeedbackAnalysis) -> Dict[str, Any]:
        """Build detailed evaluation_details with feedback metrics by topic"""
        if not analysis or analysis.total_feedbacks == 0:
            return {
                "feedback_metrics": {
                    "total_feedbacks": 0,
                    "overall_satisfaction": 0.0,
                    "topic_breakdown": {},
                    "top_strengths": [],
                    "top_weaknesses": [],
                },
            }

        # Calculate feedback rates by topic
        topic_breakdown = {}
        for topic_id, topic_analysis in analysis.topic_analyses.items():
            topic_breakdown[topic_id] = {
                "topic_name": topic_analysis.topic_name,
                "total_count": topic_analysis.total_count,
                "positive_count": topic_analysis.positive_count,
                "negative_count": topic_analysis.negative_count,
                "satisfaction_rate": topic_analysis.satisfaction_rate,
                "feedback_percentage": (topic_analysis.total_count / analysis.total_feedbacks) * 100
                if analysis.total_feedbacks > 0
                else 0,
            }

        # Get top 3 most frequent topics (points forts)
        sorted_by_frequency = sorted(analysis.topic_analyses.items(), key=lambda x: x[1].total_count, reverse=True)
        top_strengths = []
        for topic_id, topic_analysis in sorted_by_frequency[:3]:
            if topic_analysis.total_count > 0:
                top_strengths.append(
                    {
                        "topic_id": topic_id,
                        "topic_name": topic_analysis.topic_name,
                        "count": topic_analysis.total_count,
                        "satisfaction_rate": topic_analysis.satisfaction_rate,
                        "reason": "Topic le plus fréquent",
                    }
                )

        # Get top 3 weakest topics (lowest satisfaction rates)
        sorted_by_satisfaction = sorted(analysis.topic_analyses.items(), key=lambda x: x[1].satisfaction_rate)
        top_weaknesses = []
        for topic_id, topic_analysis in sorted_by_satisfaction[:3]:
            if topic_analysis.total_count > 0 and topic_analysis.satisfaction_rate < 0.7:
                top_weaknesses.append(
                    {
                        "topic_id": topic_id,
                        "topic_name": topic_analysis.topic_name,
                        "count": topic_analysis.total_count,
                        "satisfaction_rate": topic_analysis.satisfaction_rate,
                        "reason": f"Low satisfaction rate ({topic_analysis.satisfaction_rate:.0%})",
                    }
                )

        return {
            "feedback_metrics": {
                "total_feedbacks": analysis.total_feedbacks,
                "overall_satisfaction": analysis.overall_satisfaction,
                "topic_breakdown": topic_breakdown,
                "top_strengths": top_strengths,
                "top_weaknesses": top_weaknesses,
            }
        }
