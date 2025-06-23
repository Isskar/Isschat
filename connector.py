"""
Abstract base class for data pipeline connectors with trigger support.

This module defines a comprehensive connector architecture for data ingestion pipelines
that handles:
- Connection management with source/destination
- Communication protocol management
- Data transformation to compatible formats
- Authentication and authorization
- Different trigger types (manual, cron, event-driven)

Based on the discussion about connector requirements:
1. Establish connection with source/destination
2. Manage communication protocols
3. Transform data to compatible format
4. Handle authentication and authorization
5. Support different trigger types (event-driven, cron, manual)
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Iterator, Callable
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
import asyncio


# === ENUMERATIONS ===


class ConnectionStatus(Enum):
    """Connection status enumeration"""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"
    AUTHENTICATED = "authenticated"


class ProtocolType(Enum):
    """Supported communication protocols"""

    HTTP = "http"
    HTTPS = "https"
    FTP = "ftp"
    SFTP = "sftp"
    DATABASE = "database"
    FILE_SYSTEM = "file_system"
    API_REST = "api_rest"
    WEBSOCKET = "websocket"


class TriggerType(Enum):
    """Types de déclencheurs pour les connecteurs"""

    MANUAL = "manual"  # Push manuel de données
    CRON = "cron"  # Planification basée sur cron
    EVENT_DRIVEN = "event_driven"  # Déclenché par des événements


class TriggerStatus(Enum):
    """Statut des triggers"""

    INACTIVE = "inactive"
    ACTIVE = "active"
    PAUSED = "paused"
    ERROR = "error"


# === DATA CLASSES ===


@dataclass
class Document:
    """Document representation compatible with existing system"""

    page_content: str
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert document to dictionary format for vector store."""
        return {"page_content": self.page_content, "metadata": self.metadata}

    @property
    def content(self) -> str:
        """Alias for page_content to maintain backward compatibility."""
        return self.page_content


@dataclass
class ConnectionConfig:
    """Configuration for connector connection"""

    host: str
    port: Optional[int] = None
    protocol: ProtocolType = ProtocolType.HTTPS
    timeout: int = 30
    retry_attempts: int = 3
    ssl_verify: bool = True
    additional_params: Dict[str, Any] = None


@dataclass
class AuthenticationConfig:
    """Authentication configuration"""

    auth_type: str  # "basic", "oauth", "token", "certificate", etc.
    credentials: Dict[str, Any]
    token_refresh_url: Optional[str] = None
    token_expiry: Optional[int] = None


@dataclass
class TransformationConfig:
    """Data transformation configuration"""

    input_format: str  # "json", "xml", "csv", "binary", etc.
    output_format: str = "document"  # Always convert to Document format
    encoding: str = "utf-8"
    transformation_rules: Dict[str, Any] = None


@dataclass
class TriggerConfig:
    """Configuration des triggers"""

    trigger_type: TriggerType
    cron_expression: Optional[str] = None  # Pour CRON: "0 */6 * * *"
    event_source: Optional[str] = None  # Pour EVENT_DRIVEN: webhook URL, queue name, etc.
    batch_size: int = 100  # Taille des lots pour le traitement
    max_retries: int = 3
    retry_delay: int = 60  # Délai entre les tentatives (secondes)
    enabled: bool = True


@dataclass
class TriggerEvent:
    """Représentation d'un événement de trigger"""

    trigger_id: str
    trigger_type: TriggerType
    timestamp: datetime
    payload: Optional[Dict[str, Any]] = None
    source: Optional[str] = None
    metadata: Dict[str, Any] = None


# === MAIN CONNECTOR CLASS ===


class BaseConnector(ABC):
    """
    Abstract base class for data pipeline connectors with trigger support.

    Responsibilities:
    - Establish connection with source/destination
    - Manage communication protocols
    - Transform data to compatible format
    - Handle authentication and authorization
    - Manage different trigger types (manual, cron, event-driven)
    """

    def __init__(
        self,
        connection_config: ConnectionConfig,
        trigger_config: TriggerConfig,
        auth_config: Optional[AuthenticationConfig] = None,
        transform_config: Optional[TransformationConfig] = None,
    ):
        # Configuration existante
        self.connection_config = connection_config
        self.auth_config = auth_config
        self.transform_config = transform_config or TransformationConfig("json")

        # Nouvelle configuration pour les triggers
        self.trigger_config = trigger_config
        self._trigger_status = TriggerStatus.INACTIVE
        self._trigger_task: Optional[asyncio.Task] = None
        self._event_handlers: Dict[str, Callable] = {}
        self._last_execution: Optional[datetime] = None
        self._next_execution: Optional[datetime] = None

        # État existant
        self._connection_status = ConnectionStatus.DISCONNECTED
        self._authenticated = False

    # === CONNECTION MANAGEMENT ===

    @abstractmethod
    async def connect(self) -> bool:
        """
        Establish connection with the data source/destination.

        Returns:
            bool: True if connection successful
        """
        pass

    @abstractmethod
    async def disconnect(self) -> bool:
        """
        Close connection with the data source/destination.

        Returns:
            bool: True if disconnection successful
        """
        pass

    @abstractmethod
    def validate_connection(self) -> bool:
        """
        Validate current connection status.

        Returns:
            bool: True if connection is valid and active
        """
        pass

    def get_connection_status(self) -> ConnectionStatus:
        """Get current connection status."""
        return self._connection_status

    # === PROTOCOL MANAGEMENT ===

    @abstractmethod
    def setup_protocol(self) -> bool:
        """
        Setup communication protocol based on configuration.

        Returns:
            bool: True if protocol setup successful
        """
        pass

    @abstractmethod
    async def send_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send request using configured protocol.

        Args:
            request_data: Request payload

        Returns:
            Dict[str, Any]: Response data
        """
        pass

    def get_protocol_info(self) -> Dict[str, Any]:
        """Get information about current protocol configuration."""
        return {
            "protocol": self.connection_config.protocol.value,
            "host": self.connection_config.host,
            "port": self.connection_config.port,
            "ssl_verify": self.connection_config.ssl_verify,
        }

    # === AUTHENTICATION & AUTHORIZATION ===

    @abstractmethod
    async def authenticate(self) -> bool:
        """
        Perform authentication with the data source.

        Returns:
            bool: True if authentication successful
        """
        pass

    @abstractmethod
    async def refresh_authentication(self) -> bool:
        """
        Refresh authentication tokens/credentials.

        Returns:
            bool: True if refresh successful
        """
        pass

    @abstractmethod
    def check_authorization(self, resource: str, action: str) -> bool:
        """
        Check if current credentials have authorization for specific action.

        Args:
            resource: Resource identifier
            action: Action to perform (read, write, delete, etc.)

        Returns:
            bool: True if authorized
        """
        pass

    def is_authenticated(self) -> bool:
        """Check if connector is currently authenticated."""
        return self._authenticated

    # === DATA TRANSFORMATION ===

    @abstractmethod
    def transform_input_data(self, raw_data: Any) -> List[Document]:
        """
        Transform raw input data to Document format.

        Args:
            raw_data: Raw data from source

        Returns:
            List[Document]: Transformed documents
        """
        pass

    @abstractmethod
    def transform_output_data(self, documents: List[Document]) -> Any:
        """
        Transform Document format to destination format.

        Args:
            documents: List of documents to transform

        Returns:
            Any: Data in destination format
        """
        pass

    def validate_data_format(self, data: Any) -> bool:
        """
        Validate if data matches expected format.

        Args:
            data: Data to validate

        Returns:
            bool: True if format is valid
        """
        try:
            if self.transform_config.input_format == "json":
                import json

                if isinstance(data, str):
                    json.loads(data)
                elif isinstance(data, dict):
                    json.dumps(data)
                return True
            # Add other format validations as needed
            return True
        except Exception:
            return False

    # === TRIGGER MANAGEMENT ===

    async def start_trigger(self) -> bool:
        """
        Start the trigger based on configuration.

        Returns:
            bool: True if trigger started successfully
        """
        if self._trigger_status == TriggerStatus.ACTIVE:
            return True

        try:
            if self.trigger_config.trigger_type == TriggerType.CRON:
                self._trigger_task = asyncio.create_task(self._run_cron_trigger())
            elif self.trigger_config.trigger_type == TriggerType.EVENT_DRIVEN:
                self._trigger_task = asyncio.create_task(self._run_event_trigger())
            # MANUAL ne nécessite pas de tâche en arrière-plan

            self._trigger_status = TriggerStatus.ACTIVE
            return True
        except Exception as e:
            self._trigger_status = TriggerStatus.ERROR
            return False

    async def stop_trigger(self) -> bool:
        """
        Stop the active trigger.

        Returns:
            bool: True if trigger stopped successfully
        """
        if self._trigger_task:
            self._trigger_task.cancel()
            try:
                await self._trigger_task
            except asyncio.CancelledError:
                pass
            self._trigger_task = None

        self._trigger_status = TriggerStatus.INACTIVE
        return True

    async def pause_trigger(self) -> bool:
        """Pause the trigger temporarily."""
        if self._trigger_status == TriggerStatus.ACTIVE:
            self._trigger_status = TriggerStatus.PAUSED
            return True
        return False

    async def resume_trigger(self) -> bool:
        """Resume a paused trigger."""
        if self._trigger_status == TriggerStatus.PAUSED:
            self._trigger_status = TriggerStatus.ACTIVE
            return True
        return False

    # === TRIGGER IMPLEMENTATIONS ===

    async def _run_cron_trigger(self):
        """Run cron-based trigger loop."""
        while self._trigger_status in [TriggerStatus.ACTIVE, TriggerStatus.PAUSED]:
            if self._trigger_status == TriggerStatus.ACTIVE:
                if self._should_execute_cron():
                    await self._execute_trigger(
                        TriggerEvent(
                            trigger_id=f"cron_{datetime.now().isoformat()}",
                            trigger_type=TriggerType.CRON,
                            timestamp=datetime.now(),
                            source="cron_scheduler",
                        )
                    )

            await asyncio.sleep(60)  # Check every minute

    async def _run_event_trigger(self):
        """Run event-driven trigger loop."""
        await self.setup_event_listener()

        while self._trigger_status in [TriggerStatus.ACTIVE, TriggerStatus.PAUSED]:
            if self._trigger_status == TriggerStatus.ACTIVE:
                event = await self.wait_for_event()
                if event:
                    await self._execute_trigger(event)
            else:
                await asyncio.sleep(1)

    async def manual_trigger(self, payload: Optional[Dict[str, Any]] = None) -> bool:
        """
        Manually trigger data processing.

        Args:
            payload: Optional data payload for manual trigger

        Returns:
            bool: True if manual trigger executed successfully
        """
        if self.trigger_config.trigger_type != TriggerType.MANUAL:
            return False

        event = TriggerEvent(
            trigger_id=f"manual_{datetime.now().isoformat()}",
            trigger_type=TriggerType.MANUAL,
            timestamp=datetime.now(),
            payload=payload,
            source="manual_user",
        )

        return await self._execute_trigger(event)

    async def _execute_trigger(self, event: TriggerEvent) -> bool:
        """
        Execute the trigger processing pipeline.

        Args:
            event: Trigger event information

        Returns:
            bool: True if execution successful
        """
        try:
            self._last_execution = datetime.now()

            # Pre-execution hook
            await self.on_trigger_start(event)

            # Main processing
            if event.trigger_type == TriggerType.MANUAL and event.payload:
                # Pour manual, on peut avoir des données directes
                documents = self.transform_input_data(event.payload)
            else:
                # Pour cron et event-driven, on extrait les données
                documents = await self.extract_data(event.payload)

            # Process in batches
            batch_size = self.trigger_config.batch_size
            for i in range(0, len(documents), batch_size):
                batch = documents[i : i + batch_size]
                await self.process_batch(batch, event)

            # Post-execution hook
            await self.on_trigger_complete(event, len(documents))

            return True

        except Exception as e:
            await self.on_trigger_error(event, e)
            return False

    # === ABSTRACT TRIGGER METHODS ===

    @abstractmethod
    async def setup_event_listener(self) -> bool:
        """
        Setup event listener for event-driven triggers.
        Implementation depends on event source (webhook, queue, file watcher, etc.)

        Returns:
            bool: True if setup successful
        """
        pass

    @abstractmethod
    async def wait_for_event(self) -> Optional[TriggerEvent]:
        """
        Wait for and return the next event.

        Returns:
            Optional[TriggerEvent]: Next event or None if timeout
        """
        pass

    @abstractmethod
    async def process_batch(self, documents: List[Document], event: TriggerEvent) -> bool:
        """
        Process a batch of documents.

        Args:
            documents: Batch of documents to process
            event: Trigger event that initiated this processing

        Returns:
            bool: True if batch processed successfully
        """
        pass

    # === TRIGGER HOOKS ===

    async def on_trigger_start(self, event: TriggerEvent):
        """Hook called when trigger execution starts."""
        pass

    async def on_trigger_complete(self, event: TriggerEvent, processed_count: int):
        """Hook called when trigger execution completes successfully."""
        pass

    async def on_trigger_error(self, event: TriggerEvent, error: Exception):
        """Hook called when trigger execution fails."""
        pass

    # === HIGH-LEVEL OPERATIONS ===

    async def extract_data(self, query_params: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        High-level method to extract data from source.

        Args:
            query_params: Optional query parameters

        Returns:
            List[Document]: Extracted documents
        """
        if not self.validate_connection():
            await self.connect()

        if self.auth_config and not self.is_authenticated():
            await self.authenticate()

        raw_data = await self._fetch_raw_data(query_params)
        return self.transform_input_data(raw_data)

    async def load_data(self, documents: List[Document]) -> bool:
        """
        High-level method to load data to destination.

        Args:
            documents: Documents to load

        Returns:
            bool: True if load successful
        """
        if not self.validate_connection():
            await self.connect()

        if self.auth_config and not self.is_authenticated():
            await self.authenticate()

        transformed_data = self.transform_output_data(documents)
        return await self._send_data(transformed_data)

    # === ABSTRACT HELPER METHODS ===

    @abstractmethod
    async def _fetch_raw_data(self, query_params: Optional[Dict[str, Any]] = None) -> Any:
        """Fetch raw data from source."""
        pass

    @abstractmethod
    async def _send_data(self, data: Any) -> bool:
        """Send data to destination."""
        pass

    # === UTILITY METHODS ===

    def _should_execute_cron(self) -> bool:
        """Check if cron trigger should execute now."""
        if not self.trigger_config.cron_expression:
            return False

        # Implementation simplifiée - en réalité utiliser croniter ou similar
        # Exemple: "0 */6 * * *" = toutes les 6 heures
        now = datetime.now()
        if self._next_execution is None or now >= self._next_execution:
            # Calculer la prochaine exécution basée sur l'expression cron
            self._calculate_next_execution()
            return True
        return False

    def _calculate_next_execution(self):
        """Calculate next execution time based on cron expression."""
        # Implementation avec croniter ou logique cron personnalisée
        # Pour l'exemple, on ajoute 6 heures
        self._next_execution = datetime.now() + timedelta(hours=6)

    def get_trigger_status(self) -> Dict[str, Any]:
        """Get comprehensive trigger status information."""
        return {
            "trigger_type": self.trigger_config.trigger_type.value,
            "status": self._trigger_status.value,
            "last_execution": self._last_execution.isoformat() if self._last_execution else None,
            "next_execution": self._next_execution.isoformat() if self._next_execution else None,
            "enabled": self.trigger_config.enabled,
            "batch_size": self.trigger_config.batch_size,
        }

    def get_connector_info(self) -> Dict[str, Any]:
        """Get comprehensive connector information."""
        return {
            "connection_status": self._connection_status.value,
            "authenticated": self._authenticated,
            "protocol_info": self.get_protocol_info(),
            "transform_config": {
                "input_format": self.transform_config.input_format,
                "output_format": self.transform_config.output_format,
                "encoding": self.transform_config.encoding,
            },
            "trigger_status": self.get_trigger_status(),
        }

    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        return {
            "connection_valid": self.validate_connection(),
            "authenticated": self.is_authenticated(),
            "protocol_status": "active" if self._connection_status == ConnectionStatus.CONNECTED else "inactive",
            "trigger_status": self.get_trigger_status(),
            "trigger_active": self._trigger_status == TriggerStatus.ACTIVE,
            "last_check": datetime.now().isoformat(),
        }


# === EXAMPLE IMPLEMENTATIONS ===


class ExampleAPIConnector(BaseConnector):
    """
    Example implementation of a REST API connector.
    This shows how to implement the abstract methods for a specific use case.
    """

    async def connect(self) -> bool:
        """Connect to REST API endpoint."""
        try:
            # Implementation specific connection logic
            self._connection_status = ConnectionStatus.CONNECTED
            return True
        except Exception:
            self._connection_status = ConnectionStatus.ERROR
            return False

    async def disconnect(self) -> bool:
        """Disconnect from API."""
        self._connection_status = ConnectionStatus.DISCONNECTED
        return True

    def validate_connection(self) -> bool:
        """Validate API connection."""
        return self._connection_status == ConnectionStatus.CONNECTED

    def setup_protocol(self) -> bool:
        """Setup HTTP/HTTPS protocol."""
        return True

    async def send_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Send HTTP request."""
        # Implementation with requests or aiohttp
        return {"status": "success", "data": request_data}

    async def authenticate(self) -> bool:
        """Authenticate with API."""
        self._authenticated = True
        return True

    async def refresh_authentication(self) -> bool:
        """Refresh API token."""
        return True

    def check_authorization(self, resource: str, action: str) -> bool:
        """Check API permissions."""
        return self._authenticated

    def transform_input_data(self, raw_data: Any) -> List[Document]:
        """Transform API response to documents."""
        if isinstance(raw_data, dict) and "items" in raw_data:
            documents = []
            for item in raw_data["items"]:
                doc = Document(
                    page_content=str(item.get("content", "")), metadata={"source": "api", "id": item.get("id")}
                )
                documents.append(doc)
            return documents
        return []

    def transform_output_data(self, documents: List[Document]) -> Any:
        """Transform documents to API format."""
        return {"items": [{"content": doc.page_content, "metadata": doc.metadata} for doc in documents]}

    async def setup_event_listener(self) -> bool:
        """Setup webhook listener."""
        return True

    async def wait_for_event(self) -> Optional[TriggerEvent]:
        """Wait for webhook event."""
        # Implementation with webhook server or queue polling
        await asyncio.sleep(5)  # Simulate waiting
        return None

    async def process_batch(self, documents: List[Document], event: TriggerEvent) -> bool:
        """Process document batch."""
        # Implementation specific processing
        return True

    async def _fetch_raw_data(self, query_params: Optional[Dict[str, Any]] = None) -> Any:
        """Fetch data from API."""
        # Implementation with actual API call
        return {"items": [{"id": 1, "content": "example data"}]}

    async def _send_data(self, data: Any) -> bool:
        """Send data to API."""
        # Implementation with actual API call
        return True


# === USAGE EXAMPLES ===


def create_cron_connector_example():
    """Example: Create a connector with cron trigger."""
    connection_config = ConnectionConfig(host="api.example.com", protocol=ProtocolType.HTTPS)

    trigger_config = TriggerConfig(
        trigger_type=TriggerType.CRON,
        cron_expression="0 */6 * * *",  # Every 6 hours
        batch_size=50,
    )

    auth_config = AuthenticationConfig(auth_type="token", credentials={"token": "your-api-token"})

    return ExampleAPIConnector(
        connection_config=connection_config, trigger_config=trigger_config, auth_config=auth_config
    )


def create_event_driven_connector_example():
    """Example: Create a connector with event-driven trigger."""
    connection_config = ConnectionConfig(host="webhook.example.com", protocol=ProtocolType.HTTPS)

    trigger_config = TriggerConfig(
        trigger_type=TriggerType.EVENT_DRIVEN, event_source="webhook://api.example.com/data-changed", batch_size=100
    )

    return ExampleAPIConnector(connection_config=connection_config, trigger_config=trigger_config)


def create_manual_connector_example():
    """Example: Create a connector with manual trigger."""
    connection_config = ConnectionConfig(host="api.example.com", protocol=ProtocolType.HTTPS)

    trigger_config = TriggerConfig(trigger_type=TriggerType.MANUAL, batch_size=200)

    return ExampleAPIConnector(connection_config=connection_config, trigger_config=trigger_config)


# === USAGE DEMONSTRATION ===


async def main_example():
    """Demonstrate connector usage."""
    # Create manual connector
    connector = create_manual_connector_example()

    # Connect and authenticate
    await connector.connect()
    await connector.authenticate()

    # Manual trigger with data
    await connector.manual_trigger({"data": "new_data_payload"})

    # Health check
    health = await connector.health_check()
    print(f"Connector health: {health}")

    # Disconnect
    await connector.disconnect()


if __name__ == "__main__":
    # Run example
    asyncio.run(main_example())
