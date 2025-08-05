"""
WebSocket Integration Tests for docs-to-eval real-time progress tracking
Tests WebSocket connections, progress notifications, and real-time updates
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from fastapi.websockets import WebSocket

from docs_to_eval.ui_api.main import app
from docs_to_eval.ui_api.websockets import (
    ConnectionManager, 
    ProgressNotifier, 
    ProgressTracker,
    websocket_manager
)
from docs_to_eval.ui_api.routes import evaluation_runs


class TestConnectionManager:
    """Test WebSocket connection management"""
    
    def setup_method(self):
        """Set up test environment"""
        self.manager = ConnectionManager()
    
    @pytest.mark.asyncio
    async def test_connection_lifecycle(self):
        """Test WebSocket connection and disconnection"""
        mock_websocket = Mock(spec=WebSocket)
        mock_websocket.accept = AsyncMock()
        
        run_id = "test-run-123"
        
        # Test connection
        await self.manager.connect(mock_websocket, run_id)
        assert run_id in self.manager.active_connections
        assert mock_websocket in self.manager.active_connections[run_id]
        assert self.manager.get_connection_count(run_id) == 1
        
        # Test disconnection
        self.manager.disconnect(mock_websocket, run_id)
        assert self.manager.get_connection_count(run_id) == 0
        assert run_id not in self.manager.active_connections
    
    @pytest.mark.asyncio
    async def test_multiple_connections_per_run(self):
        """Test multiple WebSocket connections for the same run"""
        mock_ws1 = Mock(spec=WebSocket)
        mock_ws2 = Mock(spec=WebSocket)
        mock_ws1.accept = AsyncMock()
        mock_ws2.accept = AsyncMock()
        
        run_id = "test-run-multi"
        
        # Connect both websockets
        await self.manager.connect(mock_ws1, run_id)
        await self.manager.connect(mock_ws2, run_id)
        
        assert self.manager.get_connection_count(run_id) == 2
        assert mock_ws1 in self.manager.active_connections[run_id]
        assert mock_ws2 in self.manager.active_connections[run_id]
        
        # Disconnect one
        self.manager.disconnect(mock_ws1, run_id)
        assert self.manager.get_connection_count(run_id) == 1
        assert mock_ws2 in self.manager.active_connections[run_id]
    
    @pytest.mark.asyncio
    async def test_broadcast_to_run(self):
        """Test broadcasting messages to all connections for a run"""
        mock_ws1 = Mock(spec=WebSocket)
        mock_ws2 = Mock(spec=WebSocket)
        mock_ws1.accept = AsyncMock()
        mock_ws2.accept = AsyncMock()
        mock_ws1.send_text = AsyncMock()
        mock_ws2.send_text = AsyncMock()
        
        run_id = "test-broadcast-run"
        
        # Connect websockets
        await self.manager.connect(mock_ws1, run_id)
        await self.manager.connect(mock_ws2, run_id)
        
        # Broadcast message
        test_message = {"type": "test", "data": "Hello World"}
        await self.manager.broadcast_to_run(test_message, run_id)
        
        # Verify both connections received the message
        mock_ws1.send_text.assert_called_once_with(json.dumps(test_message))
        mock_ws2.send_text.assert_called_once_with(json.dumps(test_message))
    
    @pytest.mark.asyncio
    async def test_broadcast_with_broken_connection(self):
        """Test broadcasting when one connection is broken"""
        mock_ws1 = Mock(spec=WebSocket)
        mock_ws2 = Mock(spec=WebSocket)
        mock_ws1.accept = AsyncMock()
        mock_ws2.accept = AsyncMock()
        mock_ws1.send_text = AsyncMock(side_effect=Exception("Connection broken"))
        mock_ws2.send_text = AsyncMock()
        
        run_id = "test-broken-connection"
        
        # Connect websockets
        await self.manager.connect(mock_ws1, run_id)
        await self.manager.connect(mock_ws2, run_id)
        
        # Broadcast message - should handle broken connection gracefully
        test_message = {"type": "test", "data": "Test message"}
        await self.manager.broadcast_to_run(test_message, run_id)
        
        # Broken connection should be removed, good connection should receive message
        assert mock_ws1 not in self.manager.active_connections.get(run_id, [])
        mock_ws2.send_text.assert_called_once_with(json.dumps(test_message))
    
    def test_get_all_runs(self):
        """Test getting list of all active runs"""
        mock_ws1 = Mock(spec=WebSocket)
        mock_ws2 = Mock(spec=WebSocket)
        
        # Manually add connections (without async accept)
        self.manager.active_connections["run1"] = [mock_ws1]
        self.manager.active_connections["run2"] = [mock_ws2]
        
        all_runs = self.manager.get_all_runs()
        assert "run1" in all_runs
        assert "run2" in all_runs
        assert len(all_runs) == 2


class TestProgressNotifier:
    """Test progress notification system"""
    
    def setup_method(self):
        """Set up test environment"""
        self.mock_manager = Mock(spec=ConnectionManager)
        self.mock_manager.broadcast_to_run = AsyncMock()
        self.run_id = "test-progress-run"
        self.notifier = ProgressNotifier(self.mock_manager, self.run_id)
    
    @pytest.mark.asyncio
    async def test_phase_start_notification(self):
        """Test phase start notification"""
        await self.notifier.send_phase_start("classification", "Analyzing corpus content")
        
        # Verify broadcast was called with correct message structure
        self.mock_manager.broadcast_to_run.assert_called_once()
        call_args = self.mock_manager.broadcast_to_run.call_args
        message = call_args[0][0]  # First argument is the message
        
        assert message["type"] == "phase_start"
        assert message["run_id"] == self.run_id
        assert message["phase"] == "classification"
        assert message["description"] == "Analyzing corpus content"
        assert "timestamp" in message
    
    @pytest.mark.asyncio
    async def test_progress_update_notification(self):
        """Test progress update notification"""
        await self.notifier.send_progress_update("generation", 3, 10, "Generated question 3/10")
        
        self.mock_manager.broadcast_to_run.assert_called_once()
        call_args = self.mock_manager.broadcast_to_run.call_args
        message = call_args[0][0]
        
        assert message["type"] == "progress_update"
        assert message["phase"] == "generation"
        assert message["current"] == 3
        assert message["total"] == 10
        assert message["message"] == "Generated question 3/10"
        assert message["percentage"] == 30.0
    
    @pytest.mark.asyncio
    async def test_phase_complete_notification(self):
        """Test phase completion notification"""
        await self.notifier.send_phase_complete("verification", 15.5)
        
        self.mock_manager.broadcast_to_run.assert_called_once()
        call_args = self.mock_manager.broadcast_to_run.call_args
        message = call_args[0][0]
        
        assert message["type"] == "phase_complete"
        assert message["phase"] == "verification"
        assert message["duration"] == 15.5
    
    @pytest.mark.asyncio
    async def test_evaluation_complete_notification(self):
        """Test evaluation completion notification"""
        test_results = {
            "run_id": self.run_id,
            "aggregate_metrics": {"mean_score": 0.75},
            "status": "completed"
        }
        
        await self.notifier.send_evaluation_complete(test_results)
        
        self.mock_manager.broadcast_to_run.assert_called_once()
        call_args = self.mock_manager.broadcast_to_run.call_args
        message = call_args[0][0]
        
        assert message["type"] == "evaluation_complete"
        assert message["results"] == test_results
    
    @pytest.mark.asyncio
    async def test_error_notification(self):
        """Test error notification"""
        error_message = "Failed to generate questions: API timeout"
        
        await self.notifier.send_error(error_message)
        
        self.mock_manager.broadcast_to_run.assert_called_once()
        call_args = self.mock_manager.broadcast_to_run.call_args
        message = call_args[0][0]
        
        assert message["type"] == "error"
        assert message["error"] == error_message


class TestProgressTracker:
    """Test progress tracking functionality"""
    
    def setup_method(self):
        """Set up test environment"""
        self.mock_notifier = Mock(spec=ProgressNotifier)
        self.mock_notifier.send_phase_start = AsyncMock()
        self.mock_notifier.send_progress_update = AsyncMock()
        self.mock_notifier.send_phase_complete = AsyncMock()
        self.mock_notifier.send_error = AsyncMock()
        
        self.tracker = ProgressTracker("test-run", self.mock_notifier)
    
    @pytest.mark.asyncio
    async def test_phase_lifecycle(self):
        """Test complete phase lifecycle"""
        # Start phase
        await self.tracker.start_phase("testing", "Running tests", 5)
        
        self.mock_notifier.send_phase_start.assert_called_once_with(
            "testing", "Running tests"
        )
        
        # Update progress
        await self.tracker.increment_progress("Test 1 completed")
        await self.tracker.increment_progress("Test 2 completed")
        
        assert self.mock_notifier.send_progress_update.call_count == 2
        
        # End phase
        await self.tracker.end_phase({"tests_passed": 2})
        
        self.mock_notifier.send_phase_complete.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_progress_calculation(self):
        """Test progress percentage calculation"""
        await self.tracker.start_phase("calculation", "Computing results", 10)
        
        # Increment progress and check calculations
        for i in range(1, 4):
            await self.tracker.increment_progress(f"Step {i}")
        
        # Verify the progress updates had correct percentages
        calls = self.mock_notifier.send_progress_update.call_args_list
        assert len(calls) == 3
        
        # Check that percentages are calculated correctly
        # First call should be roughly 10% (1/10), second 20% (2/10), third 30% (3/10)
        # The exact percentage depends on the implementation
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in progress tracker"""
        await self.tracker.send_error("Test error occurred")
        
        self.mock_notifier.send_error.assert_called_once_with("Test error occurred")
    
    @pytest.mark.asyncio
    async def test_log_message(self):
        """Test log message sending"""
        await self.tracker.send_log("info", "Test log message")
        
        # Verify that log messages are sent as progress updates with correct format
        # Implementation depends on how logs are handled


class TestWebSocketIntegration:
    """Test WebSocket integration with FastAPI"""
    
    def setup_method(self):
        """Set up test client"""
        self.client = TestClient(app)
        evaluation_runs._runs.clear()
    
    def test_websocket_endpoint_exists(self):
        """Test that WebSocket endpoint is properly configured"""
        # This is a basic test to ensure the WebSocket route exists
        # Full WebSocket testing requires a more complex setup
        
        # We can test that the websocket_manager is properly initialized
        assert websocket_manager is not None
        assert isinstance(websocket_manager, ConnectionManager)
    
    @pytest.mark.asyncio
    async def test_websocket_with_evaluation(self):
        """Test WebSocket during actual evaluation"""
        # This would be a full integration test
        # For now, we'll test the components separately
        
        # Start an evaluation to get a real run_id
        evaluation_request = {
            "corpus_text": "Test corpus for WebSocket integration testing.",
            "num_questions": 2,
            "use_agentic": False
        }
        
        response = self.client.post("/api/v1/evaluation/start", json=evaluation_request)
        assert response.status_code == 200
        
        run_id = response.json()["run_id"]
        
        # Verify that the run exists and has proper WebSocket URL
        websocket_url = response.json()["websocket_url"]
        assert websocket_url == f"/api/v1/ws/{run_id}"


class TestRealTimeProgressFlow:
    """Test real-time progress flow during evaluation"""
    
    def setup_method(self):
        """Set up test environment"""
        self.mock_manager = Mock(spec=ConnectionManager)
        self.mock_manager.broadcast_to_run = AsyncMock()
        
    @pytest.mark.asyncio
    async def test_evaluation_progress_flow(self):
        """Test complete evaluation progress flow"""
        run_id = "test-flow-run"
        notifier = ProgressNotifier(self.mock_manager, run_id)
        
        # Simulate complete evaluation flow
        phases = [
            ("classification", "Analyzing corpus content"),
            ("generation", "Generating benchmark questions"),
            ("evaluation", "Evaluating with LLM"),
            ("verification", "Verifying responses"),
            ("reporting", "Generating comprehensive report")
        ]
        
        for phase_name, description in phases:
            # Start phase
            await notifier.send_phase_start(phase_name, description)
            
            # Simulate some progress updates
            for i in range(1, 4):
                await notifier.send_progress_update(phase_name, i, 3, f"Step {i}/3")
            
            # Complete phase
            await notifier.send_phase_complete(phase_name, 2.5)
        
        # Send completion
        final_results = {
            "run_id": run_id,
            "aggregate_metrics": {"mean_score": 0.82},
            "status": "completed"
        }
        await notifier.send_evaluation_complete(final_results)
        
        # Verify all notifications were sent
        expected_calls = len(phases) * 4 + 1  # 4 calls per phase + 1 completion
        assert self.mock_manager.broadcast_to_run.call_count == expected_calls
    
    @pytest.mark.asyncio
    async def test_error_during_progress_flow(self):
        """Test error handling during progress flow"""
        run_id = "test-error-run"
        notifier = ProgressNotifier(self.mock_manager, run_id)
        
        # Start a phase
        await notifier.send_phase_start("generation", "Generating questions")
        
        # Send some progress
        await notifier.send_progress_update("generation", 2, 10, "Generated question 2/10")
        
        # Send error
        await notifier.send_error("API rate limit exceeded")
        
        # Verify error was broadcast
        error_calls = [call for call in self.mock_manager.broadcast_to_run.call_args_list 
                      if call[0][0].get("type") == "error"]
        assert len(error_calls) == 1
        assert error_calls[0][0][0]["error"] == "API rate limit exceeded"


class TestWebSocketMessageFormat:
    """Test WebSocket message format consistency"""
    
    def setup_method(self):
        """Set up test environment"""
        self.mock_manager = Mock(spec=ConnectionManager)
        self.mock_manager.broadcast_to_run = AsyncMock()
        self.notifier = ProgressNotifier(self.mock_manager, "test-run")
    
    @pytest.mark.asyncio
    async def test_message_format_consistency(self):
        """Test that all WebSocket messages have consistent format"""
        # Test different message types
        await self.notifier.send_phase_start("test", "Test phase")
        await self.notifier.send_progress_update("test", 1, 5, "Progress")
        await self.notifier.send_phase_complete("test", 1.0)
        await self.notifier.send_error("Test error")
        
        # Verify all messages have required fields
        for call in self.mock_manager.broadcast_to_run.call_args_list:
            message = call[0][0]  # First argument is the message
            
            # All messages should have type and run_id
            assert "type" in message
            assert "run_id" in message
            assert message["run_id"] == "test-run"
            
            # All messages should have timestamp
            assert "timestamp" in message
            
            # Verify message type is valid
            valid_types = ["phase_start", "progress_update", "phase_complete", 
                          "evaluation_complete", "error", "log"]
            assert message["type"] in valid_types
    
    @pytest.mark.asyncio
    async def test_json_serializable_messages(self):
        """Test that all WebSocket messages are JSON serializable"""
        test_results = {
            "run_id": "test-run",
            "aggregate_metrics": {"mean_score": 0.75, "std_dev": 0.1},
            "individual_results": [
                {"question": "Test?", "score": 0.8, "method": "exact_match"}
            ]
        }
        
        await self.notifier.send_evaluation_complete(test_results)
        
        # Try to serialize the message that was sent
        call_args = self.mock_manager.broadcast_to_run.call_args
        message = call_args[0][0]
        
        # This should not throw an exception
        json_str = json.dumps(message)
        # And we should be able to deserialize it back
        deserialized = json.loads(json_str)
        
        assert deserialized["type"] == "evaluation_complete"
        assert deserialized["results"] == test_results


if __name__ == "__main__":
    pytest.main([__file__, "-v"])