"""
WebSocket manager for real-time progress updates
"""

import json
import asyncio
import threading
from typing import Dict, List, Any, Optional
from fastapi import WebSocket, WebSocketDisconnect
from ..utils.logging import get_logger


class ConnectionManager:
    """Manage WebSocket connections with thread safety"""
    
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}
        self.logger = get_logger(self.__class__.__name__)
        self._lock = threading.Lock()  # Protect concurrent access
    
    async def connect(self, websocket: WebSocket, run_id: str):
        """Accept new WebSocket connection"""
        await websocket.accept()
        
        with self._lock:  # Thread-safe connection management
            if run_id not in self.active_connections:
                self.active_connections[run_id] = []
            
            self.active_connections[run_id].append(websocket)
        
        self.logger.info(f"WebSocket connected for run {run_id}")
    
    def disconnect(self, websocket: WebSocket, run_id: str):
        """Remove WebSocket connection"""
        with self._lock:  # Thread-safe connection management
            if run_id in self.active_connections:
                if websocket in self.active_connections[run_id]:
                    self.active_connections[run_id].remove(websocket)
                
                # Clean up empty run_id entries
                if not self.active_connections[run_id]:
                    del self.active_connections[run_id]
        
        self.logger.info(f"WebSocket disconnected for run {run_id}")
    
    async def send_personal_message(self, message: Dict[str, Any], websocket: WebSocket):
        """Send message to specific WebSocket"""
        try:
            await websocket.send_text(json.dumps(message))
        except Exception as e:
            self.logger.error(f"Error sending personal message: {e}")
    
    async def broadcast_to_run(self, message: Dict[str, Any], run_id: str):
        """Broadcast message to all connections for a specific run"""
        # Get a thread-safe copy of connections
        with self._lock:
            if run_id not in self.active_connections:
                return
            # Create a copy of the connection list to avoid modification during iteration
            connections = self.active_connections[run_id].copy()
        
        for connection in connections:
            try:
                await connection.send_text(json.dumps(message))
            except Exception as e:
                self.logger.error(f"Error broadcasting to run {run_id}: {e}")
                # Remove broken connection
                self.disconnect(connection, run_id)
    
    def get_connection_count(self, run_id: str) -> int:
        """Get number of active connections for a run"""
        with self._lock:
            return len(self.active_connections.get(run_id, []))
    
    def get_all_runs(self) -> List[str]:
        """Get list of all active run IDs"""
        with self._lock:
            return list(self.active_connections.keys())


class ProgressNotifier:
    """Send progress notifications through WebSocket"""
    
    def __init__(self, connection_manager: ConnectionManager, run_id: str):
        self.connection_manager = connection_manager
        self.run_id = run_id
        self.logger = get_logger(self.__class__.__name__)
        # Track progress broadcasts per phase for light throttling
        self._progress_state: Dict[str, int] = {}
    
    async def send_phase_start(self, phase: str, description: str = ""):
        """Notify phase start"""
        message = {
            "type": "phase_start",
            "run_id": self.run_id,
            "phase": phase,
            "description": description,
            "timestamp": asyncio.get_event_loop().time()
        }
        await self.connection_manager.broadcast_to_run(message, self.run_id)
        self.logger.debug(f"Phase start notification sent: {phase}")
    
    async def send_phase_complete(self, phase: str, duration: float, results: Optional[Dict[str, Any]] = None):
        """Notify phase completion"""
        message = {
            "type": "phase_complete",
            "run_id": self.run_id,
            "phase": phase,
            "duration": duration,
            "results": results or {},
            "timestamp": asyncio.get_event_loop().time()
        }
        await self.connection_manager.broadcast_to_run(message, self.run_id)
        self.logger.debug(f"Phase complete notification sent: {phase}")
    
    async def send_progress_update(self, phase: str, completed: int, total: int, message: str = ""):
        """Send progress update"""
        progress_percent = (completed / total * 100) if total > 0 else 0
        
        # Light throttling to avoid excessive updates: emit on odd steps and on completion
        try:
            last = self._progress_state.get(phase, 0)
            should_emit = (completed == total) or (completed % 2 == 1 and completed != last)
            if not should_emit:
                return
            self._progress_state[phase] = completed
        except Exception:
            # Fallback to always emit if throttling logic fails
            pass

        message_data = {
            "type": "progress_update",
            "run_id": self.run_id,
            "phase": phase,
            "current": completed,
            "total": total,
            # For test compatibility, provide both keys
            "progress_percent": round(progress_percent, 1),
            "percentage": round(progress_percent, 1),
            "message": message,
            "timestamp": asyncio.get_event_loop().time()
        }
        await self.connection_manager.broadcast_to_run(message_data, self.run_id)
    
    async def send_error(self, error: str, phase: str = "", details: Optional[Dict[str, Any]] = None):
        """Send error notification"""
        message = {
            "type": "error",
            "run_id": self.run_id,
            "phase": phase,
            "error": error,
            "details": details or {},
            "timestamp": asyncio.get_event_loop().time()
        }
        await self.connection_manager.broadcast_to_run(message, self.run_id)
        self.logger.error(f"Error notification sent: {error}")
    
    async def send_log_message(self, level: str, message: str, details: Optional[Dict[str, Any]] = None):
        """Send log message"""
        message_data = {
            "type": "log",
            "run_id": self.run_id,
            "level": level,
            "message": message,
            "details": details or {},
            "timestamp": asyncio.get_event_loop().time()
        }
        await self.connection_manager.broadcast_to_run(message_data, self.run_id)
    
    async def send_evaluation_complete(self, results: Dict[str, Any]):
        """Send evaluation completion notification"""
        message = {
            "type": "evaluation_complete",
            "run_id": self.run_id,
            "results": results,
            "progress_percent": 100.0,
            "percentage": 100.0,
            "timestamp": asyncio.get_event_loop().time()
        }
        await self.connection_manager.broadcast_to_run(message, self.run_id)
        self.logger.info(f"Evaluation complete notification sent for run {self.run_id}")
    
    async def send_custom_message(self, message_type: str, data: Dict[str, Any]):
        """Send custom message"""
        message = {
            "type": message_type,
            "run_id": self.run_id,
            "data": data,
            "timestamp": asyncio.get_event_loop().time()
        }
        await self.connection_manager.broadcast_to_run(message, self.run_id)


class EvaluationProgressTracker:
    """Track evaluation progress and send updates"""
    
    def __init__(self, notifier: ProgressNotifier):
        self.notifier = notifier
        self.current_phase = None
        self.phase_start_time = None
        self.phase_progress = {}
    
    async def start_phase(self, phase: str, description: str = "", total_items: int = 0):
        """Start a new phase"""
        if self.current_phase:
            await self.end_phase()
        
        self.current_phase = phase
        self.phase_start_time = asyncio.get_event_loop().time()
        self.phase_progress = {
            "completed": 0,
            "total": total_items,
            "description": description
        }
        
        await self.notifier.send_phase_start(phase, description)
    
    async def update_progress(self, completed: int, message: str = ""):
        """Update progress for current phase"""
        if not self.current_phase:
            return
        
        self.phase_progress["completed"] = completed
        
        await self.notifier.send_progress_update(
            self.current_phase,
            completed,
            self.phase_progress["total"],
            message
        )
    
    async def increment_progress(self, increment: int = 1, message: str = ""):
        """Increment progress by specified amount"""
        if not self.current_phase:
            return
        
        self.phase_progress["completed"] += increment
        
        await self.notifier.send_progress_update(
            self.current_phase,
            self.phase_progress["completed"],
            self.phase_progress["total"],
            message
        )
    
    async def end_phase(self, results: Optional[Dict[str, Any]] = None):
        """End current phase"""
        if not self.current_phase or not self.phase_start_time:
            return
        
        duration = asyncio.get_event_loop().time() - self.phase_start_time
        
        await self.notifier.send_phase_complete(
            self.current_phase,
            duration,
            results
        )
        
        self.current_phase = None
        self.phase_start_time = None
        self.phase_progress = {}
    
    async def send_error(self, error: str, details: Optional[Dict[str, Any]] = None):
        """Send error for current phase"""
        await self.notifier.send_error(error, self.current_phase or "", details)
    
    async def send_log(self, level: str, message: str, details: Optional[Dict[str, Any]] = None):
        """Send log message"""
        await self.notifier.send_log_message(level, message, details)


# Global connection manager instance
websocket_manager = ConnectionManager()


def get_progress_notifier(run_id: str) -> ProgressNotifier:
    """Get progress notifier for a run"""
    return ProgressNotifier(websocket_manager, run_id)


def get_progress_tracker(run_id: str) -> EvaluationProgressTracker:
    """Get progress tracker for a run"""
    notifier = get_progress_notifier(run_id)
    return EvaluationProgressTracker(notifier)

# Backward compatibility exports expected by tests
# Alias names to match older API
ProgressTracker = EvaluationProgressTracker
ProgressManager = ConnectionManager


async def handle_websocket_connection(websocket: WebSocket, run_id: str):
    """Handle WebSocket connection lifecycle"""
    await websocket_manager.connect(websocket, run_id)
    
    try:
        while True:
            # Keep connection alive and handle any incoming messages
            data = await websocket.receive_text()
            
            # Parse incoming message
            try:
                message = json.loads(data)
                await handle_client_message(websocket, run_id, message)
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Invalid JSON message"
                }))
            
    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket, run_id)
    except Exception as e:
        websocket_manager.logger.error(f"WebSocket error for run {run_id}: {e}")
        websocket_manager.disconnect(websocket, run_id)


async def handle_client_message(websocket: WebSocket, run_id: str, message: Dict[str, Any]):
    """Handle incoming message from client"""
    message_type = message.get("type")
    
    if message_type == "ping":
        # Respond to ping
        await websocket.send_text(json.dumps({
            "type": "pong",
            "timestamp": asyncio.get_event_loop().time()
        }))
    
    elif message_type == "subscribe":
        # Client wants to subscribe to specific events
        # Could implement event filtering here
        await websocket.send_text(json.dumps({
            "type": "subscribed",
            "run_id": run_id
        }))
    
    elif message_type == "get_status":
        # Client requesting current status
        # Could return current evaluation status
        await websocket.send_text(json.dumps({
            "type": "status",
            "run_id": run_id,
            "connections": websocket_manager.get_connection_count(run_id)
        }))
    
    else:
        # Unknown message type
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": f"Unknown message type: {message_type}"
        }))