from __future__ import annotations

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from ..session import session_store
from ..agent.orchestrator import run_agent_response
from ..logging_utils import log_workflow_start, log_artifact_storage

router = APIRouter(prefix="/api", tags=["chat"])


class ChatRequest(BaseModel):
    sessionId: str
    message: str


class ChatResponse(BaseModel):
    id: str
    content: str
    sessionId: str
    progress: list[str]
    artifactIndex: int
    execStatus: str | None = None


class Artifact(BaseModel):
    code: str | None = None
    exec: dict | None = None
    violations: list[str] | None = None
    question: str | None = None


class ArtifactsResponse(BaseModel):
    sessionId: str
    artifacts: list[Artifact]


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    session = session_store.get(req.sessionId)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or expired")

    session_store.add_message(req.sessionId, role="user", content=req.message)
    artifact = run_agent_response(session.payload, req.message, str(session.data_path))
    assistant_text = artifact["answer"]
    progress = artifact["progress"]
    idx = session_store.add_artifact(
        req.sessionId,
        {
            "code": artifact.get("code"),
            "exec": artifact.get("exec"),
            "violations": artifact.get("violations"),
            "question": req.message,
        },
    )
    session_store.add_message(
        req.sessionId,
        role="assistant",
        content=assistant_text,
        extra={"artifact_index": idx},
    )
    return ChatResponse(
        id=str(session_store.message_count(req.sessionId)),
        content=assistant_text,
        sessionId=req.sessionId,
        progress=progress,
        artifactIndex=idx,
        execStatus=artifact.get("exec", {}).get("status"),
    )


@router.get("/sessions/{session_id}/artifacts", response_model=ArtifactsResponse)
async def list_artifacts(session_id: str):
    session = session_store.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    return ArtifactsResponse(
        sessionId=session_id,
        artifacts=[Artifact(**a) for a in session.artifacts],
    )


@router.websocket("/chat/ws")
async def chat_ws(websocket: WebSocket):
    await websocket.accept()
    try:
        init = await websocket.receive_json()
        session_id = init.get("sessionId")
        message = init.get("message")
        if not session_id or not message:
            await websocket.send_json(
                {"type": "error", "message": "sessionId and message are required"}
            )
            await websocket.close()
            return
        session = session_store.get(session_id)
        if not session:
            await websocket.send_json(
                {"type": "error", "message": "Session not found or expired"}
            )
            await websocket.close()
            return

        # Log workflow start
        log_workflow_start(session_id, message, session.payload)

        # Record user message
        session_store.add_message(session_id, role="user", content=message)

        # Use the main agent orchestrator with real-time streaming
        await websocket.send_json({"type": "stage", "value": "planning"})

        # Planning stage
        plan = f"Answer the question: {message} using dataset metadata only."

        await websocket.send_json({"type": "stage", "value": "code_gen"})

        # Code generation stage
        from ..agent.llm import generate_analysis_code

        code = generate_analysis_code(plan, session.payload, message)

        # Send the generated code to frontend for display
        await websocket.send_json({"type": "code", "value": code})

        await websocket.send_json({"type": "stage", "value": "safety_check"})

        # Safety check stage
        from ..agent.safety import validate_code

        violations = validate_code(code)

        if violations:
            await websocket.send_json({"type": "stage", "value": "safety_blocked"})
            # Handle safety blocked case
            from ..agent.orchestrator import run_basic_analysis

            heuristic = run_basic_analysis(session.payload, message)
            final_answer = (
                heuristic
                + "\n\n(Script blocked by safety: "
                + "; ".join(violations)
                + ")"
            )

            # Stream the heuristic response
            chunk_size = 30
            for i in range(0, len(final_answer), chunk_size):
                chunk = final_answer[i : i + chunk_size]
                await websocket.send_json({"type": "content", "value": chunk})

            artifact_data = {
                "code": code,
                "exec": {"status": "blocked"},
                "violations": violations,
                "question": message,
            }
            idx = session_store.add_artifact(session_id, artifact_data)

            await websocket.send_json({"type": "stage", "value": "completed"})
            session_store.add_message(
                session_id,
                role="assistant",
                content=final_answer,
                extra={"artifact_index": idx},
            )
            await websocket.send_json(
                {
                    "type": "done",
                    "value": {
                        "id": str(session_store.message_count(session_id)),
                        "content": final_answer,
                        "sessionId": session_id,
                    },
                }
            )
            await websocket.close()
            return

        await websocket.send_json({"type": "stage", "value": "sandbox_exec"})

        # Sandbox execution stage
        from ..agent.sandbox import execute_analysis_script

        exec_result = execute_analysis_script(code, dataset_path=str(session.data_path))

        await websocket.send_json({"type": "stage", "value": "summarize"})

        # Summarization stage with real-time streaming
        if exec_result.get("status") == "ok" and exec_result.get("logs"):
            # Stream LLM summary in real-time
            try:
                from ..agent.llm import stream_summary_chunks

                accumulated_answer = ""
                for chunk in stream_summary_chunks(
                    session.payload, message, exec_result, code
                ):
                    accumulated_answer += chunk
                    await websocket.send_json({"type": "content", "value": chunk})
                    if len(accumulated_answer) > 4000:  # Prevent runaway responses
                        break

                final_answer = accumulated_answer
            except Exception as e:
                # Fallback if LLM streaming fails
                from ..agent.orchestrator import run_basic_analysis

                logs = exec_result.get("logs", "").strip()
                if logs:
                    final_answer = (
                        f"Analysis Results:\n{logs}\n\n"
                        + run_basic_analysis(session.payload, message)
                    )
                else:
                    final_answer = (
                        run_basic_analysis(session.payload, message)
                        + f"\n\n(Exec status: {exec_result.get('status', 'n/a')}, Error: {str(e)})"
                    )

                # Stream the fallback response
                chunk_size = 30
                for i in range(0, len(final_answer), chunk_size):
                    chunk = final_answer[i : i + chunk_size]
                    await websocket.send_json({"type": "content", "value": chunk})
        else:
            # Execution failed - use heuristic analysis
            from ..agent.orchestrator import run_basic_analysis

            heuristic = run_basic_analysis(session.payload, message)
            final_answer = (
                heuristic + f"\n\n(Exec status: {exec_result.get('status', 'n/a')})"
            )

            # Stream the heuristic response
            chunk_size = 30
            for i in range(0, len(final_answer), chunk_size):
                chunk = final_answer[i : i + chunk_size]
                await websocket.send_json({"type": "content", "value": chunk})

        # Store artifact
        artifact_data = {
            "code": code,
            "exec": exec_result,
            "violations": violations,
            "question": message,
        }
        idx = session_store.add_artifact(session_id, artifact_data)

        log_artifact_storage(session_id, idx, artifact_data)

        await websocket.send_json({"type": "stage", "value": "completed"})

        # Store assistant message with artifact reference
        session_store.add_message(
            session_id,
            role="assistant",
            content=final_answer,
            extra={"artifact_index": idx},
        )

        await websocket.send_json(
            {
                "type": "done",
                "value": {
                    "id": str(session_store.message_count(session_id)),
                    "content": final_answer,
                    "sessionId": session_id,
                    "artifactIndex": idx,
                    "execStatus": exec_result.get("status"),
                },
            }
        )
        await websocket.close()
    except WebSocketDisconnect:
        # client disconnected, nothing to do
        return
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
        try:
            await websocket.close()
        except Exception:
            pass
