import unittest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from tests.test_auth_context import make_token
from src.agent.services.http_service import ServiceApiError
from src.main import app


class MainApiHelpersTests(unittest.TestCase):
    def test_chat_endpoint_returns_agent_message(self):
        client = TestClient(app)

        with patch(
            "src.main.ainvoke_ticket_agent",
            new_callable=AsyncMock,
            return_value={"messages": [type("Message", (), {"content": "There are 2 events."})()]},
            create=True,
        ) as ainvoke_agent:
            response = client.post(
                "/api/chat",
                json={"message": "Any events?", "thread_id": "test"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"message": "There are 2 events.", "thread_id": "test"})
        ainvoke_agent.assert_awaited_once_with("Any events?", thread_id="anon:test")

    def test_chat_endpoint_scopes_authorization_for_downstream_tools(self):
        client = TestClient(app)
        auth_token = object()

        with (
            patch(
                "src.main.ainvoke_ticket_agent",
                new_callable=AsyncMock,
                return_value={"messages": [type("Message", (), {"content": "held"})()]},
                create=True,
            ),
            patch("src.main.set_downstream_authorization", return_value=auth_token) as set_auth,
            patch("src.main.reset_downstream_authorization") as reset_auth,
        ):
            response = client.post(
                "/api/chat",
                json={"message": "hold seat", "thread_id": "test-auth"},
                headers={
                    "Authorization": (
                        f"Bearer {make_token(secret='dev-only-secret', sub='77777777-7777-7777-7777-777777777777', role='USER')}"
                    )
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(set_auth.call_args.args[0].startswith("Bearer "))
        reset_auth.assert_called_once_with(auth_token)

    def test_chat_endpoint_rejects_invalid_access_token(self):
        client = TestClient(app)

        response = client.post(
            "/api/chat",
            json={"message": "hold seat", "thread_id": "test-auth"},
            headers={"Authorization": "Bearer invalid-token"},
        )

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json(), {"detail": "Access token is invalid or expired."})

    def test_chat_endpoint_scopes_thread_id_by_authenticated_user(self):
        client = TestClient(app)
        user_id = "77777777-7777-7777-7777-777777777777"

        with patch(
            "src.main.ainvoke_ticket_agent",
            new_callable=AsyncMock,
            return_value={"messages": [type("Message", (), {"content": "ok"})()]},
            create=True,
        ) as ainvoke_agent:
            response = client.post(
                "/api/chat",
                json={"message": "hello", "thread_id": "default"},
                headers={"Authorization": f"Bearer {make_token(secret='dev-only-secret', sub=user_id, role='USER')}"},
            )

        self.assertEqual(response.status_code, 200)
        ainvoke_agent.assert_awaited_once_with("hello", thread_id=f"user:{user_id}:default")

    def test_chat_endpoint_logs_request_and_full_agent_reply(self):
        client = TestClient(app)
        reply = "Event Service called successfully and returned 0 events."

        with (
            patch(
                "src.main.ainvoke_ticket_agent",
                new_callable=AsyncMock,
                return_value={"messages": [type("Message", (), {"content": reply})()]},
                create=True,
            ),
            self.assertLogs("src.main", level="INFO") as logs,
        ):
            response = client.post(
                "/api/chat",
                json={"message": "list events", "thread_id": "test-logs"},
            )

        self.assertEqual(response.status_code, 200)
        joined_logs = "\n".join(logs.output)
        self.assertIn("chat request received", joined_logs)
        self.assertIn("test-logs", joined_logs)
        self.assertIn("chat agent reply", joined_logs)
        self.assertIn(reply, joined_logs)

    def test_chat_endpoint_returns_bad_gateway_for_downstream_service_error(self):
        client = TestClient(app)

        with patch(
            "src.main.ainvoke_ticket_agent",
            new_callable=AsyncMock,
            side_effect=ServiceApiError("Event Service error 500: code=500, message=failed to count events"),
            create=True,
        ):
            response = client.post(
                "/api/chat",
                json={"message": "Any events?", "thread_id": "test"},
            )

        self.assertEqual(response.status_code, 502)
        self.assertEqual(
            response.json(),
            {
                "detail": "Event Service error 500: code=500, message=failed to count events",
            },
        )

    def test_record_page_serves_microphone_wav_uploader(self):
        client = TestClient(app)

        response = client.get("/record")

        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"])
        self.assertIn("Start recording", response.text)
        self.assertIn("/api/recognize/songfinder?startTime=0", response.text)
        self.assertIn("audio/wav", response.text)
        self.assertIn("recording.wav", response.text)

    def test_recognize_song_calls_songfinder_service(self):
        audio_bytes = b"RIFF....WAVEfmt "

        with patch(
            "src.main.recognize_song_bytes",
            new_callable=AsyncMock,
            return_value={"success": True, "song": "Saved Song", "movie": None, "track": {"title": "Saved Song"}},
        ) as recognize_song_bytes:
            client = TestClient(app)

            response = client.post(
                "/api/recognize/songfinder?startTime=0",
                files={"audio": ("recording.wav", audio_bytes, "audio/wav")},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {"success": True, "song": "Saved Song", "movie": None, "track": {"title": "Saved Song"}},
        )

        recognize_song_bytes.assert_awaited_once_with(audio_bytes, start_time=0)


if __name__ == "__main__":
    unittest.main()
