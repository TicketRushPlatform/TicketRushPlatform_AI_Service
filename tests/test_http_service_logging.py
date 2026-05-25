import unittest

import httpx

from src.agent.services.http_service import (
    JsonHttpServiceClient,
    reset_downstream_authorization,
    set_downstream_authorization,
)


class HttpServiceLoggingTests(unittest.IsolatedAsyncioTestCase):
    async def test_request_json_logs_downstream_request_and_response(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"data": []})

        client = JsonHttpServiceClient(
            "Event Service",
            "http://event-service.local/api/v1",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )

        with self.assertLogs("src.agent.services.http_service", level="INFO") as logs:
            payload = await client.get_json("/events", params={"page": 1, "page_size": 5})

        self.assertEqual(payload, {"data": []})
        joined_logs = "\n".join(logs.output)
        self.assertIn("downstream request", joined_logs)
        self.assertIn("Event Service", joined_logs)
        self.assertIn("GET", joined_logs)
        self.assertIn("/events", joined_logs)
        self.assertIn("downstream response", joined_logs)
        self.assertIn("200", joined_logs)

    async def test_request_json_forwards_request_scoped_authorization_header(self):
        seen_authorization = None

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal seen_authorization
            seen_authorization = request.headers.get("authorization")
            return httpx.Response(200, json={"data": []})

        client = JsonHttpServiceClient(
            "Booking Service",
            "http://booking-service.local/api/v1",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )
        token = set_downstream_authorization("Bearer test-token")
        try:
            await client.get_json("/showtimes/00000000-0000-0000-0000-000000000001/seats")
        finally:
            reset_downstream_authorization(token)

        self.assertEqual(seen_authorization, "Bearer test-token")


if __name__ == "__main__":
    unittest.main()
