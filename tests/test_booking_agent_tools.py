import unittest

import httpx

from src.agent.auth_context import AuthenticatedUser, reset_authenticated_user, set_authenticated_user
from src.agent.services.booking_service import BookingServiceClient
from src.agent.tools.booking_tools import create_booking_service_tools


class BookingAgentToolsTests(unittest.IsolatedAsyncioTestCase):
    async def test_hold_seats_tool_calls_booking_service_with_required_payload(self):
        seen_request = None

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal seen_request
            seen_request = request
            return httpx.Response(201, json={"data": {"id": "booking_1"}, "message": "held"})

        http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        service_client = BookingServiceClient(
            base_url="http://booking-service.local/api/v1",
            http_client=http_client,
        )
        hold_tool = create_booking_service_tools(service_client)[0]

        token = set_authenticated_user(AuthenticatedUser(user_id="user_1", role="USER"))
        try:
            result = await hold_tool.ainvoke(
                {
                    "showtime_id": "show_1",
                    "seat_ids": ["A1", "A2"],
                }
            )
        finally:
            reset_authenticated_user(token)

        self.assertEqual(result["data"]["id"], "booking_1")
        self.assertEqual(seen_request.method, "POST")
        self.assertEqual(seen_request.url.path, "/api/v1/bookings/hold")
        self.assertEqual(
            seen_request.read(),
            b'{"seat_ids":["A1","A2"],"showtime_id":"show_1","user_id":"user_1"}',
        )

    async def test_hold_seats_tool_requires_authenticated_user_context(self):
        service_client = BookingServiceClient(
            base_url="http://booking-service.local/api/v1",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, json={}))),
        )
        hold_tool = create_booking_service_tools(service_client)[0]

        with self.assertRaises(ValueError) as error:
            await hold_tool.ainvoke({"showtime_id": "show_1", "seat_ids": ["A1"]})

        self.assertIn("authenticated user", str(error.exception))

    async def test_get_bookings_by_user_tool_calls_paginated_endpoint(self):
        seen_request = None

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal seen_request
            seen_request = request
            return httpx.Response(
                200,
                json={
                    "data": [{"id": "booking_1"}],
                    "page": 2,
                    "page_size": 5,
                    "total_items": 1,
                    "total_pages": 1,
                },
            )

        http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        service_client = BookingServiceClient(
            base_url="http://booking-service.local/api/v1",
            http_client=http_client,
        )
        get_user_bookings_tool = create_booking_service_tools(service_client)[2]

        token = set_authenticated_user(AuthenticatedUser(user_id="user_1", role="USER"))
        try:
            result = await get_user_bookings_tool.ainvoke({"page": 2, "page_size": 5})
        finally:
            reset_authenticated_user(token)

        self.assertEqual(result["data"][0]["id"], "booking_1")
        self.assertEqual(seen_request.method, "GET")
        self.assertEqual(seen_request.url.path, "/api/v1/bookings/user/user_1")
        self.assertEqual(seen_request.url.params["page"], "2")
        self.assertEqual(seen_request.url.params["page_size"], "5")

    def test_booking_tools_include_each_supported_booking_api(self):
        http_client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, json={})))
        service_client = BookingServiceClient(
            base_url="http://booking-service.local/api/v1",
            http_client=http_client,
        )

        tools = create_booking_service_tools(service_client)

        self.assertEqual(
            [tool.name for tool in tools],
            [
                "hold_booking_seats",
                "release_expired_booking_holds",
                "get_bookings_by_user",
                "get_booking_by_id",
                "cancel_booking",
                "confirm_booking",
                "get_showtime_seats",
                "get_showtime_seats_ws_endpoint",
            ],
        )
        self.assertEqual(
            tools[-1].invoke({"showtime_id": "show/1"})["websocket_url"],
            "ws://booking-service.local/api/v1/showtimes/show%2F1/seats/ws",
        )


if __name__ == "__main__":
    unittest.main()
