import unittest

import httpx
from langchain_core.tools import BaseTool

from src.agent.services.event_service import EventServiceClient, EventServiceError
from src.agent.tools.event_tools import GetEventByIdInput, ListEventsInput
from src.agent.tools.event_tools import create_event_service_tools


class EventAgentToolsTests(unittest.IsolatedAsyncioTestCase):
    async def test_list_events_tool_calls_event_service_with_filters(self):
        seen_request = None

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal seen_request
            seen_request = request
            return httpx.Response(
                200,
                json={
                    "data": [{"id": "evt_1", "name": "Interstellar"}],
                    "page": 2,
                    "page_size": 5,
                    "total_items": 1,
                    "total_pages": 1,
                },
            )

        http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        service_client = EventServiceClient(
            base_url="http://event-service.local/api/v1",
            http_client=http_client,
        )
        list_tool = create_event_service_tools(service_client)[0]

        result = await list_tool.ainvoke(
            {
                "page": 2,
                "page_size": 5,
                "event_type": "MOVIE",
                "search": "Interstellar",
            }
        )

        self.assertEqual(result["data"][0]["name"], "Interstellar")
        self.assertEqual(seen_request.method, "GET")
        self.assertEqual(seen_request.url.path, "/api/v1/events")
        self.assertEqual(seen_request.url.params["page"], "2")
        self.assertEqual(seen_request.url.params["page_size"], "5")
        self.assertEqual(seen_request.url.params["type"], "MOVIE")
        self.assertEqual(seen_request.url.params["search"], "Interstellar")

    async def test_get_event_by_id_tool_calls_event_detail_endpoint(self):
        seen_request = None

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal seen_request
            seen_request = request
            return httpx.Response(
                200,
                json={"data": {"id": "evt_42", "name": "Jazz Night"}, "message": "OK"},
            )

        http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        service_client = EventServiceClient(
            base_url="http://event-service.local/api/v1",
            http_client=http_client,
        )
        get_tool = create_event_service_tools(service_client)[1]

        result = await get_tool.ainvoke({"event_id": "evt_42"})

        self.assertEqual(result["data"]["id"], "evt_42")
        self.assertEqual(seen_request.method, "GET")
        self.assertEqual(seen_request.url.path, "/api/v1/events/evt_42")

    async def test_get_event_by_id_encodes_event_id_path_segment(self):
        seen_request = None

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal seen_request
            seen_request = request
            return httpx.Response(200, json={"data": {"id": "movie/42"}, "message": "OK"})

        http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        service_client = EventServiceClient(
            base_url="http://event-service.local/api/v1",
            http_client=http_client,
        )
        get_tool = create_event_service_tools(service_client)[1]

        await get_tool.ainvoke({"event_id": "movie/42"})

        self.assertEqual(str(seen_request.url), "http://event-service.local/api/v1/events/movie%2F42")

    async def test_list_events_tool_rejects_response_outside_paginated_schema(self):
        http_client = httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json={"data": [], "page": 1, "page_size": 20})
            )
        )
        service_client = EventServiceClient(
            base_url="http://event-service.local/api/v1",
            http_client=http_client,
        )
        list_tool = create_event_service_tools(service_client)[0]

        with self.assertRaises(EventServiceError) as error:
            await list_tool.ainvoke({})

        self.assertIn("PaginatedResponse", str(error.exception))

    async def test_get_event_by_id_tool_rejects_response_outside_success_schema(self):
        http_client = httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"message": "OK"}))
        )
        service_client = EventServiceClient(
            base_url="http://event-service.local/api/v1",
            http_client=http_client,
        )
        get_tool = create_event_service_tools(service_client)[1]

        with self.assertRaises(EventServiceError) as error:
            await get_tool.ainvoke({"event_id": "evt_42"})

        self.assertIn("SuccessResponse", str(error.exception))

    async def test_event_service_http_error_uses_error_response_schema(self):
        http_client = httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(404, json={"code": 404, "message": "event not found"})
            )
        )
        service_client = EventServiceClient(
            base_url="http://event-service.local/api/v1",
            http_client=http_client,
        )
        get_tool = create_event_service_tools(service_client)[1]

        with self.assertRaises(EventServiceError) as error:
            await get_tool.ainvoke({"event_id": "missing"})

        self.assertIn("404", str(error.exception))
        self.assertIn("event not found", str(error.exception))

    def test_tools_have_explicit_names_descriptions_and_schemas(self):
        service_client = EventServiceClient(
            base_url="http://event-service.local/api/v1",
            http_client=httpx.AsyncClient(
                transport=httpx.MockTransport(lambda request: httpx.Response(200, json={}))
            ),
        )

        list_tool, get_tool = create_event_service_tools(service_client)

        self.assertIsInstance(list_tool, BaseTool)
        self.assertEqual(list_tool.name, "list_events")
        self.assertIn("GET /events", list_tool.description)
        self.assertIn("page_size", list_tool.args)
        self.assertIn("event_type", list_tool.args)
        self.assertIs(list_tool.args_schema, ListEventsInput)
        self.assertIsInstance(get_tool, BaseTool)
        self.assertEqual(get_tool.name, "get_event_by_id")
        self.assertIn("GET /events/{id}", get_tool.description)
        self.assertIn("event_id", get_tool.args)
        self.assertIs(get_tool.args_schema, GetEventByIdInput)


if __name__ == "__main__":
    unittest.main()
