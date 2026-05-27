import unittest

import httpx
from langchain_core.tools import BaseTool

from src.agent.services.user_service import UserServiceClient, UserServiceError
from src.agent.tools.user_tools import (
    GetCurrentUserInput,
    GetUserByIdInput,
    GetUserStatsInput,
    ListUsersInput,
    create_user_service_tools,
)


USER_PAYLOAD = {
    "id": "user_1",
    "email": "ada@example.com",
    "full_name": "Ada Lovelace",
    "avatar_url": None,
    "gender": None,
    "age": None,
    "address": None,
    "phone_number": None,
    "bio": None,
    "provider": "LOCAL",
    "role": "USER",
    "status": "ACTIVE",
    "assigned_roles": [],
    "permissions": [],
    "created_at": "2026-05-27T00:00:00",
    "updated_at": "2026-05-27T00:00:00",
}


class UserAgentToolsTests(unittest.IsolatedAsyncioTestCase):
    async def test_get_current_user_tool_calls_me_endpoint(self):
        seen_request = None

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal seen_request
            seen_request = request
            return httpx.Response(200, json=USER_PAYLOAD)

        service_client = UserServiceClient(
            base_url="http://user-service.local",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )
        me_tool = create_user_service_tools(service_client)[0]

        result = await me_tool.ainvoke({})

        self.assertEqual(result["email"], "ada@example.com")
        self.assertEqual(seen_request.method, "GET")
        self.assertEqual(seen_request.url.path, "/users/me")

    async def test_list_users_tool_calls_users_endpoint_and_filters_locally(self):
        seen_request = None
        admin_payload = {
            **USER_PAYLOAD,
            "id": "admin_1",
            "email": "admin@example.com",
            "full_name": "Admin User",
            "role": "ADMIN",
        }

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal seen_request
            seen_request = request
            return httpx.Response(200, json=[USER_PAYLOAD, admin_payload])

        service_client = UserServiceClient(
            base_url="http://user-service.local",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )
        list_tool = create_user_service_tools(service_client)[1]

        result = await list_tool.ainvoke({"role": "ADMIN", "max_results": 10})

        self.assertEqual(result["total_matches"], 1)
        self.assertEqual(result["data"][0]["email"], "admin@example.com")
        self.assertEqual(seen_request.method, "GET")
        self.assertEqual(seen_request.url.path, "/users")

    async def test_get_user_by_id_tool_calls_user_detail_endpoint(self):
        seen_request = None

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal seen_request
            seen_request = request
            return httpx.Response(200, json=USER_PAYLOAD)

        service_client = UserServiceClient(
            base_url="http://user-service.local",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )
        get_tool = create_user_service_tools(service_client)[2]

        result = await get_tool.ainvoke({"user_id": "user/1"})

        self.assertEqual(result["full_name"], "Ada Lovelace")
        self.assertEqual(str(seen_request.url), "http://user-service.local/users/user%2F1")

    async def test_get_user_stats_tool_calls_stats_endpoint(self):
        seen_request = None

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal seen_request
            seen_request = request
            return httpx.Response(
                200,
                json={
                    "total_users": 10,
                    "active_users": 8,
                    "blocked_users": 2,
                    "admin_count": 1,
                    "age_groups": [{"label": "18-24", "value": 3}],
                    "gender_mix": [{"label": "Unknown", "value": 10}],
                },
            )

        service_client = UserServiceClient(
            base_url="http://user-service.local",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )
        stats_tool = create_user_service_tools(service_client)[3]

        result = await stats_tool.ainvoke({})

        self.assertEqual(result["total_users"], 10)
        self.assertEqual(result["admin_count"], 1)
        self.assertEqual(seen_request.method, "GET")
        self.assertEqual(seen_request.url.path, "/users/stats")

    async def test_user_service_http_error_uses_error_response_schema(self):
        service_client = UserServiceClient(
            base_url="http://user-service.local",
            http_client=httpx.AsyncClient(
                transport=httpx.MockTransport(
                    lambda request: httpx.Response(
                        403,
                        json={"code": "FORBIDDEN", "message": "Admin role is required.", "details": {}},
                    )
                )
            ),
        )
        stats_tool = create_user_service_tools(service_client)[3]

        with self.assertRaises(UserServiceError) as error:
            await stats_tool.ainvoke({})

        self.assertIn("403", str(error.exception))
        self.assertIn("Admin role is required", str(error.exception))

    def test_tools_have_explicit_names_descriptions_and_schemas(self):
        service_client = UserServiceClient(
            base_url="http://user-service.local",
            http_client=httpx.AsyncClient(
                transport=httpx.MockTransport(lambda request: httpx.Response(200, json={}))
            ),
        )

        me_tool, list_tool, get_tool, stats_tool = create_user_service_tools(service_client)

        self.assertIsInstance(me_tool, BaseTool)
        self.assertEqual(me_tool.name, "get_current_user")
        self.assertIn("GET /users/me", me_tool.description)
        self.assertIs(me_tool.args_schema, GetCurrentUserInput)
        self.assertEqual(list_tool.name, "list_users")
        self.assertIn("GET /users", list_tool.description)
        self.assertIn("max_results", list_tool.args)
        self.assertIs(list_tool.args_schema, ListUsersInput)
        self.assertEqual(get_tool.name, "get_user_by_id")
        self.assertIn("GET /users/{user_id}", get_tool.description)
        self.assertIn("user_id", get_tool.args)
        self.assertIs(get_tool.args_schema, GetUserByIdInput)
        self.assertEqual(stats_tool.name, "get_user_stats")
        self.assertIn("GET /users/stats", stats_tool.description)
        self.assertIs(stats_tool.args_schema, GetUserStatsInput)


if __name__ == "__main__":
    unittest.main()
