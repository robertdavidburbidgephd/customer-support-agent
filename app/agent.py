# ruff: noqa
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import google.auth
from pydantic import BaseModel, Field

from google.adk.agents import LlmAgent
from google.adk.apps import App
from google.adk.events import Event, EventActions
from google.adk.models import Gemini
from google.adk.workflow import Workflow, Edge, START, node
from google.genai import types

_, project_id = google.auth.default()
os.environ["GOOGLE_CLOUD_PROJECT"] = project_id
os.environ["GOOGLE_CLOUD_LOCATION"] = "global"
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"

# Shared Gemini model configuration using gemini-2.5-flash
shared_model = Gemini(
    model="gemini-2.5-flash",
    retry_options=types.HttpRetryOptions(attempts=3),
)


# Output schema for classification LlmAgent
class ClassificationOutput(BaseModel):
    is_shipping_related: bool = Field(
        description="True if the query is about shipping rates, package tracking, delivery, or returns. False otherwise."
    )


# 1. Node to extract query and save to state
@node
def save_query(node_input: types.Content) -> Event:
    query_text = ""
    if node_input and node_input.parts:
        query_text = "".join(part.text for part in node_input.parts if part.text)
    return Event(
        output=query_text,
        actions=EventActions(state_delta={"user_query": query_text}),
    )


# 2. Classification Agent to determine routing
classifier_agent = LlmAgent(
    name="classifier_agent",
    model=shared_model,
    instruction=(
        "You are a routing assistant. Analyze the user's query and classify if it is related "
        "to shipping (rates, tracking, delivery, returns). "
        "Respond ONLY with the structured output indicating is_shipping_related."
    ),
    output_schema=ClassificationOutput,
)


# 3. Router node to branch the execution flow
@node
def route_query(node_input: dict, user_query: str) -> Event:
    is_related = node_input.get("is_shipping_related", False)
    if is_related:
        return Event(output=user_query, actions=EventActions(route="shipping"))
    else:
        return Event(output=user_query, actions=EventActions(route="unrelated"))


# 4. Shipping FAQ Agent to handle shipping queries
shipping_faq_agent = LlmAgent(
    name="shipping_faq_agent",
    model=shared_model,
    instruction=(
        "You are an energetic, extremely helpful, and playful customer support representative for a shipping company! 🚚🎉 "
        "Answer the user's questions about shipping rates, tracking, delivery, or returns with lots of enthusiasm and emojis. "
        "Always be sure to highlight that we offer FREE shipping on all orders over $50! 🥳✨"
    ),
)


# 5. Decline node to decline unrelated queries politely
@node
def decline_node(node_input: str):
    message = (
        "I'm sorry, but I can only answer questions related to shipping (such as rates, tracking, delivery, or returns). "
        "Please let me know if you have a question on those topics!"
    )
    yield Event(
        content=types.Content(role="model", parts=[types.Part.from_text(text=message)])
    )
    yield Event(output=message)


# Define edges using Edge objects for robust type safety
edges = [
    Edge(from_node=START, to_node=save_query),
    Edge(from_node=save_query, to_node=classifier_agent),
    Edge(from_node=classifier_agent, to_node=route_query),
    Edge(from_node=route_query, to_node=shipping_faq_agent, route="shipping"),
    Edge(from_node=route_query, to_node=decline_node, route="unrelated"),
]

# Graph-based Workflow agent definition
root_agent = Workflow(
    name="customer_support_workflow",
    edges=edges,
    description="Classifies customer queries into shipping or unrelated, routing to appropriate handlers.",
)

# Application container setup matching standard conventions
app = App(
    root_agent=root_agent,
    name="app",
)
