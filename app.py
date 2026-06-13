import json
import os
import re
from datetime import datetime

import pandas as pd
import requests
import streamlit as st
from openai import OpenAI

st.set_page_config(
    page_title="AI Travel Planner India",
    page_icon="🧳",
    layout="wide",
)

st.title("🧳 AI Travel Planner for India")
st.write("AI-powered trip planning with day-wise itinerary, budget breakdown, and map/weather context.")

# -----------------------------
# Helpers
# -----------------------------

def _get_env(name: str) -> str | None:
    v = os.getenv(name)
    if not v:
        return None
    return v.strip() or None


def extract_json(text: str) -> dict:
    """Best-effort extraction of a single JSON object from model output."""
    text = text.strip()

    # Common case: the model returns pure JSON
    if text.startswith("{") and text.endswith("}"):
        return json.loads(text)

    # Try to find the first {...} block
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        raise ValueError("No JSON object found in model output")
    return json.loads(m.group(0))


def call_openai_itinerary(user_message: str) -> dict:
    """Calls OpenAI to generate a strict JSON itinerary."""
    api_key = _get_env("OPENAI_API_KEY")

    # Allow local dev without API key via stub
    if not api_key:
        # Create light variation so different prompts don't look identical
        seed = abs(hash(user_message)) % 1000
        duration = 3
        m = re.search(r"(\d+)\s*(day|days)", user_message, flags=re.IGNORECASE)
        if m:
            duration = int(m.group(1))
        budget = 10000
        m2 = re.search(r"₹\s*([0-9,]+)", user_message)
        if m2:
            budget = int(m2.group(1).replace(",", ""))

        cities = ["Araku Valley", "Gokarna", "Tarkarli", "Munnar", "Coorg", "Rishikesh"]
        destination = cities[seed % len(cities)]

        generated_at = datetime.now().isoformat() + "Z"

        return {
            "meta": {
                "mode": "stub",
                "generated_at": generated_at,
            },
            "trip_summary": {
                "start_city": "Hyderabad",
                "duration_days": 3,
                "budget_total_inr": 10000,
                "interests": ["relaxing", "scenic"],
                "travel_style": "relaxed",
                "vegetarian_friendly": True,
            },
            "itinerary": [
                {
                    "day": 1,
                    "title": "Scenic arrival + local food",
                    "destination": "Araku Valley",
                    "activities": [
                        {"time": "morning", "text": "Travel to Araku and check-in"},
                        {"time": "afternoon", "text": "Coffee plantation visit + viewpoint"},
                        {"time": "evening", "text": "Local veg dinner and sunset spot"},
                    ],
                    "restaurants": [
                        {"name": "Local Veg Restaurant (suggested)", "budget_per_person_inr": 400},
                    ],
                    "estimated_cost_inr": 3200,
                    "map_queries": ["Araku Valley coffee plantation viewpoint"],
                    "weather_note": "Use forecast for rain/heat and adjust outdoor stops.",
                },
                {
                    "day": 2,
                    "title": "Waterfalls + culture",
                    "destination": "Araku Valley",
                    "activities": [
                        {"time": "morning", "text": "Waterfalls + scenic walk"},
                        {"time": "afternoon", "text": "Tribal museum / local crafts"},
                        {"time": "evening", "text": "Café stop with local snacks"},
                    ],
                    "restaurants": [
                        {"name": "Veg Café (suggested)", "budget_per_person_inr": 500},
                    ],
                    "estimated_cost_inr": 3500,
                    "map_queries": ["Araku waterfalls viewpoint"],
                    "weather_note": "If raining, swap to museum/café blocks.",
                },
                {
                    "day": 3,
                    "title": "Morning viewpoints + departure",
                    "destination": "Araku Valley",
                    "activities": [
                        {"time": "morning", "text": "Sunrise viewpoint"},
                        {"time": "afternoon", "text": "Last-minute shopping + check-out"},
                        {"time": "evening", "text": "Return journey"},
                    ],
                    "restaurants": [
                        {"name": "Quick Veg Lunch (suggested)", "budget_per_person_inr": 300},
                    ],
                    "estimated_cost_inr": 2800,
                    "map_queries": ["Araku Valley sunrise viewpoint"],
                    "weather_note": "Pack light rain protection if forecast shows showers.",
                },
            ],
            "budget_breakdown": {
                "accommodation_inr": 3000,
                "transport_local_inr": 2500,
                "food_inr": 1400,
                "attractions_inr": 600,
                "buffer_inr": 2000,
                "total_estimated_inr": 10000,
            },
        }

    client = OpenAI(api_key=api_key)

    system_prompt = """
You are a senior travel planner for trips within India.
Return ONLY strict JSON (no markdown, no commentary).

Generate a complete trip plan given the user's message.

JSON schema:
{
  "meta": {
    "mode": "openai",
    "generated_at": "ISO8601Z"
  },
  "trip_summary": {
    "start_city": string,
    "duration_days": number,
    "budget_total_inr": number,
    "interests": [string],
    "travel_style": string,
    "vegetarian_friendly": boolean
  },
  "itinerary": [
    {
      "day": number,
      "title": string,
      "destination": string,
      "activities": [
        {"time": string, "text": string}
      ],
      "restaurants": [
        {"name": string, "budget_per_person_inr": number}
      ],
      "estimated_cost_inr": number,
      "map_queries": [string],
      "weather_note": string
    }
  ],
  "budget_breakdown": {
    "accommodation_inr": number,
    "transport_local_inr": number,
    "food_inr": number,
    "attractions_inr": number,
    "buffer_inr": number,
    "total_estimated_inr": number
  }
}

Rules:
- budget_total_inr must be respected (approx; allow buffer).
- duration_days must match the user's duration; if missing, default to 3.
- Provide weather_note as guidance (not actual forecast).
- map_queries should be concise search strings for Google Maps.
- restaurants should be vegetarian-friendly suggestions when appropriate.
""".strip()

    resp = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0.4,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    )

    content = resp.choices[0].message.content or ""
    return extract_json(content)


# -----------------------------
# Weather/Maps stubs
# -----------------------------

def build_maps_link(query: str) -> str:
    return "https://www.google.com/maps/search/?api=1&query=" + requests.utils.quote(query)


# -----------------------------
# UI
# -----------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Trip controls")
    st.caption("Provide a prompt like: budget, days, start city, interests, vegetarian preference, pace.")

    sample = st.text_area(
        "Sample prompt",
        value="I have ₹15000, 4 days, starting from Hyderabad. I want a relaxing mountain trip with good vegetarian food and scenic viewpoints.",
        height=120,
    )

    if st.button("Use sample prompt"):
        st.session_state.user_prompt = sample

prompt = st.chat_input("Describe your trip... (budget, days, start city, interests, veg preference, pace)")

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Planning your itinerary..."):
            try:
                plan = call_openai_itinerary(prompt)
                st.session_state.messages.append({"role": "assistant", "content": "Itinerary generated."})
            except Exception as e:
                err = f"Failed to generate plan: {e}"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})
                plan = None

            if plan:
                trip = plan.get("trip_summary", {})
                meta = plan.get("meta", {})

                st.subheader("Trip Summary")
                cols = st.columns(4)
                cols[0].metric("Start", trip.get("start_city", ""))
                cols[1].metric("Days", trip.get("duration_days", ""))
                cols[2].metric("Budget (₹)", trip.get("budget_total_inr", ""))
                cols[3].metric("Style", trip.get("travel_style", ""))

                st.caption("Interests: " + ", ".join(trip.get("interests", []) or []))
                st.caption("Vegetarian-friendly: " + str(trip.get("vegetarian_friendly", False)))

                st.divider()
                st.subheader("Day-wise Itinerary")

                for day in plan.get("itinerary", []):
                    st.markdown(f"### Day {day.get('day')} — {day.get('title')}")
                    st.write(f"**Destination:** {day.get('destination')}")

                    acts = day.get("activities", [])
                    for a in acts:
                        st.markdown(f"- **{a.get('time')}**: {a.get('text')}")

                    rests = day.get("restaurants", [])
                    if rests:
                        st.markdown("**Suggested restaurants (veg-friendly):**")
                        for r in rests:
                            st.markdown(
                                f"- {r.get('name')} · est. ₹{r.get('budget_per_person_inr', 0)}/person"
                            )

                    st.markdown(f"**Estimated cost for day:** ₹{day.get('estimated_cost_inr', 0)}")

                    mq = day.get("map_queries", [])
                    if mq:
                        st.markdown("**Map links:**")
                        links = [f"[{q}]({build_maps_link(q)})" for q in mq[:3]]
                        st.markdown(" · ".join(links), unsafe_allow_html=False)

                    wn = day.get("weather_note")
                    if wn:
                        st.info("Weather note: " + wn)

                    st.divider()

                st.subheader("Budget Breakdown")
                bb = plan.get("budget_breakdown", {})
                df = pd.DataFrame(
                    [
                        ["Accommodation", bb.get("accommodation_inr", 0)],
                        ["Local transport", bb.get("transport_local_inr", 0)],
                        ["Food", bb.get("food_inr", 0)],
                        ["Attractions", bb.get("attractions_inr", 0)],
                        ["Buffer", bb.get("buffer_inr", 0)],
                    ],
                    columns=["Category", "Estimated (₹)"],
                )

                st.dataframe(df, use_container_width=True, hide_index=True)
                st.success("Total estimated budget: ₹" + str(bb.get("total_estimated_inr", 0)))

                if meta:
                    st.caption("Generated mode: " + str(meta.get("mode", "")))

