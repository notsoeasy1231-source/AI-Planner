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
        # Parse a few things from the prompt to create variation
        seed = abs(hash(user_message)) % 10_000
        m_days = re.search(r"(\d+)\s*(day|days)", user_message, flags=re.IGNORECASE)
        duration = int(m_days.group(1)) if m_days else 4

        m_budget = re.search(r"₹\s*([0-9,]+)", user_message)
        budget = int(m_budget.group(1).replace(",", "")) if m_budget else 20000

        # Try to infer start city
        start_city = "Bengaluru"
        m_city = re.search(r"starting from\s*([A-Za-z\s]+?)(\.|,|$)", user_message, flags=re.IGNORECASE)
        if m_city:
            start_city = m_city.group(1).strip()

        vegetarian_friendly = bool(re.search(r"veg|vegetarian|pure veg", user_message, flags=re.IGNORECASE))

        hill_stations = ["Coorg (Madikeri)", "Munnar", "Coorg", "Chikmagalur", "Nainital", "Darjeeling", "Kodaikanal", "Shillong"]
        # Bias towards “peaceful hill station” vibe
        destination_cycle = [
            hill_stations[(seed + 0) % len(hill_stations)],
            hill_stations[(seed + 1) % len(hill_stations)],
            hill_stations[(seed + 2) % len(hill_stations)],
            hill_stations[(seed + 3) % len(hill_stations)],
        ]

        # Split budget across categories (roughly)
        # Make it feel higher-end by allocating more to stays + attractions
        accom = int(budget * 0.45)
        food = int(budget * 0.20)
        local_transport = int(budget * 0.15)
        attractions = int(budget * 0.10)
        buffer = budget - (accom + food + local_transport + attractions)

        # Create multi-day, multi-destination itinerary
        itinerary = []
        for d in range(1, duration + 1):
            dest = destination_cycle[(d - 1) % len(destination_cycle)]
            day_kind = [
                ("arrival + gentle viewpoint", "morning"),
                ("nature walk + panoramic views", "afternoon"),
                ("café + heritage / viewpoints", "evening"),
                ("scenic day trip + relaxed return", "morning"),
            ][(seed + d) % 4]

            title = f"{dest}: {day_kind[0].title()}"
            # Ensure activities have the expected schema
            activities = []
            activities.append({"time": "morning", "text": f"Slow start: check-in / breakfast + short viewpoint drive near {dest}"})
            activities.append({"time": "afternoon", "text": f"Nature experience: {('guided trail' if d % 2 == 0 else 'serene gardens')} + panoramic lookout"})
            activities.append({"time": "evening", "text": "Comfort dinner at a curated veg place + relaxed sunset / stargazing spot"})

            restaurants = [
                {
                    "name": "Curated Vegetarian Dining (highly recommended)",
                    "budget_per_person_inr": max(350, int(food / max(duration, 1) / 2)),
                }
            ]

            # Distribute per-day costs
            estimated_cost = int(budget / max(duration, 1))

            map_queries = [
                f"{dest} scenic viewpoint",
                f"{dest} nature walk trail",
                f"{dest} vegetarian restaurant",
            ]

            itinerary.append(
                {
                    "day": d,
                    "title": title,
                    "destination": dest,
                    "activities": activities,
                    "restaurants": restaurants,
                    "estimated_cost_inr": estimated_cost,
                    "map_queries": map_queries,
                    "weather_note": "If it rains, swap outdoor viewpoints/walks with indoor cafés, heritage spots, and scenic by-lane drives.",
                }
            )

        generated_at = datetime.now().isoformat() + "Z"

        return {
            "meta": {
                "mode": "stub",
                "generated_at": generated_at,
            },
            "trip_summary": {
                "start_city": start_city,
                "duration_days": duration,
                "budget_total_inr": budget,
                "interests": ["relaxing", "scenic"],
                "travel_style": "high-end relaxed",
                "vegetarian_friendly": vegetarian_friendly,
            },
            "itinerary": itinerary,
            "budget_breakdown": {
                "accommodation_inr": accom,
                "transport_local_inr": local_transport,
                "food_inr": food,
                "attractions_inr": attractions,
                "buffer_inr": buffer,
                "total_estimated_inr": budget,
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

