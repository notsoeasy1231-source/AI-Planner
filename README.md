# AI-Planner
personalized itineraries based on user preferences such as budget, trip duration, interests, travel style, adventure level, group type, and season. The application integrates real-time travel data, accommodation suggestions, transportation options, weather forecasts, maps, and AI recommendations into a single platform

## Setup (local)

### Run
- `pip install -r requirements.txt`
- `streamlit run app.py`

### Environment variables
- `OPENAI_API_KEY` (optional; if missing, the app uses a stub itinerary generator)
- `OPENAI_MODEL` (optional; default `gpt-4o-mini`)
- `GOOGLE_MAPS_API_KEY` (optional; used to create richer Google Maps links if present)

### Notes
This repository contains a simple MVP. External integrations like Places/Directions/Weather are implemented as best-effort (they won’t block the UI if keys are missing).

