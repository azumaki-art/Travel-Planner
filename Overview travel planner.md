# Travel Planning Robot with LangGraph

An AI travel planner that chats with you in a web page and builds a full trip
plan step by step: it finds attractions, looks up their exact locations, plans
the transport between them, and finally suggests restaurants and hotels nearby.

It is built with **LangGraph** (the agent workflow), **LangChain** (the model and
tool layer) and **Gradio** (the web interface).

---

## 1. Overview

The project is a single "agent" that can call six tools. The agent runs in a loop:

```
User message -> LLM decides -> calls a tool -> tool returns data -> LLM decides again
                                                                        |
                                                                  ... until done
                                                                        |
                                                              final answer to the user
```

The LLM is the "brain": it reads your request, understands what information is
still missing, and picks the right tool for each step. All the real data
(attractions, coordinates, routes, nearby places) comes from external services,
so the answers are based on real information rather than guesses.

Two external services are used:

| Service | What it is used for |
| --- | --- |
| A chat LLM (DeepSeek / Google Gemini / OpenAI) | Understanding the request and writing the itinerary |
| Google Maps Platform + Trip.com + web search | Real coordinates, routes, attractions, restaurants, hotels |

You talk to it through a simple web page. The page shows the chat on the right
and a live "Debug Info" panel on the left, where you can watch which tool is
running at that moment.

---

## 2. Main Features

- **Chat-based planning** – just describe your trip in normal language, no forms.
- **Streaming answers** – the reply appears word by word while it is generated.
- **Live tool log** – every tool call and its output is shown in the Debug Info panel.
- **Attraction discovery** – reads the Trip.com travel guide to get the top
  attractions of a city, with rating, review count, ticket price, opening hours,
  recommended visiting time, address and coordinates.
- **Coordinate lookup** – turns a place name into `longitude,latitude` using the
  Google Geocoding API, and returns *all* places with that name so the agent can
  pick the right one.
- **Public transport and driving routes** – distance, travel time and
  step-by-step directions (including which metro/train line to take) from the
  Google Routes API.
- **Nearby search** – restaurants, cafés, hotels and other places of interest
  around any point, with rating, price level, opening status and a Google Maps link.
- **General web search** – used to fill in gaps (for example when the user has no
  destination yet and needs suggestions).
- **Multiple model providers** – Google Gemini, OpenAI and DeepSeek can all be used
  by changing one environment variable.
- **Separate conversation per browser session** – each session gets its own
  LangGraph thread id, so two users never mix their chats.
- **Long turns are protected** – the history sent to the model is trimmed, and the
  LangGraph recursion limit is raised, so a long planning turn does not fail.
- **Key checking at startup** – missing or placeholder API keys give a clear error
  message instead of a confusing library error.

---

## 3. Project Structure

The repository root is the `TravelPlanner` folder.

```
TravelPlanner/
├── webrun.py                 # Entry point: Gradio web UI + streaming of the agent
├── .env                      # Your secret API keys (NEVER commit this file)
├── requirements.txt          # Python dependencies
│
├── graph/
│   └── graph.py              # Builds the LangGraph workflow and compiles the app
│
├── agents/
│   └── agents.py             # The agent node: builds the prompt, calls the LLM
│
├── models/
│   └── factory.py            # Chooses the LLM (Gemini / OpenAI / DeepSeek) by name
│
├── prompts/
│   └── main.py               # The big system prompt that teaches the agent its job
│
├── states/
│   └── state.py              # PublicState: the data passed between graph nodes
│
├── tools/                    # All tools the agent can call
│   ├── __init__.py           # Exports the tool list used by graph.py
│   ├── web_search.py         # General web search
│   ├── attractions.py        # Trip.com attraction search (biggest tool)
│   ├── locations.py          # Place name -> coordinates (Google Geocoding)
│   ├── transportation.py     # Route planning (Google Routes API)
│   ├── nearby.py             # Nearby restaurants / hotels (Google Places API)
│   ├── save.py               # Saves useful findings
│   └── search_backend.py     # Shared web-search helper with retries
│
└── utils/
    └── helper.py             # Small helpers: time, thread id, API key reading
```

**Reading order if you are new to the code:** `webrun.py` → `graph/graph.py` →
`agents/agents.py` → `prompts/main.py` → `tools/`.

> Note: `requirements.txt` may sit one level above `TravelPlanner/` in your local
> copy. Copy it into the repository root before pushing, so that other people can
> find it (see section 7).

---

## 4. Getting Started

Follow these steps after downloading or cloning the project. They work on
Windows, macOS and Linux.

### Step 0 – What you need

- **Python 3.11 or newer** (developed on Python 3.13.2)
- **pip**
- An **internet connection** (the tools call online APIs)
- An **API key for a chat model** (DeepSeek, Google Gemini or OpenAI)
- A **Google Maps API key**

### Step 1 – Get a chat model API key

Pick one provider and create a key:

- DeepSeek – <https://platform.deepseek.com>
- Google Gemini – <https://aistudio.google.com/apikey>
- OpenAI – <https://platform.openai.com/api-keys>

### Step 2 – Get a Google Maps API key

1. Go to the [Google Cloud Console](https://console.cloud.google.com/).
2. Create a project and enable billing (Google gives a monthly free credit).
3. Enable these **three** APIs:
   - **Geocoding API** – used to turn a place name into coordinates
   - **Places API (New)** – used to search for restaurants and hotels
   - **Routes API** – used to plan the transport between places
4. Create an API key under *APIs & Services → Credentials*.

> If you skip an API, the matching tool will fail with a clear error message that
> says which API is missing.

### Step 3 – Download the project

```bash
git clone <your-repository-url>
cd <your-repository-name>
```

Or simply download the ZIP from GitHub and unzip it.

### Step 4 – Create a virtual environment

**Windows (PowerShell)**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 5 – Install the dependencies

Run this **in the folder that contains `requirements.txt`**:

```bash
pip install -r requirements.txt
```

### Step 6 – Create your `.env` file

Create a file named `.env` next to `webrun.py` (inside `TravelPlanner/`) and put
your own keys in it:

```env
# --- Chat model (set the one you use) ---
DEEPSEEK_API_KEY=your_deepseek_key_here
# Gemini_API_KEY=your_gemini_key_here
# OPENAI_API_KEY=your_openai_key_here

# --- Google Maps ---
MAP_API_KEY=your_google_maps_key_here

# --- Optional: which model to use ---
# MODEL_NAME=deepseek-v4-flash
```

Notes:

- The app also accepts `GEMINI_API_KEY` / `GOOGLE_API_KEY` for Gemini, and
  `GOOGLE_MAPS_API_KEY` / `GOOGLE_MAP_API_KEY` for Google Maps.
- If `MODEL_NAME` is not set, the app uses the default written in `webrun.py`.
  Use a name that matches one of the supported providers, for example
  `deepseek-chat`, `gemini-2.5-flash` or `gpt-4o-mini`.

### Step 7 – Run the app

Run it **from inside the `TravelPlanner` folder** (this matters, see section 7):

```bash
cd TravelPlanner
python webrun.py
```

The terminal prints a local URL, usually:

```
http://127.0.0.1:7860
```

Open that address in your browser, type something like
*"I want to visit Tokyo for 3 days"*, and press **Send**.

### Optional – Test a single tool without the UI

Every tool file can be run on its own to check that its API key works:

```bash
python tools/locations.py
python tools/transportation.py
python tools/nearby.py
python tools/attractions.py
```

---

## 5. Main Modules and Tools

### `webrun.py` – the web interface and entry point

Starts the Gradio page and streams one user turn through the agent. It listens to
LangGraph events and forwards them to the UI:

- model tokens → appended to the chat bubble (streaming effect)
- `on_tool_start` / `on_tool_end` → written into the Debug Info panel

It also raises the graph `recursion_limit` to 100, because one planning turn can
make many tool calls (one coordinate lookup per attraction, then one route per leg).

### `graph/graph.py` – the workflow

Builds a two-node LangGraph graph:

| Node | Job |
| --- | --- |
| `agent` | The LLM decides: answer the user, or call a tool |
| `tools` | Runs the tool the agent asked for and returns the result |

The flow is `START → agent → (tools → agent)* → END`. It uses `InMemorySaver` as
the checkpointer, which is what keeps the conversation of each thread id together.

`init_app(model_name, is_async=True)` returns the compiled app. Set
`is_async=False` to use the synchronous agent (useful for simple scripts).

### `agents/agents.py` – the agent node

Contains the `Agent` base class and two subclasses:

- **`AsyncAgent`** – uses `astream`, so tokens arrive one by one (used by the web UI)
- **`SyncAgent`** – uses `invoke`, waits for the complete answer

It also holds three helper functions:

- `_build_prompt` – builds the message list: the system prompt first, then the
  history, with the first user message embedded in the prompt template.
- `_trim_history` – keeps only the most recent messages within a character budget,
  so a long turn does not re-send a huge transcript on every model call. It never
  starts the window with a tool result, which would be invalid for the model.
- `_merge_chunks` – joins streamed chunks into one message, keeping the tool calls.

### `models/factory.py` – the model chooser

`LLMFactory.get_llm(model, temperature)` returns the right chat model based on the
name prefix:

| Name starts with | Provider used | Key read from |
| --- | --- | --- |
| `gemini` | Google Gemini | `Gemini_API_KEY`, `GEMINI_API_KEY`, `GOOGLE_API_KEY` |
| `gpt` | OpenAI | `OPENAI_API_KEY` |
| `deepseek` | DeepSeek (OpenAI-compatible endpoint) | `DEEPSEEK_API_KEY`, `DeepSeek_API_KEY` |

Anything else raises a `ValueError` telling you which names are supported.

### `prompts/main.py` – the instructions for the agent

One long prompt that defines the agent's role, its rules, the list of tools, and a
15-step planning workflow (find destination → attractions → coordinates →
itinerary → routes → restaurants and hotels → final plan). It also asks the model
to think in a fixed format (`Task / Review / Thought / Plan`) before calling a
tool, and forbids making up information.

### `states/state.py` – the shared state

```python
class PublicState(TypedDict):
    messages: Annotated[list, add_messages]
```

A single `messages` list. `add_messages` tells LangGraph to *append* new messages
instead of overwriting the list each time a node runs.

### `tools/` – the six tools

| File | Tool | Backed by |
| --- | --- | --- |
| `web_search.py` | `web_search` | DuckDuckGo (`ddgs`) |
| `attractions.py` | `get_attractions_information` | Trip.com travel guide pages |
| `locations.py` | `get_location_coordinate` | Google Geocoding API |
| `transportation.py` | `route_planning` | Google Routes API |
| `nearby.py` | `search_nearby_poi` | Google Places API (New) |
| `save.py` | `save_info_and_clear_history` | No API – just remembers findings |

Other files in this folder:

- `__init__.py` – the import list that `graph.py` uses to build the tool set.
- `search_backend.py` – one shared web-search function with retries and backoff,
  so the flaky search backend does not break a planning turn.

### `utils/helper.py` – small shared helpers

- `get_current_local_datetime()` – the current time, injected into the prompt.
- `get_thread_id()` – a fresh UUID for each browser session.
- `get_api_key(*names, service=...)` – returns the first non-empty key from the
  environment, or raises a clear error if it is missing or still a placeholder.

---

## 6. Tool Reference: Parameters and Return Values

These are the six tools the agent can call. Every parameter has a description that
the model can read, so the agent usually fills them correctly on its own.

> **Coordinate format:** everywhere in this project, coordinates are written as
> `"longitude,latitude"` — **longitude first**. Example: `"139.7967,35.7148"`.

---

### 6.1 `web_search`

General English web search, used for background information and to fill gaps.

| Parameter | Type | Required | Default | Meaning |
| --- | --- | --- | --- | --- |
| `keywords` | string | yes | – | What to search for. The more precise, the better. |
| `max_results` | integer | no | `10` | How many results to return (clamped to 1–20). |

**Returns:** a list of results, each one a dictionary:

```python
[{"title": "...", "href": "https://...", "body": "short snippet ..."}, ...]
```

If nothing is found, it returns a plain string message instead, for example
`No web results found for "..."`.

---

### 6.2 `get_attractions_information`

Finds the top attractions of a city using the Trip.com travel guide.

| Parameter | Type | Required | Default | Meaning |
| --- | --- | --- | --- | --- |
| `destination` | string | yes | – | A specific city, e.g. `"Tokyo"`. A Trip.com id such as `"tokyo-294"` also works. |
| `max_attractions` | integer | no | `10` | How many attractions to return (clamped to 1–10). |
| `include_details` | boolean | no | `True` | When true, also opens each attraction page for opening hours, visit time and address. Slower but more useful. |

**Returns:** a dictionary:

```python
{
  "destination": {
      "name": "Tokyo",
      "trip_slug": "tokyo",
      "trip_district_id": "294",
      "guide_url": "https://www.trip.com/travel-guide/destination/tokyo-294/..."
  },
  "overview": "short description of the destination",
  "attractions": [
      {
        "name": "Senso-ji Temple",
        "poi_id": "12345",
        "url": "https://www.trip.com/travel-guide/attraction/tokyo/...",
        "description": "short description",
        "district": "Asakusa",
        "price": "Free entry",            # or "From US$ 16.99", or "not available"
        "rating": "4.6",
        "review_count": "12345",
        "trip_score": "...",
        "category": "Historic landmark",
        "recommended_duration": "1-2 hours",
        "opening_hours": "08:00-17:00",
        "address": "...",
        "coordinate": "139.7967,35.7148",
        "open_now": True,
        "telephone": "..."
      },
      ...
  ]
}
```

Not every field is always present — Trip.com does not publish all of them for
every attraction, and the agent is told to judge the missing ones from experience.

---

### 6.3 `get_location_coordinate`

Turns a place name into coordinates.

| Parameter | Type | Required | Default | Meaning |
| --- | --- | --- | --- | --- |
| `location` | string | yes | – | The place name, e.g. `"Senso-ji Temple"`. |
| `city` | string | no | `""` | The city or region, used to avoid mixing up places with the same name. |

**Returns:** a **list** of matches, because many places share a name. Each item:

```python
[{
    "name": "4 Chome-3-1 Asakusa, Taito City, Tokyo, Japan",
    "address": "same as name, the full formatted address",
    "coordinate": "139.7967,35.7148",
    "city": "Taito City",
    "country": "Japan",
    "place_id": "ChIJ...."
}, ...]
```

If nothing is found it returns a plain string message asking for a more specific
name.

---

### 6.4 `route_planning`

Plans how to travel between two coordinates.

| Parameter | Type | Required | Default | Meaning |
| --- | --- | --- | --- | --- |
| `origin` | string | yes | – | Start point as `"longitude,latitude"`. |
| `destination` | string | yes | – | End point as `"longitude,latitude"`. |
| `travel_mode` | string | no | `"TRANSIT"` | One of `TRANSIT` (public transport), `WALK`, `DRIVE`, `BICYCLE`. |

**Returns:** a dictionary:

```python
{
  "origin": "139.7967,35.7148",
  "destination": "139.7454,35.6586",
  "travel_mode": "TRANSIT",
  "distance_meters": 5200,
  "distance": "5.20 km",
  "duration_seconds": 1500,
  "duration": "25 min",
  "transit_fare": "180 JPY",     # or "not available"
  "steps": [
      {
        "travel_mode": "TRANSIT",
        "distance_meters": 800,
        "duration_seconds": 600,
        "instruction": "Take the Ginza Line ...",
        "transit_line": "Ginza Line",     # only for TRANSIT steps
        "vehicle_type": "SUBWAY",
        "headsign": "Shibuya",
        "stop_count": 4,
        "departure_stop": "Asakusa",
        "arrival_stop": "Ueno"
      },
      ...
  ]
}
```

If no route is found, it returns a plain string message suggesting another mode.

---

### 6.5 `search_nearby_poi`

Finds restaurants, hotels, shops and other places around a point.

| Parameter | Type | Required | Default | Meaning |
| --- | --- | --- | --- | --- |
| `location` | string | yes | – | Center point as `"longitude,latitude"`. |
| `poi_types` | string | no | `""` | Comma-separated Google place types, e.g. `"restaurant,cafe"` or `"lodging"`. |
| `keyword` | string | no | `""` | A specific place name, e.g. `"Starbucks"`. Use only when the user names one. |
| `radius` | integer | no | `2000` | Search radius in meters (clamped to 1–50000). |
| `max_results` | integer | no | `10` | How many places to return (clamped to 1–20). |

You must give at least one of `keyword` or `poi_types`, otherwise the tool raises
an error explaining that.

**Returns:** a dictionary:

```python
{
  "center_point": "139.7967,35.7148",
  "search_radius_meters": 2000,
  "search_result_count": 10,
  "pois": [
      {
        "name": "Example Ramen",
        "type": "Ramen restaurant",
        "address": "1-2-3 Asakusa, Tokyo",
        "distance_meters": 320,          # straight-line distance from the center
        "coordinate": "139.7990,35.7130",
        "rating": 4.4,
        "review_count": 812,
        "price_level": "PRICE_LEVEL_INEXPENSIVE",
        "open_now": True,
        "google_maps_url": "https://maps.google.com/?cid=..."
      },
      ...
  ]
}
```

Results are sorted by distance, nearest first. If nothing is found, it returns a
plain string message suggesting a bigger radius or different keywords.

---

### 6.6 `save_info_and_clear_history`

Saves the useful findings collected so far, so important details are not lost
during a long conversation.

| Parameter | Type | Required | Default | Meaning |
| --- | --- | --- | --- | --- |
| `information_to_save` | string | yes | – | The useful information gathered so far, as detailed as possible. |

**Returns:** a pair `(content, artifact)`:

- `content` – the short text `"Information saved."`, which is the only part the model sees
- `artifact` – the full `information_to_save` text

This tool does not call any API. It exists to keep important details close to the
agent while the conversation grows.

---

## 7. Common Problems

### The app starts but every answer fails with a "Missing API key" error

Your `.env` file is missing a key, or still contains a placeholder. Check that
`Gemini_API_KEY` / `DEEPSEEK_API_KEY` / `OPENAI_API_KEY` and `MAP_API_KEY` are set
to real values, then restart the app. The `.env` file must be next to `webrun.py`.

### `ModuleNotFoundError: No module named 'graph'` (or `tools`, `agents`)

The app must be started **from inside the `TravelPlanner` folder**, because that
folder is the import root:

```bash
cd TravelPlanner
python webrun.py
```

### `ModuleNotFoundError` for any third-party package

The dependencies are not installed in the environment you are using. Activate
your virtual environment and run:

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not in the project root, copy it there (in some working
copies it sits one level above `TravelPlanner/`).

### The coordinate tools fail with "Geocoding API is not enabled" or an HTTP error

A Google API is not enabled for your key, or billing is not set up. Enable
**Geocoding API**, **Places API (New)** and **Routes API** in the Google Cloud
Console for the same project as the key.

### Attraction search returns "No attractions were found"

The destination name may be too small or misspelled, or Trip.com changed its page
layout. Try a bigger nearby city, or pass the Trip.com destination id directly,
for example `"tokyo-294"`.

### The agent is very slow on long trips

Each attraction needs a coordinate lookup and each leg needs a route lookup, so a
3-day plan can make dozens of calls. Two things help:

- lower `max_attractions` so fewer places are looked up
- set `HISTORY_CHAR_BUDGET` (default `60000`) to a smaller number so less text is
  re-sent to the model, or set it to `0` to turn trimming off entirely

### "GraphRecursionError" or the turn stops early

The graph allows 100 steps. A very long planning turn can exceed that. Split the
request into smaller pieces (first the itinerary, then hotels), or ask for fewer
days at a time.

### The page loads but nothing happens when I press Send

Check the terminal window for an error. Common causes are an invalid model name
for the provider (see `models/factory.py` for the accepted prefixes) or a
model/API key mismatch.

### Which Python version should I use?

Python 3.11 or newer. The project was developed and tested on Python 3.13.2.

### Do I need Selenium or Chrome?

No. Earlier versions scraped Trip.com with Selenium, but the attraction tool now
uses plain HTTP requests and BeautifulSoup, so no browser or ChromeDriver is
needed. Some unused packages (`selenium`, `googlemaps`, `duckduckgo_search`) may
still appear in `requirements.txt`.

---

### Important: before you push this project to GitHub

- **Never commit the `.env` file.** It contains your private API keys. Add a
  `.gitignore` with at least these lines:

  ```gitignore
  .env
  .venv/
  __pycache__/
  *.pyc
  .idea/
  ```

- If a key was ever committed by accident, **revoke and regenerate it** in the
  provider console — deleting the file does not remove it from Git history.
- Ship a `.env.example` file with placeholder values instead, so other people
  know which variables they need.
