from surf_spot_finder.agents.smolagents import load_smolagent


def test_google_maps_tool(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "FOO")

    no_google_maps_agent = load_smolagent("gemini/gemini-2.0-flash", "GEMINI_API_KEY")
    assert sorted(list(no_google_maps_agent.tools.keys())) == [
        "final_answer",
        "visit_webpage",
        "web_search",
    ]

    monkeypatch.setenv("GOOGLE_MAPS_API_KEY", "BAR")
    google_maps_agent = load_smolagent("gemini/gemini-2.0-flash", "GEMINI_API_KEY")
    assert sorted(list(google_maps_agent.tools.keys())) == [
        "final_answer",
        "maps_directions",
        "maps_distance_matrix",
        "maps_elevation",
        "maps_geocode",
        "maps_place_details",
        "maps_reverse_geocode",
        "maps_search_places",
        "visit_webpage",
        "web_search",
    ]
