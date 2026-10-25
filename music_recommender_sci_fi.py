"""
NEBULA — a local, Streamlit-based song recommender.

Run from PyCharm's terminal:
    pip install -r requirements.txt
    streamlit run music_recommender_sci_fi.py

Keep both model files in the same folder as this file:
    music_recommender.pkl
    bollywood_recommender.pkl

Each pickle should contain a "songs" DataFrame and "tfidf_matrix". Bollywood
artist names are read from its "subtitle" column; its "thumbnail" artwork is
used directly when available.

Spotify credentials are optional. Album artwork is looked up through Apple's
public iTunes Search API first (no API key), with Spotify as an optional fallback.
Only load a pickle file you trust.
"""

from __future__ import annotations

import difflib
import html
import os
import pickle
import re
import unicodedata
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st

try:
    import spotipy
    from spotipy.oauth2 import SpotifyClientCredentials
except ImportError:
    spotipy = None
    SpotifyClientCredentials = None


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
MODEL_FILES = {
    "global": "music_recommender.pkl",
    "bollywood": "bollywood_recommender.pkl",
}
DEFAULT_RECOMMENDATIONS = 5
ARTWORK_CACHE_SECONDS = 60 * 60 * 12
ITUNES_SEARCH_URL = "https://itunes.apple.com/search"
ITUNES_USER_AGENT = "NebulaMusicRecommender/1.0"

st.set_page_config(
    page_title="NEBULA · Music Intelligence",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------------------------------- #
# Sci-fi interface
# --------------------------------------------------------------------------- #
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Space+Grotesk:wght@400;500;600;700&display=swap');

      :root {
        --nebula-bg: #070a12;
        --nebula-panel: #0d1220;
        --nebula-line: rgba(119, 151, 198, .18);
        --nebula-cyan: #66f0e2;
        --nebula-violet: #a78bfa;
        --nebula-muted: #92a0b8;
      }
      html, body, [class*="css"] {
        font-family: 'Space Grotesk', sans-serif;
      }
      .stApp {
        background:
          radial-gradient(ellipse at 82% 3%, rgba(67, 48, 135, .22), transparent 34%),
          radial-gradient(ellipse at 5% 28%, rgba(30, 133, 146, .12), transparent 32%),
          var(--nebula-bg);
        color: #eef3ff;
      }
      [data-testid="stHeader"] {
        background: rgba(7, 10, 18, .68);
      }
      [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b1020 0%, #090d18 100%);
        border-right: 1px solid var(--nebula-line);
      }
      [data-testid="stSidebar"] > div:first-child { padding-top: 1.2rem; }
      .block-container {
        max-width: 1440px;
        padding-top: 2.3rem;
        padding-bottom: 4rem;
      }
      .hero {
        position: relative;
        overflow: hidden;
        padding: 2rem 2.2rem 1.9rem;
        margin-bottom: 1.65rem;
        border: 1px solid rgba(102, 240, 226, .22);
        border-radius: 22px;
        background:
          linear-gradient(115deg, rgba(15, 28, 43, .94), rgba(19, 17, 40, .82)),
          #101626;
        box-shadow: 0 18px 70px rgba(0,0,0,.22), inset 0 1px rgba(255,255,255,.04);
      }
      .hero:after {
        content: "";
        position: absolute;
        width: 270px; height: 270px; right: 6%; top: -165px;
        border: 1px solid rgba(102, 240, 226, .2);
        border-radius: 50%;
        box-shadow: 0 0 0 28px rgba(102,240,226,.025), 0 0 0 58px rgba(167,139,250,.025);
      }
      .hero-kicker, .eyebrow, .source-label {
        font-family: 'DM Mono', monospace;
        letter-spacing: .16em;
        text-transform: uppercase;
      }
      .hero-kicker {
        color: var(--nebula-cyan);
        font-size: .72rem;
        margin-bottom: .65rem;
      }
      .hero-title {
        font-size: clamp(2.35rem, 5vw, 4.2rem);
        font-weight: 700;
        line-height: .98;
        letter-spacing: -.055em;
        margin: 0;
        color: #f3f6ff;
      }
      .hero-title span {
        color: var(--nebula-cyan);
        text-shadow: 0 0 25px rgba(102,240,226,.25);
      }
      .hero-copy {
        max-width: 650px;
        color: #aebbd1;
        font-size: 1rem;
        margin: 1rem 0 0;
      }
      .section-heading {
        font-size: 1.28rem;
        font-weight: 600;
        letter-spacing: -.025em;
        margin: .3rem 0 .2rem;
      }
      .section-subtitle {
        color: var(--nebula-muted);
        font-size: .9rem;
        margin-bottom: 1rem;
      }
      .eyebrow { font-size: .65rem; color: var(--nebula-cyan); }
      .source-label { font-size: .57rem; color: #9ba8c0; }
      .stButton > button {
        min-height: 3rem;
        border: 1px solid rgba(102,240,226,.55);
        border-radius: 11px;
        background: linear-gradient(100deg, #66f0e2, #9af6e5);
        color: #071116;
        font-weight: 700;
        letter-spacing: .03em;
        box-shadow: 0 0 25px rgba(102,240,226,.13);
        transition: transform .16s ease, box-shadow .16s ease;
      }
      .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 0 32px rgba(102,240,226,.25);
        border-color: #b0fff4;
        color: #071116;
      }
      [data-baseweb="select"] > div, [data-baseweb="input"] > div {
        background: #0c1220;
        border-color: rgba(141,162,198,.24);
        border-radius: 10px;
      }
      div[data-testid="stVerticalBlockBorderWrapper"] {
        background: linear-gradient(155deg, rgba(18,25,41,.92), rgba(11,15,27,.92));
        border-color: var(--nebula-line);
        border-radius: 16px;
      }
      div[data-testid="stImage"] img {
        border-radius: 11px;
        border: 1px solid rgba(255,255,255,.08);
      }
      .cover-placeholder {
        display: flex; align-items: center; justify-content: center;
        aspect-ratio: 1 / 1; width: 100%; border-radius: 11px;
        color: var(--nebula-cyan); font-size: 3.7rem;
        background:
          radial-gradient(circle at 50% 45%, rgba(102,240,226,.17), transparent 43%),
          linear-gradient(145deg, #141d31, #111020);
        border: 1px solid rgba(102,240,226,.17);
      }
      .track-title {
        min-height: 2.6rem; margin: .85rem 0 .2rem;
        color: #f0f3fc; font-size: .98rem; font-weight: 600;
        line-height: 1.3; overflow-wrap: anywhere;
      }
      .track-artist {
        min-height: 1.3rem; margin: 0 0 .8rem;
        color: #9aa9c0; font-size: .82rem; overflow-wrap: anywhere;
      }
      .match-pill {
        display: inline-block; padding: .22rem .5rem;
        border: 1px solid rgba(102,240,226,.22);
        border-radius: 999px; color: var(--nebula-cyan);
        background: rgba(102,240,226,.06);
        font: .63rem 'DM Mono', monospace; letter-spacing: .06em;
      }
      .sidebar-brand {
        color: #eff4ff; font-size: 1.05rem; font-weight: 700;
        letter-spacing: .05em; margin-bottom: .2rem;
      }
      .sidebar-note {
        color: #8d9bb3; font-size: .8rem; line-height: 1.5;
      }
      .status-line {
        padding: .75rem .9rem; margin: .35rem 0 1rem;
        border: 1px solid rgba(102,240,226,.17); border-radius: 10px;
        background: rgba(102,240,226,.045);
        color: #aebbd1; font: .68rem 'DM Mono', monospace;
      }
      .status-dot {
        display: inline-block; width: 7px; height: 7px; margin-right: .5rem;
        border-radius: 50%; background: var(--nebula-cyan);
        box-shadow: 0 0 10px rgba(102,240,226,.85);
      }
      hr { border-color: var(--nebula-line) !important; }
      @media (max-width: 700px) {
        .hero { padding: 1.5rem; }
        .block-container { padding-top: 1rem; }
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <section class="hero">
      <div class="hero-kicker">✦ NEBULA / PERSONAL MUSIC INTELLIGENCE</div>
      <h1 class="hero-title">Find your next<br><span>frequency.</span></h1>
      <p class="hero-copy">
        Pick a track from either the global or Bollywood collection. Each lyric
        model finds close matches; album artwork loads automatically when available.
      </p>
    </section>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Model loading and recommendations
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading your song universe…")
def load_model(model_filename: str):
    """Load the user's trusted model once per Streamlit process."""
    model_path = Path(__file__).resolve().parent / model_filename
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_filename}. Place it in the same folder "
            "as this Python file."
        )

    with model_path.open("rb") as model_file:
        data = pickle.load(model_file)

    if not isinstance(data, dict) or "songs" not in data or "tfidf_matrix" not in data:
        raise ValueError(
            "The pickle must be a dictionary containing 'songs' and 'tfidf_matrix'."
        )

    songs = data["songs"].reset_index(drop=True).copy()
    column_names = {str(column).strip().casefold(): column for column in songs.columns}

    song_column = next(
        (column_names[name] for name in ("song", "title", "track", "track_name", "song_name")
         if name in column_names),
        None,
    )
    if song_column is None:
        raise ValueError(
            "The songs table needs a song-title column such as 'song' or 'title'."
        )
    if song_column != "song":
        songs["song"] = songs[song_column]

    artist_column = next(
        (
            column_names[name]
            for name in (
                "artist",
                "artist_name",
                "artists",
                "singer",
                "singer_name",
                "subtitle",
                "performer",
            )
            if name in column_names
        ),
        None,
    )
    if artist_column is not None and artist_column != "artist":
        songs["artist"] = songs[artist_column]
    elif artist_column is None:
        songs["artist"] = "Unknown artist"

    matrix = data["tfidf_matrix"].tocsr()
    if matrix.shape[0] != len(songs):
        raise ValueError(
            f"Model mismatch: {len(songs)} songs but {matrix.shape[0]} TF-IDF rows."
        )

    songs = songs.copy()
    songs["song"] = songs["song"].fillna("Unknown track").astype(str)
    songs["artist"] = songs["artist"].fillna("Unknown artist").astype(str)
    labels = (songs["song"] + "  —  " + songs["artist"]).tolist()
    normalized_titles = songs["song"].str.strip().str.casefold().tolist()
    return songs, matrix, labels, normalized_titles


def recommend(
    song_index: int,
    count: int,
    skip_same_title: bool,
    matrix,
    normalized_titles,
):
    """Return (row index, cosine similarity) pairs, in descending order."""
    similarities = (matrix[song_index] @ matrix.T).toarray().ravel()
    order = np.argsort(-similarities)

    seen = {normalized_titles[song_index]} if skip_same_title else set()
    picks = []
    for candidate in order:
        candidate = int(candidate)
        if candidate == song_index:
            continue
        if skip_same_title:
            title = normalized_titles[candidate]
            if title in seen:
                continue
            seen.add(title)
        picks.append((candidate, float(similarities[candidate])))
        if len(picks) >= count:
            break
    return picks


# --------------------------------------------------------------------------- #
# Album artwork: free iTunes lookup, optional Spotify fallback
# --------------------------------------------------------------------------- #
def _get_credential(name: str) -> Optional[str]:
    """Read a credential from Streamlit secrets, then the local environment."""
    try:
        value = st.secrets.get(name)
    except Exception:
        value = None
    return value or os.getenv(name)


@st.cache_resource
def get_spotify_client():
    client_id = _get_credential("SPOTIFY_CLIENT_ID")
    client_secret = _get_credential("SPOTIFY_CLIENT_SECRET")
    if not (client_id and client_secret and spotipy and SpotifyClientCredentials):
        return None
    try:
        auth = SpotifyClientCredentials(
            client_id=client_id,
            client_secret=client_secret,
        )
        return spotipy.Spotify(auth_manager=auth, requests_timeout=6, retries=1)
    except Exception:
        return None


def _normalize(value: str) -> str:
    value = unicodedata.normalize("NFKD", str(value))
    value = value.encode("ascii", "ignore").decode("ascii").casefold()
    return re.sub(r"[^a-z0-9]+", " ", value).strip()


def _similarity(left: str, right: str) -> float:
    left_norm, right_norm = _normalize(left), _normalize(right)
    if not left_norm or not right_norm:
        return 0.0
    if left_norm == right_norm:
        return 1.0
    if left_norm in right_norm or right_norm in left_norm:
        return 0.9
    sequence_score = difflib.SequenceMatcher(None, left_norm, right_norm).ratio()
    left_words, right_words = set(left_norm.split()), set(right_norm.split())
    token_score = len(left_words & right_words) / max(len(left_words), len(right_words), 1)
    return max(sequence_score, token_score)


def _search_itunes(song_name: str, artist_name: str) -> Optional[str]:
    query = f"{song_name} {artist_name}".strip()
    response = requests.get(
        ITUNES_SEARCH_URL,
        params={"term": query, "entity": "song", "limit": 12},
        headers={"User-Agent": ITUNES_USER_AGENT},
        timeout=6,
    )
    response.raise_for_status()
    results = response.json().get("results", [])

    best_url = None
    best_score = 0.0
    for item in results:
        title = item.get("trackName", "")
        artist = item.get("artistName", "")
        artwork = item.get("artworkUrl100")
        title_score = _similarity(song_name, title)
        artist_score = _similarity(artist_name, artist)

        # Keep the match conservative to avoid displaying another song's cover.
        if title_score < 0.48 or artist_score < 0.38 or not artwork:
            continue
        score = 0.66 * title_score + 0.34 * artist_score
        if score > best_score:
            best_score = score
            best_url = re.sub(r"\d+x\d+bb", "600x600bb", artwork, count=1)
    return best_url


def _artist_matches_spotify(track: dict, wanted_artist: str) -> bool:
    wanted = _normalize(wanted_artist)
    if not wanted:
        return False
    for artist in track.get("artists", []):
        found = _normalize(artist.get("name", ""))
        if found and (wanted in found or found in wanted):
            return True
    return False


def _search_spotify(song_name: str, artist_name: str) -> Optional[str]:
    client = get_spotify_client()
    if client is None:
        return None
    song = song_name.replace('"', "")
    artist = artist_name.replace('"', "")
    for query in (f'track:"{song}" artist:"{artist}"', f"{song} {artist}"):
        tracks = client.search(q=query, type="track", limit=5).get("tracks", {})
        for track in tracks.get("items", []):
            images = track.get("album", {}).get("images", [])
            if _artist_matches_spotify(track, artist) and images:
                return images[0].get("url")
    return None


@st.cache_data(show_spinner=False, ttl=ARTWORK_CACHE_SECONDS)
def get_song_artwork(song_name: str, artist_name: str):
    """Return (cover URL or None, source label, short lookup error or None)."""
    errors = []
    try:
        url = _search_itunes(song_name, artist_name)
        if url:
            return url, "APPLE MUSIC CATALOG", None
    except Exception as error:
        errors.append(f"Apple lookup: {type(error).__name__}")

    if get_spotify_client() is not None:
        try:
            url = _search_spotify(song_name, artist_name)
            if url:
                return url, "SPOTIFY", None
        except Exception as error:
            errors.append(f"Spotify lookup: {type(error).__name__}")

    return None, "ARTWORK UNAVAILABLE", " · ".join(errors) if errors else None


def _dataset_artwork(song_row):
    """Use artwork already bundled in a dataset when it has a remote URL."""
    for column in ("thumbnail", "artwork_url", "cover_url", "image_url", "image"):
        value = song_row.get(column)
        if isinstance(value, str):
            value = value.strip()
            if value.startswith(("https://", "http://")):
                return value
    return None


def _reset_recommender_state(selection_key, results_key, source_key):
    """Reset only the active tab's selection and previous recommendations."""
    for key in (selection_key, results_key, source_key):
        st.session_state.pop(key, None)


def render_recommender_tab(
    tab_title: str,
    model_filename: str,
    key_prefix: str,
    recommendation_count: int,
    skip_same_title: bool,
):
    """Render one complete recommender using its own model and widget state."""
    try:
        songs, matrix, labels, normalized_titles = load_model(model_filename)
    except FileNotFoundError:
        st.error(
            f"Model file not found: {model_filename}. Add it beside "
            "music_recommender_sci_fi.py in your GitHub repository."
        )
        return
    except Exception as error:
        st.error(f"Could not load the {tab_title} model: {error}")
        return

    if len(songs) < 2:
        st.error(f"The {tab_title} model needs at least two songs.")
        return

    st.markdown(
        f'<div class="eyebrow">01 / {html.escape(tab_title.upper())} COLLECTION</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-heading">Choose your starting point</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="section-subtitle">Search {len(songs):,} tracks by song '
        "or artist; the same lyric-similarity model finds your matches.</div>",
        unsafe_allow_html=True,
    )

    search_key = f"{key_prefix}_song_artist_search"
    selection_key = f"{key_prefix}_selected_track"
    results_key = f"{key_prefix}_results"
    source_key = f"{key_prefix}_source_idx"

    search_query = st.text_input(
        "Search by song or artist",
        placeholder="Type a song title or artist…",
        key=search_key,
        on_change=_reset_recommender_state,
        args=(selection_key, results_key, source_key),
    )

    if search_query.strip():
        query = search_query.strip()
        song_matches = songs["song"].str.contains(
            query, case=False, na=False, regex=False
        )
        artist_matches = songs["artist"].str.contains(
            query, case=False, na=False, regex=False
        )
        matching_indices = np.flatnonzero(
            (song_matches | artist_matches).to_numpy()
        ).tolist()
    else:
        matching_indices = list(range(len(labels)))

    if not matching_indices:
        st.warning(f'No songs or artists found for “{html.escape(search_query.strip())}”.')
        st.session_state.pop(results_key, None)
        st.session_state.pop(source_key, None)
        return

    if search_query.strip():
        st.caption(f"{len(matching_indices):,} matching tracks")

    selected_idx = st.selectbox(
        "Matching tracks",
        options=matching_indices,
        format_func=labels.__getitem__,
        label_visibility="collapsed",
        key=selection_key,
    )

    if st.button(
        "✦  GENERATE MY MIX",
        use_container_width=True,
        key=f"{key_prefix}_generate_mix",
    ):
        st.session_state[results_key] = recommend(
            selected_idx,
            recommendation_count,
            skip_same_title,
            matrix,
            normalized_titles,
        )
        st.session_state[source_key] = selected_idx

    if results_key not in st.session_state:
        return

    if st.session_state.get(source_key) != selected_idx:
        st.caption("Your selection changed. Generate a new mix to refresh the results.")
        return

    picks = st.session_state[results_key]
    st.markdown("---")
    st.markdown(
        '<div class="eyebrow">02 / YOUR SIGNAL MATCHES</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-heading">Recommended for this track</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="section-subtitle">Starting from '
        f'<strong>{html.escape(songs.at[selected_idx, "song"])}</strong> · '
        f'{html.escape(songs.at[selected_idx, "artist"])}</div>',
        unsafe_allow_html=True,
    )

    if not picks:
        st.warning("No other tracks matched the current filters.")
        return

    artwork_errors = []
    columns = st.columns(min(4, len(picks)), gap="medium")
    for position, (song_idx, score) in enumerate(picks):
        song_name = songs.at[song_idx, "song"]
        artist_name = songs.at[song_idx, "artist"]
        cover_url = _dataset_artwork(songs.iloc[song_idx])
        if cover_url:
            source = "MODEL THUMBNAIL"
            lookup_error = None
        else:
            cover_url, source, lookup_error = get_song_artwork(
                song_name, artist_name
            )
        if lookup_error:
            artwork_errors.append(lookup_error)

        with columns[position % len(columns)]:
            with st.container(border=True):
                if cover_url:
                    st.image(cover_url, use_container_width=True)
                else:
                    st.markdown(
                        '<div class="cover-placeholder">♫</div>',
                        unsafe_allow_html=True,
                    )
                st.markdown(
                    f'<div class="track-title">{html.escape(song_name)}</div>'
                    f'<div class="track-artist">{html.escape(artist_name)}</div>'
                    f'<span class="match-pill">MATCH {score:.0%}</span>'
                    f'<p class="source-label" style="margin-top:.85rem">'
                    f'{html.escape(source)}</p>',
                    unsafe_allow_html=True,
                )

    if artwork_errors:
        st.caption(
            "Some artwork lookups could not reach a catalog. "
            "Recommendations still work; retry when your internet connection is available."
        )


# --------------------------------------------------------------------------- #
# App controls and display
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown('<div class="sidebar-brand">✦ NEBULA</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sidebar-note">TWO-COLLECTION MUSIC CONSOLE</div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown(
        '<div class="status-line"><span class="status-dot"></span>'
        "DUAL MODELS &nbsp;·&nbsp; TF-IDF / COSINE</div>",
        unsafe_allow_html=True,
    )
    recommendation_count = st.slider(
        "Tracks in your mix",
        min_value=3,
        max_value=8,
        value=DEFAULT_RECOMMENDATIONS,
        help="Choose how many similar songs to show.",
    )
    skip_same_title = st.checkbox(
        "Skip alternate versions",
        value=True,
        help="Avoid recommending another recording with the same song title.",
    )
    st.markdown("---")
    st.markdown('<div class="eyebrow">ARTWORK LOOKUP</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sidebar-note">Free Apple catalog search runs first. '
        "Spotify credentials are optional and only used as a fallback.</div>",
        unsafe_allow_html=True,
    )
    spotify_ready = get_spotify_client() is not None
    if spotify_ready:
        st.caption("Spotify fallback: connected")
    else:
        st.caption("Spotify fallback: not configured")
    st.markdown("---")
    st.markdown('<div class="eyebrow">COLLECTIONS</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sidebar-note">Global · music_recommender.pkl<br>'
        'Bollywood · bollywood_recommender.pkl</div>',
        unsafe_allow_html=True,
    )

global_tab, bollywood_tab = st.tabs(["✦ GLOBAL / MILLION SONG", "✦ BOLLYWOOD"])

with global_tab:
    render_recommender_tab(
        "Global",
        MODEL_FILES["global"],
        "global",
        recommendation_count,
        skip_same_title,
    )

with bollywood_tab:
    render_recommender_tab(
        "Bollywood",
        MODEL_FILES["bollywood"],
        "bollywood",
        recommendation_count,
        skip_same_title,
    )

st.markdown("---")
st.markdown(
    '<div class="sidebar-note" style="text-align:center">'
    "NEBULA · Two collections, one recommendation console."
    "</div>",
    unsafe_allow_html=True,
)
