# Project Specification: Netflix + Max Merger Streamlit App

## Overview
7-page Streamlit application analyzing a hypothetical Netflix + Warner Bros (Max) merger. Each page serves a specific analytical purpose.

---

## Page 0: Home (Home.py)

**Purpose:** High-level merger dashboard

**Requirements:**
- Hero metrics (4 columns): Combined catalog size, avg IMDb, cast/crew count, genre count
- Merger impact (3 columns):
  - Volume boost: Bar chart Netflix vs Max vs Combined
  - Quality shift: IMDb histogram (Netflix-only vs Combined)
  - Genre expansion: Top genres comparison
- Top titles: 2 tabs (By Rating, By Popularity), grid layout, clickable cards
- Content timeline: Stacked area chart by decade
- Geographic footprint: World map + top 10 countries bar chart
- Quick navigation cards to other pages

---

## Page 1: Explore Catalog (pages/01_Explore_Catalog.py)

**Purpose:** Searchable database with similarity recommendations

**Requirements:**
- Search & filter bar:
  - Autocomplete search
  - Type (Movies/Shows), year range, min IMDb, genre multi-select
  - Sort by dropdown, results count
- Two-panel layout:
  - Left: Results list (paginated, 50 per page)
  - Right: Detail view with full metadata
- Cast & crew teaser (collapsed expander)
- **Similar Titles (KEY FEATURE):**
  - TF-IDF + genre overlap similarity
  - Toggle scope: Netflix+Max only vs All platforms
  - Top 10 recommendations with similarity scores
  - Clicking loads new title in detail view

---

## Page 2: Platform Comparisons (pages/02_Platform_Comparisons.py)

**Purpose:** Competitive analysis vs other platforms

**Requirements:**
- Platform selector: Netflix+Max locked, choose up to 3 competitors
- Toggle: Absolute vs Relative (normalized) mode
- Volume comparison: Grouped bar chart + table
- Quality comparison: Box plot + threshold table
- **Unified Genre Analysis (KEY FEATURE):**
  - Heatmap: Top 15 genres × selected platforms
  - Drill-down on click: Count, avg IMDb, top 3 titles, "Leader" badge
- Streamlined content profile: Age certification, international share, era focus
- Market positioning matrix: Scatter plot (size vs quality)

---

## Page 3: Platform DNA (pages/03_Platform_DNA.py)

**Purpose:** Explain platform identity through patterns

**Requirements:**
- Platform personality profile cards:
  - Genre mix donut chart
  - Era focus by decade
  - Quality tier indicator
  - 3-5 defining traits
  - Side-by-side comparison option
- **Content Landscape Visualization (KEY FEATURE):**
  - UMAP 2D plot (each point = title, colored by platform)
  - Clickable areas show sample titles
  - Interpretation text for overlaps/divergences
- "What Platform Are You?" matcher:
  - User inputs: Top genres, preference sliders
  - Output: Best match platform, bar chart, explanation

---

## Page 4: Discovery Engine (pages/04_Discovery_Engine.py)

**Purpose:** Full recommendation toolkit (3 entry points)

**Requirements:**
- **Tab 1: Similar to a Title**
  - Autocomplete search
  - Scope selector, result count control, quality filter
  - "Why similar?" expanders
- **Tab 2: Preference-Based**
  - Inputs: Genres, min IMDb, type, runtime, year range
  - Popular vs hidden gems slider
  - Platform scope
  - Ranked list with fit scores
- **Tab 3: Vibe Search (NLP)**
  - Text prompt input
  - Sentence embeddings over descriptions
  - Top matches with relevance scores
- Session-based recommendation history (last 10 sets)

---

## Page 5: Strategic Insights (pages/05_Strategic_Insights.py)

**Purpose:** Executive-style merger analysis

**Requirements:**
- Merger value dashboard: Genre gap coverage, overlap rate, quality lift, diversity metric
- **Content Overlap Analysis (KEY FEATURE):**
  - Identify Netflix-Max overlaps
  - Overlap rates by genre, type, certification, decade
  - Audit table with confidence scores
- **Gap Analysis Tool:**
  - Choose perspective (merged entity or competitor)
  - Identify gaps: Low share genres, low quality areas, era gaps, geographic gaps
  - Prioritized gap list with severity labels
- Competitive positioning summary:
  - Where competitor leads, where merged entity leads
  - Battleground genres, white space areas
  - Strategic recommendation text
- Market impact simulation (optional): Market share charts, HHI calculation

---

## Page 6: Interactive Lab (pages/06_Interactive_Lab.py)

**Purpose:** High-engagement creative features

**Requirements:**
- **Feature 1: Build Your Streaming Service (Draft Game)**
  - Budget + draft titles from dataset
  - Title value function (IMDb + popularity + recency)
  - Live dashboard: Spend remaining, avg IMDb, genre distribution, diversity score
  - Compare your service vs Netflix+Max
- **Feature 2: Hypothetical Title Predictor**
  - User inputs: Description, type, genres, runtime, year, country, certification, budget tier
  - Predict IMDb score range with confidence band
  - Recommend directors/actors based on genre track record
  - Suggest best-fit platform
- **Feature 3: Insight Generator**
  - Select scope (platform, genre, decade, all)
  - Generate 5-8 data-backed insights
  - "Surprise me" random insight button

---

## Page 7: Cast & Crew Network (pages/07_Cast_Crew_Network.py)

**Purpose:** Explore people behind content

**Requirements:**
- Person search & profile:
  - Autocomplete search by name
  - Filters: Role (actor/director), min title count, platform scope
  - Profile: Key stats (avg IMDb, votes, career span, top genre)
  - Top collaborators list (clickable)
  - Filmography table (sortable, paginated)
  - Career trend line (IMDb over time)
- Rankings tabs (Directors vs Actors):
  - Most titles
  - Highest average IMDb
  - Most popular (sum TMDB popularity)
  - Side-by-side comparison (2-3 people)
- Optional: Connection finder mini-game (actor-to-actor path)

---

## Global Components (All Pages)

### Navigation
- Top bar: App title, current page indicator, settings icon
- Sidebar: Page links, global filters

### Global Filters (Persist via session state)
- Platform view: Netflix+Max combined, Netflix only, Max only, All platforms
- Content type: Movies, Shows
- Year range slider
- Minimum IMDb slider
- Genre multi-select
- Reset filters button

### Quick Stats Panel
- Titles matching current filter state
- Average IMDb for current filter state
- Updates when filters change

---

## Technical Requirements

### Data
- Cleaned data stored as parquet files
- Cached at app startup using `@st.cache_data`

### Models
- TF-IDF vectorizers cached with `@st.cache_resource`
- Sentence embeddings precomputed offline
- UMAP/t-SNE coordinates precomputed offline

### Performance
- Precomputed artifacts for expensive operations
- Lazy loading for details (credits load on title selection)
- Paginated lists and tables
- Only render filtered subsets

### Transparency
- Footer on every page: Data disclosure (mid-2023 snapshot), hypothetical merger disclaimer
- "How it works" expanders for predictions/similarity
- Link to methodology and GitHub repo

---

## Success Criteria

### Technical
- Pages load reliably and quickly (<2s)
- Global filters apply consistently
- Similarity and recommendations return sensible results
- Successfully deployed on Streamlit Cloud

### Product & UX
- Easy to navigate
- Clear separation of purposes per page
- Outputs are explainable with tooltips

### Academic
- Demonstrates real data engineering (merge, clean, validate)
- Demonstrates ML/NLP thoughtfully (similarity, embeddings, clustering)
- Demonstrates decision-support thinking (overlap, gap analysis, strategy)
- Transparent about limitations and snapshot timing
