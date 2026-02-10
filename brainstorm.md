# BRAINSTORM.md
Unified Streaming Catalog Intelligence Platform (Netflix + Max Merger Simulation)

This document is the full product plan for our Streamlit capstone app. It defines the final app structure, page-by-page scope, global components, performance approach, development timeline, and success criteria. It is written so our team can build iteratively and stay consistent across weeks.

---

## 1) Final App Structure (7 Pages)

Home.py
├── pages/1_Explore_Catalog.py
├── pages/2_Platform_Comparisons.py
├── pages/3_Platform_DNA.py
├── pages/4_Discovery_Engine.py
├── pages/5_Strategic_Insights.py
├── pages/6_Interactive_Lab.py
└── pages/7_Cast_Crew_Network.py

## 2) PAGE 0: HOME (Home.py) | Merger Dashboard

### Purpose
High-level overview of the Netflix + Max merger impact. Sets context, shows scale, and gives a clean summary. Deeper dives happen on later pages.

### Layout

#### Section 1: Hero Metrics (4 columns)
- Combined catalog size (titles count)
- Combined average IMDb score (and delta vs Netflix-only)
- Total cast and crew count (unique people in credits)
- Unique genre count

#### Section 2: Merger Impact Snapshot (3 columns)
**Column 1: Volume Boost**
- Bar chart: Netflix-only vs Max-only vs Combined
- Supporting label showing percentage increase vs Netflix-only

**Column 2: Quality Shift**
- Histogram of IMDb scores: Netflix-only vs Combined
- Highlight count of “high-rated titles” above a threshold (example: IMDb > 8.0)

**Column 3: Genre Expansion**
- Compare Netflix-only vs Combined for top genres by share
- Highlight biggest lifts in share or raw count for key genres

#### Section 3: Top Titles in Merged Catalog
- Two tabs:
  - By Rating: Top titles by IMDb score (with a minimum vote filter)
  - By Popularity: Top titles by TMDB popularity
- Grid layout cards (4 columns x 5 rows)
- Each card shows title, year, type, IMDb score + votes, and platform badge
- Clicking a card routes to Explore page with that title preloaded

#### Section 4: Content Timeline
- Stacked area chart: Netflix vs Max titles by decade
- Toggle: Movies vs Shows

#### Section 5: Geographic Footprint
- World map (choropleth): production countries by content volume
- Side bar chart: top 10 countries
- Simple insight line: “Content from X countries”

#### Section 6: Quick Navigation Cards
- “Explore Full Catalog” → Explore page
- “Compare With Competitors” → Platform Comparisons page
- “Get Recommendations” → Discovery Engine page

---

## 3) PAGE 1: Explore Catalog (pages/1_Explore_Catalog.py)

### Purpose
The core searchable database and the most “product-like” browsing experience. Users can search, filter, click a title, and get similar titles.

### Layout

#### Top: Unified Search and Filter Bar
- Search box with autocomplete
- Compact filters:
  - Type (All, Movies, Shows)
  - Year range slider
  - Minimum IMDb slider (or quick thresholds like 6+, 7+, 8+)
  - Genre multi-select
- Sort by dropdown (Relevance, IMDb, Year, Popularity, Title)
- Results count text (“Showing X titles”)

#### Main Area: Two-Panel Layout

**Left Panel (Results List)**
- Compact list items:
  - Title + year
  - Type icon
  - IMDb score + vote count (abbreviated)
  - First 1 to 2 genres
  - Platform badge
- Pagination (example: 50 per page)
- Click title loads the right panel

**Right Panel (Detail View)**
- Header: title + year + type
- One-line metrics row: IMDb score, votes, TMDB score, popularity
- Full description
- Metadata in a clean two-column grid:
  - Genres, runtime
  - Year, certification
  - Countries, platform
  - Seasons (if show)

**Cast and Crew Teaser (collapsed by default)**
- Expander showing top actors and directors
- Button to open Cast and Crew Network page with that person or title context

**Similar Titles (key feature)**
- Button: “Find Similar Titles”
- Similarity logic:
  - TF-IDF on description (primary signal)
  - Genre overlap (secondary signal)
- Toggle for scope:
  - Netflix + Max only
  - All platforms (if we include additional datasets)
- Output: 10 similar titles in a grid with similarity score and platform badge
- Clicking a recommendation loads it immediately in the detail view

---

## 4) PAGE 2: Platform Comparisons (pages/2_Platform_Comparisons.py)

### Purpose
Compare Netflix + Max (as the “merged entity”) against competitor platforms. Focus is competitive positioning and category-level differences.

### Layout

#### Platform Selector
- Netflix + Max is locked on as baseline
- User can choose up to 3 additional platforms (Disney+, Prime, Paramount+, Apple TV+)
- Toggle: Absolute vs Relative (normalized) mode
  - Absolute: raw counts
  - Relative: percentages or per-1000-title comparisons for fairness

#### Section 1: Volume Comparison
- Grouped bar chart: total titles by platform
- Stacked breakdown: movies vs shows
- Table with exact counts

#### Section 2: Quality Comparison
- Box plot: IMDb score distribution by platform
- Table:
  - Average IMDb
  - Percent of titles above thresholds (example: >7.0, >8.0)

#### Section 3: Unified Genre Analysis
- Heatmap:
  - Rows: top 15 genres
  - Columns: selected platforms
  - Cells: count or percent depending on view mode
- Drill-down on click:
  - Title count per platform (bar)
  - Avg IMDb per platform (bar)
  - Top 3 titles per platform in that genre (cards)
  - “Leader” badge for best average quality within that genre

#### Section 4: Streamlined Content Profile
- Age certification distribution (stacked bars)
- International share metric:
  - % of titles where production countries include non-US
- Era focus:
  - Median release year per platform
  - Optional year distribution visualization

#### Section 5: Market Positioning Matrix
- Scatter plot:
  - X: catalog size
  - Y: avg IMDb
  - Bubble: total TMDB popularity
- Quadrants labeled (example: “Large + High Quality”, “Small + High Quality”)

---

## 5) PAGE 3: Platform DNA (pages/3_Platform_DNA.py)

### Purpose
Explain platform identity in a more interpretive way. This answers “what makes each platform feel different” using patterns, clustering, and profile summaries.

### Layout

#### Section 1: Platform Personality Profile Cards
- Select a platform from dropdown
- DNA card includes:
  - Genre mix donut chart
  - Era focus by decade
  - Quality tier indicator (avg IMDb)
  - 3 to 5 defining traits generated from comparisons (simple and readable)

Optional:
- Side-by-side comparison of 2 platforms using the same DNA card format

#### Section 2: Content Landscape Visualization
- UMAP or t-SNE plot precomputed offline
- Each point is a title, colored by platform
- Clicking an area shows a small sample list of titles from that cluster
- Provide short interpretation text:
  - where Netflix and Max overlap
  - where they diverge (clusters dominated by one platform)

#### Section 3: “What Platform Are You?” Matcher
- User inputs:
  - top genres (multi-select)
  - preference sliders:
    - classics vs new releases
    - hidden gems vs blockbusters
    - shorter vs longer runtimes
    - family-friendly vs mature
- Output:
  - best match platform and match percent
  - bar chart match percent across all platforms
  - brief explanation tied directly to the input preferences

---

## 6) PAGE 4: Discovery Engine (pages/4_Discovery_Engine.py)

### Purpose
Full recommendation toolkit with three simple entry points. This is the “I want something to watch” experience.

### Layout

#### Tab 1: Similar to a Title
- Search a title (autocomplete)
- Choose scope:
  - Netflix + Max only
  - All platforms
  - specific platform
- Control number of results (5 to 20)
- Optional quality filter (minimum IMDb)
- Output:
  - recommendation cards with similarity score
  - expander “Why similar?” showing genre overlap and key description terms

#### Tab 2: Preference-Based Recommendations
- Inputs:
  - genres (1 to 5)
  - minimum IMDb
  - movies/shows/both
  - runtime preference
  - year range
  - slider: popular vs hidden gems (weights TMDB popularity)
  - platform scope
- Output:
  - ranked list with fit score
  - short “why it matches” text per item

#### Tab 3: Vibe Search (NLP)
- Text prompt input (what user is in the mood for)
- Optional filters: IMDb min, year range, platform scope
- Uses sentence embeddings over descriptions
- Output:
  - top matches with relevance score and short highlighted theme match

#### Session-Based Recommendation History
- Store last 10 recommendation sets in the current session
- Allow user marking:
  - Interested
  - Watched
  - Not interested

---

## 7) PAGE 5: Strategic Insights (pages/5_Strategic_Insights.py)

### Purpose
Executive-style analysis for the merged entity. Focus on merger impact, overlaps, gaps, and positioning.

### Layout

#### Section 1: Merger Value Dashboard (headline metrics)
- Genre gap coverage metric
- Overlap rate metric
- Quality lift metric
- Diversity metric (genre entropy or similar)

#### Section 2: Content Overlap Analysis
- Identify overlap between Netflix and Max using matching logic
- Show overlap rates by:
  - genre
  - type (movie/show)
  - certification
  - decade
- Provide an audit table for matched pairs with confidence scores

#### Section 3: Gap Analysis Tool
- Choose platform perspective (Netflix + Max or competitor)
- Define gaps using measurable rules:
  - low share genres
  - low quality genres (avg IMDb below threshold)
  - era gaps (low titles in certain decades)
  - geographic gaps (few titles from key regions)
- Output:
  - prioritized gap list with severity labels
  - clear description of “what kind of content is missing” in plain terms

#### Section 4: Competitive Positioning Summary
- Select one competitor platform
- Output:
  - where competitor leads
  - where merged entity leads
  - battleground genres (close competition)
  - white space areas (weak for both)
- Provide short strategic recommendation text based on findings

#### Section 5: Market Impact Simulation (optional but impressive)
- Market share charts:
  - size share
  - quality-weighted share
- HHI calculation for concentration
- Clearly label as a simulated market based on datasets

---

## 8) PAGE 6: Interactive Lab (pages/6_Interactive_Lab.py)

### Purpose
High-engagement features that show creativity and technical depth. This page is intentionally more playful but still data-driven.

### Feature 1: Build Your Streaming Service (Draft Game)
- User gets a budget and drafts titles from the dataset
- Title “value” function uses a mix of:
  - IMDb score
  - TMDB popularity
  - recency
- Live dashboard while drafting:
  - spend remaining
  - avg IMDb
  - genre distribution
  - diversity score
- Compare your drafted “service” vs Netflix + Max on key metrics

### Feature 2: Hypothetical Title Predictor
- User inputs:
  - description
  - type
  - genres
  - runtime
  - year (future range)
  - country
  - certification
  - budget tier
- Model predicts IMDb score range with confidence band
- Also recommends top directors and actors based on genre track record
- Outputs which platform the concept fits best based on platform strengths

### Feature 3: Insight Generator
- User selects scope (platform, genre, decade, all platforms)
- Generates 5 to 8 specific, data-backed insights
- Includes a “Surprise me” random insight button

---

## 9) PAGE 7: Cast and Crew Network (pages/7_Cast_Crew_Network.py)

### Purpose
Explore the people behind the content. Show connections, collaborator networks, and rankings.

### Layout

#### Section 1: Person Search and Profile
- Search by name with autocomplete
- Filters:
  - role (actor/director)
  - minimum title count
  - platform scope
- Profile includes:
  - key stats (avg IMDb, total votes, career span, top genre)
  - top collaborators list (click to jump)
  - filmography table with sorting and pagination
  - career trend line (IMDb over time)

#### Section 2: Rankings
- Tabs:
  - Directors
  - Actors
- Ranking toggle:
  - most titles
  - highest average IMDb
  - most popular (sum TMDB popularity)
- Compare 2 to 3 people side-by-side on key metrics

Optional:
- “Connection finder” mini-game (actor to actor path) if time allows

---

## 10) Global Components (Used Across All Pages)

### Navigation
- Top bar with app title, current page indicator, and simple settings icon
- Sidebar includes:
  - page navigation links
  - global filters that persist using Streamlit session state

### Global Filters (persisted)
- Platform view:
  - Netflix + Max combined
  - Netflix only
  - Max only
  - All platforms (enables competitor platform selection)
- Content type: Movies, Shows
- Year range
- Minimum IMDb
- Genre multi-select
- Reset filters button

### Quick Stats Panel
- Titles matching current filter state
- Average IMDb for current filter state
- Updates only when filters apply, to reduce performance load

---

## 11) Performance and Caching Strategy

### Data loading
- Cleaned data stored as parquet files
- Cached at app startup using Streamlit caching

### Model loading
- Cache ML models as resources
- Keep vectorizers and embeddings precomputed when possible

### Precomputed artifacts
- Platform-level stats and distributions saved as parquet
- UMAP or t-SNE coordinates precomputed offline
- Network graphs may be precomputed as node and edge tables, then filtered on demand

### Lazy loading rules
- Credits details load when a title or person is selected
- Heavy plots only render for filtered subsets
- Paginate lists and tables to avoid rendering large objects

---

## 12) Visual Design System (High-level)

- Consistent metric cards, charts, and table styling
- Platform badges and combined entity badge
- Dark mode friendly color scheme and readable typography
- Consistent chart labeling and tooltips for clarity

---

## 13) Transparency and Documentation

### Footer (every page)
- Data disclosure: mid-2023 snapshot
- Hypothetical merger disclaimer
- Link to methodology details
- Link to GitHub repo

### Methodology disclosure patterns
- When showing predictions or similarity, label output as estimated
- Provide short, plain-English “how it works” expanders

---

## 14) Revised Development Timeline (12 weeks)

### Week 1: Foundation
- Merge all platform datasets
- Clean and validate fields
- Export parquet files
- Build reusable loading functions with caching

### Week 2: Home + Infrastructure
- Home dashboard
- Multi-page structure
- Global filters with session state
- Helper utilities for consistent plots

### Week 3: Explore Catalog
- Search, filters, two-panel UI
- Title detail view
- Similar titles recommender

### Week 4: Platform Comparisons
- Volume, quality, genre heatmap + drill-down
- Market positioning matrix
- Profile metrics

### Week 5: Platform DNA
- DNA cards
- UMAP or t-SNE visualization
- Platform matcher tool

### Week 6: Discovery Engine
- Title-based similarity tool
- Preference-based recommendations
- Recommendation history

### Week 7: Discovery Engine + Strategic Insights start
- Vibe search embeddings
- Strategic Insights: merger dashboard + gap analysis base

### Week 8: Strategic Insights
- Overlap analysis + audit table
- Competitive positioning summaries
- Market impact simulation (optional)

### Week 9: Interactive Lab
- Build Your Service draft game
- Hypothetical title predictor + cast and crew suggestions
- Insight generator

### Week 10: Cast and Crew Network
- Person profiles
- Collaborator analysis
- Rankings and comparisons
- Optional connection finder

### Week 11: Integration and Optimization
- Make global filters consistent across pages
- Caching and performance tuning
- Bug fixes and user testing
- Final UI polish pass

### Week 12: Final Polish and Deployment
- Documentation and methodology write-up
- Final visuals and styling consistency
- Deploy to Streamlit Cloud
- Presentation rehearsals and demo path

---

## 15) Feature Priority Matrix

### Must-have
- Data pipeline (clean merged dataset + parquet)
- Home dashboard
- Explore Catalog page + detail cards
- Platform Comparisons page
- Basic recommender (title similarity)
- Strategic overlap + gap analysis core
- Cast and crew profiles (basic)

### Should-have (core differentiation)
- Platform DNA page with UMAP/t-SNE
- Vibe search embeddings
- Strategic positioning summaries vs competitors
- Interactive Lab at least one feature (draft game or title predictor)

### Nice-to-have
- Market impact HHI simulation
- Connection finder game
- Extra insight generation depth

---

## 16) Success Criteria

### Technical
- Pages load reliably and quickly
- Global filters apply consistently
- Similarity and recommendation tools return sensible results
- Caching prevents long delays
- Deployed successfully on Streamlit Cloud

### Product and UX
- Easy to navigate
- Clear separation of purposes per page
- Outputs are explainable with short tooltips and expanders

### Academic
- Demonstrates real data engineering work (merge, clean, validate)
- Demonstrates ML and NLP thoughtfully (similarity, embeddings, clustering)
- Demonstrates decision-support thinking (overlap, gap analysis, strategy module)
- Transparent about limitations and snapshot timing

---