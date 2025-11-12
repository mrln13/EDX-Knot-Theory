# EDX-Knot-Theory

This repository accompanies the paper **"Convergence-Guided EDS Mapping for Bulk Rock Composition Using Knot Theory"**.

## Overview

We present an adaptive SEM–EDS mapping strategy that combines a deterministic
billiard-knot tiling trajectory with a data-driven convergence criterion. The
goal is to obtain representative bulk rock compositions from large-area EDS
maps more efficiently, with fewer tiles and reduced beam time.

### Abstract (from the manuscript)

Bulk rock composition (BRC) controls phase equilibria, mineral stability and
reconstructed pressure–temperature paths, but is traditionally obtained from
destructive whole-rock analyses that lack spatial context. SEM–EDS mapping
offers a route to BRC that preserves textural information, yet full thin-section
maps are time-consuming and dose-intensive, and conventional raster scans can
introduce grid-alignment bias.

We present an adaptive SEM–EDS mapping strategy that combines a deterministic
billiard-knot tiling trajectory with a data-driven convergence criterion. The
stage follows a periodic billiard path that visits tile centres in a
non-redundant, non-grid-aligned order, promoting uniform coverage and reducing
sensitivity to sample fabric. We track the cumulative bulk composition in
centred log-ratio (clr) space using an exponentially weighted moving average and
monitor the clr-distance between successive states, terminating acquisition once
this metric stabilizes.

We test this approach on two contrasting rocks, a strongly foliated pelitic
schist and a more equigranular diorite. Across both datasets, the billiard-knot
trajectory reaches convergence in fewer tiles than raster and spiral patterns,
reducing tiles-to-convergence by roughly 10–20%. Stable BRC estimates are
obtained after only ~7.7% and ~5.5% of the available tiles in the schist and
diorite mosaics, respectively, with tighter clustering of convergence steps
across starting positions. These gains translate directly into reduced beam time
for large-area EDS mapping. The combination of structured non-raster scan paths
with a compositional convergence metric is broadly applicable to other spectral
imaging and streaming compositional data.

## Repository contents

This repository consists of two main components:

1. **Trajectory generation**
   - Code for constructing billiard-knot trajectories on a rectangular canvas.
   - Routines to search over integer pairs (p, q) and tile sizes (h, w) under
     user-defined constraints to ensure that the trajectory passes through all
     tile centres exactly once.
   - Utilities to export the resulting tile centre coordinates for use in
     SEM/EDS acquisition software.

2. **Retrospective convergence analysis**
   - Scripts to reorder existing tile data to emulate different scan patterns
     (billiard-knot, raster with flyback, inward spiral).
   - Functions to compute cumulative compositions over successive tiles for
     each pattern.
   - Implementation of convergence monitoring in clr space using an
     exponentially weighted moving average (EWMA) and the clr-distance between
     successive states.
   - Code to compute summary statistics (e.g. tiles to convergence, mean,
     standard deviation) and to generate the figures used in the manuscript.

Further details on file structure and usage (e.g. required dependencies and
example commands) are provided in the subdirectory-specific README files.
