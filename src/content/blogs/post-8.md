---
title: "Toko Lens: Multimodal E-Commerce Visual Search Engine"
description: "End-to-end multimodal search system built on 3.4M+ scraped Tokopedia products, combining ViT-based image embeddings and BGE-M3 text embeddings inside FAISS indices (HNSW + IVF-PQ), served through a FastAPI backend and a plain HTML/JS frontend deployed on Hugging Face Spaces."
author: "Fati Buulolo"
image:
  url: "../../assets/images-015/display.png"
  alt: "Toko Lens: Multimodal E-Commerce Visual Search Engine"
pubDate: 2026-05-01
tags:
  ["Python", "FastAPI", "FAISS", "ViT", "BGE-M3", "Multimodal", "Web Scraping", "Hugging Face", "HTML", "E-Commerce", "Computer Vision", "NLP"]
---


## Overview

Finding a product on an e-commerce platform when you only have a photo and no idea what it is called is a problem that keyword search cannot solve. Toko Lens is a multimodal visual search engine built specifically for Tokopedia, the largest Indonesian e-commerce marketplace. Given a product image (and an optional text query and price range), the system retrieves the most visually and semantically similar products from a catalogue of over 3.4 million scraped listings in real time.

The system combines **Vision Transformer (ViT)** image embeddings, **BGE-M3** multilingual text embeddings, and two purpose-built **FAISS** indices served through a **FastAPI** backend. The frontend is a plain **HTML/CSS/JavaScript** single-page application, and the entire stack is deployed on **Hugging Face Spaces**.

---

## Problem Statement

A shopper sees a product in a photo (a sneaker style, a furniture piece, a fashion item) and wants to find it or something close to it on Tokopedia. Typing a product name requires prior knowledge of the category and vocabulary. Visual search removes that dependency: the image itself becomes the query, and an optional text refinement narrows the results further.

This project operationalises that idea at scale:

1. A **data pipeline** to scrape and store a representative catalogue of Tokopedia products with images.
2. A **multimodal embedding pipeline** to encode both images and product titles into dense vector spaces.
3. A **retrieval system** combining approximate nearest-neighbour search over both modalities.
4. A **web application** that accepts image uploads, optional crop selection, text queries, and price filters, and returns ranked results with live product images.

---

## Data Pipeline

### Phase 1: Shop Scraping

The pipeline begins by collecting all shops labelled **Mall** on Tokopedia. Mall-labelled stores represent verified, high-quality merchants and provide a consistent product quality baseline. The scraper collected **13,165 unique shop names**.

For each shop name, a second scraping pass retrieved the corresponding **Shop ID (SID)**. The SID is the internal Tokopedia identifier required to query product listings from the platform's GraphQL API.

### Phase 2: Product Scraping (7 Phases)

Using the collected SIDs, the pipeline queried the product catalogue of every shop, extracting four fields per product: `Product_Name`, `Price`, `Rating`, and `URL_Product`. Because Colab free-tier runtime sessions have hard time limits and occasionally crash mid-run, the scraping was divided into **7 sequential phases**, each resuming from the last completed shop. The combined result was a raw product table of **3,614,752 listings**.

The full product dataset is publicly available on Kaggle:
**[tokopedia-products-with-images-dataset-v1](https://www.kaggle.com/datasets/fati22/tokopedia-products-with-images-dataset-v1)**

### Phase 3: Image Scraping (15 Phases)

Each product URL was then visited to retrieve its primary product image. This stage was the most time-intensive: image downloads are bandwidth-bound, and each Colab session could only process a fraction of the full catalogue before timing out. The scraping was therefore split into **15 phases**, each picking up from where the previous left off.

A subset of products could not be scraped for images. These were mostly prescription-grade medical products whose listing pages display a doctor-prescription notice instead of a product image, making automated image retrieval impossible.

Final image count after all 15 phases: **3,475,088 product images**, matching a cleaned product table of the same size (down from 3,614,752 after removing image-less entries).

The image dataset is also publicly available on Kaggle:
**[tokopedia-images-product](https://www.kaggle.com/datasets/fati22/tokopedia-images-product)**

---

## Embedding Pipeline

### Text Embedding (BGE-M3)

Product titles were encoded using **[BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3)**, a state-of-the-art multilingual dense retrieval model that handles Indonesian, English, and code-mixed text natively. Each title was encoded into a **1024-dimensional** float32 vector.

### Image Embedding (ViT)

Product images were encoded using **[google/vit-base-patch16-224](https://huggingface.co/google/vit-base-patch16-224)**, the Vision Transformer base variant pretrained on ImageNet-21k. Each image was resized to 224 x 224, processed through the standard ViT patch tokenizer, and the `[CLS]` token from the final hidden state was taken as the image representation, yielding a **768-dimensional** float32 vector.

### Phased Execution (8 Parquet Files)

The embedding of 3.4M products is computationally heavy. Both the title and image embedding runs were divided into **4 phases each**, producing 4 output Parquet files per modality (8 Parquet files total). Each Parquet file stores a chunk of product IDs alongside their corresponding embedding vectors.

---

## FAISS Index Construction

Two separate FAISS indices were built from the Parquet files, one for image embeddings and one for title embeddings.

### Image Index: IndexHNSWFlat

```python
M_hnsw = 32
ef     = 200

core_index = faiss.IndexHNSWFlat(dim, M_hnsw)
core_index.hnsw.efConstruction = ef
final_index = faiss.IndexIDMap2(core_index)
```

HNSW (Hierarchical Navigable Small World) was chosen for the image index because it delivers very low query latency with high recall and does not require a training phase, which is an important property when indexing 3.4M vectors in a single sequential pass across 7 Parquet files.

### Text Index: IndexIVFPQ

```python
nlist = 1000
m     = 64
nbits = 8

quantizer  = faiss.IndexFlatL2(dim)
core_index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, nbits)
final_index = faiss.IndexIDMap2(core_index)
```

IVF-PQ (Inverted File with Product Quantization) was chosen for the text index because product quantization compresses the 1024-dimensional float32 vectors significantly, reducing RAM footprint at serving time while still supporting fast approximate search with a configurable `nprobe` parameter.

Both indices use `IndexIDMap2` as a wrapper so that product IDs (`ID_Product`) are stored alongside the vectors and returned directly in search results, eliminating the need for a separate ID lookup table.

After construction, both indices were saved as `.faiss` files and pushed to a Hugging Face dataset repository for use at serving time:

```
Viewww/tokopedia-search-indices
  tokopedia_img_hnsw.faiss
  tokopedia_txt_ivfpq.faiss
  metadata/metadata_tokopedia.db
```

---

## System Architecture

### Backend: FastAPI

The backend is a **Python FastAPI** application deployed as a Hugging Face Space. On startup it downloads the FAISS indices and SQLite metadata database from the Hugging Face Hub using `hf_hub_download`, loads the ViT and BGE-M3 models into memory, and exposes two endpoints.

**`POST /search`** accepts a product image, an optional search text, a minimum price, and a maximum price. The search pipeline proceeds as follows:

```
1. Decode uploaded image with Pillow
2. Encode image with ViT → 768-d float32 vector
3. Search image FAISS index (HNSW) → top-2000 candidate IDs
4. If search text provided:
     Encode text with BGE-M3 → 1024-d float32 vector
     Search text FAISS index (IVF-PQ, nprobe=64) → top-5000 IDs
     Intersect with image candidates
5. Query SQLite for metadata of surviving candidate IDs
6. Filter by price range
7. Re-rank by original FAISS image similarity order
8. Return top-100 results as JSON
```

**`GET /get-image`** fetches the live primary image URL for a given product from Tokopedia's GraphQL API in real time, since image URLs are not stored statically in the metadata database.

### Frontend: HTML / CSS / JavaScript

The frontend is a single-page application built with plain **HTML, CSS, and JavaScript** (no frontend framework required), also deployed as a Hugging Face Space. It provides the following interaction flow:

1. The user uploads a product image from their device.
2. An in-browser crop tool lets the user isolate the region of interest before searching.
3. An optional text field accepts a product name or keyword to refine the search beyond pure visual similarity.
4. A price range slider narrows results to a specified budget.
5. On submission, the frontend sends a `multipart/form-data` request to the FastAPI backend and renders the returned product cards with names, prices, ratings, and live images fetched from `GET /get-image`.

---

## How the Search Works

```
User uploads image
       │
       ▼
  [Optional crop]
       │
       ▼
  [Optional text query + price range]
       │
       ▼
  FastAPI backend
  ├─ ViT encodes image   ─────────────► HNSW index search → top-2000 visual matches
  └─ BGE-M3 encodes text ─────────────► IVF-PQ index search → top-5000 text matches
                                                    │
                                         Intersection of both candidate sets
                                                    │
                                         SQLite metadata lookup + price filter
                                                    │
                                         Re-rank by visual similarity order
                                                    │
                                         Return top-100 product cards
                                                    │
                                       Live image URLs fetched from Tokopedia GQL
```

---

## Tools and Libraries

| Category | Library / Tool |
|---|---|
| Data Scraping | Python `requests`, Tokopedia GraphQL API |
| Data Storage | SQLite, Parquet, Kaggle Datasets |
| Image Embedding | ViT Base Patch16-224 (`google/vit-base-patch16-224`) via Hugging Face Transformers |
| Text Embedding | BGE-M3 (`BAAI/bge-m3`) via Sentence Transformers |
| Vector Search | FAISS (IndexHNSWFlat, IndexIVFPQ, IndexIDMap2) |
| Backend | FastAPI, Uvicorn |
| Frontend | HTML, CSS, JavaScript |
| Model / Index Hosting | Hugging Face Hub (Datasets + Spaces) |
| Training / Embedding Environment | Google Colab, Kaggle Notebooks (GPU) |

---

## Source Code and Demo

<strong style="background-color:#c9b99a; padding: 2px 6px; border-radius: 4px;">
  <a href="https://huggingface.co/spaces/Viewww/TokoLens/tree/main">Frontend Space</a>
</strong>

<strong style="background-color:#c9b99a; padding: 2px 6px; border-radius: 4px; margin-left: 8px;">
  <a href="https://huggingface.co/spaces/Viewww/image-retrieval-backend/tree/main">Backend Space</a>
</strong>

<strong style="background-color:#c9b99a; padding: 2px 6px; border-radius: 4px; margin-left: 8px;">
  <a href="https://viewww-tokolens.static.hf.space">Live Demo</a>
</strong>
