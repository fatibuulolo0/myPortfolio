---
title: "Toko Lens: Multimodal E-Commerce Visual Search Engine"
description: "End-to-end multimodal visual search system built on 3.4M+ scraped Tokopedia products, using a unified CLIP embedding space (laion/CLIP-ViT-H-14) to serve both image and text queries against a single FAISS index, delivered through a FastAPI backend and a plain HTML/JS frontend on Hugging Face Spaces."
author: "Fati Buulolo"
image:
  url: "../../assets/images-015/display_tokolens.png"
  alt: "Toko Lens: Multimodal E-Commerce Visual Search Engine"
pubDate: 2026-05-01
tags:
  ["Python", "FastAPI", "FAISS", "CLIP", "Multimodal", "Web Scraping", "Hugging Face", "HTML", "E-Commerce", "Computer Vision"]
---


## Overview

Finding a product on an e-commerce platform when you only have a photo and no idea what it is called is a problem that keyword search cannot solve. Toko Lens is a multimodal visual search engine built for Tokopedia, the largest Indonesian e-commerce marketplace. Given a product image (and an optional text keyword plus price range), the system retrieves the most visually and semantically similar products from a catalogue of over 3.4 million scraped listings in real time.

The retrieval system is built around **CLIP (laion/CLIP-ViT-H-14-laion2B-s32B-b79K)**, a large-scale vision-language model trained on two billion image-text pairs from the internet. CLIP encodes both images and text into the same 1024-dimensional vector space, which means a single FAISS index is sufficient to handle image queries, text queries, and combinations of both. The backend is a **FastAPI** application and the frontend is a plain **HTML/CSS/JavaScript** single-page application, both deployed on **Hugging Face Spaces**.

---

## Problem Statement

A shopper encounters a product in a photo and wants to find it or something similar on Tokopedia. Typing a product name requires prior knowledge of the category and vocabulary. Visual search removes that barrier: the image itself becomes the query, and optional text input narrows results further.

This project operationalises that idea at scale:

1. A **data pipeline** to scrape a representative catalogue of Tokopedia products with images.
2. A **CLIP embedding pipeline** to encode all product images into a unified vector space.
3. A **single-index retrieval system** that handles both image and text queries using the same FAISS index.
4. A **web application** that accepts image uploads, in-browser crop selection, text queries, and price filters, and returns ranked results with live product images.

---

## Data Pipeline

### Phase 1: Shop Scraping

The pipeline begins by collecting all shops labelled **Mall** on Tokopedia. Mall-labelled stores represent verified, high-quality merchants and provide a consistent product quality baseline. The scraper collected **13,165 unique shop names**.

For each shop name, a second scraping pass retrieved the corresponding **Shop ID (SID)**, the internal Tokopedia identifier required to query product listings from the platform's GraphQL API.

### Phase 2: Product Scraping (7 Phases)

Using the collected SIDs, the pipeline queried the product catalogue of every shop, extracting four fields per product: `Product_Name`, `Price`, `Rating`, and `URL_Product`. Because Colab free-tier runtime sessions have hard time limits and occasionally crash mid-run, the scraping was divided into **7 sequential phases**, each resuming from the last completed shop. The combined result was a raw product table of **3,614,752 listings**.

The full product dataset is publicly available on Kaggle:
**[tokopedia-products-with-images-dataset-v1](https://www.kaggle.com/datasets/fati22/tokopedia-products-with-images-dataset-v1)**

### Phase 3: Image Scraping (15 Phases)

Each product URL was visited to retrieve its primary product image. This was the most time-intensive stage because image downloads are bandwidth-bound and each Colab session could only process a fraction of the catalogue before timing out. The scraping was split into **15 phases**, each resuming from where the previous one stopped.

A subset of products could not be scraped for images. These were mostly prescription-grade medical products whose listing pages display a doctor-prescription notice instead of a product image, making automated image retrieval impossible.

Final image count after all 15 phases: **3,475,088 product images**, matching a cleaned product table of the same size (down from 3,614,752 after removing image-less entries).

The image dataset is also publicly available on Kaggle:
**[tokopedia-images-product](https://www.kaggle.com/datasets/fati22/tokopedia-images-product)**

---

## Embedding Pipeline

### Why CLIP

The previous version of this system used two separate models: ViT for image embeddings and BGE-M3 for text embeddings. Those two models produce vectors in incompatible spaces (768-dim vs 1024-dim), requiring two separate FAISS indices and an intersection operation at query time.

CLIP solves this at the architecture level. Because it was trained end-to-end using contrastive loss on 2 billion image-text pairs, its image encoder and text encoder are trained jointly to produce vectors that occupy the same semantic space. A photo of a laptop and the phrase "laptop gaming" land close together in this shared space even though they come from completely different input modalities. This makes CLIP the natural choice for a product search system where users may query by image, by text, or by both.

### Model: laion/CLIP-ViT-H-14-laion2B-s32B-b79K

The model used for all embeddings is **`laion/CLIP-ViT-H-14-laion2B-s32B-b79K`**, an open CLIP model trained by LAION on 2 billion curated image-text pairs. Key specifications:

| Property | Value |
|---|---|
| Vision backbone | ViT-H/14 (1280-dim internal, 32 encoder layers) |
| Text backbone | Transformer (1024-dim internal, 24 encoder layers) |
| Output dimension | 1024-dim (after visual and text projection layers) |
| Training data | LAION-2B (2 billion image-text pairs) |
| Similarity metric | Cosine similarity (L2-normalized dot product) |

Both the image encoder and the text encoder project their outputs to the same **1024-dimensional** space through learned linear projection layers (`visual_projection` and `text_projection`). After L2 normalization, vectors from both modalities can be compared directly using inner product.

### Embedding Process (14 Phases)

Embedding 3.4 million product images with a ViT-H/14 backbone is computationally heavy. The embedding run was divided into **14 sequential phases** on Kaggle GPU notebooks (Tesla T4), each processing a slice of the dataset and saving one output Parquet file. The batch size was set to 32 to stay within T4 VRAM limits.

Each Parquet file stores:

```
ID_Product    (int64)     Product identifier matching the SQLite metadata
Judul         (str)       Product title, kept for traceability
Image_Embedding (list)    1024-dim float32 CLIP image vector, L2-normalized
```

Only image embeddings are stored in the Parquet files. Text embeddings for the product catalogue are not needed because the CLIP text encoder is applied at query time directly on the user's keyword, and its output searches the same image-based FAISS index.

```python
with torch.no_grad():
    vision_outputs = model.vision_model(pixel_values=imgs)
    image_features = model.visual_projection(vision_outputs.pooler_output)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
```

---

## FAISS Index Construction

### Single Index: IndexHNSWFlat with Inner Product

Because all vectors are already L2-normalized and cosine similarity equals inner product for unit vectors, the index uses `faiss.METRIC_INNER_PRODUCT` to align with CLIP's native similarity metric.

```python
d      = 1024
M      = 32
ef     = 200

core_index = faiss.IndexHNSWFlat(d, M, faiss.METRIC_INNER_PRODUCT)
core_index.hnsw.efConstruction = ef
index  = faiss.IndexIDMap2(core_index)
```

HNSW (Hierarchical Navigable Small World) was chosen because it delivers very low query latency with high recall and requires no training phase, which is important when indexing 3.4 million vectors in a single sequential pass across 14 Parquet files. `IndexIDMap2` wraps the core index so that `ID_Product` values are stored alongside vectors and returned directly in search results.

After construction, the index was saved and pushed to a Hugging Face dataset repository:

```
Viewww/tokopedia-search-indices
  tokopedia_img_CLIP_hnsw.faiss
  metadata/metadata_tokopedia.db
```

---

## System Architecture

### Backend: FastAPI

The backend is a **Python FastAPI** application deployed as a Hugging Face Space. On startup it downloads the FAISS index and SQLite metadata database from the Hugging Face Hub, then loads the CLIP model in **float16** precision to fit within the Space's RAM budget.

```python
model_clip = CLIPModel.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
```

Two helper functions handle embedding at inference time:

```python
def embed_image(pil_img):
    inputs       = processor(images=pil_img, return_tensors="pt")
    pixel_values = inputs['pixel_values'].to(device)
    with torch.no_grad():
        vision_outputs = model_clip.vision_model(pixel_values=pixel_values)
        img_emb        = model_clip.visual_projection(vision_outputs.pooler_output)
        img_emb        = img_emb / img_emb.norm(dim=-1, keepdim=True)
    return img_emb.float().cpu().numpy().astype('float32')

def embed_text(text):
    text_inputs = processor(
        text=text, padding=True, truncation=True,
        max_length=77, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        text_outputs = model_clip.text_model(**text_inputs)
        txt_emb      = model_clip.text_projection(text_outputs.pooler_output)
        txt_emb      = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
    return txt_emb.float().cpu().numpy().astype('float32')
```

**`POST /search`** accepts a product image, an optional text keyword, a minimum price, and a maximum price. The search pipeline proceeds as follows:

```
1. Decode uploaded image with Pillow
2. Encode image with CLIP vision encoder + visual_projection
   → 1024-dim L2-normalized float32 vector
3. Search the single HNSW FAISS index → top-2000 candidate IDs

4. If user provided a text keyword:
     Encode keyword with CLIP text encoder + text_projection
     → 1024-dim L2-normalized float32 vector
     Search the SAME FAISS index → top-5000 IDs
     Keep only candidate IDs that appear in both result sets,
     preserving the original visual similarity ranking order

5. Query SQLite for metadata of surviving candidate IDs
6. Filter by price range (price_min to price_max)
7. Sort by original FAISS visual similarity rank
8. Return top-100 results as JSON
```

**`GET /get-image`** fetches the live primary image URL for a given product from Tokopedia's GraphQL API in real time, since image URLs are not stored statically in the metadata database.

### Frontend: HTML / CSS / JavaScript

The frontend is a single-page application built with plain **HTML, CSS, and JavaScript**, also deployed as a Hugging Face Space. It provides the following interaction flow:

1. The user uploads a product image from their device or captures one with the device camera.
2. A built-in crop tool (Cropper.js) lets the user isolate the specific object of interest before submitting the query.
3. An optional text field accepts a product name or keyword to refine the search beyond visual similarity.
4. A price range input narrows results to a target budget.
5. The frontend sends a `multipart/form-data` request to the FastAPI backend and renders the returned product cards with names, prices, ratings, and live images fetched via `GET /get-image`.

---

## How the Search Works

```
User uploads image (or captures via camera)
              │
              ▼
     [Optional: crop with Cropper.js]
              │
              ▼
     [Optional: add text keyword + price range]
              │
              ▼
         FastAPI backend
              │
              ├─ CLIP vision encoder
              │         │ 1024-dim vector
              │         ▼
              │   HNSW FAISS search → top-2000 visual candidates
              │
              └─ (if text provided)
                  CLIP text encoder
                        │ 1024-dim vector (same space)
                        ▼
                  HNSW FAISS search → top-5000 text candidates
                        │
                  Intersect with visual candidates
                        │
              SQLite metadata lookup
                        │
              Price range filter
                        │
              Sort by visual similarity rank
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
| Image and Text Embedding | CLIP ViT-H/14 (`laion/CLIP-ViT-H-14-laion2B-s32B-b79K`) via Hugging Face Transformers |
| Vector Search | FAISS (IndexHNSWFlat, IndexIDMap2, METRIC_INNER_PRODUCT) |
| Backend | FastAPI, Uvicorn |
| Frontend | HTML, CSS, JavaScript, Cropper.js |
| Model / Index Hosting | Hugging Face Hub (Datasets + Spaces) |
| Embedding Environment | Kaggle Notebooks (GPU: Tesla T4) |
| Scraping Environment | Google Colab |

---

## Source Code and Demo

<strong style="background-color:#c9b99a; padding: 2px 6px; border-radius: 4px;">
  <a href="https://huggingface.co/spaces/Viewww/TokoLens/tree/main">Frontend Space</a>
</strong>

<strong style="background-color:#c9b99a; padding: 2px 6px; border-radius: 4px; margin-left: 8px;">
  <a href="https://huggingface.co/spaces/sole1l/tokolens-backend/tree/main">Backend Space</a>
</strong>

<strong style="background-color:#c9b99a; padding: 2px 6px; border-radius: 4px; margin-left: 8px;">
  <a href="https://viewww-tokolens.static.hf.space">Live Demo</a>
</strong>
