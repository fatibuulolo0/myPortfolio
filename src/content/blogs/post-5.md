---
title: "E-Commerce Behavior Mining: Sequential Pattern Mining with PrefixSpan & Association Rules"
description: "Mining over 5.9 million e-commerce user interaction events using PrefixSpan to discover sequential behavioral patterns, compute brand-level conversion rates, and visualize the complete customer journey funnel."
author: "Fati Buulolo"
image:
  url: "../../assets/image-012/cover.png"
  alt: "E-Commerce Behavior Mining with PrefixSpan"
pubDate: 2026-04-02
tags:
  ["Python", "PrefixSpan", "Sequential Pattern Mining", "Association Rules", "E-Commerce"]
---


## Overview

Understanding how customers behave and not just what they buy, is a core challenge in e-commerce analytics. Purchase outcomes are rarely the result of a single action. They emerge from a sequence of interactions: a user views a product, compares alternatives, adds to cart, browses again, then decides.

This project applies **Sequential Pattern Mining (SPM)** using the **PrefixSpan** algorithm to extract ordered behavioral sequences from approximately 5.97 million raw e-commerce event logs. Association rule confidence is then used to compute **behavioral conversion rates** at each step of the customer journey, and the results are visualized through an interactive **Sankey diagram**.

---

## Problem Statement

Standard product analytics tools report aggregate metrics like total views, cart rates, purchase rates. What they miss is sequential context: the order in which events happen and how one action influences the next.

Two core questions drive this project:

1. What are the most frequent ordered behavioral sequences users follow before purchasing?
2. Given a specific sequence of prior actions, how likely is a user to convert at the next step?

Answering these questions requires moving beyond tabular aggregations into sequence-aware pattern mining, which is what PrefixSpan is designed to solve.

---

## Dataset

| Property | Value |
|---|---|
| Source | E-Commerce Behavior Event Logs |
| Files | `sequence_event1.json`, `sequence_event2.json` |
| Total Event Sequences | **5,974,703** |
| Minimum Support Threshold | 0.1% of total events (~5,974 occurrences) |
| Event Types | `view`, `cart`, `purchase` |
| Brands Covered | Samsung, Apple, Xiaomi, Unknown |

Each sequence represents a user session encoded as an ordered list of `event_brand` tokens, for example `['view_samsung', 'cart_samsung', 'purchase_samsung']`. The two event files were combined into a single sequence corpus before mining.

---

## Methodology

### Sequential Pattern Mining — PrefixSpan

PrefixSpan is a projection-based sequential pattern mining algorithm that discovers all frequent ordered subsequences without generating candidates. It works by iteratively projecting the database onto suffix subsequences for each frequent prefix, making it efficient on large datasets.

```
Input  : User event sequences
         e.g. ['view_samsung', 'cart_samsung', 'purchase_samsung']

Process: PrefixSpan — min_support = 0.1% of total events

Output : All frequent ordered patterns with occurrence counts
```

### Conversion Rate via Association Rule Confidence

For every frequent sequential pattern of length > 1, an association rule is derived:

```
Pattern  : [A, B]
Rule     : A → B
Confidence = freq(A, B) / freq(A)
```

This confidence is interpreted as a behavioral conversion rate which, which represent the probability that a user who performed action A will subsequently perform action B within the same session sequence.

---

## Results

### Top Frequent Sequential Patterns

The 15 most frequent patterns across the entire dataset reveal dominant browsing behavior per brand:

| Rank | Pattern | Frequency |
|---|---|---|
| 1 | `view_Unknown` | 1,512,378 |
| 2 | `view_samsung` | 1,285,599 |
| 3 | `view_Unknown → view_Unknown` | 1,051,460 |
| 4 | `view_apple` | 1,002,335 |
| 5 | `view_samsung → view_samsung` | 971,106 |
| 6 | `view_apple → view_apple` | 791,250 |
| 7 | `view_xiaomi` | 737,611 |
| 8 | `view_Unknown × 3` | 670,225 |
| 9 | `view_samsung × 3` | 583,094 |
| 10 | `view_xiaomi → view_xiaomi` | 552,483 |

The dominance of repeated single-brand view patterns like `view_samsung → view_samsung`, `view_apple → view_apple` confirms that users engage in multi-session consideration loops before committing to any commercial action.

---

### Business-Critical Patterns

Filtering for patterns that involve `cart` or `purchase` isolates the commercially relevant journeys:

| Rank | Pattern | Frequency |
|---|---|---|
| 1 | `cart_samsung` | 185,786 |
| 2 | `view_samsung → cart_samsung` | 184,926 |
| 3 | `purchase_samsung` | 149,358 |
| 4 | `view_samsung → purchase_samsung` | 148,934 |
| 5 | `cart_apple` | 136,037 |
| 6 | `view_apple → cart_apple` | 135,572 |
| 7 | `cart_samsung → view_samsung` | 127,766 |
| 8 | `view_samsung → cart_samsung → view_samsung` | 127,086 |
| 9 | `purchase_apple` | 121,635 |
| 10 | `view_apple → purchase_apple` | 121,342 |
| 11 | `cart_samsung → purchase_samsung` | 104,910 |
| 12 | `view_samsung → cart_samsung → purchase_samsung` | 104,409 |

---

### Conversion Rate Analysis

Using association rule confidence, the behavioral conversion rate is computed for each step-to-step transition:

| Pattern (Antecedent → Consequent) | Events | Conversion Rate |
|---|---|---|
| `view_samsung` → `cart_samsung` | 184,926 | **14.38%** |
| `view_samsung` → `purchase_samsung` | 148,934 | **11.58%** |
| `view_apple` → `cart_apple` | 135,572 | **13.53%** |
| `view_apple` → `purchase_apple` | 121,342 | **12.11%** |
| `cart_samsung` → `purchase_samsung` | 104,910 | **56.47%** |
| `view_samsung → cart_samsung` → `purchase_samsung` | 104,409 | **56.46%** |
| `cart_samsung` → `view_samsung` | 127,766 | **68.77%** |
| `view_samsung → cart_samsung` → `view_samsung` | 127,086 | **68.72%** |
| `cart_apple` → `view_apple` | 92,655 | **68.11%** |
| `view_apple → cart_apple` → `view_apple` | 92,293 | **68.08%** |
| `view_samsung → view_samsung` → `cart_samsung` | 94,500 | **9.73%** |

The data reveals a critical inflection point at the cart stage: the view-to-cart rate sits around **13–14%**, while cart-to-purchase jumps to **~56%**. Once a user adds an item to cart, the likelihood of purchase is nearly four times higher than at the view stage.

---

## Customer Journey Visualization

The Sankey diagram below maps weighted event transitions across the top 30 business-critical patterns. Each node represents a behavioral state (`view`, `cart`, or `purchase` per brand), and each flow represents the volume of users moving between states.

<iframe
  src="../../assets/image-012/sankey_visualization.html"
  width="100%"
  height="580px"
  frameborder="0"
  style="border-radius: 8px; margin: 1rem 0;">
</iframe>

The diagram makes visible what the tables imply: the majority of user flow concentrates between Samsung and Apple journeys, with significant return loops from `cart` back to `view` which show a behavioral signature of comparison shopping and price hesitation before final commitment.

---

## Business Insights

### The Golden Path

The most high-value sequential journey identified from the data:

```
view_samsung  →  cart_samsung  →  purchase_samsung
```

This three-step path was completed 104,409 times and represents the clearest signal of a high-intent buyer. Any product or marketing intervention that accelerates a user through this path yields direct revenue impact.

---

### The Cart is the Conversion Tipping Point

The view-to-cart rate (~14%) and cart-to-purchase rate (~56%) are not close, they are separated by a factor of four. This makes the cart stage the single highest-leverage point in the funnel. Reducing friction between `view` and `cart` through urgency signals, social proof, or personalized offers for repeat viewers will produce more downstream conversions than any equivalent effort at a later stage.

---

### Cart Abandonment Is a Browse-Back Loop

The pattern `cart_samsung → view_samsung` achieved a confidence of 68.77%, meaning nearly 7 in 10 users who add Samsung to cart return to browse before purchasing. This is not abandonment, it is deliberation. These users are still in the funnel. A well-timed cart recovery notification or a price-match prompt at this moment can recover a large share of these sessions.

---

### Samsung Leads in Volume, Apple Leads in Efficiency

| Brand | Views | Purchases | View-to-Purchase Rate |
|---|---|---|---|
| Samsung | 1,285,599 | 149,358 | 11.6% |
| Apple | 1,002,335 | 121,635 | 12.1% |

Apple converts a slightly higher share of its viewers into buyers. Samsung's greater raw volume comes with a broader and less committed audience. Each brand warrants a different retention strategy: Samsung users benefit from comparison features and decision-support tools; Apple users, already more decisive, respond better to upsell and accessory recommendations at checkout.

---

### Xiaomi Has an Unresolved Funnel Drop

Xiaomi is the third most-viewed brand in the dataset with 737,611 view events, yet it does not appear in any of the top commercial conversion patterns. The data implies that Xiaomi attracts significant awareness but fails to convert it. This gap between discovery and action points to product page friction, pricing misalignment, or insufficient purchase intent among Xiaomi's viewing audience.

---

## Conclusion

This project demonstrates that sequential pattern mining surfaces behavioral signals that aggregate funnel metrics simply cannot. The critical findings: a 56% cart-to-purchase conversion rate; a 68% browse-back rate from cart; and Xiaomi's invisible funnel are all invisible in a standard view/cart/purchase breakdown. They only emerge when user behavior is analyzed as an ordered sequence rather than a collection of independent events.

PrefixSpan, combined with association rule confidence scoring, provides a scalable and interpretable framework for this kind of analysis. The Sankey diagram translates the pattern output into a visual format that makes these findings immediately actionable for product, marketing, and merchandising teams.

---

## Tools and Libraries

- **Python** — pandas, collections
- **Sequential Pattern Mining** — PrefixSpan 0.5.2
- **Visualization** — Plotly (interactive Sankey diagram)
- **Environment** — Google Colab
- **Data Storage** — Google Drive

---

## Source Code

<strong style="background-color:#c9b99a; padding: 2px 6px; border-radius: 4px;">
  <a href="https://github.com/FatiBuuloloo/E-commerce_Shopper_Behavior_Mining_PrefixSpan_-_Association_Rule-mini_project-012">View on GitHub</a>
</strong>
