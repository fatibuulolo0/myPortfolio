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

<div class="max-w-2xl mx-auto my-8 overflow-hidden border border-zinc-200 rounded-2xl shadow-sm">
  <table class="w-full text-left border-collapse">
    <thead class="bg-[#c9b99a] text-zinc-900">
      <tr>
        <th colspan="2" class="p-4 text-center font-bold tracking-wide uppercase text-xs border-b border-zinc-300">
          Dataset Characteristics & Configuration
        </th>
      </tr>
    </thead>
    <tbody class="kanit-light text-sm">
      <tr class="border-b border-zinc-100 hover:bg-zinc-50 transition-colors">
        <td class="p-4 font-bold bg-zinc-50/30 w-1/3">Source</td>
        <td class="p-4">E-Commerce Behavior Event Logs</td>
      </tr>
      <tr class="border-b border-zinc-100 hover:bg-zinc-50 transition-colors">
        <td class="p-4 font-bold bg-zinc-50/30">Files</td>
        <td class="p-4 font-mono text-xs">
          sequence_event1.json, sequence_event2.json
        </td>
      </tr>
      <tr class="border-b border-zinc-100 hover:bg-zinc-50 transition-colors">
        <td class="p-4 font-bold bg-zinc-50/30">Total Event Sequences</td>
        <td class="p-4 font-bold">5,974,703</td>
      </tr>
      <tr class="border-b border-zinc-100 hover:bg-zinc-50 transition-colors">
        <td class="p-4 font-bold bg-zinc-50/30">Minimum Support Threshold</td>
        <td class="p-4">0.1% of total events (~5,974 occurrences)</td>
      </tr>
      <tr class="border-b border-zinc-100 hover:bg-zinc-50 transition-colors">
        <td class="p-4 font-bold bg-zinc-50/30">Event Types</td>
        <td class="p-4 font-mono text-xs">
          view, cart, purchase
        </td>
      </tr>
      <tr class="hover:bg-zinc-50 transition-colors">
        <td class="p-4 font-bold bg-zinc-50/30">Brands Covered</td>
        <td class="p-4">
          Samsung, Apple, Xiaomi, Unknown
        </td>
      </tr>
    </tbody>
  </table>
</div>

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

<div class="overflow-x-auto my-8 border border-zinc-200 rounded-xl shadow-sm">
  <table class="w-full text-left border-collapse min-w-[600px]">
    <thead>
      <tr class="bg-[#c9b99a] text-zinc-900 text-sm">
        <th class="p-4 font-bold border-b border-zinc-300 w-16 text-center">Rank</th>
        <th class="p-4 font-bold border-b border-zinc-300">Pattern</th>
        <th class="p-4 font-bold border-b border-zinc-300 text-right">Frequency</th>
      </tr>
    </thead>
    <tbody class="kanit-light text-sm">
      <tr class="border-b border-zinc-100 hover:bg-zinc-50 transition-colors">
        <td class="p-3 text-center font-bold">1</td>
        <td class="p-3 font-mono text-xs"><code>view_Unknown</code></td>
        <td class="p-3 text-right">1,512,378</td>
      </tr>
      <tr class="border-b border-zinc-100 hover:bg-zinc-50 transition-colors">
        <td class="p-3 text-center font-bold">2</td>
        <td class="p-3 font-mono text-xs"><code>view_samsung</code></td>
        <td class="p-3 text-right">1,285,599</td>
      </tr>
      <tr class="border-b border-zinc-100 hover:bg-zinc-50 transition-colors">
        <td class="p-3 text-center font-bold">3</td>
        <td class="p-3 font-mono text-xs"><code>view_Unknown</code> → <code>view_Unknown</code></td>
        <td class="p-3 text-right">1,051,460</td>
      </tr>
      <tr class="border-b border-zinc-100 hover:bg-zinc-50 transition-colors">
        <td class="p-3 text-center font-bold">4</td>
        <td class="p-3 font-mono text-xs"><code>view_apple</code></td>
        <td class="p-3 text-right">1,002,335</td>
      </tr>
      <tr class="border-b border-zinc-100 hover:bg-zinc-50 transition-colors">
        <td class="p-3 text-center font-bold">5</td>
        <td class="p-3 font-mono text-xs"><code>view_samsung</code> → <code>view_samsung</code></td>
        <td class="p-3 text-right">971,106</td>
      </tr>
      <tr class="border-b border-zinc-100 hover:bg-zinc-50 transition-colors">
        <td class="p-3 text-center font-bold">6</td>
        <td class="p-3 font-mono text-xs"><code>view_apple</code> → <code>view_apple</code></td>
        <td class="p-3 text-right">791,250</td>
      </tr>
      <tr class="border-b border-zinc-100 hover:bg-zinc-50 transition-colors">
        <td class="p-3 text-center font-bold">7</td>
        <td class="p-3 font-mono text-xs"><code>view_xiaomi</code></td>
        <td class="p-3 text-right">737,611</td>
      </tr>
      <tr class="border-b border-zinc-100 hover:bg-zinc-50 transition-colors">
        <td class="p-3 text-center font-bold">8</td>
        <td class="p-3 font-mono text-xs"><code>view_Unknown</code> → <code>view_Unknown</code> → <code>view_Unknown</code></td>
        <td class="p-3 text-right">670,225</td>
      </tr>
      <tr class="border-b border-zinc-100 hover:bg-zinc-50 transition-colors">
        <td class="p-3 text-center font-bold">9</td>
        <td class="p-3 font-mono text-xs"><code>view_samsung</code> → <code>view_samsung</code> → <code>view_samsung</code></td>
        <td class="p-3 text-right">583,094</td>
      </tr>
      <tr class="hover:bg-zinc-50 transition-colors">
        <td class="p-3 text-center font-bold">10</td>
        <td class="p-3 font-mono text-xs"><code>view_xiaomi</code> → <code>view_xiaomi</code></td>
        <td class="p-3 text-right">552,483</td>
      </tr>
    </tbody>
  </table>
</div>

The dominance of repeated single-brand view patterns like `view_samsung → view_samsung`, `view_apple → view_apple` confirms that users engage in multi-session consideration loops before committing to any commercial action.

---

### Business-Critical Patterns

Filtering for patterns that involve `cart` or `purchase` isolates the commercially relevant journeys:

<div class="overflow-x-auto my-8 border border-zinc-200 rounded-xl shadow-sm">
  <table class="w-full text-left border-collapse min-w-[600px]">
    <thead>
      <tr class="bg-[#c9b99a] text-zinc-900 text-sm">
        <th class="p-4 font-bold border-b border-zinc-300 w-16 text-center">Rank</th>
        <th class="p-4 font-bold border-b border-zinc-300">Pattern</th>
        <th class="p-4 font-bold border-b border-zinc-300 text-right">Frequency</th>
      </tr>
    </thead>
    <tbody class="kanit-light text-sm">
      <tr class="border-b border-zinc-100 hover:bg-zinc-50 transition-colors">
        <td class="p-3 text-center font-bold">1</td>
        <td class="p-3 font-mono text-xs"><code>cart_samsung</code></td>
        <td class="p-3 text-right">185,786</td>
      </tr>
      <tr class="border-b border-zinc-100 hover:bg-zinc-50 transition-colors">
        <td class="p-3 text-center font-bold">2</td>
        <td class="p-3 font-mono text-xs"><code>view_samsung</code> → <code>cart_samsung</code></td>
        <td class="p-3 text-right">184,926</td>
      </tr>
      <tr class="border-b border-zinc-100 hover:bg-zinc-50 transition-colors">
        <td class="p-3 text-center font-bold">3</td>
        <td class="p-3 font-mono text-xs"><code>purchase_samsung</code></td>
        <td class="p-3 text-right">149,358</td>
      </tr>
      <tr class="border-b border-zinc-100 hover:bg-zinc-50 transition-colors">
        <td class="p-3 text-center font-bold">4</td>
        <td class="p-3 font-mono text-xs"><code>view_samsung</code> → <code>purchase_samsung</code></td>
        <td class="p-3 text-right">148,934</td>
      </tr>
      <tr class="border-b border-zinc-100 hover:bg-zinc-50 transition-colors">
        <td class="p-3 text-center font-bold">5</td>
        <td class="p-3 font-mono text-xs"><code>cart_apple</code></td>
        <td class="p-3 text-right">136,037</td>
      </tr>
      <tr class="border-b border-zinc-100 hover:bg-zinc-50 transition-colors">
        <td class="p-3 text-center font-bold">6</td>
        <td class="p-3 font-mono text-xs"><code>view_apple</code> → <code>cart_apple</code></td>
        <td class="p-3 text-right">135,572</td>
      </tr>
      <tr class="border-b border-zinc-100 hover:bg-zinc-50 transition-colors">
        <td class="p-3 text-center font-bold">7</td>
        <td class="p-3 font-mono text-xs"><code>cart_samsung</code> → <code>view_samsung</code></td>
        <td class="p-3 text-right">127,766</td>
      </tr>
      <tr class="border-b border-zinc-100 hover:bg-zinc-50 transition-colors">
        <td class="p-3 text-center font-bold">8</td>
        <td class="p-3 font-mono text-xs"><code>view_samsung</code> → <code>cart_samsung</code> → <code>view_samsung</code></td>
        <td class="p-3 text-right">127,086</td>
      </tr>
      <tr class="border-b border-zinc-100 hover:bg-zinc-50 transition-colors">
        <td class="p-3 text-center font-bold">9</td>
        <td class="p-3 font-mono text-xs"><code>purchase_apple</code></td>
        <td class="p-3 text-right">121,635</td>
      </tr>
      <tr class="border-b border-zinc-100 hover:bg-zinc-50 transition-colors">
        <td class="p-3 text-center font-bold">10</td>
        <td class="p-3 font-mono text-xs"><code>view_apple</code> → <code>purchase_apple</code></td>
        <td class="p-3 text-right">121,342</td>
      </tr>
      <tr class="border-b border-zinc-100 hover:bg-zinc-50 transition-colors">
        <td class="p-3 text-center font-bold">11</td>
        <td class="p-3 font-mono text-xs"><code>cart_samsung</code> → <code>purchase_samsung</code></td>
        <td class="p-3 text-right">104,910</td>
      </tr>
      <tr class="hover:bg-zinc-50 transition-colors">
        <td class="p-3 text-center font-bold">12</td>
        <td class="p-3 font-mono text-xs"><code>view_samsung</code> → <code>cart_samsung</code> → <code>purchase_samsung</code></td>
        <td class="p-3 text-right">104,409</td>
      </tr>
    </tbody>
  </table>
</div>

---

### Conversion Rate Analysis

Using association rule confidence, the behavioral conversion rate is computed for each step-to-step transition:

<div class="overflow-x-auto my-8 border border-zinc-200 rounded-xl shadow-sm">
  <table class="w-full text-left border-collapse min-w-[700px]">
    <thead>
      <tr class="bg-[#c9b99a] text-zinc-900 text-sm">
        <th class="p-4 font-bold border-b border-zinc-300">Pattern (Antecedent → Consequent)</th>
        <th class="p-4 font-bold border-b border-zinc-300">Events</th>
        <th class="p-4 font-bold border-b border-zinc-300 text-center">Conversion Rate</th>
      </tr>
    </thead>
    <tbody class="kanit-light text-sm">
      <tr class="border-b border-zinc-100 hover:bg-zinc-50 transition-colors">
        <td class="p-3 font-mono text-xs"><code>view_samsung</code> → <code>cart_samsung</code></td>
        <td class="p-3">184,926</td>
        <td class="p-3 text-center font-bold">14.38%</td>
      </tr>
      <tr class="border-b border-zinc-100 bg-zinc-50/30 hover:bg-zinc-50 transition-colors">
        <td class="p-3 font-mono text-xs"><code>view_samsung</code> → <code>purchase_samsung</code></td>
        <td class="p-3">148,934</td>
        <td class="p-3 text-center font-bold">11.58%</td>
      </tr>
      <tr class="border-b border-zinc-100 hover:bg-zinc-50 transition-colors">
        <td class="p-3 font-mono text-xs"><code>view_apple</code> → <code>cart_apple</code></td>
        <td class="p-3">135,572</td>
        <td class="p-3 text-center font-bold">13.53%</td>
      </tr>
      <tr class="border-b border-zinc-100 bg-zinc-50/30 hover:bg-zinc-50 transition-colors">
        <td class="p-3 font-mono text-xs"><code>view_apple</code> → <code>purchase_apple</code></td>
        <td class="p-3">121,342</td>
        <td class="p-3 text-center font-bold">12.11%</td>
      </tr>
      <tr class="border-b border-zinc-100 hover:bg-zinc-50 transition-colors">
        <td class="p-3 font-mono text-xs"><code>cart_samsung</code> → <code>purchase_samsung</code></td>
        <td class="p-3">104,910</td>
        <td class="p-3 text-center font-bold">56.47%</td>
      </tr>
      <tr class="border-b border-zinc-100 bg-zinc-50/30 hover:bg-zinc-50 transition-colors">
        <td class="p-3 font-mono text-xs"><code>view_samsung → cart_samsung</code> → <code>purchase_samsung</code></td>
        <td class="p-3">104,409</td>
        <td class="p-3 text-center font-bold">56.46%</td>
      </tr>
      <tr class="border-b border-zinc-100 hover:bg-zinc-50 transition-colors">
        <td class="p-3 font-mono text-xs"><code>cart_samsung</code> → <code>view_samsung</code></td>
        <td class="p-3">127,766</td>
        <td class="p-3 text-center font-bold">68.77%</td>
      </tr>
      <tr class="border-b border-zinc-100 bg-zinc-50/30 hover:bg-zinc-50 transition-colors">
        <td class="p-3 font-mono text-xs"><code>view_samsung → cart_samsung</code> → <code>view_samsung</code></td>
        <td class="p-3">127,086</td>
        <td class="p-3 text-center font-bold">68.72%</td>
      </tr>
      <tr class="border-b border-zinc-100 hover:bg-zinc-50 transition-colors">
        <td class="p-3 font-mono text-xs"><code>cart_apple</code> → <code>view_apple</code></td>
        <td class="p-3">92,655</td>
        <td class="p-3 text-center font-bold">68.11%</td>
      </tr>
      <tr class="border-b border-zinc-100 bg-zinc-50/30 hover:bg-zinc-50 transition-colors">
        <td class="p-3 font-mono text-xs"><code>view_apple → cart_apple</code> → <code>view_apple</code></td>
        <td class="p-3">92,293</td>
        <td class="p-3 text-center font-bold">68.08%</td>
      </tr>
      <tr class="hover:bg-zinc-50 transition-colors">
        <td class="p-3 font-mono text-xs"><code>view_samsung → view_samsung</code> → <code>cart_samsung</code></td>
        <td class="p-3">94,500</td>
        <td class="p-3 text-center font-bold">9.73%</td>
      </tr>
    </tbody>
  </table>
</div>

The data reveals a critical inflection point at the cart stage: the view-to-cart rate sits around **13–14%**, while cart-to-purchase jumps to **~56%**. Once a user adds an item to cart, the likelihood of purchase is nearly four times higher than at the view stage.

---

## Customer Journey Visualization

The Sankey diagram below maps weighted event transitions across the top 30 business-critical patterns. Each node represents a behavioral state (`view`, `cart`, or `purchase` per brand), and each flow represents the volume of users moving between states.

<iframe
  src="/myPortfolio/sankey_visualization.html" 
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

<div style="overflow-x: auto; margin-bottom: 20px;">
  <table style="width: 100%; border-collapse: collapse; text-align: left;">
    <thead>
      <tr style="border-bottom: 2px solid #c9b99a;">
        <th style="padding: 12px; font-weight: bold;">Brand</th>
        <th style="padding: 12px; font-weight: bold;">Views</th>
        <th style="padding: 12px; font-weight: bold;">Purchases</th>
        <th style="padding: 12px; font-weight: bold;">Rate</th>
      </tr>
    </thead>
    <tbody>
      <tr style="border-bottom: 1px solid #eee;">
        <td style="padding: 12px;">Samsung</td>
        <td style="padding: 12px;">1,285,599</td>
        <td style="padding: 12px;">149,358</td>
        <td style="padding: 12px;">11.6%</td>
      </tr>
      <tr>
        <td style="padding: 12px;">Apple</td>
        <td style="padding: 12px;">1,002,335</td>
        <td style="padding: 12px;">121,635</td>
        <td style="padding: 12px;">12.1%</td>
      </tr>
    </tbody>
  </table>
</div>

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
