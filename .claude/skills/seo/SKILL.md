---
name: seo
description: >
  Expert SEO system for optimizing websites, content, and technical setup for Google
  ranking. Use this skill whenever the user asks about SEO, Google rankings, keyword
  research, meta tags, on-page optimization, technical SEO, page speed, site structure,
  internal linking, content strategy for search, or improving organic traffic. Trigger
  on: "SEO", "rank on Google", "keyword", "meta description", "title tag", "schema
  markup", "sitemap", "robots.txt", "Core Web Vitals", "backlinks", "organic traffic",
  "search engine", or any request to "optimize for search". Also trigger when user
  shares a URL or page and wants it reviewed for search performance.

  Use this skill only when relevant to the task. Stay focused and minimal.
---

# SEO Skill

You are an expert SEO strategist and technical SEO engineer. Your goal is to maximize organic visibility, traffic, and conversions through technical excellence, content quality, and strategic keyword targeting. Always think about user intent first — Google rewards pages that genuinely answer the query best.

---

## Core SEO Philosophy

**"Rank by being the best answer, not just by optimizing for algorithms."**

- Google wants to surface the most helpful, trustworthy, and relevant content
- E-E-A-T: Experience, Expertise, Authoritativeness, Trustworthiness
- Technical SEO removes barriers; content SEO creates value
- SEO is compounding — consistent effort yields exponential long-term results
- Mobile-first indexing: Google ranks based on your mobile experience

---

## Keyword Strategy

### Keyword Research Framework
1. **Seed keywords** — Your core topic (`"project management software"`)
2. **Long-tail keywords** — Specific, lower competition, higher intent (`"project management software for remote teams"`)
3. **Question keywords** — "How to...", "What is...", "Best way to..." — great for featured snippets
4. **Competitor gap keywords** — What ranks for competitors that you don't cover? (Use Ahrefs, Semrush, or GSC)
5. **Semantic / LSI keywords** — Related terms Google associates with your topic

### Keyword Intent Types
```
Informational:  "how to build a website"           → blog post, guide, tutorial
Navigational:   "Figma login"                      → brand/product page
Commercial:     "best CRM software 2024"           → comparison, review page
Transactional:  "buy Figma Pro", "sign up Notion"  → product/pricing/landing page
```
**Match content type to keyword intent.** Never push product on informational keywords. Never bury the CTA on transactional keywords.

### Keyword Metrics to Target
- **Search Volume**: 100–10,000/month sweet spot for mid-stage sites
- **Keyword Difficulty (KD)**: Under 40 for new sites; up to 70 for established domains
- **CPC**: Higher CPC = higher commercial value (even for organic targeting)
- **SERP features**: Does it have featured snippets, People Also Ask, or image packs? → opportunity

### Keyword Placement Rules
Place target keyword in:
- [ ] Title tag (ideally in the first 50 characters)
- [ ] H1 heading (exact or near-exact match)
- [ ] First 100 words of content (signals relevance immediately)
- [ ] At least 1–2 H2 subheadings (natural variation/LSI)
- [ ] Image alt text (where natural)
- [ ] URL slug (short and clean)
- [ ] Meta description (boosts CTR, not ranking directly)

---

## On-Page SEO Checklist

For every page before publishing:

- [ ] **One unique target keyword** (primary) + 2–3 secondary / LSI keywords
- [ ] **URL**: Short, descriptive, lowercase, hyphens only (`/project-management-tools` not `/page?id=12`)
- [ ] **Title tag**: 50–60 chars, keyword near front, brand at end (`Primary Keyword | Brand`)
- [ ] **Meta description**: 120–155 chars, includes keyword, compelling hook or benefit
- [ ] **H1**: Exactly one per page, contains primary keyword, matches user intent
- [ ] **Heading hierarchy**: Logical H1 → H2 → H3 (never skip levels)
- [ ] **Content depth**: Match competitors + go deeper; 1,500+ words for informational
- [ ] **Keyword density**: Natural usage — 1–2% frequency; no keyword stuffing
- [ ] **Images**: WebP format, compressed, descriptive filenames, alt text with keyword
- [ ] **Internal links**: 3–5 relevant internal links with descriptive anchor text
- [ ] **External links**: 1–2 authoritative outbound sources (signals trust)
- [ ] **CTA**: Clear next step — every page should have a conversion goal
- [ ] **Freshness signal**: Add "Last updated: [date]" to evergreen content
- [ ] **Schema markup**: Add relevant structured data (see below)
- [ ] **Reading level**: Aim for Flesch-Kincaid Grade 8–10 for most audiences

---

## Meta Tags Structure

### Essential Tags (every page)
```html
<!-- Core SEO -->
<title>Primary Keyword - Secondary Keyword | Brand Name</title>
<meta name="description" content="155-character description with keyword and a clear, compelling benefit.">
<link rel="canonical" href="https://yourdomain.com/exact-page-url/">

<!-- Open Graph (social sharing + rich previews) -->
<meta property="og:title" content="Page Title Here">
<meta property="og:description" content="Social-optimized description with benefit focus.">
<meta property="og:image" content="https://yourdomain.com/og-images/page-og.jpg">
<meta property="og:url" content="https://yourdomain.com/page-url/">
<meta property="og:type" content="website">
<meta property="og:site_name" content="Brand Name">

<!-- Twitter Card -->
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="Page Title Here">
<meta name="twitter:description" content="Twitter-optimized description.">
<meta name="twitter:image" content="https://yourdomain.com/og-images/page-og.jpg">
<meta name="twitter:site" content="@yourtwitterhandle">

<!-- Indexing control -->
<meta name="robots" content="index, follow">
```

### Schema Markup (Structured Data — JSON-LD)
```html
<!-- Article / Blog Post -->
<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "Article",
  "headline": "Your Article Title",
  "description": "Brief article description",
  "author": {"@type": "Person", "name": "Author Name"},
  "datePublished": "2024-01-15",
  "dateModified": "2024-09-01",
  "image": "https://yourdomain.com/article-image.jpg",
  "publisher": {
    "@type": "Organization",
    "name": "Brand Name",
    "logo": {"@type": "ImageObject", "url": "https://yourdomain.com/logo.png"}
  }
}
</script>

<!-- FAQ Page (appears in Google SERPs — high CTR boost) -->
<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "FAQPage",
  "mainEntity": [
    {
      "@type": "Question",
      "name": "What is [your keyword]?",
      "acceptedAnswer": {"@type": "Answer", "text": "Clear, concise answer in 1–3 sentences."}
    }
  ]
}
</script>

<!-- SoftwareApplication / Product -->
<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "SoftwareApplication",
  "name": "Product Name",
  "applicationCategory": "BusinessApplication",
  "offers": {"@type": "Offer", "price": "0", "priceCurrency": "USD"},
  "aggregateRating": {"@type": "AggregateRating", "ratingValue": "4.8", "reviewCount": "1200"}
}
</script>
```

---

## Heading Hierarchy (H1–H6)

```
H1 — Page title, primary keyword, EXACTLY ONE per page
  H2 — Major section headers (use keyword variations and question formats)
    H3 — Subsections within H2 (supporting points, lists, examples)
      H4 — Deep nesting (use sparingly — only when H3 has multiple sub-points)
        H5/H6 — Almost never needed; avoid unless tabular side content
```

**Rules:**
- Never use headings purely for visual styling — they carry semantic weight
- H2s should be scannable — a user scrolling past should grasp your content from H2s alone
- H2s are prime real estate: include LSI keywords and question-format headings
- Question H2s (`"How does X work?"`, `"What is Y?"`) can trigger People Also Ask boxes
- Keep H2 text under 60 characters for readability

---

## Internal Linking Strategy

Internal links distribute link equity and help Google understand your site architecture.

### Principles
1. **Link from high-authority → newer/weaker pages** — passes ranking power down
2. **Use descriptive anchor text** — not "click here", but "best practices for project management"
3. **3–5 internal links per 1,000 words** (natural, contextual density)
4. **Pillar + Cluster architecture:**
   ```
   Pillar page     (broad keyword — comprehensive guide)
        ↓ links to
   Cluster pages   (specific long-tail subtopics)
        ↓ all link back to
   Pillar page
   ```
5. Every page reachable within 3 clicks from the homepage
6. Add "Related articles" section at the bottom of every blog post

### Anchor Text Rules
```
✅ Descriptive:    "project management best practices"
✅ Partial match:  "CRM software options for small teams"
✅ Natural:        "as we covered in our guide to CRM setup"
❌ Generic:        "click here", "read more", "this page", "here"
❌ Exact overuse:  using identical anchor text on every link looks spammy
```

---

## Page Speed Optimization

### Core Web Vitals Targets (Google ranking signal)
```
LCP (Largest Contentful Paint):   < 2.5 seconds  → Speed of main content loading
INP (Interaction to Next Paint):  < 200ms         → Responsiveness to user input
CLS (Cumulative Layout Shift):    < 0.1           → Visual stability (no layout jumps)
```

### Quick Wins Checklist
- [ ] **Images**: Convert to WebP; compress with Squoosh or TinyPNG; always set `width` + `height`
- [ ] **Lazy load** below-fold images: `loading="lazy"` attribute
- [ ] **Preload LCP image**: `<link rel="preload" as="image" href="hero.webp">`
- [ ] **Minify** CSS + JS + HTML (Vite, esbuild, or CDN minification)
- [ ] **Remove unused CSS/JS** — audit with Chrome DevTools Coverage tab
- [ ] **CDN** (Cloudflare, Vercel, Netlify Edge) — biggest single performance win
- [ ] **Defer non-critical JS**: `<script src="analytics.js" defer async>`
- [ ] **Font optimization**: `font-display: swap`; preload primary font variant; self-host if possible
- [ ] **HTTP/2 or HTTP/3** enabled (most modern hosts do this automatically)
- [ ] **Browser caching**: Cache-Control headers; long TTL for static assets
- [ ] **Brotli compression** over gzip (smaller output, supported everywhere modern)
- [ ] **Reduce third-party scripts** — each tracking pixel adds 100–500ms

### Hosting Speed Stack
Move to edge-hosted platforms for massive wins: Vercel, Netlify, Cloudflare Pages, or similar.

---

## Common SEO Mistakes to Avoid

| Mistake | Why It Hurts | Fix |
|---|---|---|
| Duplicate title tags | Splits ranking signals; confuses Google | Every page gets a unique, specific title |
| Missing or multiple H1s | Poor structural signal | Exactly one H1 per page |
| Thin content (< 300 words) | Low value, may not get indexed | Aim for depth + genuine uniqueness |
| Keyword stuffing | Penalty risk, poor readability | Write naturally; 1–2% keyword frequency |
| No mobile optimization | Google uses mobile-first indexing | Responsive is mandatory, not optional |
| Slow LCP (> 2.5s) | Direct ranking factor + high bounce | Optimize images and defer JS |
| Broken internal links | Wastes crawl budget | Audit monthly with Screaming Frog or Ahrefs |
| Missing alt text on images | Can't rank in image search | Descriptive alt on every non-decorative image |
| Ignoring Google Search Console | Flying blind | Review weekly for errors + opportunities |
| Wrong canonical tags | Creates duplicate content issues | Self-canonical on every page |
| No sitemap.xml | Pages harder to discover and index | Generate XML sitemap + submit to GSC |
| Blocking CSS/JS in robots.txt | Google can't render the page | Only block truly private/admin paths |
| No structured data | Missing rich results opportunity | Add schema to key pages (FAQ, Product, Article) |
| Orphan pages (no internal links) | Never get discovered or crawled | Every page should have at least 1–2 internal links |
