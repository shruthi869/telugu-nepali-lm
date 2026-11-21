# Global Voices Nepali pilot review — 2026-08-15

This snapshot approves a bounded, manually initiated crawl of born-digital
Nepali articles published at `https://ne.globalvoices.org/`.

## Rights evidence

- Source: https://ne.globalvoices.org/
- Site-wide license linked in the source footer:
  https://creativecommons.org/licenses/by/3.0/
- The Nepali footer states that the site operates under a Creative Commons
  license and links CC BY 3.0. The embedded license metadata permits
  reproduction, distribution, and derivative works with attribution and notice.

Only the article text is approved. Images, repeated image captions, social-media
embeds, videos, comments, linked third-party pages, and other media are excluded.
Each retained record must keep its article URL and the required Global Voices
attribution.

## Crawl evidence and limits

- Robots policy reviewed 2026-08-15:
  https://ne.globalvoices.org/robots.txt
- Discovery sitemap reviewed 2026-08-15:
  https://ne.globalvoices.org/wp-sitemap.xml
- Robots declares a 10-second crawl delay and disallows administrative paths.
- The collector must obey that delay, stay on `ne.globalvoices.org`, accept only
  dated article paths, and begin with a source-specific 20-article pilot.
- Raw responses are immutable and the crawl resumes through `pull_state.sqlite`.
- Raising the page cap requires an explicit command-line override after the
  pilot's language and clean-token yield has been reviewed.

## Attribution

Retain the article URL and credit the named author/translator and Global Voices
under CC BY 3.0. The preprocessing record also carries this license expression
and attribution requirement.
