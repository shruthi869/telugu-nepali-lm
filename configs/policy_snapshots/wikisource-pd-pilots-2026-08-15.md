# Wikisource public-domain pilot review — 2026-08-15

This snapshot approves two narrowly scoped, manually initiated pilots. It does
not approve an all-pages crawl of either Wikisource installation.

## Shared policy evidence

- Multilingual Wikisource copyright policy, reviewed 2026-08-15:
  https://wikisource.org/wiki/Wikisource:Copyright_policy
- Wikimedia robot policy, reviewed 2026-08-15:
  https://wikitech.wikimedia.org/wiki/Robot_policy
- Wikimedia Terms of Use, reviewed 2026-08-15:
  https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use

The multilingual copyright policy says works must be public domain or available
under CC BY-SA 4.0 and that a license template should identify the source work's
status. Contributor-created changes remain subject to Wikimedia attribution and
share-alike requirements. Collection therefore retains page URL, revision ID,
template evidence, and the required attribution in every accepted record.

## Telugu pilot

- Source: https://te.wikisource.org/
- Official local copyright policy:
  https://te.wikisource.org/wiki/వికీసోర్స్:Copyright_policy
- Approved discovery scope: namespace-0 direct members of `Category:PD India`.
- Required item assertion: the rendered page must itself remain in `PD India`.
- Mandatory exclusions: `Non-free DLI`, `Copyrighted`, `Copyvio`, `Fair use`,
  or any configured equivalent.

The local policy explicitly says thousands of `Non-free DLI` books are read-only
and identifies that template. Those pages are not approved for this corpus even
if they also appear through another category. The collector rejects them before
text extraction. This pilot does not approve other Telugu categories.

## Nepali pilot

- Source: https://wikisource.org/wiki/Main_Page/ne
- Approved discovery scope: namespace-0 direct members of `Category:PD-Nepal`.
- Category reviewed 2026-08-15:
  https://wikisource.org/wiki/Category:PD-Nepal
- Required item assertion: the rendered page must itself remain in `PD-Nepal`.
- Mandatory exclusions: `Copyrighted`, `Copyvio`, `Fair use`, or any configured
  equivalent.

At review time the category contained 74 direct pages. This is a small provenance
pilot, not a claim that Wikisource can supply the project's full manual-token
target.

## Operational limits

- One request at a time and at most four requests per second.
- A source-specific page cap is mandatory and may only be raised deliberately.
- Raw responses are immutable and collection resumes through `pull_state.sqlite`.
- A page without its required public-domain category is retained only as a
  rejection audit record and never enters candidate training text.
- Any later terms, category, or filter change requires a new dated snapshot.
