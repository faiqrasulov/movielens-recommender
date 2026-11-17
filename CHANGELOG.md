# Changelog

Notable changes to this project, newest first.

## 0.1.0 (2026-11-17)

First tagged release. It contains everything below.

## 2026-11-16

- Package versions pinned in `requirements.lock`, with a `make install-locked` target.
- README: EASE^R and iALS are statistically tied and are described that way.

## 2026-11-15

- First version: popularity, item-kNN, iALS, BPR and EASE^R from scratch, compared on three split protocols with bootstrap intervals.
- README: the main protocol is called a per-user temporal split; the earlier wording "leakage-free" was wrong and is removed.
