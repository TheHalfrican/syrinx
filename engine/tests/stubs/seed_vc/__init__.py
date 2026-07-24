"""Stand-in for the GPL ``seed-vc`` package (which lives only in .venv-seedvc).

The worker is a thin adapter over seed-vc's public API, so the protocol can be
tested against a fake that honours the same call shapes — no weights, no GPU,
no GPL code in this repo.
"""
