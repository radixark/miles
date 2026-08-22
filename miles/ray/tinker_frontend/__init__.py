"""HTTP frontend speaking the official tinker SDK's REST protocol (/api/v1).

Verified against ``tinker==0.24.1`` wheel source and live captured traffic:
an UNMODIFIED SDK pointed at this server (``base_url`` + ``api_key``) drives
training and sampling. The frontend is a thin protocol gateway — every
training verb becomes one operation on the backend ledger, sampling proxies
to the sglang router, and no training semantics live here.
"""
