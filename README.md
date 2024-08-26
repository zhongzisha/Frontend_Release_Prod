comment the lines as the following `/local/NCI_HERE_app/venv/lib/python3.9/site-packages/urllib3/__init__.py`
```
try:
    import ssl
except ImportError:
    pass
else:
    if not ssl.OPENSSL_VERSION.startswith("OpenSSL "):  # Defensive:
        warnings.warn(
            "urllib3 v2 only supports OpenSSL 1.1.1+, currently "
            f"the 'ssl' module is compiled with {ssl.OPENSSL_VERSION!r}. "
            "See: https://github.com/urllib3/urllib3/issues/3020",
            exceptions.NotOpenSSLWarning,
        )
    elif ssl.OPENSSL_VERSION_INFO < (1, 1, 1):  # Defensive:
        # raise ImportError(
        #    "urllib3 v2 only supports OpenSSL 1.1.1+, currently "
        #    f"the 'ssl' module is compiled with {ssl.OPENSSL_VERSION!r}. "
        #    "See: https://github.com/urllib3/urllib3/issues/2168"
        #)
        pass
```