"""TA package.

Making ``ta`` a regular package guarantees that every module is imported
under a single stable name (``ta.src.*``) regardless of the working
directory.  This fixes recurring Numba ``cache=True`` crashes caused by
the same files being compiled/imported under two different module names
(``src.*`` when running from ``ta/`` and ``ta.src.*`` when running from
the repository root): a cache entry written under one name cannot be
unpickled under the other.
"""
