"""
omni_eda
========

Fully automated Exploratory Data Analysis for pandas DataFrames.

Quick start
-----------
>>> from omni_eda import OmniEDA
>>> eda = OmniEDA()
>>> results = eda.run(df)
>>> eda.generate_report("report.html")

or, in one line:

>>> from omni_eda import OmniEDA
>>> OmniEDA("data.csv").generate_report("report.html")
"""

from __future__ import annotations

__version__ = "0.3.0"

from omni_eda.analyzer import OmniEDA
from omni_eda.config import EDAConfig

__all__ = ["OmniEDA", "EDAConfig", "__version__"]
