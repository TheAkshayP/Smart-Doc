"""
Disable Google generative AI telemetry where possible to avoid weird
telemetry errors from some versions of google-generativeai / telemetry hooks.
"""

import os
try:
    import google.generativeai as genai
    # set_options/telemetry may not exist in older/newer versions; guard it.
    try:
        genai.set_options(telemetry_enabled=False)
    except Exception:
        # some versions have genai.configure/other ways; just ignore if not available
        pass
except Exception:
    # If the package is not installed yet when this module is imported, ignore.
    pass
