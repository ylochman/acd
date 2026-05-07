try:
    from ._acd import acd_cpp as acd_core
except Exception as e:
    print(
        "Warning: C++ ACD solver not available, using Python implementation which is much slower. To use the C++ solver, please follow the installation instructions in the README."
    )
    from .acd_py import acd_py as acd_core
